/*	CornerRounder.cpp

	Corner Rounder - third learning plugin, and the first that is NOT a point
	operation. Two things are new versus Chromatic Aberration / Gradient Map:

	  1. MULTI-PASS. We cannot compute a pixel from its input pixel alone. Pass
	     one builds a signed distance field from the alpha; pass two bands that
	     field once per stroke and composites.

	  2. BUFFER EXPANSION. An outward stroke lives OUTSIDE the layer's bounds.
	     If PreRender doesn't grow the output rect, AE clips the stroke to the
	     layer and you get the classic "my glow is cut off" bug.

	Ported from python-proto/corner_rounder/cr_step1..5.

	Revision History
	Version		Change											Engineer	Date
	=======		======											========	======
	1.0			Initial scaffold from Python prototype		aldai		8/17/2026
*/

#define _CRT_SECURE_NO_WARNINGS	// we build param names with sprintf

#if HAS_CUDA
	#include <cuda_runtime.h>
	// cuda_runtime.h defines these; our header needs its own versions.
	#undef MAJOR_VERSION
	#undef MINOR_VERSION
#endif

#include "CornerRounder.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <thread>
#include <atomic>


/* ---- GPU rendering -------------------------------------------------------
   CR_GPU_RENDER 1: PreRender opts each F32 frame into GPU (JFA path); AE still
   auto-falls back to the CPU SmartRender for 8/16-bit, non-CUDA devices, Mercury
   Software Only, or OOM. GPU render only happens when BOTH the global
   SUPPORTS_GPU_RENDER_F32 flag is advertised AND PreRender sets
   GPU_RENDER_POSSIBLE. Flip to 0 to force everything back onto the CPU path. */
#define CR_GPU_RENDER 1

#if HAS_CUDA
// Defined in CornerRounder_Kernel.cu. No extern "C": nvcc uses the same MSVC
// host compiler (-ccbin), so the mangled names match. The whole corner-round
// pipeline (bbox crop, open/close via JFA per metric, profile blend, coverage +
// preserve-AA + amount, nearest-colour fill) lives in the one launcher.
extern void CR_CornerRound_CUDA(
	const float *src, float *dst, int srcPitch, int dstPitch,
	int W, int H, int offX, int offY, int inW, int inH,
	float thr, float rv, float rc, float feather, float amount,
	float profile, int preserveAA, int pad);
#endif

#if HAS_METAL
// POD mirror of CRGpuParams in CornerRounder_Metal.mm / _Kernel_Metal.h. The host
// fills geometry + params; the .mm fills wx0/wy0/CW/CH after the device bbox.
// LAYOUT MUST MATCH the Metal struct exactly.
struct CRGpuParams {
	int   W, H;
	int   srcPitch, dstPitch;
	int   offX, offY;
	int   inW, inH;
	int   wx0, wy0, CW, CH;
	float thr, feather, amount;
	int   preserveAA, pad;
};

// Defined in CornerRounder_Metal.mm (Objective-C++). extern "C" so the .cpp links
// against unmangled names.
extern "C" bool CR_MetalCompile (void *devicePV, void **outData, char *errBuf, int errLen);
extern "C" void CR_MetalDispose (void *dataPV);
extern "C" bool CR_CornerRound_Metal (void *devicePV, void *queuePV, void *dataPV,
									  void *srcMemPV, void *dstMemPV, CRGpuParams p,
									  float rv, float rc, float profile);
#endif

/* Set once in GlobalSetup. Needed to reach the AEGP stream suites, which are
   the ONLY way to change param visibility at runtime in AE. */
static AEGP_PluginID	S_cr_id = 0L;

/* =========================================================================
   Boilerplate
   ========================================================================= */

static PF_Err
About (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	suites.ANSICallbacksSuite1()->sprintf(
		out_data->return_msg,
		"%s v%d.%d\r%s",
		STR(StrID_Name), MAJOR_VERSION, MINOR_VERSION, STR(StrID_Description));

	return PF_Err_NONE;
}

static PF_Err
GlobalSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err				err = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	out_data->my_version = PF_VERSION(	MAJOR_VERSION, MINOR_VERSION,
										BUG_VERSION, STAGE_VERSION, BUILD_VERSION);

	// Needed for the DynamicStream suite used to show/hide stroke groups.
	ERR(suites.UtilitySuite3()->AEGP_RegisterWithAEGP(NULL, STR(StrID_Name), &S_cr_id));

	// I_EXPAND_BUFFER: the classic path may return more pixels than it got.
	// SEND_UPDATE_PARAMS_UI: needed to sync stroke-group visibility on load.
	out_data->out_flags  =  PF_OutFlag_DEEP_COLOR_AWARE |
							PF_OutFlag_I_EXPAND_BUFFER |
							PF_OutFlag_SEND_UPDATE_PARAMS_UI;

	// SmartFX + float + MFR. Must match PiPL OutFlags_2 exactly.
	out_data->out_flags2 =  PF_OutFlag2_SUPPORTS_SMART_RENDER |
							PF_OutFlag2_FLOAT_COLOR_AWARE |
							PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

	// Advertise CUDA GPU support (F32 only). Premiere negotiates GPU
	// differently, so gate it to AE. This alone makes AE call GPU_DEVICE_SETUP;
	// a frame only renders on GPU if PreRender ALSO opts in (see CR_GPU_RENDER).
	if (in_data->appl_id != 'PrMr') {
		out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}

	return err;
}

/* =========================================================================
   Parameters. A FIXED list (no dynamic groups): Radius, Link, Convex/Concave
   radii, Corner Profile, Edge Softness, Amount, Alpha Threshold, Preserve AA.
   ========================================================================= */

static PF_Err
ParamsSetup (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err				err = PF_Err_NONE;
	PF_ParamDef			def;

	// --- master radius -----------------------------------------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Radius_Param_Name),
							0, CR_RADIUS_MAX, 0, 50, CR_RADIUS_DFLT,
							PF_Precision_TENTHS, 0, 0, RADIUS_DISK_ID);

	// --- link convex+concave to the master radius --------------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX(	STR(StrID_Link_Param_Name), TRUE, 0, LINK_DISK_ID);

	// --- separate convex / concave radii (used when Link is off) -----------
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Convex_Param_Name),
							0, CR_RADIUS_MAX, 0, 50, CR_RADIUS_DFLT,
							PF_Precision_TENTHS, 0, 0, CONVEX_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Concave_Param_Name),
							0, CR_RADIUS_MAX, 0, 50, CR_RADIUS_DFLT,
							PF_Precision_TENTHS, 0, 0, CONCAVE_DISK_ID);

	// --- corner profile: 0 circular .. 100 squircle (Round style only) -----
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Profile_Param_Name),
							0, 100, 0, 100, 0,
							PF_Precision_TENTHS, 0, 0, PROFILE_DISK_ID);

	// --- corner style per side: Round | Bevel | Miter ----------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(	STR(StrID_ConvexStyle_Param_Name),
					3, CR_STYLE_ROUND, STR(StrID_Style_Choices), CONVEX_STYLE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(	STR(StrID_ConcaveStyle_Param_Name),
					3, CR_STYLE_ROUND, STR(StrID_Style_Choices), CONCAVE_STYLE_DISK_ID);

	// --- edge softness (AA feather, px) ------------------------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Feather_Param_Name),
							0, 10, 0, 4, CR_FEATHER_DFLT,
							PF_Precision_HUNDREDTHS, 0, 0, FEATHER_DISK_ID);

	// --- amount: blend source -> rounded -----------------------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Amount_Param_Name),
							0, 100, 0, 100, 100,
							PF_Precision_TENTHS, 0, 0, AMOUNT_DISK_ID);

	// --- alpha threshold for "inside" --------------------------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Threshold_Param_Name),
							0, 100, 0, 100, 50,
							PF_Precision_TENTHS, 0, 0, THRESHOLD_DISK_ID);

	// --- preserve source anti-aliasing on unmoved edges --------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX(	STR(StrID_Preserve_Param_Name), TRUE, 0, PRESERVE_DISK_ID);

	// --- optional matte: scales Amount per pixel (protect / restrict corners) ---
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER(	STR(StrID_Matte_Param_Name), PF_LayerDefault_NONE, MATTE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP(	STR(StrID_MatteChannel_Param_Name),
					2, CR_MATTE_LUMA, STR(StrID_MatteChannel_Choices),
					MATTE_CHANNEL_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX(	STR(StrID_MatteInvert_Param_Name), FALSE, 0, MATTE_INVERT_DISK_ID);

	out_data->num_params = CR_NUM_PARAMS;
	return err;
}

/* =========================================================================
   Distance transforms.

   Three EXACT metrics; the corner slider blends between them. Each is O(N)
   and INDEPENDENT of stroke width - that's the whole point of Step 4.
   ========================================================================= */

#define CR_INF 1e20f

/* The "unreachable" sentinel, AFTER a transform has run.

   The two transform families disagreed on scale: CR_EDT works on SQUARED
   distances and finishes with sqrtf, so an unreachable pixel comes out at
   sqrt(1e20) = 1e10, while CR_Chamfer never square-roots and leaves 1e20.
   CR_CornerField blends those together - and for Concave it EXTRAPOLATES
   along (bevel - round) - so mixing a 1e20 with a 1e10 produces a wildly
   wrong magnitude from two values that were both just meant to say "far".
   Capping every finished field at one sane finite value keeps the blend
   well-behaved and keeps the sentinel from escaping into the composite. */
#define CR_FAR 1.0e7f

static void
CR_ClampField (float *buf, A_long n)
{
	for (A_long i = 0; i < n; i++) {
		float v = buf[i];
		if (!(v >= 0.0f))	v = 0.0f;		// also catches NaN
		if (v > CR_FAR)		v = CR_FAR;
		buf[i] = v;
	}
}

/* Split [0,n) into a few contiguous ranges and run fn(lo,hi) on each, one per
   thread. fn MUST be self-contained: touch only the memory its [lo,hi) owns,
   and allocate any scratch it needs ITSELF so every thread has private scratch.
   Small n, or a single CPU, just runs inline.

   Why this is safe for the FH transform: the Euclidean distance transform is
   SEPARABLE - a 1D pass down every column, then a 1D pass across every row.
   Within a pass the lines are completely independent (column x only ever
   touches column x), so handing a block of columns to each thread needs no
   locks at all. The join between the two passes is the only barrier.

   A note on MFR: we declared SUPPORTS_THREADED_RENDERING, so AE already renders
   several FRAMES at once. This intra-frame threading can therefore oversubscribe
   the CPU. We cap the thread count and only thread large buffers to keep that in
   check; the OS scheduler absorbs the rest. It still pays off on the case that
   feels slow - a single interactive frame (scrubbing, a held still), where MFR
   has only one frame to work with and every core would otherwise sit idle. */
template <class F>
static PF_Err
CR_ParallelRanges (A_long n, F fn)
{
	if (n <= 0) return PF_Err_NONE;

	A_long hw = (A_long)std::thread::hardware_concurrency();
	if (hw < 1) hw = 1;
	A_long T = (hw < 8) ? hw : 8;			// cap so MFR + this stays sane

	// Threads are not free; a small buffer is faster left inline.
	if (n < 256 || T <= 1) { fn(0, n); return PF_Err_NONE; }
	if (T > n) T = n;

	std::atomic<bool> failed(false);
	auto run = [&](A_long lo, A_long hi) {
		// An exception must never cross a thread boundary (that calls
		// std::terminate); catch here and report through the flag instead.
		try { fn(lo, hi); }
		catch (...) { failed.store(true); }
	};

	const A_long chunk = (n + T - 1) / T;
	std::vector<std::thread> pool;
	pool.reserve((size_t)(T - 1));
	for (A_long t = 1; t < T; t++) {
		A_long lo = t * chunk, hi = lo + chunk;
		if (hi > n) hi = n;
		if (lo >= hi) break;
		pool.emplace_back(run, lo, hi);
	}
	run(0, (chunk < n) ? chunk : n);		// main thread takes the first block
	for (auto &th : pool) th.join();

	return failed.load() ? PF_Err_OUT_OF_MEMORY : PF_Err_NONE;
}

/* Felzenszwalb-Huttenlocher 1D squared distance transform: the lower envelope
   of parabolas f[i] + (x-i)^2. Operates on a strided line so the same routine
   serves both columns and rows (the transform is separable). */
static void
CR_DT1D (float *f, A_long n, A_long stride, float *dScratch, A_long *v, float *z)
{
	A_long k = 0;
	v[0] = 0;
	z[0] = -CR_INF;
	z[1] =  CR_INF;

	for (A_long q = 1; q < n; q++) {
		float fq = f[q * stride];
		float s;
		for (;;) {
			float fv = f[v[k] * stride];
			s = ((fq + (float)q * q) - (fv + (float)v[k] * v[k])) /
				(2.0f * (float)q - 2.0f * (float)v[k]);
			if (s <= z[k] && k > 0)	k--;
			else					break;
		}
		k++;
		v[k]	 = q;
		z[k]	 = s;
		z[k + 1] = CR_INF;
	}

	k = 0;
	for (A_long q = 0; q < n; q++) {
		while (z[k + 1] < (float)q) k++;
		float dx = (float)(q - v[k]);
		dScratch[q] = dx * dx + f[v[k] * stride];
	}
	for (A_long q = 0; q < n; q++) f[q * stride] = dScratch[q];
}

/* Exact Euclidean distance to the nearest "source" pixel (src != 0).
   buf must be W*H; on return it holds the distance in pixels. */
static void
CR_EDT (float *buf, A_long W, A_long H)
{
	// Columns are independent -> a block of columns per thread, each with its
	// own scratch of height H.
	CR_ParallelRanges(W, [=](A_long x0, A_long x1) {
		std::vector<float>	scratch(H), z(H + 1);
		std::vector<A_long>	v(H);
		for (A_long x = x0; x < x1; x++)
			CR_DT1D(buf + x, H, W, &scratch[0], &v[0], &z[0]);
	});
	// Then rows, same idea, scratch of width W.
	CR_ParallelRanges(H, [=](A_long y0, A_long y1) {
		std::vector<float>	scratch(W), z(W + 1);
		std::vector<A_long>	v(W);
		for (A_long y = y0; y < y1; y++)
			CR_DT1D(buf + y * W, W, 1, &scratch[0], &v[0], &z[0]);
	});
	// The final sqrt is per-pixel independent too.
	CR_ParallelRanges(W * H, [=](A_long i0, A_long i1) {
		for (A_long i = i0; i < i1; i++) buf[i] = sqrtf(buf[i]);
	});
	CR_ClampField(buf, W * H);
}

/* Same FH transform, but it also reports WHICH pixel was nearest.

   Why we need the feature and not just the distance: a distance field built
   from integer offsets only takes values sqrt(dx^2+dy^2), which in any 1px
   window is a handful of distinct numbers (measured: 3). The smoothstep has
   nothing to ramp across, so the anti-aliasing does nothing - worst of all on
   the chamfer metrics, whose values are pure integers, which is exactly why
   Concave showed no AA at all.
   The fix (cr_step5): the nearest OPAQUE pixel's own alpha says where the true
   contour lies relative to its center, at (a - 0.5) along the outward normal.
   Subtracting that lands every distance on the real sub-pixel contour and took
   the field from 3 to 37 distinct values per pixel in the prototype.
   We compute the feature once with the L2 transform and reuse it for the other
   metrics: the correction is bounded by 0.5px, so borrowing it is a sub-pixel
   approximation (unlike swapping the NORM, which was wrong by up to 73px). */
static void
CR_EDT_Feature (const float *seed, A_long W, A_long H, A_long *featIdx)
{
	// Shared across threads: written to disjoint columns in pass 1, then read
	// in pass 2. The join between the two CR_ParallelRanges calls is the only
	// barrier needed - no locks.
	std::vector<float>	g((size_t)W * H);
	std::vector<A_long>	ys((size_t)W * H);
	float  *gP  = &g[0];
	A_long *ysP = &ys[0];

	// --- pass 1: down each column, remember the source ROW ------------------
	CR_ParallelRanges(W, [=](A_long x0, A_long x1) {
		std::vector<float>	line(H), z(H + 1);
		std::vector<A_long>	v(H);
		for (A_long x = x0; x < x1; x++) {
			for (A_long y = 0; y < H; y++) line[y] = seed[y * W + x];

			A_long k = 0; v[0] = 0; z[0] = -CR_INF; z[1] = CR_INF;
			for (A_long q = 1; q < H; q++) {
				float s;
				for (;;) {
					s = ((line[q] + (float)q * q) - (line[v[k]] + (float)v[k] * v[k])) /
						(2.0f * (float)q - 2.0f * (float)v[k]);
					if (k > 0 && s <= z[k]) k--; else break;
				}
				k++; v[k] = q; z[k] = s; z[k + 1] = CR_INF;
			}
			k = 0;
			for (A_long q = 0; q < H; q++) {
				while (z[k + 1] < (float)q) k++;
				float dy = (float)(q - v[k]);
				gP[q * W + x]  = dy * dy + line[v[k]];
				ysP[q * W + x] = v[k];
			}
		}
	});

	// --- pass 2: across each row, remember the source COLUMN ----------------
	CR_ParallelRanges(H, [=](A_long y0, A_long y1) {
		std::vector<float>	line(W), z(W + 1);
		std::vector<A_long>	v(W);
		for (A_long y = y0; y < y1; y++) {
			for (A_long x = 0; x < W; x++) line[x] = gP[y * W + x];

			A_long k = 0; v[0] = 0; z[0] = -CR_INF; z[1] = CR_INF;
			for (A_long q = 1; q < W; q++) {
				float s;
				for (;;) {
					s = ((line[q] + (float)q * q) - (line[v[k]] + (float)v[k] * v[k])) /
						(2.0f * (float)q - 2.0f * (float)v[k]);
					if (k > 0 && s <= z[k]) k--; else break;
				}
				k++; v[k] = q; z[k] = s; z[k + 1] = CR_INF;
			}
			k = 0;
			for (A_long q = 0; q < W; q++) {
				while (z[k + 1] < (float)q) k++;
				A_long fx = v[k];					// nearest column
				A_long fy = ysP[y * W + fx];		// its row, from pass 1
				if (fx < 0) fx = 0;  if (fx >= W) fx = W - 1;
				if (fy < 0) fy = 0;  if (fy >= H) fy = H - 1;
				featIdx[y * W + q] = fy * W + fx;
			}
		}
	});
}

/* Two-pass chamfer. diagonal=false -> L1 (bevel); true -> L-inf (miter).

   NOT threaded, unlike CR_EDT: a chamfer is a raster scan where each pixel
   reads neighbours that were JUST written (up/left on the forward pass,
   down/right on the backward pass), so rows are not independent and there is
   nothing to hand a thread without a wavefront/antidiagonal rewrite. It is also
   the cheaper transform and only runs for Bevel/Miter, so the Euclidean path -
   the default, and the expensive one - is where threading actually matters. */
static void
CR_Chamfer (float *buf, A_long W, A_long H, bool diagonal)
{
	#define CR_RELAX(ny, nx, cost)											\
		if ((ny) >= 0 && (ny) < H && (nx) >= 0 && (nx) < W) {				\
			float cand = buf[(ny) * W + (nx)] + (cost);						\
			if (cand < buf[y * W + x]) buf[y * W + x] = cand;				\
		}

	for (A_long y = 0; y < H; y++) {
		for (A_long x = 0; x < W; x++) {
			CR_RELAX(y - 1, x, 1.0f);  CR_RELAX(y, x - 1, 1.0f);
			if (diagonal) { CR_RELAX(y - 1, x - 1, 1.0f); CR_RELAX(y - 1, x + 1, 1.0f); }
		}
	}
	for (A_long y = H - 1; y >= 0; y--) {
		for (A_long x = W - 1; x >= 0; x--) {
			CR_RELAX(y + 1, x, 1.0f);  CR_RELAX(y, x + 1, 1.0f);
			if (diagonal) { CR_RELAX(y + 1, x + 1, 1.0f); CR_RELAX(y + 1, x - 1, 1.0f); }
		}
	}
	#undef CR_RELAX
	CR_ClampField(buf, W * H);		// same finite "far" value as CR_EDT
}

static float
CR_Smoothstep (float e0, float e1, float x)
{
	float d = e1 - e0;
	if (d < 1e-6f) return x < e0 ? 0.0f : 1.0f;
	float t = (x - e0) / d;
	if (t < 0.0f) t = 0.0f;
	if (t > 1.0f) t = 1.0f;
	return t * t * (3.0f - 2.0f * t);
}

/* L1 (Manhattan) distance transform, THREADED. Unlike the raster chamfer, the L1
   DT is SEPARABLE: a 1D pass down every column (columns independent) then a 1D
   pass across every row (rows independent), each a forward+backward "+1" sweep.
   Both passes parallelize with CR_ParallelRanges, exactly like CR_EDT - which is
   why Bevel is fast while the raster chamfer (used for L-inf) stays serial.
   Verified bit-exact against scipy's cityblock transform. buf: 0 at source,
   CR_INF elsewhere; on return holds the L1 distance. */
static void
CR_ChamferL1 (float *buf, A_long W, A_long H)
{
	// columns
	CR_ParallelRanges(W, [=](A_long x0, A_long x1) {
		for (A_long x = x0; x < x1; x++) {
			float c = CR_INF;
			for (A_long y = 0; y < H; y++) {
				float *p = &buf[y * W + x];
				c = (*p == 0.0f) ? 0.0f : c + 1.0f;
				if (c < *p) *p = c;
			}
			c = CR_INF;
			for (A_long y = H - 1; y >= 0; y--) {
				float *p = &buf[y * W + x];
				c = (*p == 0.0f) ? 0.0f : c + 1.0f;
				if (c < *p) *p = c;
			}
		}
	});
	// rows
	CR_ParallelRanges(H, [=](A_long y0, A_long y1) {
		for (A_long y = y0; y < y1; y++) {
			float *row = &buf[y * W];
			for (A_long x = 1; x < W; x++)     { float d = row[x - 1] + 1.0f; if (d < row[x]) row[x] = d; }
			for (A_long x = W - 2; x >= 0; x--) { float d = row[x + 1] + 1.0f; if (d < row[x]) row[x] = d; }
		}
	});
	CR_ClampField(buf, W * H);
}

/* Metric codes: 0 = L2 (round), 1 = L1 (bevel), 2 = L-inf (miter). */
static void
CR_TransformMetric (float *buf, A_long W, A_long H, A_long metric)
{
	if (metric == 0)      CR_EDT(buf, W, H);		// Euclidean (threaded)
	else if (metric == 1) CR_ChamferL1(buf, W, H);	// L1 bevel  (threaded)
	else                  CR_Chamfer(buf, W, H, true);	// L-inf miter (serial raster)
}

/* Build the signed rounded field g (+inside) of close(open(S, rv), rc) with a
   possibly DIFFERENT metric for the open (om) and close (cm) passes - which is
   what gives per-side corner styles. Standalone (not the SmartRender lambda) so
   it can run at full res OR on a supersampled buffer. Includes the inscribed-
   circle clamp. inside is a 0/1 mask of size W*H; g is written W*H. */
static void
CR_BuildG (A_long om, A_long cm, float rv, float rc,
		   A_long W, A_long H, const char *inside, float *g)
{
	const A_long N = W * H;
	std::vector<float>	f1((size_t)N), f2((size_t)N);
	std::vector<char>	Oset((size_t)N);
	std::vector<float>	dOutE;
	float	rvE = rv, rcE = rc;
	bool	didOpen = false;

	if (rv >= 0.5f || rc >= 0.5f) {
		for (A_long k = 0; k < N; k++) f1[k] = inside[k] ? CR_INF : 0.0f;
		CR_TransformMetric(&f1[0], W, H, om);			// d_in(S) under open metric
		float maxDin = 0.0f;
		for (A_long k = 0; k < N; k++)
			if (f1[k] < CR_FAR * 0.5f && f1[k] > maxDin) maxDin = f1[k];
		float cap = maxDin - 0.5f; if (cap < 0.0f) cap = 0.0f;
		if (rvE > cap) rvE = cap;
		if (rcE > cap) rcE = cap;
		if (rvE >= 0.5f) {
			for (A_long k = 0; k < N; k++) f2[k] = (f1[k] > rvE) ? 0.0f : CR_INF;
			CR_TransformMetric(&f2[0], W, H, om);		// d_out(E)
			for (A_long k = 0; k < N; k++) Oset[k] = (f2[k] <= rvE) ? 1 : 0;
			dOutE = f2;
			didOpen = true;
		}
	}
	if (!didOpen)
		for (A_long k = 0; k < N; k++) Oset[k] = inside[k];

	if (rcE >= 0.5f) {
		for (A_long k = 0; k < N; k++) f1[k] = Oset[k] ? 0.0f : CR_INF;
		CR_TransformMetric(&f1[0], W, H, cm);			// d_out(O) under close metric
		std::vector<char> Dset((size_t)N);
		for (A_long k = 0; k < N; k++) Dset[k] = (f1[k] <= rcE) ? 1 : 0;
		for (A_long k = 0; k < N; k++) f1[k] = Dset[k] ? CR_INF : 0.0f;
		CR_TransformMetric(&f1[0], W, H, cm);			// d_in(D)
		std::vector<float> f3((size_t)N);
		for (A_long k = 0; k < N; k++) f3[k] = Dset[k] ? 0.0f : CR_INF;
		CR_TransformMetric(&f3[0], W, H, cm);			// d_out(D)
		for (A_long k = 0; k < N; k++) g[k] = (f1[k] - f3[k]) - rcE;
	} else if (didOpen) {
		for (A_long k = 0; k < N; k++) g[k] = rvE - dOutE[k];
	} else {
		for (A_long k = 0; k < N; k++) f2[k] = inside[k] ? 0.0f : CR_INF;
		CR_TransformMetric(&f2[0], W, H, cm);
		for (A_long k = 0; k < N; k++) g[k] = f1[k] - f2[k];
	}
}

/* style popup (CR_STYLE_*) -> metric code (0 L2 / 1 L1 / 2 L-inf). */
static A_long
CR_StyleMetric (A_long style)
{
	if (style == CR_STYLE_BEVEL) return 1;
	if (style == CR_STYLE_MITER) return 2;
	return 0;										// Round (and any unexpected value)
}

/* =========================================================================
   SmartFX phase 1: PRE-RENDER.

   No pixels exist yet. This phase only NEGOTIATES REGIONS:
     - read params (they drive how much room we need)
     - ask for the input we intend to read
     - declare the output rect, GROWN by the furthest outward stroke
   Everything expensive belongs in SmartRender; the distance field in
   particular cannot be built here because there are no pixels to read.
   ========================================================================= */

static PF_Err
PreRender (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_PreRenderExtra	*extra )
{
	PF_Err				err = PF_Err_NONE, err2 = PF_Err_NONE;
	PF_RenderRequest	req = extra->input->output_request;
	PF_CheckoutResult	in_result;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	PF_Handle infoH = suites.HandleSuite1()->host_new_handle(sizeof(CRInfo));
	if (!infoH) return PF_Err_OUT_OF_MEMORY;

	CRInfo *info = reinterpret_cast<CRInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) { suites.HandleSuite1()->host_dispose_handle(infoH); return PF_Err_OUT_OF_MEMORY; }

	extra->output->pre_render_data = infoH;
	AEFX_CLR_STRUCT(*info);

	// Distances are authored at full res; scale them to the current resolution
	// or strokes will be the wrong size in half/quarter previews.
	PF_FpLong dsX = (PF_FpLong)in_data->downsample_x.num / in_data->downsample_x.den;
	PF_FpLong dsY = (PF_FpLong)in_data->downsample_y.num / in_data->downsample_y.den;
	PF_FpLong ds  = (dsX + dsY) * 0.5;

	#define CR_CHECKOUT(idx, var)									\
		PF_ParamDef var;											\
		AEFX_CLR_STRUCT(var);										\
		ERR(PF_CHECKOUT_PARAM(in_data, (idx), in_data->current_time,		\
							  in_data->time_step, in_data->time_scale, &var));

	{
		CR_CHECKOUT(CR_RADIUS,		radius_p);
		CR_CHECKOUT(CR_LINK,		link_p);
		CR_CHECKOUT(CR_CONVEX,		convex_p);
		CR_CHECKOUT(CR_CONCAVE,		concave_p);
		CR_CHECKOUT(CR_PROFILE,		profile_p);
		CR_CHECKOUT(CR_CONVEX_STYLE,	cvxsty_p);
		CR_CHECKOUT(CR_CONCAVE_STYLE,	ccvsty_p);
		CR_CHECKOUT(CR_FEATHER,		feather_p);
		CR_CHECKOUT(CR_AMOUNT,		amount_p);
		CR_CHECKOUT(CR_THRESHOLD,	thresh_p);
		CR_CHECKOUT(CR_PRESERVE_AA,	preserve_p);
		CR_CHECKOUT(CR_MATTE_CHANNEL, mchan_p);
		CR_CHECKOUT(CR_MATTE_INVERT,  minv_p);

		if (!err) {
			PF_FpLong radius  = radius_p.u.fs_d.value;
			A_long    link    = link_p.u.bd.value;
			PF_FpLong convex  = link ? radius : convex_p.u.fs_d.value;
			PF_FpLong concave = link ? radius : concave_p.u.fs_d.value;
			info->convexR	= convex  * ds;
			info->concaveR	= concave * ds;
			info->profile	= profile_p.u.fs_d.value / 100.0;
			info->convexStyle  = cvxsty_p.u.pd.value;
			info->concaveStyle = ccvsty_p.u.pd.value;
			info->feather	= feather_p.u.fs_d.value * ds;
			info->amount	= amount_p.u.fs_d.value / 100.0;
			info->threshold	= thresh_p.u.fs_d.value / 100.0;
			info->preserveAA= preserve_p.u.bd.value;
			info->matteChannel = mchan_p.u.pd.value;
			info->matteInvert  = minv_p.u.bd.value;
			info->maxRadius	= (info->convexR > info->concaveR)
							? info->convexR : info->concaveR;
		}

		ERR2(PF_CHECKIN_PARAM(in_data, &radius_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &link_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &convex_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &concave_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &profile_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &cvxsty_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &ccvsty_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &feather_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &amount_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &thresh_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &preserve_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &mchan_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &minv_p));
	}
	#undef CR_CHECKOUT

	info->hasMatte = FALSE;
	AEFX_CLR_STRUCT(info->matteRect);

	// Remember what AE actually asked for BEFORE we widen the input request.
	const PF_LRect reqRect = extra->input->output_request.rect;

	A_long grow = (A_long)ceil(info->maxRadius + info->feather + 1.0);
	if (grow < 0) grow = 0;

	// We must READ input over the region we intend to WRITE, plus the reach of
	// the distance field, or strokes near the request edge will be wrong.
	req.rect.left	-= grow;
	req.rect.top	-= grow;
	req.rect.right	+= grow;
	req.rect.bottom	+= grow;
	req.preserve_rgb_of_zero_alpha = TRUE;
	req.field = PF_Field_FRAME;

	ERR(extra->cb->checkout_layer(	in_data->effect_ref,
									CR_INPUT, CR_INPUT, &req,
									in_data->current_time, in_data->time_step,
									in_data->time_scale, &in_result));

	// Optional matte layer, checked out over the same region. If a layer is
	// connected its result_rect is non-empty; store it so SmartRender can align
	// and sample it. A matte forces the CPU path (the GPU kernel has no second
	// world), which AE handles as a normal fallback.
	{
		PF_CheckoutResult matte_result;
		AEFX_CLR_STRUCT(matte_result);
		PF_Err mErr = extra->cb->checkout_layer(in_data->effect_ref,
									CR_MATTE, CR_MATTE, &req,
									in_data->current_time, in_data->time_step,
									in_data->time_scale, &matte_result);
		if (!mErr &&
			matte_result.result_rect.right  > matte_result.result_rect.left &&
			matte_result.result_rect.bottom > matte_result.result_rect.top) {
			info->hasMatte  = TRUE;
			info->matteRect = matte_result.result_rect;
		}
	}

	if (!err) {
		// *** THE BUFFER EXPANSION ***
		// Without this the outward strokes are clipped to the layer bounds.
		//
		// The two rects are NOT symmetric, and getting this wrong throws
		// "result rect must not exceed request rect in PF_Cmd_SMART_PRE_RENDER":
		//   max_result_rect - the largest area this effect could EVER cover.
		//                     It MAY exceed the request; this is what grows the
		//                     layer's bounds so the stroke isn't clipped.
		//   result_rect     - what we promise to fill for THIS request. It must
		//                     stay INSIDE output_request.rect. AE only asked for
		//                     that region, so covering more is a contract error
		//                     (it also happens per-tile, so the request is often
		//                     much smaller than the whole frame).
		// An EMPTY input (a text layer on a frame with no visible glyphs, a
		// fully-transparent frame) has nothing to stroke. Growing an empty rect
		// would invent a `grow`-sized box out of nothing and declare we cover
		// it - which is how you get a phantom rectangle the size of the layer
		// bounds appearing when the artwork isn't there. Stay empty instead.
		const bool inputEmpty =
			(in_result.result_rect.right  <= in_result.result_rect.left) ||
			(in_result.result_rect.bottom <= in_result.result_rect.top);

		if (inputEmpty) {
			extra->output->result_rect     = in_result.result_rect;
			extra->output->max_result_rect = in_result.max_result_rect;
			info->inRect  = in_result.result_rect;
			info->outRect = in_result.result_rect;
			info->pad     = 0;
			extra->output->solid = FALSE;
			suites.HandleSuite1()->host_unlock_handle(infoH);
			return err;
		}

		// Bound the working margin to ~half the shape's smaller dimension (an upper
		// bound on the inscribed-circle radius - the most any corner can round).
		// Corner rounding never grows the silhouette past its own bbox, so this is
		// enough for correct AA and the concave dilate, and it keeps a huge Radius
		// on 4K from allocating enormous buffers. (The convex radius is further
		// capped at the true inradius per-frame in SmartRender / the GPU kernel.)
		const A_long bbW = in_result.result_rect.right  - in_result.result_rect.left;
		const A_long bbH = in_result.result_rect.bottom - in_result.result_rect.top;
		const A_long halfMin = ((bbW < bbH) ? bbW : bbH) / 2;
		const A_long featPx  = (A_long)ceil(info->feather) + 2;
		A_long padEff = grow;
		if (padEff > halfMin + featPx) padEff = halfMin + featPx;
		if (padEff < 0) padEff = 0;

		PF_LRect m = in_result.max_result_rect;
		m.left -= padEff;  m.top -= padEff;  m.right += padEff;  m.bottom += padEff;
		extra->output->max_result_rect = m;

		PF_LRect r = in_result.result_rect;
		r.left -= padEff;  r.top -= padEff;  r.right += padEff;  r.bottom += padEff;

		// clip the promise to what was actually asked for
		if (r.left   < reqRect.left)	r.left   = reqRect.left;
		if (r.top    < reqRect.top)		r.top    = reqRect.top;
		if (r.right  > reqRect.right)	r.right  = reqRect.right;
		if (r.bottom > reqRect.bottom)	r.bottom = reqRect.bottom;
		if (r.right  < r.left)			r.right  = r.left;
		if (r.bottom < r.top)			r.bottom = r.top;
		extra->output->result_rect = r;

		// Stash rects so SmartRender can map output pixels to input ones.
		// This MUST be the result_rect we just declared - the output world
		// covers exactly what we promised, NOT the (larger) request rect.
		// Using reqRect here shifted the art down-right by (r.left - reqRect.left),
		// an error that shrank as more strokes grew r outward toward reqRect -
		// which is exactly how the bug presented: "it shifts back with each
		// new stroke".
		info->inRect	= in_result.result_rect;
		info->outRect	= r;
		info->pad		= padEff;	// SmartRender pads its working buffer by this

		extra->output->solid = FALSE;

#if CR_GPU_RENDER
		// Opt THIS frame into GPU rendering. Only takes effect when the global
		// SUPPORTS_GPU_RENDER_F32 flag is also set and AE has a supported (CUDA)
		// device; otherwise AE silently uses PF_Cmd_SMART_RENDER (CPU). While
		// CR_GPU_RENDER is 0 we never set this, so the GPU path stays dormant
		// and every frame renders on the CPU.
		// A matte layer needs a second input world the GPU kernel doesn't take,
		// and Bevel/Miter use the 4x-supersampled CPU path - fall back to the CPU
		// in both cases (AE handles it as a normal fallback).
		const bool styled = (info->convexStyle  != CR_STYLE_ROUND) ||
							 (info->concaveStyle != CR_STYLE_ROUND);
		if (!info->hasMatte && !styled)
			extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
#endif
	}

	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}

/* =========================================================================
   SmartFX phase 2: SMART RENDER.

   Now the pixels exist. This is where the multi-pass work happens:
     pass 1  input alpha -> mask -> the exact metric fields -> signed field
     pass 2  band the field once per stroke, composite, then source over
   ========================================================================= */

/* Pull the alpha of input pixel (u,v) as a float 0..1, any bit depth. */
static float
CR_ReadAlpha (const PF_EffectWorld *w, A_long bitdepth, A_long u, A_long v)
{
	if (u < 0 || v < 0 || u >= w->width || v >= w->height) return 0.0f;
	const char *row = (const char*)w->data + (size_t)v * w->rowbytes;
	switch (bitdepth) {
		case 32: return ((const PF_PixelFloat*)row)[u].alpha;
		case 16: return ((const PF_Pixel16*)row)[u].alpha / (float)PF_MAX_CHAN16;
		default: return ((const PF_Pixel8*)row)[u].alpha  / 255.0f;
	}
}

static void
CR_ReadRGBA (const PF_EffectWorld *w, A_long bitdepth, A_long u, A_long v, float *rgba)
{
	rgba[0] = rgba[1] = rgba[2] = rgba[3] = 0.0f;
	if (u < 0 || v < 0 || u >= w->width || v >= w->height) return;
	const char *row = (const char*)w->data + (size_t)v * w->rowbytes;
	switch (bitdepth) {
		case 32: {
			const PF_PixelFloat *p = &((const PF_PixelFloat*)row)[u];
			rgba[0] = p->red; rgba[1] = p->green; rgba[2] = p->blue; rgba[3] = p->alpha;
		} break;
		case 16: {
			const PF_Pixel16 *p = &((const PF_Pixel16*)row)[u];
			rgba[0] = p->red / (float)PF_MAX_CHAN16;
			rgba[1] = p->green / (float)PF_MAX_CHAN16;
			rgba[2] = p->blue / (float)PF_MAX_CHAN16;
			rgba[3] = p->alpha / (float)PF_MAX_CHAN16;
		} break;
		default: {
			const PF_Pixel8 *p = &((const PF_Pixel8*)row)[u];
			rgba[0] = p->red / 255.0f; rgba[1] = p->green / 255.0f;
			rgba[2] = p->blue / 255.0f; rgba[3] = p->alpha / 255.0f;
		} break;
	}
}

/* Bilinear source alpha at a FRACTIONAL input position (u,v). Out-of-range
   samples read 0. Used to place the supersampled edge sub-pixel-accurately. */
static float
CR_SampleAlphaBilinear (const PF_EffectWorld *w, A_long bitdepth, float u, float v)
{
	float fu = (float)floor(u), fv = (float)floor(v);
	A_long u0 = (A_long)fu, v0 = (A_long)fv;
	float tu = u - fu, tv = v - fv;
	float a00 = CR_ReadAlpha(w, bitdepth, u0,     v0);
	float a10 = CR_ReadAlpha(w, bitdepth, u0 + 1, v0);
	float a01 = CR_ReadAlpha(w, bitdepth, u0,     v0 + 1);
	float a11 = CR_ReadAlpha(w, bitdepth, u0 + 1, v0 + 1);
	float a0 = a00 + (a10 - a00) * tu;
	float a1 = a01 + (a11 - a01) * tu;
	return a0 + (a1 - a0) * tv;
}

/* Matte weight 0..1 at matte-world pixel (u,v). channel = luma or alpha. Returns
   0 outside the matte world (so a small matte rounds only where it paints; use
   Invert for the opposite). Luma is rec709. */
static float
CR_MatteWeight (const PF_EffectWorld *w, A_long bitdepth, A_long u, A_long v,
				A_long channel)
{
	if (u < 0 || v < 0 || u >= w->width || v >= w->height) return 0.0f;
	float rgba[4];
	CR_ReadRGBA(w, bitdepth, u, v, rgba);
	if (channel == CR_MATTE_ALPHA) return rgba[3];
	return 0.2126f * rgba[0] + 0.7152f * rgba[1] + 0.0722f * rgba[2];	// luma
}

static void
CR_WriteRGBA (PF_EffectWorld *w, A_long bitdepth, A_long x, A_long y, const float *rgbaIn)
{
	char *row = (char*)w->data + (size_t)y * w->rowbytes;

	// Sanitize before writing. A NaN slips straight through a naive clamp
	// (every comparison against NaN is false) and an infinity clamps to FULL -
	// which is how a stray non-finite alpha turns into a solid opaque block
	// instead of nothing. Alpha is a coverage value and belongs in [0,1];
	// colour is left unclamped on the float path so HDR still works.
	float rgba[4];
	for (A_long c = 0; c < 4; c++) {
		float v = rgbaIn[c];
		if (!(v == v))	v = 0.0f;				// NaN
		rgba[c] = v;
	}
	if (!(rgba[3] >= 0.0f))	rgba[3] = 0.0f;
	if (rgba[3] > 1.0f)		rgba[3] = 1.0f;

	#define CR_CL(v, mx) ((v) < 0 ? 0 : ((v) > (mx) ? (mx) : (v)))
	switch (bitdepth) {
		case 32: {
			PF_PixelFloat *p = &((PF_PixelFloat*)row)[x];
			p->red = rgba[0]; p->green = rgba[1]; p->blue = rgba[2]; p->alpha = rgba[3];
		} break;
		case 16: {
			PF_Pixel16 *p = &((PF_Pixel16*)row)[x];
			p->red   = (A_u_short)CR_CL(rgba[0] * PF_MAX_CHAN16 + 0.5f, (float)PF_MAX_CHAN16);
			p->green = (A_u_short)CR_CL(rgba[1] * PF_MAX_CHAN16 + 0.5f, (float)PF_MAX_CHAN16);
			p->blue  = (A_u_short)CR_CL(rgba[2] * PF_MAX_CHAN16 + 0.5f, (float)PF_MAX_CHAN16);
			p->alpha = (A_u_short)CR_CL(rgba[3] * PF_MAX_CHAN16 + 0.5f, (float)PF_MAX_CHAN16);
		} break;
		default: {
			PF_Pixel8 *p = &((PF_Pixel8*)row)[x];
			p->red   = (A_u_char)CR_CL(rgba[0] * 255.0f + 0.5f, 255.0f);
			p->green = (A_u_char)CR_CL(rgba[1] * 255.0f + 0.5f, 255.0f);
			p->blue  = (A_u_char)CR_CL(rgba[2] * 255.0f + 0.5f, 255.0f);
			p->alpha = (A_u_char)CR_CL(rgba[3] * 255.0f + 0.5f, 255.0f);
		} break;
	}
	#undef CR_CL
}

static PF_Err
SmartRender (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_SmartRenderExtra	*extra )
{
	// PORT STEP 2: the CIRCULAR corner rounder on the CPU. Reuses the FH-EDT
	// engine (CR_EDT / CR_EDT_Feature) and the crop-to-bbox+pad framework from
	// Buildable Stroke. Pipeline (see python-proto/corner_rounder/cr_step3..5):
	//   threshold alpha -> open (erode,dilate) -> close (dilate,erode) as
	//   distance-field thresholds -> one signed field g (+inside) -> smoothstep
	//   AA -> preserve source AA on unmoved edges -> Amount blend; colour from
	//   the nearest opaque source pixel (edge extend) so added pixels don't
	//   fringe black.  Corner PROFILE is still circular here (metric blend TBD).
	PF_Err				err = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	PF_EffectWorld		*inputP = NULL, *outputP = NULL, *matteP = NULL;

	PF_Handle infoH = reinterpret_cast<PF_Handle>(extra->input->pre_render_data);
	CRInfo *info = reinterpret_cast<CRInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) return PF_Err_BAD_CALLBACK_PARAM;

	ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CR_INPUT, &inputP));
	ERR(extra->cb->checkout_output(in_data->effect_ref, &outputP));
	if (info->hasMatte)
		ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CR_MATTE, &matteP));

	if (!err && inputP && outputP) {
		const A_long	bd	 = extra->input->bitdepth;
		const A_long	W	 = outputP->width;
		const A_long	H	 = outputP->height;
		const A_long	pad	 = info->pad;
		const A_long	PW	 = W + 2 * pad;
		const A_long	PH	 = H + 2 * pad;
		const A_long	offX = info->inRect.left - info->outRect.left;
		const A_long	offY = info->inRect.top  - info->outRect.top;

		#define CR_IN_X(px)	((px) - pad - offX)
		#define CR_IN_Y(py)	((py) - pad - offY)
		#define CR_PASSTHROUGH()										\
			for (A_long y = 0; y < H; y++)							\
				for (A_long x = 0; x < W; x++) {					\
					float s4[4];									\
					CR_ReadRGBA(inputP, bd, x - offX, y - offY, s4);\
					CR_WriteRGBA(outputP, bd, x, y, s4);			\
				}

		const float	thr		= (float)info->threshold;
		const float	rv		= (float)info->convexR;
		const float	rc		= (float)info->concaveR;
		float		feath	= (float)info->feather;
		if (feath < 1e-3f) feath = 1e-3f;
		const float	amount	= (float)info->amount;
		const bool	preserve = (info->preserveAA != 0);

		try {
			// No rounding, or Amount 0 -> exact passthrough (matches cr_step5).
			if ((rv < 0.5f && rc < 0.5f) || amount <= 1e-6f) {
				CR_PASSTHROUGH();
				suites.HandleSuite1()->host_unlock_handle(infoH);
				return err;
			}

			// --- find the shape bbox in padded space -------------------------
			A_long minX = PW, minY = PH, maxX = -1, maxY = -1;
			for (A_long py = 0; py < PH; py++)
				for (A_long px = 0; px < PW; px++)
					if (CR_ReadAlpha(inputP, bd, CR_IN_X(px), CR_IN_Y(py)) >= thr) {
						if (px < minX) minX = px;  if (px > maxX) maxX = px;
						if (py < minY) minY = py;  if (py > maxY) maxY = py;
					}

			// No pixel above threshold -> no contour -> passthrough (empty-shape
			// guard; the same text-animator-opacity case Buildable Stroke hit).
			if (maxX < 0) {
				CR_PASSTHROUGH();
				suites.HandleSuite1()->host_unlock_handle(infoH);
				return err;
			}

			// --- crop everything to (bbox + pad) -----------------------------
			const A_long wx0 = MAX(0, minX - pad);
			const A_long wy0 = MAX(0, minY - pad);
			const A_long wx1 = MIN(PW - 1, maxX + pad);
			const A_long wy1 = MIN(PH - 1, maxY + pad);
			const A_long CW  = wx1 - wx0 + 1;
			const A_long CH  = wy1 - wy0 + 1;
			const A_long N   = CW * CH;

			std::vector<float>	alpha((size_t)N);
			std::vector<char>	inside((size_t)N);
			for (A_long cy = 0; cy < CH; cy++)
				for (A_long cx = 0; cx < CW; cx++) {
					float a = CR_ReadAlpha(inputP, bd, CR_IN_X(cx + wx0), CR_IN_Y(cy + wy0));
					alpha[(size_t)cy * CW + cx]  = a;
					inside[(size_t)cy * CW + cx] = (a >= thr) ? 1 : 0;
				}

				// --- rounded coverage -------------------------------------------
				// Round = the fast distance-field pipeline (+ optional Profile
				// squircle blend toward L-inf). Bevel/Miter use integer chamfer
				// metrics whose edges only anti-alias when SUPERSAMPLED, so those
				// run the same open/close chain on a 4x buffer and area-downsample
				// the rounded set. Convex style drives the OPEN metric, concave the
				// CLOSE metric (per-side styles). CPU-only (GPU falls back).
				const A_long cvxM = CR_StyleMetric(info->convexStyle);
				const A_long ccvM = CR_StyleMetric(info->concaveStyle);
				const bool   styled = (info->convexStyle  != CR_STYLE_ROUND) ||
									   (info->concaveStyle != CR_STYLE_ROUND);
				const bool   usePreserve = preserve && !styled;

				std::vector<float>	f1((size_t)N);		// scratch (reused for colour)
				std::vector<float>	cov((size_t)N);

				if (!styled) {
					std::vector<float> g((size_t)N);
					CR_BuildG(0, 0, rv, rc, CW, CH, &inside[0], &g[0]);		// L2 round
					const float profT = (info->profile < 0.0) ? 0.0f :
										 (info->profile > 1.0) ? 1.0f : (float)info->profile;
					if (profT > 1e-4f) {									// blend toward square
						std::vector<float> gSq((size_t)N);
						CR_BuildG(2, 2, rv, rc, CW, CH, &inside[0], &gSq[0]);
						for (A_long k = 0; k < N; k++)
							g[k] = (1.0f - profT) * g[k] + profT * gSq[k];
					}
					for (A_long k = 0; k < N; k++) {
						float c = CR_Smoothstep(-feath, feath, g[k]);
						if (usePreserve && g[k] <= 1.0f && g[k] >= -1.0f &&
							alpha[k] > 0.0f && alpha[k] < 1.0f)
							c = alpha[k];					// unmoved edge: keep source AA
						cov[k] = c;
					}
				} else {
					// 2x supersample: identical AA to 4x on bevel/miter (validated in
					// the prototype) but a quarter of the pixels/work.
					const A_long ss  = 2;
					const A_long CWs = CW * ss, CHs = CH * ss;
					const float  iox = (float)(wx0 - pad - offX);	// input coord of cropped x=0
					const float  ioy = (float)(wy0 - pad - offY);
					std::vector<char> inSS((size_t)CWs * CHs);
					for (A_long Y = 0; Y < CHs; Y++)
						for (A_long X = 0; X < CWs; X++) {
							float u = iox + ((float)X + 0.5f) / ss - 0.5f;
							float v = ioy + ((float)Y + 0.5f) / ss - 0.5f;
							inSS[(size_t)Y * CWs + X] =
								(CR_SampleAlphaBilinear(inputP, bd, u, v) >= thr) ? 1 : 0;
						}
					std::vector<float> gSS((size_t)CWs * CHs);
					CR_BuildG(cvxM, ccvM, rv * ss, rc * ss, CWs, CHs, &inSS[0], &gSS[0]);
					const float inv = 1.0f / (float)(ss * ss);
					for (A_long cy = 0; cy < CH; cy++)
						for (A_long cx = 0; cx < CW; cx++) {
							float acc = 0.0f;
							for (A_long dy = 0; dy < ss; dy++) {
								const float *row = &gSS[(size_t)(cy * ss + dy) * CWs + cx * ss];
								for (A_long dx = 0; dx < ss; dx++)
									if (row[dx] > 0.0f) acc += 1.0f;
							}
							cov[(size_t)cy * CW + cx] = acc * inv;
						}
				}

				// --- (matte-scaled) Amount blend ---------------------------------
				// A matte layer scales Amount per pixel: 0 -> keep the original
				// (protect that corner), 1 -> full rounding. Invert flips it.
				const bool	 useMatte = (info->hasMatte && matteP != NULL);
				const A_long mChan   = info->matteChannel;
				const bool	 mInv    = (info->matteInvert != 0);
				const A_long mOffX   = info->outRect.left - info->matteRect.left;
				const A_long mOffY   = info->outRect.top  - info->matteRect.top;

				std::vector<float>	outA((size_t)N);
				for (A_long k = 0; k < N; k++) {
					float amt = amount;
					if (useMatte) {
						const A_long cx = k % CW, cy = k / CW;
						const A_long ox = cx + wx0 - pad, oy = cy + wy0 - pad;
						float w = CR_MatteWeight(matteP, bd, ox + mOffX, oy + mOffY, mChan);
						if (mInv) w = 1.0f - w;
						amt *= (w < 0.0f) ? 0.0f : (w > 1.0f ? 1.0f : w);
					}
					outA[k] = (1.0f - amt) * alpha[k] + amt * cov[k];
				}

			// --- colour: nearest opaque source pixel (edge extend) -----------
			std::vector<A_long> feat((size_t)N);
			for (A_long k = 0; k < N; k++) f1[k] = inside[k] ? 0.0f : CR_INF;
			CR_EDT_Feature(&f1[0], CW, CH, &feat[0]);

			// --- write the output --------------------------------------------
			for (A_long y = 0; y < H; y++) {
				for (A_long x = 0; x < W; x++) {
					const A_long cx = x + pad - wx0;
					const A_long cy = y + pad - wy0;
					if (cx >= 0 && cx < CW && cy >= 0 && cy < CH) {
						const A_long k   = cy * CW + cx;
						const A_long kf  = feat[k];
						const A_long cxf = kf % CW, cyf = kf / CW;
						float rgba[4];
						CR_ReadRGBA(inputP, bd,
									CR_IN_X(cxf + wx0), CR_IN_Y(cyf + wy0), rgba);
						rgba[3] = outA[k];			// straight rgb + new coverage
						CR_WriteRGBA(outputP, bd, x, y, rgba);
					} else {
						float s4[4];				// outside work rect: source through
						CR_ReadRGBA(inputP, bd, x - offX, y - offY, s4);
						CR_WriteRGBA(outputP, bd, x, y, s4);
					}
				}
			}
		}
		catch (...) {
			suites.HandleSuite1()->host_unlock_handle(infoH);
			return PF_Err_OUT_OF_MEMORY;
		}

		#undef CR_IN_X
		#undef CR_IN_Y
		#undef CR_PASSTHROUGH
	}

	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}

static PF_Err
GPUDeviceSetup (
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_GPUDeviceSetupExtra	*extra )
{
	PF_Err err = PF_Err_NONE;

	// CUDA (Windows): kernel is statically linked, nothing to build here - just
	// claim F32 support. Any framework we don't handle is left on the CPU path.
	if (extra->input->what_gpu == PF_GPU_Framework_CUDA) {
		out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#if HAS_METAL
	// Metal (macOS): compile the MSL library and build the pipeline states once
	// per device, stash them in gpu_data. On a compile error, surface Metal's
	// own message in AE's error dialog so it can be read directly.
	else if (extra->input->what_gpu == PF_GPU_Framework_METAL) {
		PF_GPUDeviceSuite1 *gpu = NULL;
		if (in_data->pica_basicP->AcquireSuite(kPFGPUDeviceSuite,
				kPFGPUDeviceSuiteVersion1, (const void**)&gpu) || !gpu)
			return PF_Err_BAD_CALLBACK_PARAM;

		PF_GPUDeviceInfo info;
		AEFX_CLR_STRUCT(info);
		PF_Err e = gpu->GetDeviceInfo(in_data->effect_ref,
					extra->input->device_index, &info);
		in_data->pica_basicP->ReleaseSuite(kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
		if (e) return e;

		void *metalData = NULL;
		char  errBuf[512] = {0};
		if (!CR_MetalCompile(info.devicePV, &metalData, errBuf, sizeof(errBuf))) {
			PF_STRCPY(out_data->return_msg, "CornerRounder Metal build failed: ");
			strncat(out_data->return_msg, errBuf,
					sizeof(out_data->return_msg) - strlen(out_data->return_msg) - 1);
			out_data->out_flags |= PF_OutFlag_DISPLAY_ERROR_MESSAGE;
			return PF_Err_INTERNAL_STRUCT_DAMAGED;
		}
		extra->output->gpu_data = metalData;			// opaque; freed in Setdown
		out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
	return err;
}

static PF_Err
GPUDeviceSetdown (
	PF_InData					*in_data,
	PF_OutData					*out_data,
	PF_GPUDeviceSetdownExtra	*extra )
{
#if HAS_METAL
	// Release the Metal pipeline states built in GPUDeviceSetup.
	if (extra->input->what_gpu == PF_GPU_Framework_METAL && extra->input->gpu_data) {
		CR_MetalDispose(const_cast<void*>(extra->input->gpu_data));
	}
#endif
	// CUDA kernel is statically linked; nothing device-specific to release.
	return PF_Err_NONE;
}

#if HAS_CUDA || HAS_METAL
static PF_Err
SmartRenderGPU (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_SmartRenderExtra	*extra )
{
	PF_Err				err		= PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	PF_EffectWorld		*inputP = NULL, *outputP = NULL;

	// pre_render_data is the PF_Handle allocated in PreRender - it MUST be locked
	// to get the CRInfo pointer (treating the handle as the pointer gives garbage
	// rects and the whole layer renders wrong; this was the BS "layer disappears
	// on CUDA" bug).
	PF_Handle infoH = reinterpret_cast<PF_Handle>(extra->input->pre_render_data);
	CRInfo *info = reinterpret_cast<CRInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) return PF_Err_BAD_CALLBACK_PARAM;

	// GPU device suite comes straight off the PICA basic suite.
	PF_GPUDeviceSuite1 *gpu = NULL;
	err = in_data->pica_basicP->AcquireSuite(
			kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, (const void**)&gpu);
	if (err || !gpu) {
		suites.HandleSuite1()->host_unlock_handle(infoH);
		return err ? err : PF_Err_BAD_CALLBACK_PARAM;
	}

	ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CR_INPUT, &inputP));
	ERR(extra->cb->checkout_output(in_data->effect_ref, &outputP));

	if (!err && inputP && outputP) {
		void *src_mem = NULL, *dst_mem = NULL;
		ERR(gpu->GetGPUWorldData(in_data->effect_ref, inputP,  &src_mem));
		ERR(gpu->GetGPUWorldData(in_data->effect_ref, outputP, &dst_mem));

		if (!err) {
			const int bytesPerPixel = 16;			// GPU_BGRA128 = 4 x float32
			int srcPitch = inputP->rowbytes  / bytesPerPixel;
			int dstPitch = outputP->rowbytes / bytesPerPixel;
			int offX = info->inRect.left - info->outRect.left;
			int offY = info->inRect.top  - info->outRect.top;

#if HAS_CUDA
			if (extra->input->what_gpu == PF_GPU_Framework_CUDA) {
				CR_CornerRound_CUDA(
					(const float*)src_mem, (float*)dst_mem,
					srcPitch, dstPitch,
					outputP->width, outputP->height,
					offX, offY, inputP->width, inputP->height,
					(float)info->threshold, (float)info->convexR, (float)info->concaveR,
					(float)info->feather, (float)info->amount, (float)info->profile,
					(int)info->preserveAA, (int)info->pad);

				if (cudaPeekAtLastError() != cudaSuccess)
					err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}
#endif
#if HAS_METAL
			if (extra->input->what_gpu == PF_GPU_Framework_METAL) {
				// Metal needs the device + command queue (CUDA used the default
				// context). gpu_data holds the pipeline states built in setup.
				PF_GPUDeviceInfo devInfo;
				AEFX_CLR_STRUCT(devInfo);
				ERR(gpu->GetDeviceInfo(in_data->effect_ref,
						extra->input->device_index, &devInfo));
				if (!err) {
					CRGpuParams p;
					AEFX_CLR_STRUCT(p);
					p.W          = outputP->width;
					p.H          = outputP->height;
					p.srcPitch   = srcPitch;
					p.dstPitch   = dstPitch;
					p.offX       = offX;
					p.offY       = offY;
					p.inW        = inputP->width;
					p.inH        = inputP->height;
					p.thr        = (float)info->threshold;
					p.feather    = (float)info->feather;
					p.amount     = (float)info->amount;
					p.preserveAA = (int)info->preserveAA;
					p.pad        = (int)info->pad;
					// wx0/wy0/CW/CH are filled by CR_CornerRound_Metal after bbox.

					if (!CR_CornerRound_Metal(devInfo.devicePV, devInfo.command_queuePV,
							const_cast<void*>(extra->input->gpu_data),
							src_mem, dst_mem, p,
							(float)info->convexR, (float)info->concaveR,
							(float)info->profile))
						err = PF_Err_INTERNAL_STRUCT_DAMAGED;
				}
			}
#endif
		}
	}

	in_data->pica_basicP->ReleaseSuite(kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}
#endif // HAS_CUDA || HAS_METAL

/* Classic (non-Smart) render - simple passthrough. */
static PF_Err
Render (
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err				err = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	ERR(PF_COPY(&params[CR_INPUT]->u.ld, output, NULL, NULL));
	return err;
}

/* =========================================================================
   Registration + entry point.
   ========================================================================= */

extern "C" DllExport
PF_Err PluginDataEntryFunction2(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB2 inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT_EXT2(
		inPtr,
		inPluginDataCallBackPtr,
		"Corner Rounder",			// Name
		"aldai CornerRounder",	// Match Name
		"Learning",					// Category
		AE_RESERVED_INFO,			// Reserved Info
		"EffectMain",				// Entry point
		"https://www.adobe.com");	// support URL

	return result;
}

PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;

	try {
		switch (cmd) {
			case PF_Cmd_ABOUT:
				err = About(in_data, out_data, params, output);
				break;
			case PF_Cmd_GLOBAL_SETUP:
				err = GlobalSetup(in_data, out_data, params, output);
				break;
			case PF_Cmd_PARAMS_SETUP:
				err = ParamsSetup(in_data, out_data, params, output);
				break;
			case PF_Cmd_RENDER:					// classic path (Premiere / legacy)
				err = Render(in_data, out_data, params, output);
				break;
			case PF_Cmd_SMART_PRE_RENDER:		// SmartFX phase 1: regions
				err = PreRender(in_data, out_data, reinterpret_cast<PF_PreRenderExtra*>(extra));
				break;
			case PF_Cmd_SMART_RENDER:			// SmartFX phase 2: pixels (CPU)
				err = SmartRender(in_data, out_data, reinterpret_cast<PF_SmartRenderExtra*>(extra));
				break;
			case PF_Cmd_GPU_DEVICE_SETUP:		// per GPU device AE exposes
				err = GPUDeviceSetup(in_data, out_data,
						reinterpret_cast<PF_GPUDeviceSetupExtra*>(extra));
				break;
			case PF_Cmd_GPU_DEVICE_SETDOWN:
				err = GPUDeviceSetdown(in_data, out_data,
						reinterpret_cast<PF_GPUDeviceSetdownExtra*>(extra));
				break;
#if HAS_CUDA || HAS_METAL
			case PF_Cmd_SMART_RENDER_GPU:		// SmartFX phase 2: pixels (GPU)
				err = SmartRenderGPU(in_data, out_data,
						reinterpret_cast<PF_SmartRenderExtra*>(extra));
				break;
#endif
		}
	}
	catch(PF_Err &thrown_err){
		err = thrown_err;
	}
	return err;
}
