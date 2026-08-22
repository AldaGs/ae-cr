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
   Flip CR_GPU_RENDER to 1 once the real kernels exist. At 0 the whole GPU path
   is COMPILED and LINKED and AE's GPU_DEVICE_SETUP runs (proving the toolchain
   + device plumbing), but PreRender does NOT opt this frame into GPU, so every
   frame still renders on the CPU - output stays correct while we build. See the
   SUPPORTS_GPU_RENDER_F32 note: GPU render only happens when BOTH the global
   flag is advertised AND PreRender sets GPU_RENDER_POSSIBLE. */
#define CR_GPU_RENDER 1		// STAGE 5: multi-stroke composite, bbox-cropped

#if HAS_CUDA
// POD mirror of the kernel's CRGpuStroke - the LAYOUT MUST MATCH the struct in
// CornerRounder_Kernel.cu exactly (same fields, same order). Kept as a plain
// struct here so the .cpp doesn't have to pull CUDA headers into the AE build.
struct CRGpuStroke {
	float	lo, hi, corner, opacity;
	int		side, order, fill;
	float	colR, colG, colB;
	float	col2R, col2G, col2B;
	float	gAX, gAY, gdx, gdy, ginv;
};

// Defined in CornerRounder_Kernel.cu. No extern "C": nvcc uses the same MSVC
// host compiler (-ccbin), so the mangled names match.
extern void CR_Composite_CUDA(
	const float *src, float *dst, int srcPitch, int dstPitch,
	int W, int H, int offX, int offY, int inW, int inH,
	float thr, float feather, int subpixel, int pad,
	const CRGpuStroke *hostStrokes, int count,
	int needM, int needB, int needInside);
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
   Parameters.

   CR_MAX_STROKES groups are declared here, once. "Add Stroke" does not create
   params (AE forbids that) - it bumps a hidden count and un-hides the next
   group. See the header for why.
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
	A_char				nameAC[64];
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	// --- stroke count --------------------------------------------------------
	// A param, not sequence data: it persists + undoes for free and keeps the
	// effect thread-safe (Supervisor loses MFR by writing sequence data here).
	// Left VISIBLE on purpose: the buttons drive it, but dragging it directly
	// is often faster, and it makes the whole mechanism debuggable.
	AEFX_CLR_STRUCT(def);
	PF_ADD_SLIDER(	STR(StrID_Count_Param_Name),
					1, CR_MAX_STROKES, 1, CR_MAX_STROKES,
					1,							// default: 1 stroke
					COUNT_DISK_ID);

	// --- Add / Remove buttons ----------------------------------------------
	// SUPERVISE makes AE send PF_Cmd_USER_CHANGED_PARAM when they're clicked.
	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY;
	PF_ADD_BUTTON(	STR(StrID_Add_Param_Name),
					STR(StrID_Add_Param_Name),
					0, PF_ParamFlag_SUPERVISE, ADD_DISK_ID);

	AEFX_CLR_STRUCT(def);
	def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY;
	PF_ADD_BUTTON(	STR(StrID_Remove_Param_Name),
					STR(StrID_Remove_Param_Name),
					0, PF_ParamFlag_SUPERVISE, REMOVE_DISK_ID);

	// --- global edge controls ----------------------------------------------
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Threshold_Param_Name),
							0, 100, 0, 100, 50,
							PF_Precision_TENTHS, 0, 0, THRESHOLD_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Feather_Param_Name),
							0, 10, 0, 4, CR_FEATHER_DFLT,
							PF_Precision_HUNDREDTHS, 0, 0, FEATHER_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOXX(	STR(StrID_Subpixel_Param_Name), TRUE, 0, SUBPIXEL_DISK_ID);

	// --- the stroke groups --------------------------------------------------
	for (A_long i = 0; i < CR_MAX_STROKES; i++) {

		sprintf(nameAC, "Stroke %d", (int)(i + 1));

		// NOTE: every group is created VISIBLE. PF_PUI_INVISIBLE would hide
		// them here, but AE cannot TOGGLE that flag later - visibility at
		// runtime goes through the AEGP stream suites (see CR_SetGroupVisible).
		// Mixing the two mechanisms just fights itself, so we use streams only;
		// PF_Cmd_UPDATE_PARAMS_UI hides the extras as soon as the UI appears.
		AEFX_CLR_STRUCT(def);
		PF_ADD_TOPIC(nameAC, CR_DISK_ID(i, CRS_TOPIC));

		AEFX_CLR_STRUCT(def);
		PF_ADD_FLOAT_SLIDERX(	STR(StrID_Gap_Param_Name),
								0, CR_DIST_MAX, 0, 100, 0,
								PF_Precision_TENTHS, 0, 0, CR_DISK_ID(i, CRS_GAP));

		AEFX_CLR_STRUCT(def);
		PF_ADD_FLOAT_SLIDERX(	STR(StrID_Width_Param_Name),
								0, CR_DIST_MAX, 0, 100, CR_WIDTH_DFLT,
								PF_Precision_TENTHS, 0, 0, CR_DISK_ID(i, CRS_WIDTH));

		AEFX_CLR_STRUCT(def);
		PF_ADD_POPUP(	STR(StrID_Side_Param_Name),
						3, CR_SIDE_OUTER, STR(StrID_Side_Choices),
						CR_DISK_ID(i, CRS_SIDE));

		AEFX_CLR_STRUCT(def);
		PF_ADD_POPUP(	STR(StrID_Corner_Param_Name),
						4, CR_CORNER_ROUND, STR(StrID_Corner_Choices),
						CR_DISK_ID(i, CRS_CORNER));

		AEFX_CLR_STRUCT(def);
		PF_ADD_COLOR(	STR(StrID_Color_Param_Name),
						(i % 2) ? 20 : 230, 30, (i % 2) ? 230 : 40,
						CR_DISK_ID(i, CRS_COLOR));

		// SUPERVISE so toggling Fill fires PF_Cmd_USER_CHANGED_PARAM, which is
		// where we show/hide this stroke's gradient controls.
		AEFX_CLR_STRUCT(def);
		def.flags = PF_ParamFlag_SUPERVISE;
		PF_ADD_POPUP(	STR(StrID_Fill_Param_Name),
						2, CR_FILL_SOLID, STR(StrID_Fill_Choices),
						CR_DISK_ID(i, CRS_FILL));

		AEFX_CLR_STRUCT(def);
		PF_ADD_COLOR(	STR(StrID_Color2_Param_Name),
						(i % 2) ? 230 : 20, 200, (i % 2) ? 40 : 230,
						CR_DISK_ID(i, CRS_COLOR2));

		// Default endpoints: a horizontal ramp across the middle of the layer.
		// PF_ADD_POINT defaults are PERCENTAGES of the layer.
		AEFX_CLR_STRUCT(def);
		PF_ADD_POINT(	STR(StrID_GStart_Param_Name),
						25, 50, 0, CR_DISK_ID(i, CRS_GSTART));

		AEFX_CLR_STRUCT(def);
		PF_ADD_POINT(	STR(StrID_GEnd_Param_Name),
						75, 50, 0, CR_DISK_ID(i, CRS_GEND));

		AEFX_CLR_STRUCT(def);
		PF_ADD_FLOAT_SLIDERX(	STR(StrID_Opacity_Param_Name),
								0, 100, 0, 100, 100,
								PF_Precision_TENTHS, 0, 0, CR_DISK_ID(i, CRS_OPACITY));

		AEFX_CLR_STRUCT(def);
		PF_ADD_POPUP(	STR(StrID_Order_Param_Name),
						2, CR_ORDER_BEHIND, STR(StrID_Order_Choices),
						CR_DISK_ID(i, CRS_ORDER));

		AEFX_CLR_STRUCT(def);
		PF_END_TOPIC(CR_DISK_ID(i, CRS_TOPIC_END));
	}

	out_data->num_params = CR_NUM_PARAMS;
	return err;
}

/* =========================================================================
   Dynamic show/hide of the stroke groups.
   ========================================================================= */

/* Show or hide one stroke group.

   THE GOTCHA THAT COST US A BUILD: in After Effects, PF_PUI_INVISIBLE is only
   honored when the param is CREATED (PF_Cmd_PARAMS_SETUP). Flipping it later
   through PF_UpdateParamUI does nothing - which is why "Add Stroke" appeared
   to do nothing at all. Runtime visibility in AE goes through the AEGP
   DynamicStream suite instead, exactly as SDK sample UI/Supervisor does it:
       "Changing visibility of params in AE is handled through stream suites"
   Hiding the GROUP_START stream hides the whole group with it.

   `gradVisible` is a SECOND, independent gate for the three gradient-only
   fields (end color + the two endpoints). They show only when the group is
   visible AND the stroke's Fill is Linear Gradient, so a Solid stroke isn't
   cluttered with controls that do nothing. */
static PF_Err
CR_SetGroupVisible (
	PF_InData	*in_data,
	A_long		strokeIndex,
	bool		visible,
	bool		gradVisible )
{
	PF_Err				err = PF_Err_NONE, err2 = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	AEGP_EffectRefH		meH			= NULL;
	AEGP_StreamRefH		streamH		= NULL;

	ERR(suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(
			S_cr_id, in_data->effect_ref, &meH));

	// The topic carries the whole group; the children follow it.
	for (A_long f = CRS_TOPIC; f <= CRS_ORDER && !err; f++) {
		bool isGradField = (f == CRS_COLOR2 || f == CRS_GSTART || f == CRS_GEND);
		bool fieldVis    = isGradField ? (visible && gradVisible) : visible;
		A_Boolean hideB  = fieldVis ? FALSE : TRUE;

		streamH = NULL;
		ERR(suites.StreamSuite2()->AEGP_GetNewEffectStreamByIndex(
				S_cr_id, meH, CR_P(strokeIndex, f), &streamH));
		if (!err && streamH) {
			ERR(suites.DynamicStreamSuite2()->AEGP_SetDynamicStreamFlag(
					streamH, AEGP_DynStreamFlag_HIDDEN, FALSE, hideB));
			ERR2(suites.StreamSuite2()->AEGP_DisposeStream(streamH));
		}
	}

	if (meH) ERR2(suites.EffectSuite2()->AEGP_DisposeEffect(meH));
	return err;
}

/* Make the UI match the stored count. Called on load (UPDATE_PARAMS_UI) and
   after a button click (USER_CHANGED_PARAM). */
static PF_Err
CR_SyncGroups (
	PF_InData	*in_data,
	PF_ParamDef	*params[] )
{
	PF_Err	err		= PF_Err_NONE;
	A_long	count	= params[CR_COUNT]->u.sd.value;

	// Premiere has no stream suites; it uses the PF_PUI path and simply shows
	// every group. Bail rather than error out there.
	if (in_data->appl_id == 'PrMr') return PF_Err_NONE;

	if (count < 1)					count = 1;
	if (count > CR_MAX_STROKES)		count = CR_MAX_STROKES;

	for (A_long i = 0; i < CR_MAX_STROKES && !err; i++) {
		bool groupVis = (i < count);
		bool gradVis  = (params[CR_P(i, CRS_FILL)]->u.pd.value == CR_FILL_LINEAR);
		ERR(CR_SetGroupVisible(in_data, i, groupVis, gradVis));
	}

	// Grey out the buttons at the ends of the range. The count is already
	// clamped in UserChangedParam (so there was never a crash to prevent -
	// CR_MAX_STROKES groups always exist), but a dead-looking button is much
	// clearer than one that silently does nothing.
	//
	// TWO GOTCHAS, both learned the hard way:
	//  1. Unlike PF_PUI_INVISIBLE, PF_PUI_DISABLED *can* be toggled at runtime
	//     through PF_UpdateParamUI - no stream suite needed.
	//  2. A ParamDef straight out of PF_CHECKOUT_PARAM is NOT accepted as-is:
	//     AE threw "UpdateParamUI() passed ParamDef of wrong type". You have to
	//     re-assert param_type (and, for a button, its name pointer) on the
	//     copy before handing it back - exactly what Supervisor does before
	//     each of its PF_UpdateParamUI calls.
	// This block is COSMETIC, so it deliberately swallows its own errors: a
	// greyed-out button is never worth surfacing an error dialog to the user.
	{
		AEGP_SuiteHandler	suites(in_data->pica_basicP);
		struct { A_long idx; StrIDType str; bool disable; } btns[2] = {
			{ CR_ADD,    StrID_Add_Param_Name,    count >= CR_MAX_STROKES },
			{ CR_REMOVE, StrID_Remove_Param_Name, count <= 1 }
		};
		for (A_long b = 0; b < 2; b++) {
			PF_ParamDef p;
			AEFX_CLR_STRUCT(p);
			if (PF_CHECKOUT_PARAM(in_data, btns[b].idx, in_data->current_time,
								  in_data->time_step, in_data->time_scale,
								  &p) == PF_Err_NONE) {
				p.param_type = PF_Param_BUTTON;
				p.u.button_d.u.PF_DEF_NAMESPTR = STR(btns[b].str);
				if (btns[b].disable)	p.ui_flags |=  PF_PUI_DISABLED;
				else					p.ui_flags &= ~PF_PUI_DISABLED;

				(void)suites.ParamUtilsSuite3()->PF_UpdateParamUI(
						in_data->effect_ref, btns[b].idx, &p);
				(void)PF_CHECKIN_PARAM(in_data, &p);
			}
		}
	}
	return err;
}

static PF_Err
UserChangedParam (
	PF_InData					*in_data,
	PF_OutData					*out_data,
	PF_ParamDef					*params[],
	const PF_UserChangedParamExtra	*which )
{
	PF_Err	err		= PF_Err_NONE;
	A_long	count	= params[CR_COUNT]->u.sd.value;
	A_long	idx		= which->param_index;
	bool	countChanged = false;

	if (idx == CR_ADD) {
		if (count < CR_MAX_STROKES) count++;
		countChanged = true;
	} else if (idx == CR_REMOVE) {
		if (count > 1) count--;
		countChanged = true;
	} else {
		// The only other supervised param is a stroke's Fill popup, whose change
		// flips the gradient controls on/off. Anything else needs no UI work.
		A_long rel = idx - CR_STROKE_BASE;
		bool isFill = (rel >= 0) && (rel % CRS_PER_STROKE == CRS_FILL);
		if (!isFill) return PF_Err_NONE;
	}

	if (countChanged) {
		// Write the new count back into the hidden param so it persists.
		params[CR_COUNT]->u.sd.value = count;
		params[CR_COUNT]->uu.change_flags = PF_ChangeFlag_CHANGED_VALUE;
	}

	ERR(CR_SyncGroups(in_data, params));

	// Only a count change alters the RENDER; a Fill toggle already re-renders on
	// its own (its value feeds PreRender), and its visibility work is UI-only.
	if (!err && countChanged) out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
	return err;
}

/* =========================================================================
   Reading + resolving parameters.
   ========================================================================= */

static PF_FpLong
CR_CornerValue (A_long popup)
{
	switch (popup) {
		case CR_CORNER_MITER:	return 0.0;
		case CR_CORNER_BEVEL:	return 2.0;
		case CR_CORNER_CONCAVE:	return 3.0;
		case CR_CORNER_ROUND:
		default:				return 1.0;
	}
}

/* Turn the user's RELATIVE (gap, width) into ABSOLUTE bands.

   This is cr_step5's resolve_stack(). The distance math always measures from
   the ORIGINAL shape - there is one field and these are absolute distances on
   it. The relative model exists purely so inserting or reordering a stroke
   doesn't force the user to renumber every offset after it.
   A separate cursor per side: outer strokes stack outward, inner inward. */
static void
CR_ResolveStack (CRInfo *info)
{
	PF_FpLong cursorOuter = 0.0, cursorInner = 0.0;
	info->maxOuter = 0.0;

	for (A_long i = 0; i < info->count; i++) {
		CRStroke *s = &info->stroke[i];
		PF_FpLong gap = s->lo;		// staged: lo/hi hold gap/width until now
		PF_FpLong wid = s->hi;

		if (s->side == CR_SIDE_INNER) {
			s->lo = cursorInner + gap;
			s->hi = s->lo + wid;
			cursorInner = s->hi;
		} else if (s->side == CR_SIDE_CENTER) {
			PF_FpLong c = cursorOuter + gap;
			s->lo = c - wid * 0.5;
			s->hi = c + wid * 0.5;
			cursorOuter = s->hi;
			if (s->hi > info->maxOuter) info->maxOuter = s->hi;
		} else {
			s->lo = cursorOuter + gap;
			s->hi = s->lo + wid;
			cursorOuter = s->hi;
			if (s->hi > info->maxOuter) info->maxOuter = s->hi;
		}
	}
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

/* Blend the three exact fields by the corner slider.
   0..1 miter->round, 1..2 round->bevel, >2 extrapolate past bevel = concave.
   The fields are always ordered L-inf <= L2 <= L1, so every blend stays sane. */
static void
CR_CornerField (const float *m, const float *r, const float *b,
				float *dst, A_long n, PF_FpLong corner)
{
	float c = (float)corner;
	if (c <= 1.0f) {
		float t = c < 0.0f ? 0.0f : c;
		for (A_long i = 0; i < n; i++) dst[i] = m[i] + (r[i] - m[i]) * t;
	} else if (c <= 2.0f) {
		float t = c - 1.0f;
		for (A_long i = 0; i < n; i++) dst[i] = r[i] + (b[i] - r[i]) * t;
	} else {
		float t = c - 2.0f;
		for (A_long i = 0; i < n; i++) dst[i] = b[i] + (b[i] - r[i]) * t;
	}
}

/* Per-pixel form of CR_CornerField. Same blend, evaluated one pixel at a time
   so the render loop does not need dOut/dIn as separate full-size arrays. */
static float
CR_BlendAt (const float *m, const float *r, const float *b, A_long k, float c)
{
	if (c <= 1.0f) { float t = (c < 0.0f) ? 0.0f : c; return m[k] + (r[k] - m[k]) * t; }
	if (c <= 2.0f) { float t = c - 1.0f;              return r[k] + (b[k] - r[k]) * t; }
	{               float t = c - 2.0f;               return b[k] + (b[k] - r[k]) * t; }
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

	#define CR_CHECKOUT(idx, var)											\
		PF_ParamDef var;													\
		AEFX_CLR_STRUCT(var);												\
		ERR(PF_CHECKOUT_PARAM(in_data, (idx), in_data->current_time,		\
							  in_data->time_step, in_data->time_scale, &var));

	{
		CR_CHECKOUT(CR_COUNT,		count_p);
		CR_CHECKOUT(CR_THRESHOLD,	thresh_p);
		CR_CHECKOUT(CR_FEATHER,		feather_p);
		CR_CHECKOUT(CR_SUBPIXEL,	sub_p);

		if (!err) {
			info->count		= count_p.u.sd.value;
			info->threshold	= thresh_p.u.fs_d.value / 100.0;
			info->feather	= feather_p.u.fs_d.value * ds;
			info->subpixel	= sub_p.u.bd.value;
			if (info->count < 1)				info->count = 1;
			if (info->count > CR_MAX_STROKES)	info->count = CR_MAX_STROKES;
		}

		ERR2(PF_CHECKIN_PARAM(in_data, &count_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &thresh_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &feather_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &sub_p));
	}

	for (A_long i = 0; i < info->count && !err; i++) {
		CR_CHECKOUT(CR_P(i, CRS_GAP),		gap_p);
		CR_CHECKOUT(CR_P(i, CRS_WIDTH),		wid_p);
		CR_CHECKOUT(CR_P(i, CRS_SIDE),		side_p);
		CR_CHECKOUT(CR_P(i, CRS_CORNER),	corner_p);
		CR_CHECKOUT(CR_P(i, CRS_COLOR),		color_p);
		CR_CHECKOUT(CR_P(i, CRS_FILL),		fill_p);
		CR_CHECKOUT(CR_P(i, CRS_COLOR2),	color2_p);
		CR_CHECKOUT(CR_P(i, CRS_GSTART),	gstart_p);
		CR_CHECKOUT(CR_P(i, CRS_GEND),		gend_p);
		CR_CHECKOUT(CR_P(i, CRS_OPACITY),	op_p);
		CR_CHECKOUT(CR_P(i, CRS_ORDER),		order_p);

		if (!err) {
			CRStroke *s = &info->stroke[i];
			// Stage gap/width in lo/hi; CR_ResolveStack turns them absolute.
			s->lo		= gap_p.u.fs_d.value * ds;
			s->hi		= wid_p.u.fs_d.value * ds;
			s->side		= side_p.u.pd.value;
			s->corner	= CR_CornerValue(corner_p.u.pd.value);
			s->opacity	= op_p.u.fs_d.value / 100.0;
			s->color[0]	= color_p.u.cd.value.red   / 255.0;
			s->color[1]	= color_p.u.cd.value.green / 255.0;
			s->color[2]	= color_p.u.cd.value.blue  / 255.0;
			s->order	= order_p.u.pd.value;
			// fill: gradient colours + endpoints. Points arrive as PF_Fixed in
			// the CURRENT (downsampled) layer coordinate space - same space as
			// in_data->output_origin, so no ds scaling here (unlike distances).
			s->fill		= fill_p.u.pd.value;
			s->color2[0]= color2_p.u.cd.value.red   / 255.0;
			s->color2[1]= color2_p.u.cd.value.green / 255.0;
			s->color2[2]= color2_p.u.cd.value.blue  / 255.0;
			s->gsx		= gstart_p.u.td.x_value / 65536.0;
			s->gsy		= gstart_p.u.td.y_value / 65536.0;
			s->gex		= gend_p.u.td.x_value   / 65536.0;
			s->gey		= gend_p.u.td.y_value   / 65536.0;
		}

		ERR2(PF_CHECKIN_PARAM(in_data, &gap_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &wid_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &side_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &corner_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &color_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &fill_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &color2_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &gstart_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &gend_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &op_p));
		ERR2(PF_CHECKIN_PARAM(in_data, &order_p));
	}
	#undef CR_CHECKOUT

	CR_ResolveStack(info);

	// Remember what AE actually asked for BEFORE we widen the input request.
	const PF_LRect reqRect = extra->input->output_request.rect;

	A_long grow = (A_long)ceil(info->maxOuter + info->feather + 1.0);
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

		PF_LRect m = in_result.max_result_rect;
		m.left -= grow;  m.top -= grow;  m.right += grow;  m.bottom += grow;
		extra->output->max_result_rect = m;

		PF_LRect r = in_result.result_rect;
		r.left -= grow;  r.top -= grow;  r.right += grow;  r.bottom += grow;

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
		info->pad		= grow;		// SmartRender pads its working buffer by this

		extra->output->solid = FALSE;

#if CR_GPU_RENDER
		// Opt THIS frame into GPU rendering. Only takes effect when the global
		// SUPPORTS_GPU_RENDER_F32 flag is also set and AE has a supported (CUDA)
		// device; otherwise AE silently uses PF_Cmd_SMART_RENDER (CPU). While
		// CR_GPU_RENDER is 0 we never set this, so the GPU path stays dormant
		// and every frame renders on the CPU.
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
	PF_Err				err = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	PF_EffectWorld		*inputP = NULL, *outputP = NULL;

	PF_Handle infoH = reinterpret_cast<PF_Handle>(extra->input->pre_render_data);
	CRInfo *info = reinterpret_cast<CRInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) return PF_Err_BAD_CALLBACK_PARAM;

	ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CR_INPUT, &inputP));
	ERR(extra->cb->checkout_output(in_data->effect_ref, &outputP));

	if (!err && inputP && outputP) {
		const A_long	bd	= extra->input->bitdepth;
		const A_long	W	= outputP->width;
		const A_long	H	= outputP->height;

		// The distance field must see shape pixels from OUTSIDE the region we
		// render - up to the stroke's reach away - which is why PreRender
		// widened the INPUT request by `pad`. Work in "padded space": padded
		// pixel (px,py) maps to output (px-pad, py-pad).
		const A_long	pad	= info->pad;
		const A_long	PW	= W + 2 * pad;
		const A_long	PH	= H + 2 * pad;

		// Where the input world sits inside the output world.
		const A_long	offX = info->inRect.left - info->outRect.left;
		const A_long	offY = info->inRect.top  - info->outRect.top;

		// Output pixel (x,y) sits at LAYER coordinate (x+oox, y+ooy) - the space
		// the gradient endpoints live in, so this is what makes a gradient line
		// up with where the user dragged its handles.
		//
		// This MUST come from outRect, not in_data->output_origin_x/y.
		// output_origin is a CLASSIC-render-path concept (it describes an
		// I_EXPAND_BUFFER output); under SmartFX the output world corresponds to
		// the result_rect we declared in PreRender. Deriving it the other way
		// proves which is right - the input mapping we already trust says
		//     layer(input u) = inRect.left + u,  u = x - offX,
		//     offX = inRect.left - outRect.left
		// so  layer(output x) = inRect.left + x - (inRect.left - outRect.left)
		//                     = x + outRect.left.
		const A_long	oox = info->outRect.left;
		const A_long	ooy = info->outRect.top;

		#define CR_IN_X(px)	((px) - pad - offX)
		#define CR_IN_Y(py)	((py) - pad - offY)

		const float thr = (float)info->threshold;

		try {
			// --- pass 1: locate the shape, without allocating anything -------
			// One cheap scan gives us both "is there a shape at all" and its
			// bounding box, which is what lets us crop everything below.
			A_long minX = PW, minY = PH, maxX = -1, maxY = -1;
			for (A_long py = 0; py < PH; py++) {
				for (A_long px = 0; px < PW; px++) {
					if (CR_ReadAlpha(inputP, bd, CR_IN_X(px), CR_IN_Y(py)) >= thr) {
						if (px < minX) minX = px;
						if (px > maxX) maxX = px;
						if (py < minY) minY = py;
						if (py > maxY) maxY = py;
					}
				}
			}

			// --- NOTHING TO STROKE? ------------------------------------------
			// A frame can carry a full-size bounding box and still contain no
			// shape - a TEXT ANIMATOR driving per-character/word/line Opacity
			// to 0 is exactly this. (Animator opacity is applied INSIDE the
			// text renderer, so we receive a frame that is "present but
			// invisible"; LAYER opacity is applied AFTER effects and never
			// reaches us - which is why only the animator triggered it.)
			// No pixel above threshold means no contour, so no stroke anywhere.
			if (maxX < 0) {
				for (A_long y = 0; y < H; y++) {
					for (A_long x = 0; x < W; x++) {
						float src[4];
						CR_ReadRGBA(inputP, bd, x - offX, y - offY, src);
						CR_WriteRGBA(outputP, bd, x, y, src);
					}
				}
				suites.HandleSuite1()->host_unlock_handle(infoH);
				return err;
			}

			// --- crop every buffer to the work rect --------------------------
			// The whole shape lies inside (bbox + pad), so distances computed
			// there are exact for every pixel in it. Anything further out than
			// `pad` is beyond the outermost band and cannot be stroked, so it
			// only needs the source copied through.
			// On typical artwork - a title over an empty frame - this is the
			// single biggest saving available: the fields shrink from the whole
			// padded frame to roughly the artwork itself.
			const A_long wx0 = MAX(0, minX - pad);
			const A_long wy0 = MAX(0, minY - pad);
			const A_long wx1 = MIN(PW - 1, maxX + pad);
			const A_long wy1 = MIN(PH - 1, maxY + pad);
			const A_long CW  = wx1 - wx0 + 1;
			const A_long CH  = wy1 - wy0 + 1;
			const A_long N   = CW * CH;

			// output (x,y) -> index into the cropped buffers (caller checks range)
			#define CR_CIDX(x, y)	(((y) + pad - wy0) * CW + ((x) + pad - wx0))

			std::vector<float> alphaBuf(N);
			for (A_long cy = 0; cy < CH; cy++)
				for (A_long cx = 0; cx < CW; cx++)
					alphaBuf[cy * CW + cx] =
						CR_ReadAlpha(inputP, bd, CR_IN_X(cx + wx0), CR_IN_Y(cy + wy0));

			// --- which fields do we actually need? ---------------------------
			bool needM = false, needR = false, needB = false, needInside = false;
			for (A_long i = 0; i < info->count; i++) {
				PF_FpLong c = info->stroke[i].corner;
				if (c <= 1.0)	{ needM = true; needR = true; }
				else			{ needR = true; needB = true; }
				if (info->stroke[i].side != CR_SIDE_OUTER) needInside = true;
			}

			const bool sub = (info->subpixel != 0);

			// SUB-PIXEL SEEDS - this is what fixes Concave's anti-aliasing.
			// A chamfer propagates min(neighbour + cost), so if the SOURCES
			// start at their true fractional offset, every propagated value is
			// fractional too and the field is smooth by construction.
			// Previously the chamfer seeded a flat 0 (integer field) and we
			// nudged the result by half a pixel afterwards. That works for a
			// blend, but Concave EXTRAPOLATES (2*bevel - round), which
			// amplifies the integer quantisation about 2x - far more than a
			// half-pixel nudge can smooth. Hence: correct each field BEFORE it
			// is blended, never after.
			// An edge pixel with coverage a sits (a-0.5) inside the contour, so
			// that is exactly the offset its seed carries.
			#define CR_SEED_CHAMFER(vecName, insideIsSource)					\
				vecName.resize(N);												\
				for (A_long i2 = 0; i2 < N; i2++) {								\
					float a2 = alphaBuf[i2];									\
					bool inside2 = a2 >= thr;									\
					bool src2 = (insideIsSource) ? inside2 : !inside2;			\
					vecName[i2] = src2											\
						? (sub ? ((insideIsSource) ? -(a2 - 0.5f) : (a2 - 0.5f))\
							   : -0.5f)											\
						: CR_INF;												\
				}
			#define CR_SEED_EDT(vecName, insideIsSource)						\
				vecName.resize(N);												\
				for (A_long i2 = 0; i2 < N; i2++) {								\
					bool inside2 = alphaBuf[i2] >= thr;							\
					bool src2 = (insideIsSource) ? inside2 : !inside2;			\
					vecName[i2] = src2 ? 0.0f : CR_INF;							\
				}

			// The Euclidean field can't take fractional seeds the same way (FH
			// works on SQUARED distance), so it keeps the nearest-feature
			// correction - which is exact for L2 and already verified.
			// Applied here, per field, BEFORE any blending.
			std::vector<A_long> feat;
			std::vector<float>  seedTmp;

			std::vector<float> outM, outR, outB, inM, inR, inB;

			if (needM) { CR_SEED_CHAMFER(outM, true);  CR_Chamfer(&outM[0], CW, CH, true);  }
			if (needB) { CR_SEED_CHAMFER(outB, true);  CR_Chamfer(&outB[0], CW, CH, false); }
			if (needR) {
				CR_SEED_EDT(outR, true);
				if (sub) { seedTmp = outR; }
				CR_EDT(&outR[0], CW, CH);
				if (sub) {
					feat.resize(N);
					CR_EDT_Feature(&seedTmp[0], CW, CH, &feat[0]);
					for (A_long k = 0; k < N; k++) outR[k] -= (alphaBuf[feat[k]] - 0.5f);
				} else {
					for (A_long k = 0; k < N; k++) outR[k] -= 0.5f;
				}
				CR_ClampField(&outR[0], N);
			}

			if (needInside) {
				if (needM) { CR_SEED_CHAMFER(inM, false); CR_Chamfer(&inM[0], CW, CH, true);  }
				if (needB) { CR_SEED_CHAMFER(inB, false); CR_Chamfer(&inB[0], CW, CH, false); }
				if (needR) {
					CR_SEED_EDT(inR, false);
					if (sub) { seedTmp = inR; }
					CR_EDT(&inR[0], CW, CH);
					if (sub) {
						feat.resize(N);
						CR_EDT_Feature(&seedTmp[0], CW, CH, &feat[0]);
						for (A_long k = 0; k < N; k++) inR[k] -= (0.5f - alphaBuf[feat[k]]);
					} else {
						for (A_long k = 0; k < N; k++) inR[k] -= 0.5f;
					}
					CR_ClampField(&inR[0], N);
				}
			} else {
				inR.assign(N, 0.0f);	// unused; the blend still reads three pointers
			}
			#undef CR_SEED_CHAMFER
			#undef CR_SEED_EDT

			// free the scratch the feature pass needed
			std::vector<A_long>().swap(feat);
			std::vector<float>().swap(seedTmp);

			const float *pM = needM ? &outM[0] : &outR[0];
			const float *pR = &outR[0];
			const float *pB = needB ? &outB[0] : &outR[0];
			const float *qM = (needInside && needM) ? &inM[0] : &inR[0];
			const float *qR = &inR[0];
			const float *qB = (needInside && needB) ? &inB[0] : &inR[0];

			// --- composite ---------------------------------------------------
			bool anyFront = false;
			for (A_long i = 0; i < info->count; i++)
				if (info->stroke[i].order == CR_ORDER_FRONT) anyFront = true;

			std::vector<float> behRGB(N * 3, 0.0f), behA(N, 0.0f);
			std::vector<float> frtRGB, frtA;
			if (anyFront) { frtRGB.assign(N * 3, 0.0f); frtA.assign(N, 0.0f); }

			// Only ONE field-sized temporary now: the blended signed distance.
			// dOut/dIn used to be separate arrays; they are folded into this
			// loop instead, which is two fewer full-size buffers.
			std::vector<float> sdf(N);
			// ...plus the field's local gradient magnitude, which is what makes
			// the feather mean the same thing on every corner style. See below.
			std::vector<float> grad(N);

			PF_FpLong lastCorner = -999.0;
			for (A_long i = 0; i < info->count; i++) {
				const CRStroke *s = &info->stroke[i];

				// The field depends only on the corner style, so rebuild it
				// only when that actually changes between strokes.
				if (s->corner != lastCorner) {
					const float c = (float)s->corner;
					for (A_long k = 0; k < N; k++) {
						float dO = CR_BlendAt(pM, pR, pB, k, c);
						float dI = CR_BlendAt(qM, qR, qB, k, c);
						bool  inside = alphaBuf[k] >= thr;
						float d = inside ? -dI : dO;
						// (Removed a near-contour override that forced
						// d = -(alpha-0.5) for |d|<1. It predated per-field
						// sub-pixel seeding; now each field is already sub-pixel
						// accurate at the contour, so the override only replaced
						// good values with a differently-scaled estimate,
						// planting a small discontinuity that the |grad| step
						// below then amplified into uneven AA.)
						sdf[k] = d;
					}

					// LOCAL GRADIENT = how many field units one pixel is worth.
					//
					// The feather is specified in PIXELS, but the smoothstep
					// compares FIELD VALUES. Those are only the same thing when
					// the field has unit gradient - true of a real Euclidean
					// distance (measured: |grad| 0.96..1.01) but NOT of the
					// other corner styles. Concave, being an extrapolation,
					// reaches |grad| ~3.9 on curves, so a 0.5px feather was
					// collapsing to ~0.13px of actual transition there: the
					// anti-aliasing was not missing, it was squeezed to
					// sub-pixel width exactly where curvature is highest.
					// That is why corners came good with sub-pixel seeds while
					// ROUND SHAPES still looked hard.
					// Scaling the ramp by |grad| makes the transition a fixed
					// width in pixels for every style. For Round this is a
					// no-op, since |grad| is already 1.
					for (A_long cy = 0; cy < CH; cy++) {
						for (A_long cx = 0; cx < CW; cx++) {
							A_long k2 = cy * CW + cx;
							bool ix0 = (cx > 0), ix1 = (cx < CW - 1);
							bool iy0 = (cy > 0), iy1 = (cy < CH - 1);
							float gx = sdf[ix1 ? k2 + 1 : k2] - sdf[ix0 ? k2 - 1 : k2];
							float gy = sdf[iy1 ? k2 + CW : k2] - sdf[iy0 ? k2 - CW : k2];
							if (ix0 && ix1) gx *= 0.5f;		// central difference
							if (iy0 && iy1) gy *= 0.5f;
							float g = sqrtf(gx * gx + gy * gy);
							if (!(g > 0.25f))	g = 0.25f;	// also catches NaN
							if (g > 8.0f)		g = 8.0f;
							grad[k2] = g;
						}
					}
					lastCorner = s->corner;
				}

				float *accRGB = (s->order == CR_ORDER_FRONT) ? &frtRGB[0] : &behRGB[0];
				float *accA   = (s->order == CR_ORDER_FRONT) ? &frtA[0]   : &behA[0];

				// LINEAR GRADIENT SETUP (per stroke).
				// t = clamp( dot(P - start, end - start) / |end - start|^2 ).
				// P is the pixel's LAYER coordinate; for cropped index k that is
				//   Lx = cx - pad + wx0 + oox,  Ly = cy - pad + wy0 + ooy.
				// Folding the constants, Lx - gsx = cx + gAX and Ly - gsy = cy + gAY,
				// so the per-pixel cost is just a dot product.
				const bool  isGrad = (s->fill == CR_FILL_LINEAR);
				const float gdx  = (float)(s->gex - s->gsx);
				const float gdy  = (float)(s->gey - s->gsy);
				float gden = gdx * gdx + gdy * gdy;
				const float ginv = (gden > 1e-6f) ? (1.0f / gden) : 0.0f;
				const float gAX  = (float)(wx0 - pad + oox) - (float)s->gsx;
				const float gAY  = (float)(wy0 - pad + ooy) - (float)s->gsy;

				const float aa = (float)info->feather;
				for (A_long k = 0; k < N; k++) {
					float m = (s->side == CR_SIDE_INNER) ? -sdf[k] : sdf[k];

					// BAND COVERAGE = DIFFERENCE OF TWO RAMPS, NOT THEIR PRODUCT.
					// The product form leaves a 25%-transparent seam wherever two
					// strokes touch: each band is 0.5 at a shared boundary and
					// source-over gives 0.5+0.5*(1-0.5)=0.75. Alpha-over assumes
					// INDEPENDENT coverage, but abutting bands are COMPLEMENTARY -
					// they tile the distance axis, so coverage must SUM. Written
					// as a difference the terms telescope and the stack sums to
					// exactly 1 across every join.
					// Feather is in PIXELS; convert to field units with |grad|.
					const float aaE = aa * grad[k];
					float cov = CR_Smoothstep((float)s->lo - aaE, (float)s->lo + aaE, m) -
								CR_Smoothstep((float)s->hi - aaE, (float)s->hi + aaE, m);
					if (cov <= 0.0f) continue;
					if (cov > 1.0f) cov = 1.0f;
					cov *= (float)s->opacity;
					if (cov <= 0.0f) continue;

					// Pick the fill colour: flat, or lerp along the gradient axis.
					float col0 = (float)s->color[0];
					float col1 = (float)s->color[1];
					float col2 = (float)s->color[2];
					if (isGrad) {
						A_long cyk = k / CW, cxk = k - cyk * CW;
						float t = (((float)cxk + gAX) * gdx +
								   ((float)cyk + gAY) * gdy) * ginv;
						if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
						col0 += ((float)s->color2[0] - col0) * t;
						col1 += ((float)s->color2[1] - col1) * t;
						col2 += ((float)s->color2[2] - col2) * t;
					}

					// Disjoint bands: ACCUMULATE (premultiplied), don't composite.
					accRGB[k * 3 + 0] += col0 * cov;
					accRGB[k * 3 + 1] += col1 * cov;
					accRGB[k * 3 + 2] += col2 * cov;
					accA[k] += cov;
				}
			}

			// Overlapping stacks (a Center stroke crossing an Outside one) can
			// push accumulated coverage past 1; normalize so the premultiplied
			// colour stays consistent with the alpha.
			{
				float *accs[2]    = { &behA[0],   anyFront ? &frtA[0]   : NULL };
				float *accsRGB[2] = { &behRGB[0], anyFront ? &frtRGB[0] : NULL };
				for (A_long b = 0; b < 2; b++) {
					if (!accs[b]) continue;
					for (A_long k = 0; k < N; k++) {
						if (accs[b][k] > 1.0f) {
							float inv = 1.0f / accs[b][k];
							for (A_long c = 0; c < 3; c++) accsRGB[b][k * 3 + c] *= inv;
							accs[b][k] = 1.0f;
						}
					}
				}
			}

			// --- final stack: behind strokes, source art, front strokes -------
			for (A_long y = 0; y < H; y++) {
				const A_long py = y + pad;
				const bool rowIn = (py >= wy0 && py <= wy1);
				for (A_long x = 0; x < W; x++) {
					float src[4];
					CR_ReadRGBA(inputP, bd, x - offX, y - offY, src);

					const A_long px = x + pad;
					if (!rowIn || px < wx0 || px > wx1) {
						// Further from the shape than any band reaches.
						CR_WriteRGBA(outputP, bd, x, y, src);
						continue;
					}
					const A_long k = CR_CIDX(x, y);

					// The accumulators are PREMULTIPLIED, so the stack
					// composites in premultiplied space (a plain lerp, no
					// per-layer divide) and converts to straight at the end.
					const float sa = src[3];
					float srcPre[3];
					for (A_long c = 0; c < 3; c++) srcPre[c] = src[c] * sa;

					float midA = sa + behA[k] * (1.0f - sa);
					float midPre[3];
					for (A_long c = 0; c < 3; c++)
						midPre[c] = srcPre[c] + behRGB[k * 3 + c] * (1.0f - sa);

					float fa = anyFront ? frtA[k] : 0.0f;
					float outA = fa + midA * (1.0f - fa);
					float outPre[3];
					for (A_long c = 0; c < 3; c++) {
						float fc = anyFront ? frtRGB[k * 3 + c] : 0.0f;
						outPre[c] = fc + midPre[c] * (1.0f - fa);
					}

					float out[4];
					if (outA > 1e-6f) {
						for (A_long c = 0; c < 3; c++) out[c] = outPre[c] / outA;
					} else {
						out[0] = out[1] = out[2] = 0.0f;
					}
					out[3] = outA;
					CR_WriteRGBA(outputP, bd, x, y, out);
				}
			}

			#undef CR_CIDX
		}
		catch (std::bad_alloc &) {
			err = PF_Err_OUT_OF_MEMORY;
		}
	}

	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}

/* =========================================================================
   GPU path (CUDA only).

   Structurally a mirror of the CPU SmartRender split, following SDK sample
   Effect/SDK_Invert_ProcAmp - the only GPU sample. Three pieces:
     GPU_DEVICE_SETUP   - per GPU device AE exposes. For CUDA the kernel is
                          statically linked, so there is nothing to compile
                          here; we just confirm we support this device.
     GPU_DEVICE_SETDOWN - matching teardown (nothing to free for CUDA).
     SMART_RENDER_GPU   - per frame: get device pointers for the input/output
                          worlds and launch kernels. Reached only when PreRender
                          set GPU_RENDER_POSSIBLE (gated by CR_GPU_RENDER).

   PreRender is shared with the CPU path and already ran, so info->{inRect,
   outRect,pad,strokes...} are all populated the same way.
   ========================================================================= */

static PF_Err
GPUDeviceSetup (
	PF_InData				*in_data,
	PF_OutData				*out_data,
	PF_GPUDeviceSetupExtra	*extra )
{
	PF_Err err = PF_Err_NONE;

	// We only handle CUDA. For any other framework we simply don't claim
	// support, so AE keeps that device on the CPU path.
	if (extra->input->what_gpu == PF_GPU_Framework_CUDA) {
		out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
	return err;
}

static PF_Err
GPUDeviceSetdown (
	PF_InData					*in_data,
	PF_OutData					*out_data,
	PF_GPUDeviceSetdownExtra	*extra )
{
	// CUDA kernel is statically linked; nothing device-specific to release.
	return PF_Err_NONE;
}

#if HAS_CUDA
static PF_Err
SmartRenderGPU (
	PF_InData			*in_data,
	PF_OutData			*out_data,
	PF_SmartRenderExtra	*extra )
{
	PF_Err				err		= PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);
	PF_EffectWorld		*inputP = NULL, *outputP = NULL;

	// pre_render_data is the PF_Handle we allocated in PreRender - it must be
	// LOCKED to get the CRInfo pointer, exactly as the CPU SmartRender does.
	// (Treating the handle as the pointer gives garbage rects/params, which
	// pushes the source read out of bounds and collapses the band -> the whole
	// layer renders transparent. That was the "layer disappears on CUDA" bug.)
	PF_Handle infoH = reinterpret_cast<PF_Handle>(extra->input->pre_render_data);
	CRInfo *info = reinterpret_cast<CRInfo*>(suites.HandleSuite1()->host_lock_handle(infoH));
	if (!info) return PF_Err_BAD_CALLBACK_PARAM;

	// Acquire the GPU device suite directly off the PICA basic suite (keeps us
	// out of AEGP_SuiteHandler, which doesn't wrap it).
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
		// GPU worlds hand back device pointers, not host memory.
		void *src_mem = NULL, *dst_mem = NULL;
		ERR(gpu->GetGPUWorldData(in_data->effect_ref, inputP,  &src_mem));
		ERR(gpu->GetGPUWorldData(in_data->effect_ref, outputP, &dst_mem));

		if (!err) {
			const int bytesPerPixel = 16;			// GPU_BGRA128 = 4 x float32
			int srcPitch = inputP->rowbytes  / bytesPerPixel;
			int dstPitch = outputP->rowbytes / bytesPerPixel;
			int offX = info->inRect.left - info->outRect.left;
			int offY = info->inRect.top  - info->outRect.top;

			// STAGE 5: full multi-stroke composite, now bbox-CROPPED on the GPU
			// (info->pad = the outward reach; the kernel finds the shape bbox on
			// the device and runs the fields on just bbox+pad). Build the
			// device-bound stroke array and the "which fields are needed" flags,
			// then hand the whole thing to one composite launch.
			//
			// Output pixel (x,y) sits at LAYER (x+outRect.left, y+outRect.top)
			// under SmartFX (same origin the CPU path uses); gAX/gAY carry the
			// -gstart offset so the kernel only adds x/y. The GPU path has NO crop
			// yet, so there is no wx0/pad term the CPU folds in.
			CRGpuStroke gs[CR_MAX_STROKES];
			int needM = 0, needB = 0, needInside = 0;
			const int count = (info->count < CR_MAX_STROKES) ? info->count : CR_MAX_STROKES;

			for (int i = 0; i < count; i++) {
				const CRStroke *s = &info->stroke[i];
				gs[i].lo       = (float)s->lo;
				gs[i].hi       = (float)s->hi;
				gs[i].corner   = (float)s->corner;
				gs[i].opacity  = (float)s->opacity;
				gs[i].side     = (int)s->side;
				gs[i].order    = (int)s->order;
				gs[i].fill     = (int)s->fill;
				gs[i].colR     = (float)s->color[0];
				gs[i].colG     = (float)s->color[1];
				gs[i].colB     = (float)s->color[2];
				gs[i].col2R    = (float)s->color2[0];
				gs[i].col2G    = (float)s->color2[1];
				gs[i].col2B    = (float)s->color2[2];

				const float gdx = (float)(s->gex - s->gsx);
				const float gdy = (float)(s->gey - s->gsy);
				const float gden = gdx * gdx + gdy * gdy;
				gs[i].gdx  = gdx;
				gs[i].gdy  = gdy;
				gs[i].ginv = (gden > 1e-6f) ? (1.0f / gden) : 0.0f;
				gs[i].gAX  = (float)info->outRect.left - (float)s->gsx;
				gs[i].gAY  = (float)info->outRect.top  - (float)s->gsy;

				// Which metric fields does this corner value touch? (Mirrors the
				// CPU's needM/needR/needB; needR is always true.)
				if (s->corner <= 1.0)	needM = 1;
				else					needB = 1;
				if (s->side != CR_SIDE_OUTER) needInside = 1;
			}

			CR_Composite_CUDA(
				(const float*)src_mem, (float*)dst_mem,
				srcPitch, dstPitch,
				outputP->width, outputP->height,
				offX, offY, inputP->width, inputP->height,
				(float)info->threshold, (float)info->feather, info->subpixel,
				(int)info->pad,
				gs, count, needM, needB, needInside);

			if (cudaPeekAtLastError() != cudaSuccess)
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
		}
	}

	in_data->pica_basicP->ReleaseSuite(kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1);
	suites.HandleSuite1()->host_unlock_handle(infoH);
	return err;
}
#endif // HAS_CUDA

/* =========================================================================
   Classic render path.

   Premiere and legacy hosts don't do SmartFX. A stroke needs expansion that
   the classic path can't express well, so we pass the frame through rather
   than render something subtly wrong. AE always takes the SmartFX path.
   ========================================================================= */

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
			case PF_Cmd_USER_CHANGED_PARAM:		// Add / Remove Stroke clicked
				err = UserChangedParam(in_data, out_data, params,
						reinterpret_cast<const PF_UserChangedParamExtra*>(extra));
				break;
			case PF_Cmd_UPDATE_PARAMS_UI:		// sync group visibility on load
				err = CR_SyncGroups(in_data, params);
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
#if HAS_CUDA
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
