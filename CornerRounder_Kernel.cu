/*
	CornerRounder_Kernel.cu

	GPU render path (CUDA). Built up in stages, one capability per build:
	  Stage 1: JFA distance field + one solid round outer stroke.
	  Stage 2: all four corner styles (JFA per metric) + sub-pixel + |grad| AA.
	  Stage 3: per-stroke linear gradient fill.
	  Stage 4: inner/center sides, behind/front stacking, up to CR_MAX_STROKES.
	  Stage 5 (THIS FILE): bbox CROP for perf/memory. Everything above ran over
	           the full output world; now we find the shape's bounding box on the
	           device, run all the distance-field work on just (bbox + pad), and
	           copy the source straight through everywhere else. This is the GPU
	           twin of the CPU's "crop to work rect" pass: the work rect contains
	           the WHOLE shape, so distances inside it are exact, and any pixel
	           further than `pad` from the shape can't be stroked - it only needs
	           the source copied. On a title over an empty frame this is the
	           biggest win available (fewer pixels AND fewer JFA passes, since the
	           pass count is log2 of the cropped max dimension).

	The whole thing is a faithful port of the CPU SmartRender's math; the one
	deliberate divergence is JFA (parallel) in place of the CPU's sequential
	FH/chamfer transform. JFA also stores each pixel's nearest-seed COORDINATES,
	which gives the sub-pixel "feature" correction for free.

	COORDINATE SPACES (the thing to get right when cropping):
	  - OUTPUT space (x,y): the world AE hands us, [0,W) x [0,H). src/dst reads.
	  - INPUT space: output minus (offX,offY).
	  - CROPPED space (cx,cy): [0,CW) x [0,CH), the work rect. cx = x - wx0.
	    All the field buffers and the JFA live here. A cropped pixel maps to
	    output (wx0+cx, wy0+cy). Distances are translation-invariant, so seeds
	    storing cropped coords give the same distances as output coords would.

	Pixel format is GPU_BGRA128: float4 laid out B,G,R,A (.w = alpha). Source
	alpha is STRAIGHT; we composite in premultiplied space, matching the CPU.
*/

#include <cuda_runtime.h>

#define CR_NONE (-1)
#define CR_FAR  1.0e7f		// finite "unreachable" cap, matches the CPU field

// Distance metrics. Order by tightness: L-inf <= L2 <= L1, same as the CPU.
#define CR_METRIC_L2   0	// round
#define CR_METRIC_L1   1	// bevel
#define CR_METRIC_LINF 2	// miter

// Side / order / fill constants - MUST match CornerRounder.h.
#define CR_SIDE_OUTER   1
#define CR_SIDE_INNER   2
#define CR_SIDE_CENTER  3
#define CR_ORDER_BEHIND 1
#define CR_ORDER_FRONT  2
#define CR_FILL_LINEAR  2

// One stroke's resolved render state, mirrored from the host CRStroke. The
// gradient endpoints are pre-folded into a dot-projection (gAX/gAY carry the
// -gstart offset; ginv = 1/|end-start|^2) so the kernel only adds x/y.
struct CRGpuStroke {
	float	lo, hi, corner, opacity;
	int		side, order, fill;
	float	colR, colG, colB;			// start colour (R,G,B)
	float	col2R, col2G, col2B;		// gradient end colour
	float	gAX, gAY, gdx, gdy, ginv;	// folded linear-gradient constants
};

__device__ __forceinline__ float
cr_smoothstep(float e0, float e1, float x)
{
	float d = e1 - e0;
	if (d < 1e-6f) return x < e0 ? 0.0f : 1.0f;
	float t = (x - e0) / d;
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
	return t * t * (3.0f - 2.0f * t);
}

/* Candidate distance from (x,y) to a seed under a metric. L2 stays SQUARED
   (monotone, so the nearer test is identical and we defer the sqrt). */
__device__ __forceinline__ float
cr_metric_dist(int dx, int dy, int metric)
{
	if (metric == CR_METRIC_L1)   return (float)(abs(dx) + abs(dy));
	if (metric == CR_METRIC_LINF) return (float)max(abs(dx), abs(dy));
	return (float)(dx * dx + dy * dy);				// L2, squared
}

/* Blend the three metric fields by the corner slider (per-pixel CR_BlendAt):
   0..1 miter->round, 1..2 round->bevel, >2 EXTRAPOLATE past bevel = concave. */
__device__ __forceinline__ float
cr_blend_at(float m, float r, float b, float c)
{
	if (c <= 1.0f) { float t = (c < 0.0f) ? 0.0f : c; return m + (r - m) * t; }
	if (c <= 2.0f) { float t = c - 1.0f;               return r + (b - r) * t; }
	{               float t = c - 2.0f;                return b + (b - r) * t; }
}

/* Read input alpha under OUTPUT pixel (ox,oy); 0 if outside the input world. */
__device__ __forceinline__ float
cr_alpha_at(const float4 *src, int ox, int oy, int srcPitch,
			int offX, int offY, int inW, int inH)
{
	int iu = ox - offX, iv = oy - offY;
	if (iu < 0 || iv < 0 || iu >= inW || iv >= inH) return 0.0f;
	return src[iv * srcPitch + iu].w;
}

/* Shape bounding box in OUTPUT space, via atomics into bbox[minx,miny,maxx,maxy]
   (init {W,H,-1,-1}). One thread per output pixel; opaque pixels extend the box.
   This is the cheap scan that lets everything below crop to (bbox + pad). */
__global__ void
cr_bbox(const float4 *src, int *bbox, int W, int H, int srcPitch,
		int offX, int offY, int inW, int inH, float thr)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	if (cr_alpha_at(src, x, y, srcPitch, offX, offY, inW, inH) >= thr) {
		atomicMin(&bbox[0], x);
		atomicMin(&bbox[1], y);
		atomicMax(&bbox[2], x);
		atomicMax(&bbox[3], y);
	}
}

/* Copy the source straight to the destination (honouring the expansion offset).
   Used for the empty-shape case and, inside the composite, for pixels outside
   the work rect. */
__global__ void
cr_copythrough(const float4 *src, float4 *dst, int W, int H,
			   int srcPitch, int dstPitch, int offX, int offY, int inW, int inH)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	int iu = x - offX, iv = y - offY;
	float4 sp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (iu >= 0 && iv >= 0 && iu < inW && iv < inH)
		sp = src[iv * srcPitch + iu];
	dst[y * dstPitch + x] = sp;
}

/* Seed in CROPPED space: cropped pixel (cx,cy) maps to output (wx0+cx,wy0+cy).
   sourceInside=1 -> opaque pixels are the source (OUTER field). =0 -> transparent
   pixels are the source (INNER field). Seeds store their own CROPPED coords. */
__global__ void
cr_seed(const float4 *src, int2 *seed, int CW, int CH, int srcPitch,
		int wx0, int wy0, int offX, int offY, int inW, int inH,
		float thr, int sourceInside)
{
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	if (cx >= CW || cy >= CH) return;

	float a = cr_alpha_at(src, wx0 + cx, wy0 + cy, srcPitch, offX, offY, inW, inH);
	bool inside = (a >= thr);
	bool isSrc  = sourceInside ? inside : !inside;
	seed[cy * CW + cx] = isSrc ? make_int2(cx, cy) : make_int2(CR_NONE, CR_NONE);
}

/* One jump-flood pass at `step` under `metric`, in cropped space. */
__global__ void
cr_flood(const int2 *in, int2 *out, int CW, int CH, int step, int metric)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= CW || y >= CH) return;

	int2  best  = in[y * CW + x];
	float bestD = (best.x < 0) ? 1e30f
				: cr_metric_dist(x - best.x, y - best.y, metric);

	#pragma unroll
	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			int nx = x + dx * step, ny = y + dy * step;
			if (nx < 0 || ny < 0 || nx >= CW || ny >= CH) continue;
			int2 s = in[ny * CW + nx];
			if (s.x < 0) continue;
			float d = cr_metric_dist(x - s.x, y - s.y, metric);
			if (d < bestD) { bestD = d; best = s; }
		}
	}
	out[y * CW + x] = best;
}

/* Resolve a finished seed field to a distance in PIXELS with the sub-pixel
   correction folded in. The feature is the nearest seed's coords (cropped); read
   its alpha at output (wx0+sx, wy0+sy). Sign depends on which side is the source
   (outerField=1 -> af-0.5, else 0.5-af); sub-pixel off falls back to a flat 0.5.
   Corrected PER FIELD, BEFORE the corner blend. */
__global__ void
cr_resolve(const int2 *seed, const float4 *src, float *fld,
		   int CW, int CH, int srcPitch, int wx0, int wy0,
		   int offX, int offY, int inW, int inH,
		   int metric, int subpixel, int outerField)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= CW || y >= CH) return;

	int2 s = seed[y * CW + x];
	float d;
	if (s.x < 0) {
		d = CR_FAR;
	} else {
		int dx = x - s.x, dy = y - s.y;
		d = (metric == CR_METRIC_L2)
			? sqrtf((float)(dx * dx + dy * dy))
			: cr_metric_dist(dx, dy, metric);

		float corr = 0.5f;
		if (subpixel) {
			float af = cr_alpha_at(src, wx0 + s.x, wy0 + s.y, srcPitch,
								   offX, offY, inW, inH);
			corr = outerField ? (af - 0.5f) : (0.5f - af);
		}
		d -= corr;
	}

	if (!(d >= 0.0f)) d = 0.0f;			// also catches NaN
	if (d > CR_FAR)   d = CR_FAR;
	fld[y * CW + x] = d;
}

/* Signed distance at cropped index k for corner c: -inner inside the shape,
   +outer outside. Mirrors the CPU's sdf = inside ? -dI : dO. */
__device__ __forceinline__ float
cr_signed(const float *oM, const float *oR, const float *oB,
		  const float *iM, const float *iR, const float *iB,
		  int needInside, int k, bool inside, float c)
{
	if (inside) {
		float dI = needInside ? cr_blend_at(iM[k], iR[k], iB[k], c) : 0.0f;
		return -dI;
	}
	return cr_blend_at(oM[k], oR[k], oB[k], c);
}

/* THE composite, launched over the FULL output world. Pixels outside the work
   rect [wx0..wx0+CW-1] x [wy0..wy0+CH-1] just copy the source (they're further
   than any band reaches). Inside, one thread owns output pixel (x,y): loop every
   stroke, band it on the signed field (feather scaled by local |grad| so it
   stays a fixed pixel width on every corner style), accumulate PREMULTIPLIED
   into behind/front register accumulators, then stack behind -> art -> front. */
__global__ void
cr_composite(const float *oM, const float *oR, const float *oB,
			 const float *iM, const float *iR, const float *iB,
			 int needInside,
			 const CRGpuStroke *strokes, int count, float feather,
			 const float4 *src, float4 *dst,
			 int W, int H, int srcPitch, int dstPitch,
			 int offX, int offY, int inW, int inH, float thr,
			 int wx0, int wy0, int CW, int CH)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	// Source pixel (straight alpha). BGRA: .x=B .y=G .z=R .w=A.
	int iu = x - offX, iv = y - offY;
	float4 sp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (iu >= 0 && iv >= 0 && iu < inW && iv < inH)
		sp = src[iv * srcPitch + iu];

	// Outside the work rect: nothing to stroke, copy the source straight through.
	int cx = x - wx0, cy = y - wy0;
	if (cx < 0 || cy < 0 || cx >= CW || cy >= CH) {
		dst[y * dstPitch + x] = sp;
		return;
	}
	int k = cy * CW + cx;

	// Which side of the contour is this pixel (and its 4 neighbours) on?
	bool inC = sp.w >= thr;
	bool inL = cr_alpha_at(src, x - 1, y, srcPitch, offX, offY, inW, inH) >= thr;
	bool inR2= cr_alpha_at(src, x + 1, y, srcPitch, offX, offY, inW, inH) >= thr;
	bool inU = cr_alpha_at(src, x, y - 1, srcPitch, offX, offY, inW, inH) >= thr;
	bool inD = cr_alpha_at(src, x, y + 1, srcPitch, offX, offY, inW, inH) >= thr;

	// Neighbour CROPPED indices, clamped to the work rect (crop edges lie beyond
	// any band, so clamping there never affects a visible pixel).
	bool cx0 = (cx > 0), cx1 = (cx < CW - 1);
	bool cy0 = (cy > 0), cy1 = (cy < CH - 1);
	int kL = cx0 ? k - 1  : k, kR = cx1 ? k + 1  : k;
	int kU = cy0 ? k - CW : k, kD = cy1 ? k + CW : k;

	// Behind / front premultiplied accumulators (registers).
	float behR = 0, behG = 0, behB = 0, behA = 0;
	float frtR = 0, frtG = 0, frtB = 0, frtA = 0;

	for (int i = 0; i < count; i++) {
		const CRGpuStroke st = strokes[i];
		float c = st.corner;

		// Signed field at this pixel and its 4 neighbours, for the band and the
		// gradient magnitude (which depends on the corner style).
		float s0 = cr_signed(oM, oR, oB, iM, iR, iB, needInside, k,  inC, c);
		float sL = cr_signed(oM, oR, oB, iM, iR, iB, needInside, kL, cx0 ? inL : inC, c);
		float sR = cr_signed(oM, oR, oB, iM, iR, iB, needInside, kR, cx1 ? inR2: inC, c);
		float sU = cr_signed(oM, oR, oB, iM, iR, iB, needInside, kU, cy0 ? inU : inC, c);
		float sD = cr_signed(oM, oR, oB, iM, iR, iB, needInside, kD, cy1 ? inD : inC, c);

		float gx = sR - sL, gy = sD - sU;
		if (cx0 && cx1) gx *= 0.5f;
		if (cy0 && cy1) gy *= 0.5f;
		float g = sqrtf(gx * gx + gy * gy);
		if (!(g > 0.25f)) g = 0.25f;
		if (g > 8.0f)     g = 8.0f;

		// INNER bands measure inward (flip the signed field); OUTER/CENTER read
		// it directly, their band placement already straddles/sits outside 0.
		float m = (st.side == CR_SIDE_INNER) ? -s0 : s0;

		// Band coverage = DIFFERENCE of two ramps (telescopes to 1 across joins).
		float aaE = feather * g;
		float cov = cr_smoothstep(st.lo - aaE, st.lo + aaE, m) -
					cr_smoothstep(st.hi - aaE, st.hi + aaE, m);
		cov = cov < 0.0f ? 0.0f : (cov > 1.0f ? 1.0f : cov);
		cov *= st.opacity;
		if (cov <= 0.0f) continue;

		// Fill colour: flat or lerp along the gradient axis.
		float cR = st.colR, cG = st.colG, cB = st.colB;
		if (st.fill == CR_FILL_LINEAR) {
			float t = (((float)x + st.gAX) * st.gdx +
					   ((float)y + st.gAY) * st.gdy) * st.ginv;
			t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
			cR += (st.col2R - cR) * t;
			cG += (st.col2G - cG) * t;
			cB += (st.col2B - cB) * t;
		}

		// Disjoint bands ACCUMULATE (premultiplied); don't composite.
		if (st.order == CR_ORDER_FRONT) {
			frtR += cR * cov; frtG += cG * cov; frtB += cB * cov; frtA += cov;
		} else {
			behR += cR * cov; behG += cG * cov; behB += cB * cov; behA += cov;
		}
	}

	// Overlapping stacks can push accumulated coverage past 1; normalise so the
	// premultiplied colour stays consistent with the alpha.
	if (behA > 1.0f) { float inv = 1.0f / behA; behR *= inv; behG *= inv; behB *= inv; behA = 1.0f; }
	if (frtA > 1.0f) { float inv = 1.0f / frtA; frtR *= inv; frtG *= inv; frtB *= inv; frtA = 1.0f; }

	// Stack in premultiplied space: behind -> source -> front.
	float sa = sp.w;
	float spR = sp.z * sa, spG = sp.y * sa, spB = sp.x * sa;	// straight->premult

	float midA = sa + behA * (1.0f - sa);
	float midR = spR + behR * (1.0f - sa);
	float midG = spG + behG * (1.0f - sa);
	float midB = spB + behB * (1.0f - sa);

	float outA = frtA + midA * (1.0f - frtA);
	float outR = frtR + midR * (1.0f - frtA);
	float outG = frtG + midG * (1.0f - frtA);
	float outB = frtB + midB * (1.0f - frtA);

	float4 o;
	if (outA > 1e-6f) { o.x = outB / outA; o.y = outG / outA; o.z = outR / outA; }
	else              { o.x = o.y = o.z = 0.0f; }
	o.w = outA;

	dst[y * dstPitch + x] = o;
}

/* Run a full JFA for one metric into `fld` (cropped), reusing the ping-pong seed
   buffers. `seed0` already holds the seeds for the desired side. */
static void
cr_field_for_metric(
	const int2 *seed0, int2 *bufA, int2 *bufB,
	const float4 *src, float *fld,
	int CW, int CH, int srcPitch, int wx0, int wy0,
	int offX, int offY, int inW, int inH,
	int metric, int subpixel, int outerField, dim3 grid, dim3 block)
{
	int N = CW * CH;
	cudaMemcpy(bufA, seed0, (size_t)N * sizeof(int2), cudaMemcpyDeviceToDevice);

	int maxdim = (CW > CH) ? CW : CH;
	int step = 1;
	while (step < maxdim) step <<= 1;
	step >>= 1;

	int2 *cur = bufA, *other = bufB;
	for (; step >= 1; step >>= 1) {
		cr_flood<<<grid, block>>>(cur, other, CW, CH, step, metric);
		int2 *t = cur; cur = other; other = t;
	}

	cr_resolve<<<grid, block>>>(cur, src, fld, CW, CH, srcPitch, wx0, wy0,
								offX, offY, inW, inH, metric, subpixel, outerField);
}

/* Host launcher for STAGE 5: bbox-cropped multi-stroke composite.
   Declared extern (no extern "C") in the .cpp. `pad` is the outward reach
   (info->pad = ceil(maxOuter + feather + 1)); the work rect is (shape bbox + pad)
   clamped to the output world. */
void
CR_Composite_CUDA(
	const float *src, float *dst,
	int srcPitch, int dstPitch,
	int W, int H, int offX, int offY, int inW, int inH,
	float thr, float feather, int subpixel, int pad,
	const CRGpuStroke *hostStrokes, int count,
	int needM, int needB, int needInside)
{
	const float4 *src4 = (const float4*)src;

	int2  *seed0 = 0, *bufA = 0, *bufB = 0;
	float *oM = 0, *oR = 0, *oB = 0, *iM = 0, *iR = 0, *iB = 0;
	CRGpuStroke *devStrokes = 0;
	int *devBox = 0;

	dim3 block(16, 16, 1);
	dim3 gridFull((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, 1);

	#define CR_TRY(call) do { if ((call) != cudaSuccess) goto cleanup; } while (0)

	// --- find the shape bounding box on the device -----------------------------
	int hostBox[4] = { W, H, -1, -1 };			// minx, miny, maxx, maxy
	CR_TRY(cudaMalloc(&devBox, 4 * sizeof(int)));
	CR_TRY(cudaMemcpy(devBox, hostBox, 4 * sizeof(int), cudaMemcpyHostToDevice));
	cr_bbox<<<gridFull, block>>>(src4, devBox, W, H, srcPitch,
								 offX, offY, inW, inH, thr);
	CR_TRY(cudaMemcpy(hostBox, devBox, 4 * sizeof(int), cudaMemcpyDeviceToHost));

	// Empty shape: nothing to stroke, copy the whole frame through and return.
	if (hostBox[2] < hostBox[0] || hostBox[3] < hostBox[1]) {
		cr_copythrough<<<gridFull, block>>>(src4, (float4*)dst, W, H,
											srcPitch, dstPitch, offX, offY, inW, inH);
		cudaDeviceSynchronize();
		goto cleanup;
	}

	{
		// Work rect = shape bbox grown by pad, clamped to the output world.
		int wx0 = hostBox[0] - pad; if (wx0 < 0) wx0 = 0;
		int wy0 = hostBox[1] - pad; if (wy0 < 0) wy0 = 0;
		int wx1 = hostBox[2] + pad; if (wx1 > W - 1) wx1 = W - 1;
		int wy1 = hostBox[3] + pad; if (wy1 > H - 1) wy1 = H - 1;
		int CW  = wx1 - wx0 + 1;
		int CH  = wy1 - wy0 + 1;
		int N   = CW * CH;

		dim3 gridCrop((CW + block.x - 1) / block.x, (CH + block.y - 1) / block.y, 1);

		CR_TRY(cudaMalloc(&seed0, (size_t)N * sizeof(int2)));
		CR_TRY(cudaMalloc(&bufA,  (size_t)N * sizeof(int2)));
		CR_TRY(cudaMalloc(&bufB,  (size_t)N * sizeof(int2)));
		CR_TRY(cudaMalloc(&oR,    (size_t)N * sizeof(float)));
		if (needM) CR_TRY(cudaMalloc(&oM, (size_t)N * sizeof(float)));
		if (needB) CR_TRY(cudaMalloc(&oB, (size_t)N * sizeof(float)));
		if (needInside) {
			CR_TRY(cudaMalloc(&iR, (size_t)N * sizeof(float)));
			if (needM) CR_TRY(cudaMalloc(&iM, (size_t)N * sizeof(float)));
			if (needB) CR_TRY(cudaMalloc(&iB, (size_t)N * sizeof(float)));
		}
		CR_TRY(cudaMalloc(&devStrokes, (size_t)count * sizeof(CRGpuStroke)));
		CR_TRY(cudaMemcpy(devStrokes, hostStrokes, (size_t)count * sizeof(CRGpuStroke),
						  cudaMemcpyHostToDevice));

		// --- OUTER fields (opaque pixels are the source) ---
		cr_seed<<<gridCrop, block>>>(src4, seed0, CW, CH, srcPitch, wx0, wy0,
									 offX, offY, inW, inH, thr, 1);
		cr_field_for_metric(seed0, bufA, bufB, src4, oR, CW, CH, srcPitch, wx0, wy0,
							offX, offY, inW, inH, CR_METRIC_L2, subpixel, 1, gridCrop, block);
		if (needM) cr_field_for_metric(seed0, bufA, bufB, src4, oM, CW, CH, srcPitch, wx0, wy0,
							offX, offY, inW, inH, CR_METRIC_LINF, subpixel, 1, gridCrop, block);
		if (needB) cr_field_for_metric(seed0, bufA, bufB, src4, oB, CW, CH, srcPitch, wx0, wy0,
							offX, offY, inW, inH, CR_METRIC_L1, subpixel, 1, gridCrop, block);

		// --- INNER fields (transparent pixels are the source) ---
		if (needInside) {
			cr_seed<<<gridCrop, block>>>(src4, seed0, CW, CH, srcPitch, wx0, wy0,
										 offX, offY, inW, inH, thr, 0);
			cr_field_for_metric(seed0, bufA, bufB, src4, iR, CW, CH, srcPitch, wx0, wy0,
								offX, offY, inW, inH, CR_METRIC_L2, subpixel, 0, gridCrop, block);
			if (needM) cr_field_for_metric(seed0, bufA, bufB, src4, iM, CW, CH, srcPitch, wx0, wy0,
								offX, offY, inW, inH, CR_METRIC_LINF, subpixel, 0, gridCrop, block);
			if (needB) cr_field_for_metric(seed0, bufA, bufB, src4, iB, CW, CH, srcPitch, wx0, wy0,
								offX, offY, inW, inH, CR_METRIC_L1, subpixel, 0, gridCrop, block);
		}

		// Unused fields alias oR/iR so the composite always reads valid pointers.
		const float *pOM = needM ? oM : oR;
		const float *pOB = needB ? oB : oR;
		const float *pIR = needInside ? iR : oR;
		const float *pIM = (needInside && needM) ? iM : pIR;
		const float *pIB = (needInside && needB) ? iB : pIR;

		cr_composite<<<gridFull, block>>>(pOM, oR, pOB, pIM, pIR, pIB,
										  needInside, devStrokes, count, feather,
										  src4, (float4*)dst, W, H, srcPitch, dstPitch,
										  offX, offY, inW, inH, thr, wx0, wy0, CW, CH);
		cudaDeviceSynchronize();
	}

cleanup:
	if (seed0)      cudaFree(seed0);
	if (bufA)       cudaFree(bufA);
	if (bufB)       cudaFree(bufB);
	if (oM)         cudaFree(oM);
	if (oR)         cudaFree(oR);
	if (oB)         cudaFree(oB);
	if (iM)         cudaFree(iM);
	if (iR)         cudaFree(iR);
	if (iB)         cudaFree(iB);
	if (devStrokes) cudaFree(devStrokes);
	if (devBox)     cudaFree(devBox);
	#undef CR_TRY
}
