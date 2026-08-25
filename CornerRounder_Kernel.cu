/*
	CornerRounder_Kernel.cu

	GPU render path (CUDA) for Corner Rounder - a faithful port of the CPU
	SmartRender's circular + squircle pipeline (python-proto/corner_rounder,
	cr_step3..5). The one deliberate divergence is JFA (Jump Flooding, parallel,
	log2(n) passes) in place of the CPU's sequential FH-EDT / chamfer transform.

	PIPELINE (mirrors the CPU exactly):
	  threshold alpha -> mask; then run OPEN then CLOSE as distance-field
	  thresholds to get one signed field g (+inside the rounded shape):
	    d_in(S)              JFA seeded on OUTSIDE pixels
	    E = d_in(S) > rv     erode
	    d_out(E)             JFA seeded on E
	    O = d_out(E) <= rv   dilate  (opened set)
	    d_out(O)             JFA seeded on O
	    D = d_out(O) <= rc   dilate
	    g = d_in(D) - d_out(D) - rc     (signed sdf of D, minus rc = erode by rc)
	  rv<0.5 -> O = mask; rc<0.5 -> convex-only g = rv - d_out(E).
	  CORNER PROFILE: run the whole chain twice - L2 (round) and L-inf (square) -
	  and blend g = (1-t)*g_round + t*g_square. AA stays clean from the blend (the
	  L2 part keeps the field fractional), so no |grad| norm / sub-pixel needed.
	  coverage = smoothstep(-feather, feather, g); preserve source AA on unmoved
	  edges; Amount blends source->rounded. Colour = nearest OPAQUE source pixel
	  (JFA feature coords) so added pixels edge-extend instead of fringing black.

	COORDINATE SPACES:
	  - OUTPUT (x,y): [0,W)x[0,H), the world AE hands us. src/dst reads.
	  - INPUT: output minus (offX,offY).
	  - CROPPED (cx,cy): [0,CW)x[0,CH), the work rect (shape bbox + pad). All the
	    field buffers and the JFA live here; cropped maps to output (wx0+cx,wy0+cy).
	    Distances are translation-invariant so cropped seeds give the same result.

	Pixel format GPU_BGRA128: float4 laid out B,G,R,A (.w = alpha). Source alpha
	is STRAIGHT; we output straight rgb + the new coverage alpha (AE premultiplies).
*/

#include <cuda_runtime.h>

#define CR_NONE (-1)
#define CR_FAR  1.0e7f			// finite "unreachable" cap, matches the CPU field

#define CR_METRIC_L2   0		// round  (Euclidean)
#define CR_METRIC_LINF 2		// square (Chebyshev) - matches the CPU CR_Chamfer

__device__ __forceinline__ float
cr_smoothstep(float e0, float e1, float x)
{
	float d = e1 - e0;
	if (d < 1e-6f) return x < e0 ? 0.0f : 1.0f;
	float t = (x - e0) / d;
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
	return t * t * (3.0f - 2.0f * t);
}

/* Candidate distance from a (dx,dy) offset under a metric. L2 stays SQUARED
   (monotone -> the nearer test is identical, defer the sqrt to resolve). */
__device__ __forceinline__ float
cr_metric_dist(int dx, int dy, int metric)
{
	if (metric == CR_METRIC_LINF) return (float)max(abs(dx), abs(dy));
	return (float)(dx * dx + dy * dy);				// L2, squared
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

/* Shape bounding box in OUTPUT space via atomics into bbox[minx,miny,maxx,maxy]
   (init {W,H,-1,-1}); opaque pixels extend it. Lets everything crop to bbox+pad. */
__global__ void
cr_bbox(const float4 *src, int *bbox, int W, int H, int srcPitch,
		int offX, int offY, int inW, int inH, float thr)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	if (cr_alpha_at(src, x, y, srcPitch, offX, offY, inW, inH) >= thr) {
		atomicMin(&bbox[0], x);  atomicMin(&bbox[1], y);
		atomicMax(&bbox[2], x);  atomicMax(&bbox[3], y);
	}
}

/* Copy source straight to dst (honouring the expansion offset). Empty-shape /
   out-of-work-rect fallback. */
__global__ void
cr_copythrough(const float4 *src, float4 *dst, int W, int H,
			   int srcPitch, int dstPitch, int offX, int offY, int inW, int inH)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	int iu = x - offX, iv = y - offY;
	float4 sp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (iu >= 0 && iv >= 0 && iu < inW && iv < inH) sp = src[iv * srcPitch + iu];
	dst[y * dstPitch + x] = sp;
}

/* Seed from the source ALPHA. srcIsInside=1 -> opaque pixels are seeds (used for
   d_out(S) and the colour feature); =0 -> transparent pixels are seeds (d_in(S)).
   Seeds store their own CROPPED coords. */
__global__ void
cr_seed_alpha(const float4 *src, int2 *seed, int CW, int CH, int srcPitch,
			  int wx0, int wy0, int offX, int offY, int inW, int inH,
			  float thr, int srcIsInside)
{
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	if (cx >= CW || cy >= CH) return;
	float a = cr_alpha_at(src, wx0 + cx, wy0 + cy, srcPitch, offX, offY, inW, inH);
	bool inside = (a >= thr);
	bool isSrc  = srcIsInside ? inside : !inside;
	seed[cy * CW + cx] = isSrc ? make_int2(cx, cy) : make_int2(CR_NONE, CR_NONE);
}

/* Seed from a computed FIELD (the morphology's intermediate sets E, O, D).
   greater=1 -> seed where fld > thrVal; greater=0 -> where fld <= thrVal. */
__global__ void
cr_seed_field(const float *fld, int2 *seed, int CW, int CH, float thrVal, int greater)
{
	int cx = blockIdx.x * blockDim.x + threadIdx.x;
	int cy = blockIdx.y * blockDim.y + threadIdx.y;
	if (cx >= CW || cy >= CH) return;
	float v = fld[cy * CW + cx];
	bool isSrc = greater ? (v > thrVal) : (v <= thrVal);
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
	float bestD = (best.x < 0) ? 1e30f : cr_metric_dist(x - best.x, y - best.y, metric);

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

/* Resolve a finished seed field to a plain distance in PIXELS (L2 -> sqrt of the
   squared metric; L-inf is already linear). CR_FAR where no seed was reached. */
__global__ void
cr_resolve_dist(const int2 *seed, float *fld, int CW, int CH, int metric)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= CW || y >= CH) return;
	int2 s = seed[y * CW + x];
	float d;
	if (s.x < 0) d = CR_FAR;
	else {
		int dx = x - s.x, dy = y - s.y;
		d = (metric == CR_METRIC_L2) ? sqrtf((float)(dx * dx + dy * dy))
									 : cr_metric_dist(dx, dy, metric);
	}
	if (!(d >= 0.0f)) d = 0.0f;
	if (d > CR_FAR)   d = CR_FAR;
	fld[y * CW + x] = d;
}

/* g = a - b - rc  (close: signed_sdf(D) - rc, with a=d_in(D), b=d_out(D)). */
__global__ void
cr_field_sub(float *g, const float *a, const float *b, float rc, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) g[i] = a[i] - b[i] - rc;
}

/* g = rv - a  (convex-only: rv - d_out(E)). */
__global__ void
cr_field_rsub(float *g, const float *a, float rv, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) g[i] = rv - a[i];
}

/* g = (1-t)*g + t*gs  (corner-profile metric blend). */
__global__ void
cr_field_lerp(float *g, const float *gs, float t, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) g[i] = (1.0f - t) * g[i] + t * gs[i];
}

/* Max of a distance field (fixed-point *256 into an int, so atomicMax works),
   ignoring the CR_FAR "unreachable" sentinel. Used to find the shape's inradius
   for the inscribed-circle radius clamp. */
__global__ void
cr_reduce_maxdin(const float *f, int *outMax, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	float v = f[i];
	if (v < CR_FAR * 0.5f && v > 0.0f)
		atomicMax(outMax, (int)(v * 256.0f));
}

/* Composite over the FULL output world. Outside the work rect: copy source
   through. Inside: coverage from the signed field, preserve source AA on unmoved
   edges, Amount blend, colour from the nearest opaque source pixel. */
__global__ void
cr_composite(const float *g, const int2 *feat,
			 const float4 *src, float4 *dst, int W, int H, int srcPitch, int dstPitch,
			 int offX, int offY, int inW, int inH, float thr,
			 float feather, float amount, int preserveAA,
			 int wx0, int wy0, int CW, int CH)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	int iu = x - offX, iv = y - offY;
	float4 sp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	if (iu >= 0 && iv >= 0 && iu < inW && iv < inH) sp = src[iv * srcPitch + iu];

	int cx = x - wx0, cy = y - wy0;
	if (cx < 0 || cy < 0 || cx >= CW || cy >= CH) { dst[y * dstPitch + x] = sp; return; }
	int k = cy * CW + cx;

	float gv  = g[k];
	float cov = cr_smoothstep(-feather, feather, gv);
	float a   = sp.w;
	if (preserveAA && gv <= 1.0f && gv >= -1.0f && a > 0.0f && a < 1.0f)
		cov = a;						// unmoved edge -> keep source AA
	float outA = (1.0f - amount) * a + amount * cov;

	// Colour: nearest opaque source pixel (edge extend). feat stores cropped coords.
	float4 col = sp;
	int2 fs = feat[k];
	if (fs.x >= 0) {
		int fu = (wx0 + fs.x) - offX, fv = (wy0 + fs.y) - offY;
		if (fu >= 0 && fv >= 0 && fu < inW && fv < inH) col = src[fv * srcPitch + fu];
	}

	float4 o;
	o.x = col.x; o.y = col.y; o.z = col.z;		// straight B,G,R
	if (!(outA >= 0.0f)) outA = 0.0f;
	if (outA > 1.0f)     outA = 1.0f;
	o.w = outA;
	dst[y * dstPitch + x] = o;
}

/* ---- host orchestration ------------------------------------------------- */

/* Full JFA for one metric -> distance field `fld` (cropped). seed0 already holds
   the seeds; bufA/bufB are ping-pong scratch. */
static void
cr_jfa(const int2 *seed0, int2 *bufA, int2 *bufB, float *fld,
	   int CW, int CH, int metric, dim3 grid, dim3 block)
{
	int N = CW * CH;
	cudaMemcpy(bufA, seed0, (size_t)N * sizeof(int2), cudaMemcpyDeviceToDevice);
	int maxdim = (CW > CH) ? CW : CH;
	int step = 1; while (step < maxdim) step <<= 1; step >>= 1;
	int2 *cur = bufA, *other = bufB;
	for (; step >= 1; step >>= 1) {
		cr_flood<<<grid, block>>>(cur, other, CW, CH, step, metric);
		int2 *t = cur; cur = other; other = t;
	}
	cr_resolve_dist<<<grid, block>>>(cur, fld, CW, CH, metric);
}

/* Build the signed rounded field g (+inside) under `metric`, mirroring the CPU
   buildG lambda. Scratch fA/fB/fC are cropped float buffers. */
static void
cr_buildG(int metric, float *gOut,
		  int2 *seed0, int2 *bufA, int2 *bufB, float *fA, float *fB, float *fC,
		  int *devMax, const float4 *src, int CW, int CH, int srcPitch, int wx0, int wy0,
		  int offX, int offY, int inW, int inH, float thr, float rv, float rc,
		  dim3 gridCrop, dim3 block)
{
	int N = CW * CH;
	dim3 b1(256, 1, 1), g1((N + 255) / 256, 1, 1);

	// d_in(S): JFA seeded on OUTSIDE pixels.
	cr_seed_alpha<<<gridCrop, block>>>(src, seed0, CW, CH, srcPitch, wx0, wy0,
									   offX, offY, inW, inH, thr, 0);
	cr_jfa(seed0, bufA, bufB, fA, CW, CH, metric, gridCrop, block);	// fA = d_in(S)

	// INSCRIBED-CIRCLE CLAMP: convex erosion empties the shape once the radius
	// passes the inradius; the concave close then vanishes it too (its dilate
	// saturates the work rect, the erode-back removes everything). Cap BOTH radii
	// at the inradius so rounding tops out as the inscribed circle. Reduce d_in(S)
	// -> maxDin (D2H sync, like the bbox copy).
	float rvE = rv, rcE = rc;
	int   didOpen = 0;
	if (rv >= 0.5f || rc >= 0.5f) {
		int zero = 0;
		cudaMemcpy(devMax, &zero, sizeof(int), cudaMemcpyHostToDevice);
		cr_reduce_maxdin<<<g1, b1>>>(fA, devMax, N);
		int hMax = 0;
		cudaMemcpy(&hMax, devMax, sizeof(int), cudaMemcpyDeviceToHost);
		float cap = (hMax / 256.0f) - 0.5f; if (cap < 0.0f) cap = 0.0f;
		if (rvE > cap) rvE = cap;
		if (rcE > cap) rcE = cap;
		if (rvE >= 0.5f) {						// O = dilate(erode(S,rvE),rvE)
			cr_seed_field<<<gridCrop, block>>>(fA, seed0, CW, CH, rvE, 1);	// E = d_in(S)>rvE
			cr_jfa(seed0, bufA, bufB, fB, CW, CH, metric, gridCrop, block);	// fB = d_out(E)
			didOpen = 1;
		}
	}

	if (rcE >= 0.5f) {
		// d_out(O): O = didOpen ? d_out(E)<=rvE : the mask itself.
		if (didOpen) cr_seed_field<<<gridCrop, block>>>(fB, seed0, CW, CH, rvE, 0);
		else         cr_seed_alpha<<<gridCrop, block>>>(src, seed0, CW, CH, srcPitch,
										wx0, wy0, offX, offY, inW, inH, thr, 1);
		cr_jfa(seed0, bufA, bufB, fA, CW, CH, metric, gridCrop, block);	// fA = d_out(O)

		cr_seed_field<<<gridCrop, block>>>(fA, seed0, CW, CH, rcE, 1);	// !D = d_out(O)>rcE
		cr_jfa(seed0, bufA, bufB, fB, CW, CH, metric, gridCrop, block);	// fB = d_in(D)
		cr_seed_field<<<gridCrop, block>>>(fA, seed0, CW, CH, rcE, 0);	// D = d_out(O)<=rcE
		cr_jfa(seed0, bufA, bufB, fC, CW, CH, metric, gridCrop, block);	// fC = d_out(D)
		cr_field_sub<<<g1, b1>>>(gOut, fB, fC, rcE, N);					// g = d_in(D)-d_out(D)-rcE
	} else if (didOpen) {
		cr_field_rsub<<<g1, b1>>>(gOut, fB, rvE, N);					// convex-only: rvE-d_out(E)
	} else {
		// radius collapsed to ~0 (sub-pixel-thin shape): signed sdf of the mask.
		// fA still holds d_in(S); compute d_out(S) into fB, g = d_in(S)-d_out(S).
		cr_seed_alpha<<<gridCrop, block>>>(src, seed0, CW, CH, srcPitch, wx0, wy0,
										   offX, offY, inW, inH, thr, 1);
		cr_jfa(seed0, bufA, bufB, fB, CW, CH, metric, gridCrop, block);
		cr_field_sub<<<g1, b1>>>(gOut, fA, fB, 0.0f, N);
	}
}

/* Host launcher. Declared extern (no extern "C") in the .cpp. */
void
CR_CornerRound_CUDA(
	const float *src, float *dst, int srcPitch, int dstPitch,
	int W, int H, int offX, int offY, int inW, int inH,
	float thr, float rv, float rc, float feather, float amount,
	float profile, int preserveAA, int pad)
{
	const float4 *src4 = (const float4*)src;
	dim3 block(16, 16, 1);
	dim3 gridFull((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, 1);

	// No rounding, or Amount 0 -> straight passthrough.
	if ((rv < 0.5f && rc < 0.5f) || amount <= 1e-6f) {
		cr_copythrough<<<gridFull, block>>>(src4, (float4*)dst, W, H,
											srcPitch, dstPitch, offX, offY, inW, inH);
		cudaDeviceSynchronize();
		return;
	}

	int2  *seed0 = 0, *bufA = 0, *bufB = 0;
	float *fA = 0, *fB = 0, *fC = 0, *gR = 0, *gS = 0;
	int   *devBox = 0, *devMax = 0;

	#define CR_TRY(call) do { if ((call) != cudaSuccess) goto cleanup; } while (0)

	int hostBox[4] = { W, H, -1, -1 };
	CR_TRY(cudaMalloc(&devBox, 4 * sizeof(int)));
	CR_TRY(cudaMemcpy(devBox, hostBox, 4 * sizeof(int), cudaMemcpyHostToDevice));
	cr_bbox<<<gridFull, block>>>(src4, devBox, W, H, srcPitch, offX, offY, inW, inH, thr);
	CR_TRY(cudaMemcpy(hostBox, devBox, 4 * sizeof(int), cudaMemcpyDeviceToHost));

	if (hostBox[2] < hostBox[0] || hostBox[3] < hostBox[1]) {	// empty shape
		cr_copythrough<<<gridFull, block>>>(src4, (float4*)dst, W, H,
											srcPitch, dstPitch, offX, offY, inW, inH);
		cudaDeviceSynchronize();
		goto cleanup;
	}

	{
		int wx0 = hostBox[0] - pad; if (wx0 < 0) wx0 = 0;
		int wy0 = hostBox[1] - pad; if (wy0 < 0) wy0 = 0;
		int wx1 = hostBox[2] + pad; if (wx1 > W - 1) wx1 = W - 1;
		int wy1 = hostBox[3] + pad; if (wy1 > H - 1) wy1 = H - 1;
		int CW = wx1 - wx0 + 1, CH = wy1 - wy0 + 1, N = CW * CH;
		dim3 gridCrop((CW + block.x - 1) / block.x, (CH + block.y - 1) / block.y, 1);

		CR_TRY(cudaMalloc(&seed0, (size_t)N * sizeof(int2)));
		CR_TRY(cudaMalloc(&bufA,  (size_t)N * sizeof(int2)));
		CR_TRY(cudaMalloc(&bufB,  (size_t)N * sizeof(int2)));
		CR_TRY(cudaMalloc(&fA,    (size_t)N * sizeof(float)));
		CR_TRY(cudaMalloc(&fB,    (size_t)N * sizeof(float)));
		CR_TRY(cudaMalloc(&fC,    (size_t)N * sizeof(float)));
		CR_TRY(cudaMalloc(&gR,    (size_t)N * sizeof(float)));
		CR_TRY(cudaMalloc(&devMax, sizeof(int)));

		cr_buildG(CR_METRIC_L2, gR, seed0, bufA, bufB, fA, fB, fC, devMax, src4,
				  CW, CH, srcPitch, wx0, wy0, offX, offY, inW, inH, thr, rv, rc,
				  gridCrop, block);

		float t = (profile < 0.0f) ? 0.0f : (profile > 1.0f ? 1.0f : profile);
		if (t > 1e-4f) {
			CR_TRY(cudaMalloc(&gS, (size_t)N * sizeof(float)));
			cr_buildG(CR_METRIC_LINF, gS, seed0, bufA, bufB, fA, fB, fC, devMax, src4,
					  CW, CH, srcPitch, wx0, wy0, offX, offY, inW, inH, thr, rv, rc,
					  gridCrop, block);
			dim3 b1(256, 1, 1), g1((N + 255) / 256, 1, 1);
			cr_field_lerp<<<g1, b1>>>(gR, gS, t, N);
		}

		// Colour feature: nearest OPAQUE source pixel. JFA (L2), keep the seed
		// coords (no resolve) - the final flood buffer IS the feature field.
		cr_seed_alpha<<<gridCrop, block>>>(src4, seed0, CW, CH, srcPitch, wx0, wy0,
										   offX, offY, inW, inH, thr, 1);
		cudaMemcpy(bufA, seed0, (size_t)N * sizeof(int2), cudaMemcpyDeviceToDevice);
		int maxdim = (CW > CH) ? CW : CH;
		int step = 1; while (step < maxdim) step <<= 1; step >>= 1;
		int2 *cur = bufA, *other = bufB;
		for (; step >= 1; step >>= 1) {
			cr_flood<<<gridCrop, block>>>(cur, other, CW, CH, step, CR_METRIC_L2);
			int2 *tt = cur; cur = other; other = tt;
		}

		cr_composite<<<gridFull, block>>>(gR, cur, src4, (float4*)dst,
										  W, H, srcPitch, dstPitch, offX, offY, inW, inH,
										  thr, feather, amount, preserveAA, wx0, wy0, CW, CH);
		cudaDeviceSynchronize();
	}

cleanup:
	if (seed0)  cudaFree(seed0);
	if (bufA)   cudaFree(bufA);
	if (bufB)   cudaFree(bufB);
	if (fA)     cudaFree(fA);
	if (fB)     cudaFree(fB);
	if (fC)     cudaFree(fC);
	if (gR)     cudaFree(gR);
	if (gS)     cudaFree(gS);
	if (devBox) cudaFree(devBox);
	if (devMax) cudaFree(devMax);
	#undef CR_TRY
}
