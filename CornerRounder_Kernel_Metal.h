/*	CornerRounder_Kernel_Metal.h

	Metal (MSL) port of CornerRounder_Kernel.cu, for the macOS GPU path.

	A faithful 1:1 translation of the CUDA kernels - same JFA morphology (open then
	close as distance-field thresholds), same dual-metric (L2 round / L-inf square)
	blend, same inscribed-circle radius clamp, same edge-extend colour feature. The
	host orchestrator lives in CornerRounder_Metal.mm and mirrors CR_CornerRound_CUDA.

	The MSL source is embedded as a C string and compiled at RUNTIME via
	-[MTLDevice newLibraryWithSource:...], exactly like the other ported plugins.
	Differences from CUDA are pure API, not algorithm:
	  - [[thread_position_in_grid]] instead of blockIdx/threadIdx
	  - scalar kernel args arrive as `constant T&` (set via setBytes); the shared
	    geometry is packed into one CRGpuParams struct
	  - the device bbox / max reductions use MSL atomic_int + atomic_fetch_*
	AE hands worlds as GPU_BGRA128 (float4, .x=B .y=G .z=R .w=A).
*/

#pragma once

static const char *kCRKernelMetalString = R"CRMETAL(
#include <metal_stdlib>
using namespace metal;

#define CR_NONE (-1)
#define CR_FAR  1.0e7f

#define CR_METRIC_L2   0
#define CR_METRIC_LINF 2

// LAYOUT MUST MATCH CRGpuParams in CornerRounder_Metal.mm exactly.
struct CRGpuParams {
    int   W, H;
    int   srcPitch, dstPitch;
    int   offX, offY;
    int   inW, inH;
    int   wx0, wy0, CW, CH;
    float thr, feather, amount;
    int   preserveAA, pad;
};

inline float cr_smoothstep_(float e0, float e1, float x) {
    float d = e1 - e0;
    if (d < 1e-6f) return x < e0 ? 0.0f : 1.0f;
    float t = (x - e0) / d;
    t = clamp(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/* Candidate distance under a metric. L2 stays SQUARED (defer the sqrt). */
inline float cr_metric_dist(int dx, int dy, int metric) {
    if (metric == CR_METRIC_LINF) return (float)max(abs(dx), abs(dy));
    return (float)(dx * dx + dy * dy);
}

/* Read input alpha under OUTPUT pixel (ox,oy); 0 if outside the input world. */
inline float cr_alpha_at(const device float4 *src, int ox, int oy, int srcPitch,
                         int offX, int offY, int inW, int inH) {
    int iu = ox - offX, iv = oy - offY;
    if (iu < 0 || iv < 0 || iu >= inW || iv >= inH) return 0.0f;
    return src[iv * srcPitch + iu].w;
}

/* Shape bounding box in OUTPUT space via atomics into box[minx,miny,maxx,maxy]. */
kernel void cr_bbox(const device float4 *src [[buffer(0)]],
                    device atomic_int   *box [[buffer(1)]],
                    constant CRGpuParams &p  [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.W || y >= p.H) return;
    if (cr_alpha_at(src, x, y, p.srcPitch, p.offX, p.offY, p.inW, p.inH) >= p.thr) {
        atomic_fetch_min_explicit(&box[0], x, memory_order_relaxed);
        atomic_fetch_min_explicit(&box[1], y, memory_order_relaxed);
        atomic_fetch_max_explicit(&box[2], x, memory_order_relaxed);
        atomic_fetch_max_explicit(&box[3], y, memory_order_relaxed);
    }
}

/* Copy source straight to dst (honouring the expansion offset). */
kernel void cr_copythrough(const device float4 *src [[buffer(0)]],
                           device float4       *dst [[buffer(1)]],
                           constant CRGpuParams &p  [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.W || y >= p.H) return;
    int iu = x - p.offX, iv = y - p.offY;
    float4 sp = float4(0.0f);
    if (iu >= 0 && iv >= 0 && iu < p.inW && iv < p.inH) sp = src[iv * p.srcPitch + iu];
    dst[y * p.dstPitch + x] = sp;
}

/* Seed from the source ALPHA. srcIsInside=1 -> opaque pixels are seeds; =0 ->
   transparent pixels are seeds. Seeds store their own CROPPED coords. */
kernel void cr_seed_alpha(const device float4 *src [[buffer(0)]],
                          device int2         *seed [[buffer(1)]],
                          constant CRGpuParams &p    [[buffer(2)]],
                          constant int         &srcIsInside [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    int cx = gid.x, cy = gid.y;
    if (cx >= p.CW || cy >= p.CH) return;
    float a = cr_alpha_at(src, p.wx0 + cx, p.wy0 + cy, p.srcPitch, p.offX, p.offY, p.inW, p.inH);
    bool inside = (a >= p.thr);
    bool isSrc  = srcIsInside ? inside : !inside;
    seed[cy * p.CW + cx] = isSrc ? int2(cx, cy) : int2(CR_NONE, CR_NONE);
}

/* Seed from a computed FIELD. greater=1 -> seed where fld > thrVal; =0 -> <=. */
kernel void cr_seed_field(const device float *fld  [[buffer(0)]],
                          device int2        *seed [[buffer(1)]],
                          constant CRGpuParams &p   [[buffer(2)]],
                          constant float       &thrVal [[buffer(3)]],
                          constant int         &greater [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    int cx = gid.x, cy = gid.y;
    if (cx >= p.CW || cy >= p.CH) return;
    float v = fld[cy * p.CW + cx];
    bool isSrc = greater ? (v > thrVal) : (v <= thrVal);
    seed[cy * p.CW + cx] = isSrc ? int2(cx, cy) : int2(CR_NONE, CR_NONE);
}

/* One jump-flood pass at `step` under `metric`, in cropped space. */
kernel void cr_flood(const device int2 *inbuf [[buffer(0)]],
                     device int2       *outbuf [[buffer(1)]],
                     constant CRGpuParams &p    [[buffer(2)]],
                     constant int         &step [[buffer(3)]],
                     constant int         &metric [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.CW || y >= p.CH) return;

    int2  best  = inbuf[y * p.CW + x];
    float bestD = (best.x < 0) ? 1e30f : cr_metric_dist(x - best.x, y - best.y, metric);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx * step, ny = y + dy * step;
            if (nx < 0 || ny < 0 || nx >= p.CW || ny >= p.CH) continue;
            int2 s = inbuf[ny * p.CW + nx];
            if (s.x < 0) continue;
            float d = cr_metric_dist(x - s.x, y - s.y, metric);
            if (d < bestD) { bestD = d; best = s; }
        }
    }
    outbuf[y * p.CW + x] = best;
}

/* Resolve a finished seed field to a plain distance in PIXELS. */
kernel void cr_resolve_dist(const device int2 *seed [[buffer(0)]],
                            device float      *fld  [[buffer(1)]],
                            constant CRGpuParams &p   [[buffer(2)]],
                            constant int         &metric [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.CW || y >= p.CH) return;
    int2 s = seed[y * p.CW + x];
    float d;
    if (s.x < 0) d = CR_FAR;
    else {
        int dx = x - s.x, dy = y - s.y;
        d = (metric == CR_METRIC_L2) ? sqrt((float)(dx * dx + dy * dy))
                                     : cr_metric_dist(dx, dy, metric);
    }
    if (!(d >= 0.0f)) d = 0.0f;
    if (d > CR_FAR)   d = CR_FAR;
    fld[y * p.CW + x] = d;
}

/* g = a - b - rc. */
kernel void cr_field_sub(device float       *g [[buffer(0)]],
                         const device float *a [[buffer(1)]],
                         const device float *b [[buffer(2)]],
                         constant float     &rc [[buffer(3)]],
                         constant int       &N  [[buffer(4)]],
                         uint gid [[thread_position_in_grid]]) {
    int i = gid;
    if (i < N) g[i] = a[i] - b[i] - rc;
}

/* g = rv - a. */
kernel void cr_field_rsub(device float       *g [[buffer(0)]],
                          const device float *a [[buffer(1)]],
                          constant float     &rv [[buffer(2)]],
                          constant int       &N  [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    int i = gid;
    if (i < N) g[i] = rv - a[i];
}

/* g = (1-t)*g + t*gs. */
kernel void cr_field_lerp(device float       *g  [[buffer(0)]],
                          const device float *gs [[buffer(1)]],
                          constant float     &t  [[buffer(2)]],
                          constant int       &N  [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    int i = gid;
    if (i < N) g[i] = (1.0f - t) * g[i] + t * gs[i];
}

/* Max of a distance field (fixed-point *256 into an int), ignoring CR_FAR. */
kernel void cr_reduce_maxdin(const device float *f [[buffer(0)]],
                             device atomic_int  *outMax [[buffer(1)]],
                             constant int       &N [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    int i = gid;
    if (i >= N) return;
    float v = f[i];
    if (v < CR_FAR * 0.5f && v > 0.0f)
        atomic_fetch_max_explicit(outMax, (int)(v * 256.0f), memory_order_relaxed);
}

/* Composite over the FULL output world. */
kernel void cr_composite(const device float  *g   [[buffer(0)]],
                         const device int2   *feat [[buffer(1)]],
                         const device float4 *src  [[buffer(2)]],
                         device float4       *dst  [[buffer(3)]],
                         constant CRGpuParams &p    [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int x = gid.x, y = gid.y;
    if (x >= p.W || y >= p.H) return;

    int iu = x - p.offX, iv = y - p.offY;
    float4 sp = float4(0.0f);
    if (iu >= 0 && iv >= 0 && iu < p.inW && iv < p.inH) sp = src[iv * p.srcPitch + iu];

    int cx = x - p.wx0, cy = y - p.wy0;
    if (cx < 0 || cy < 0 || cx >= p.CW || cy >= p.CH) { dst[y * p.dstPitch + x] = sp; return; }
    int k = cy * p.CW + cx;

    float gv  = g[k];
    float cov = cr_smoothstep_(-p.feather, p.feather, gv);
    float a   = sp.w;
    if (p.preserveAA && gv <= 1.0f && gv >= -1.0f && a > 0.0f && a < 1.0f)
        cov = a;
    float outA = (1.0f - p.amount) * a + p.amount * cov;

    float4 col = sp;
    int2 fs = feat[k];
    if (fs.x >= 0) {
        int fu = (p.wx0 + fs.x) - p.offX, fv = (p.wy0 + fs.y) - p.offY;
        if (fu >= 0 && fv >= 0 && fu < p.inW && fv < p.inH) col = src[fv * p.srcPitch + fu];
    }

    float4 o;
    o.x = col.x; o.y = col.y; o.z = col.z;
    if (!(outA >= 0.0f)) outA = 0.0f;
    if (outA > 1.0f)     outA = 1.0f;
    o.w = outA;
    dst[y * p.dstPitch + x] = o;
}
)CRMETAL";
