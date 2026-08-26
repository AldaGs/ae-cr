/*	CornerRounder_Metal.mm

	macOS Metal host for the CornerRounder GPU path. Three entry points, mirrored
	on the other ported plugins:

	  CR_MetalCompile   - GPU_DEVICE_SETUP: compile the MSL library once per device
	                      and build one pipeline state per kernel; stash in gpu_data.
	  CR_MetalDispose   - GPU_DEVICE_SETDOWN: release the pipeline states.
	  CR_CornerRound_Metal - per frame: a step-for-step twin of CR_CornerRound_CUDA.
	                      bbox (readback) -> crop -> buildG per metric (with a mid
	                      inscribed-circle max reduction readback) -> profile blend
	                      -> colour-feature JFA -> composite.

	Unlike the simpler ports this pipeline has host-readback sync points inside the
	frame (bbox + maxDin), so the encoder is FLUSHED (commit + wait) at each and a
	fresh one opened. Between flushes, dispatches batch in one serial compute encoder
	(Metal auto-barriers the tracked JFA ping-pong buffers). Seeds are written
	directly into bufA (no seed0 + device->device copy, unlike the CUDA path).

	The MSL source lives in CornerRounder_Kernel_Metal.h.
*/

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <vector>

#include "CornerRounder_Kernel_Metal.h"

#define CR_METRIC_L2   0
#define CR_METRIC_LINF 2

/* LAYOUT MUST MATCH CRGpuParams in CornerRounder_Kernel_Metal.h and the copy in
   CornerRounder.cpp exactly. */
struct CRGpuParams {
	int   W, H;
	int   srcPitch, dstPitch;
	int   offX, offY;
	int   inW, inH;
	int   wx0, wy0, CW, CH;
	float thr, feather, amount;
	int   preserveAA, pad;
};

struct CRMetalGPUData {
	id<MTLComputePipelineState> bbox, copythrough, seed_alpha, seed_field, flood,
								resolve_dist, field_sub, field_rsub, field_lerp,
								reduce_maxdin, composite;
};

/* ------------------------------------------------------- compile / dispose */

extern "C" void CR_MetalDispose (void *dataPV);	// defined below; used in Compile

extern "C" bool
CR_MetalCompile (void *devicePV, void **outData, char *errBuf, int errLen)
{
	@autoreleasepool {
		id<MTLDevice> device = (id<MTLDevice>)devicePV;
		NSError *error = nil;
		NSString *source = [NSString stringWithUTF8String:kCRKernelMetalString];
		id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
		if (!lib) {
			if (errBuf && errLen > 0) {
				const char *m = error ? [[error localizedDescription] UTF8String]
									   : "unknown Metal compile error";
				strncpy(errBuf, m ? m : "nil", errLen - 1);
				errBuf[errLen - 1] = 0;
			}
			return false;
		}

		CRMetalGPUData *d = (CRMetalGPUData*)calloc(1, sizeof(CRMetalGPUData));
		bool ok = true;

		auto mk = [&](id<MTLComputePipelineState> __strong *slot, const char *name) {
			if (!ok) return;
			id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
			if (!fn) {
				if (errBuf && errLen > 0) { snprintf(errBuf, errLen, "missing kernel: %s", name); }
				ok = false; return;
			}
			NSError *e = nil;
			*slot = [device newComputePipelineStateWithFunction:fn error:&e];
			[fn release];
			if (!*slot) {
				if (errBuf && errLen > 0) {
					const char *m = e ? [[e localizedDescription] UTF8String] : "pso failed";
					snprintf(errBuf, errLen, "pso %s: %s", name, m ? m : "nil");
				}
				ok = false;
			}
		};

		mk(&d->bbox,          "cr_bbox");
		mk(&d->copythrough,   "cr_copythrough");
		mk(&d->seed_alpha,    "cr_seed_alpha");
		mk(&d->seed_field,    "cr_seed_field");
		mk(&d->flood,         "cr_flood");
		mk(&d->resolve_dist,  "cr_resolve_dist");
		mk(&d->field_sub,     "cr_field_sub");
		mk(&d->field_rsub,    "cr_field_rsub");
		mk(&d->field_lerp,    "cr_field_lerp");
		mk(&d->reduce_maxdin, "cr_reduce_maxdin");
		mk(&d->composite,     "cr_composite");

		[lib release];

		if (!ok) { CR_MetalDispose(d); return false; }
		*outData = d;
		return true;
	}
}

extern "C" void
CR_MetalDispose (void *dataPV)
{
	CRMetalGPUData *d = (CRMetalGPUData*)dataPV;
	if (!d) return;
	[d->bbox release];
	[d->copythrough release];
	[d->seed_alpha release];
	[d->seed_field release];
	[d->flood release];
	[d->resolve_dist release];
	[d->field_sub release];
	[d->field_rsub release];
	[d->field_lerp release];
	[d->reduce_maxdin release];
	[d->composite release];
	free(d);
}

/* -------------------------------------------------------------- the render */

extern "C" bool
CR_CornerRound_Metal (void *devicePV, void *queuePV, void *dataPV,
					  void *srcMemPV, void *dstMemPV, CRGpuParams p,
					  float rv, float rc, float profile)
{
	@autoreleasepool {
		id<MTLDevice>       device = (id<MTLDevice>)devicePV;
		id<MTLCommandQueue> queue  = (id<MTLCommandQueue>)queuePV;
		CRMetalGPUData     *d      = (CRMetalGPUData*)dataPV;
		id<MTLBuffer>       src    = (id<MTLBuffer>)srcMemPV;
		id<MTLBuffer>       dst    = (id<MTLBuffer>)dstMemPV;

		const MTLSize TG = MTLSizeMake(16, 16, 1);
		std::vector<id<MTLBuffer>> bufs;
		auto mkbuf = [&](size_t bytes) -> id<MTLBuffer> {
			id<MTLBuffer> b = [device newBufferWithLength:bytes
											options:MTLResourceStorageModePrivate];
			bufs.push_back(b);
			return b;
		};
		auto freeAll = [&]() { for (id<MTLBuffer> b : bufs) [b release]; bufs.clear(); };

		// Flushable encoder: batch dispatches, flush (commit+wait) at readbacks.
		id<MTLCommandBuffer> cb = nil;
		id<MTLComputeCommandEncoder> e = nil;
		auto begin = [&]() { cb = [queue commandBuffer]; e = [cb computeCommandEncoder]; };
		auto flush = [&]() { [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; e = nil; cb = nil; };
		auto disp2D = [&](int w, int h) {
			[e dispatchThreadgroups:MTLSizeMake((w+15)/16, (h+15)/16, 1) threadsPerThreadgroup:TG];
		};
		auto disp1D = [&](int n) {
			[e dispatchThreadgroups:MTLSizeMake((n+255)/256, 1, 1)
				threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
		};

		// No rounding, or Amount 0 -> straight passthrough.
		if ((rv < 0.5f && rc < 0.5f) || p.amount <= 1e-6f) {
			begin();
			[e setComputePipelineState:d->copythrough];
			[e setBuffer:src offset:0 atIndex:0];
			[e setBuffer:dst offset:0 atIndex:1];
			[e setBytes:&p length:sizeof(p) atIndex:2];
			disp2D(p.W, p.H);
			flush();
			freeAll();
			return true;
		}

		// --- shape bbox -> work rect (readback) -------------------------------
		int hostBox[4] = { p.W, p.H, -1, -1 };
		id<MTLBuffer> boxBuf = [device newBufferWithLength:4*sizeof(int)
											options:MTLResourceStorageModeShared];
		memcpy([boxBuf contents], hostBox, sizeof(hostBox));
		begin();
		[e setComputePipelineState:d->bbox];
		[e setBuffer:src    offset:0 atIndex:0];
		[e setBuffer:boxBuf offset:0 atIndex:1];
		[e setBytes:&p length:sizeof(p) atIndex:2];
		disp2D(p.W, p.H);
		flush();
		memcpy(hostBox, [boxBuf contents], sizeof(hostBox));
		[boxBuf release];

		// Empty shape -> copy through and return.
		if (hostBox[2] < hostBox[0] || hostBox[3] < hostBox[1]) {
			begin();
			[e setComputePipelineState:d->copythrough];
			[e setBuffer:src offset:0 atIndex:0];
			[e setBuffer:dst offset:0 atIndex:1];
			[e setBytes:&p length:sizeof(p) atIndex:2];
			disp2D(p.W, p.H);
			flush();
			freeAll();
			return true;
		}

		int wx0 = hostBox[0] - p.pad; if (wx0 < 0) wx0 = 0;
		int wy0 = hostBox[1] - p.pad; if (wy0 < 0) wy0 = 0;
		int wx1 = hostBox[2] + p.pad; if (wx1 > p.W - 1) wx1 = p.W - 1;
		int wy1 = hostBox[3] + p.pad; if (wy1 > p.H - 1) wy1 = p.H - 1;
		p.wx0 = wx0; p.wy0 = wy0; p.CW = wx1 - wx0 + 1; p.CH = wy1 - wy0 + 1;
		const int CW = p.CW, CH = p.CH, N = CW * CH;

		id<MTLBuffer> bufA = mkbuf((size_t)N * sizeof(int) * 2);	// int2
		id<MTLBuffer> bufB = mkbuf((size_t)N * sizeof(int) * 2);
		id<MTLBuffer> fA   = mkbuf((size_t)N * sizeof(float));
		id<MTLBuffer> fB   = mkbuf((size_t)N * sizeof(float));
		id<MTLBuffer> fC   = mkbuf((size_t)N * sizeof(float));
		id<MTLBuffer> gR   = mkbuf((size_t)N * sizeof(float));
		id<MTLBuffer> gS   = (profile > 1e-4f) ? mkbuf((size_t)N * sizeof(float)) : nil;
		id<MTLBuffer> maxBuf = [device newBufferWithLength:sizeof(int)
											options:MTLResourceStorageModeShared];

		// --- dispatch helpers (into the current open encoder) -----------------
		auto seedAlpha = [&](int srcIsInside) {
			[e setComputePipelineState:d->seed_alpha];
			[e setBuffer:src  offset:0 atIndex:0];
			[e setBuffer:bufA offset:0 atIndex:1];
			[e setBytes:&p length:sizeof(p) atIndex:2];
			[e setBytes:&srcIsInside length:sizeof(int) atIndex:3];
			disp2D(CW, CH);
		};
		auto seedField = [&](id<MTLBuffer> fld, float thrVal, int greater) {
			[e setComputePipelineState:d->seed_field];
			[e setBuffer:fld  offset:0 atIndex:0];
			[e setBuffer:bufA offset:0 atIndex:1];
			[e setBytes:&p length:sizeof(p) atIndex:2];
			[e setBytes:&thrVal length:sizeof(float) atIndex:3];
			[e setBytes:&greater length:sizeof(int) atIndex:4];
			disp2D(CW, CH);
		};
		// JFA: seeds already in bufA -> flood ping-pong -> resolve into fld.
		// Returns the final flood buffer (for the no-resolve colour feature case).
		int floodMetric = CR_METRIC_L2;		// captured by reference by jfaFlood
		auto jfaFlood = [&]() -> id<MTLBuffer> {
			int maxdim = (CW > CH) ? CW : CH;
			int step = 1; while (step < maxdim) step <<= 1; step >>= 1;
			id<MTLBuffer> cur = bufA, other = bufB;
			for (; step >= 1; step >>= 1) {
				[e setComputePipelineState:d->flood];
				[e setBuffer:cur   offset:0 atIndex:0];
				[e setBuffer:other offset:0 atIndex:1];
				[e setBytes:&p length:sizeof(p) atIndex:2];
				[e setBytes:&step length:sizeof(int) atIndex:3];
				[e setBytes:&floodMetric length:sizeof(int) atIndex:4];
				disp2D(CW, CH);
				id<MTLBuffer> t = cur; cur = other; other = t;
			}
			return cur;
		};
		auto jfa = [&](id<MTLBuffer> outFld, int metric) {
			floodMetric = metric;
			id<MTLBuffer> cur = jfaFlood();
			[e setComputePipelineState:d->resolve_dist];
			[e setBuffer:cur offset:0 atIndex:0];
			[e setBuffer:outFld offset:0 atIndex:1];
			[e setBytes:&p length:sizeof(p) atIndex:2];
			[e setBytes:&metric length:sizeof(int) atIndex:3];
			disp2D(CW, CH);
		};
		auto fieldSub = [&](id<MTLBuffer> g, id<MTLBuffer> a, id<MTLBuffer> b, float rcv) {
			[e setComputePipelineState:d->field_sub];
			[e setBuffer:g offset:0 atIndex:0];
			[e setBuffer:a offset:0 atIndex:1];
			[e setBuffer:b offset:0 atIndex:2];
			[e setBytes:&rcv length:sizeof(float) atIndex:3];
			[e setBytes:&N length:sizeof(int) atIndex:4];
			disp1D(N);
		};
		auto fieldRsub = [&](id<MTLBuffer> g, id<MTLBuffer> a, float rvv) {
			[e setComputePipelineState:d->field_rsub];
			[e setBuffer:g offset:0 atIndex:0];
			[e setBuffer:a offset:0 atIndex:1];
			[e setBytes:&rvv length:sizeof(float) atIndex:2];
			[e setBytes:&N length:sizeof(int) atIndex:3];
			disp1D(N);
		};

		// buildG: mirror the CUDA cr_buildG (leaves the encoder open on return).
		auto buildG = [&](int metric, id<MTLBuffer> gOut) {
			// d_in(S): JFA seeded on OUTSIDE pixels.
			seedAlpha(0);
			jfa(fA, metric);	// fA = d_in(S)

			float rvE = rv, rcE = rc;
			int   didOpen = 0;
			// Inscribed-circle clamp: reduce fA -> maxDin (needs a readback).
			{
				int zero = 0; memcpy([maxBuf contents], &zero, sizeof(int));
				[e setComputePipelineState:d->reduce_maxdin];
				[e setBuffer:fA offset:0 atIndex:0];
				[e setBuffer:maxBuf offset:0 atIndex:1];
				[e setBytes:&N length:sizeof(int) atIndex:2];
				disp1D(N);
				flush();				// commit + wait so we can read maxDin
				int hMax = 0; memcpy(&hMax, [maxBuf contents], sizeof(int));
				begin();
				float cap = (hMax / 256.0f) - 0.5f; if (cap < 0.0f) cap = 0.0f;
				if (rvE > cap) rvE = cap;
				if (rcE > cap) rcE = cap;
				if (rvE >= 0.5f) {						// O = dilate(erode(S,rvE),rvE)
					seedField(fA, rvE, 1);				// E = d_in(S) > rvE
					jfa(fB, metric);					// fB = d_out(E)
					didOpen = 1;
				}
			}

			if (rcE >= 0.5f) {
				if (didOpen) seedField(fB, rvE, 0);		// O = d_out(E) <= rvE
				else         seedAlpha(1);				// O = mask itself
				jfa(fA, metric);						// fA = d_out(O)

				seedField(fA, rcE, 1);					// !D = d_out(O) > rcE
				jfa(fB, metric);						// fB = d_in(D)
				seedField(fA, rcE, 0);					// D = d_out(O) <= rcE
				jfa(fC, metric);						// fC = d_out(D)
				fieldSub(gOut, fB, fC, rcE);			// g = d_in(D)-d_out(D)-rcE
			} else if (didOpen) {
				fieldRsub(gOut, fB, rvE);				// convex-only: rvE - d_out(E)
			} else {
				seedAlpha(1);							// d_out(S)
				jfa(fB, metric);
				fieldSub(gOut, fA, fB, 0.0f);			// g = d_in(S) - d_out(S)
			}
		};

		begin();
		buildG(CR_METRIC_L2, gR);

		float t = (profile < 0.0f) ? 0.0f : (profile > 1.0f ? 1.0f : profile);
		if (t > 1e-4f) {
			buildG(CR_METRIC_LINF, gS);
			[e setComputePipelineState:d->field_lerp];
			[e setBuffer:gR offset:0 atIndex:0];
			[e setBuffer:gS offset:0 atIndex:1];
			[e setBytes:&t length:sizeof(float) atIndex:2];
			[e setBytes:&N length:sizeof(int) atIndex:3];
			disp1D(N);
		}

		// Colour feature: nearest OPAQUE source pixel. JFA (L2), keep the seed
		// coords (no resolve) - the final flood buffer IS the feature field.
		seedAlpha(1);
		floodMetric = CR_METRIC_L2;
		id<MTLBuffer> feat = jfaFlood();

		[e setComputePipelineState:d->composite];
		[e setBuffer:gR   offset:0 atIndex:0];
		[e setBuffer:feat offset:0 atIndex:1];
		[e setBuffer:src  offset:0 atIndex:2];
		[e setBuffer:dst  offset:0 atIndex:3];
		[e setBytes:&p length:sizeof(p) atIndex:4];
		disp2D(p.W, p.H);
		flush();

		[maxBuf release];
		freeAll();
		return true;
	}
}
