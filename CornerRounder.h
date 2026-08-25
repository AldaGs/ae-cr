/*
	CornerRounder.h

	Corner Rounder - rounds the sharp corners of a layer's ALPHA silhouette
	(the raster cousin of Illustrator's "Round Corners"). Ported from
	python-proto/corner_rounder/cr_step1..5.

	Like Buildable Stroke (the plugin this was scaffolded from) it is a
	DISTANCE-FIELD effect, NOT a point operation:
	  - output can be LARGER than input (concave fill / squircle bulge grow the
	    silhouette) -> SmartFX buffer expansion is used
	  - it is MULTI-PASS: threshold the alpha, run open+close as distance-field
	    thresholds, resolve one signed distance field, then anti-alias it.

	Locked design (see the Python prototype for the derivations):
	  - round = close(open(S)). OPENING (erode r, dilate r) rounds CONVEX
	    corners; CLOSING (dilate r, erode r) rounds CONCAVE; together they round
	    everything while straight edges stay put.
	  - built from distance transforms, cost is INDEPENDENT of radius (reuses the
	    FH-EDT / JFA engine inherited from Buildable Stroke).
	  - the FINAL field is a true Euclidean SDF (|grad|=1), so AA is a plain
	    smoothstep over +-feather px, no gradient normalization needed.
	  - CORNER PROFILE: circular (disk) vs squircle (superellipse) - the
	    "corner style = distance metric" (L^p) idea; port does it via a metric
	    blend between the round (L2) and square metrics.
	  - PRESERVE SOURCE AA: on unmoved (straight) edges pass the source alpha
	    through instead of regenerating it.
	  - added coverage (concave / squircle) takes the NEAREST OPAQUE source colour
	    (edge extend) so new pixels don't fringe black.

	NOTE: unlike Buildable Stroke the parameter list is FIXED (no dynamic
	"Add Stroke" groups), so the AEGP stream-suite show/hide machinery is gone.
*/

#pragma once

#ifndef CORNERROUNDER_H
#define CORNERROUNDER_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define PF_DEEP_COLOR_AWARE 1	// make sure we get 16bpc pixels; AE_Effect.h checks for this.

#include "AEConfig.h"

#ifdef AE_OS_WIN
	typedef unsigned short PixelType;
	#include <Windows.h>
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"	// PF_GPUDeviceSuite1, GPU cmd structs
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AEGP_SuiteHandler.h"

#include "CornerRounder_Strings.h"

/* Versioning information */

#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

/* Slider ranges / defaults, in pixels at full resolution. 2000 max supports big
   fillets on 4K/8K footage; the effective convex radius is still capped per-frame
   at the shape's inscribed-circle radius so it maxes out as a circle rather than
   eroding the shape away. */
#define	CR_RADIUS_MAX		2000
#define	CR_RADIUS_DFLT		12
#define	CR_FEATHER_DFLT		0.75	// half-width of the AA ramp, PX. The proto
									// settled on ~0.75 for curved edges; too small
									// stair-steps under magnification.

/* Global parameter order. MUST match the order of PF_ADD_* in ParamsSetup. */
enum {
	CR_INPUT = 0,			// index 0 is always the input layer
	CR_RADIUS,				// master corner radius, px
	CR_LINK,				// checkbox: convex+concave both follow Radius
	CR_CONVEX,				// convex (outer) radius, px   (used when Link is off)
	CR_CONCAVE,				// concave (inner) radius, px  (used when Link is off)
	CR_PROFILE,				// 0..100 -> circular .. squircle (Round style only)
	CR_CONVEX_STYLE,		// popup: Round | Bevel | Miter (outer corners)
	CR_CONCAVE_STYLE,		// popup: Round | Bevel | Miter (inner corners)
	CR_FEATHER,				// AA ramp half-width, px
	CR_AMOUNT,				// 0..100 %: blend source -> rounded
	CR_THRESHOLD,			// alpha cutoff for "inside" (0..100 %)
	CR_PRESERVE_AA,			// pass source AA through on unmoved edges
	CR_MATTE,				// optional layer: scales Amount per pixel (corner exclude)
	CR_MATTE_CHANNEL,		// popup: read matte Luminance | Alpha
	CR_MATTE_INVERT,		// checkbox: flip the matte (protect vs restrict)
	CR_NUM_PARAMS
};

/* Matte Channel popup (1-based, as AE reports popups). */
#define	CR_MATTE_LUMA	1
#define	CR_MATTE_ALPHA	2

/* Corner Style popup -> distance metric. Round = L2 (fast, GPU), Bevel = L1,
   Miter = L-inf. Bevel/Miter render on a 4x-supersampled CPU path. */
#define	CR_STYLE_ROUND	1
#define	CR_STYLE_BEVEL	2
#define	CR_STYLE_MITER	3

/* Stable IDs saved in the project file. Never reuse/renumber once shipped. */
enum {
	RADIUS_DISK_ID = 1,
	LINK_DISK_ID,
	CONVEX_DISK_ID,
	CONCAVE_DISK_ID,
	PROFILE_DISK_ID,
	FEATHER_DISK_ID,
	AMOUNT_DISK_ID,
	THRESHOLD_DISK_ID,
	PRESERVE_DISK_ID,
	MATTE_DISK_ID,
	MATTE_CHANNEL_DISK_ID,
	MATTE_INVERT_DISK_ID,
	CONVEX_STYLE_DISK_ID,
	CONCAVE_STYLE_DISK_ID,
};

/* Per-render snapshot. Built in PreRender, consumed in SmartRender. All
   distances are already scaled to the current (downsampled) resolution. */
typedef struct CRInfo {
	PF_FpLong	convexR;			// convex (open) radius, px
	PF_FpLong	concaveR;			// concave (close) radius, px
	PF_FpLong	profile;			// 0..1, circular .. squircle
	A_long		convexStyle;		// CR_STYLE_ROUND/BEVEL/MITER (outer)
	A_long		concaveStyle;		// CR_STYLE_ROUND/BEVEL/MITER (inner)
	PF_FpLong	feather;			// AA half-width, px
	PF_FpLong	amount;				// 0..1 blend
	PF_FpLong	threshold;			// 0..1 alpha cutoff
	A_long		preserveAA;			// 0/1
	PF_FpLong	maxRadius;			// max(convexR, concaveR), px (rect growth)
	// Optional matte layer that scales Amount per pixel (corner exclusion).
	A_long		hasMatte;			// 0/1 - a matte layer is connected
	A_long		matteChannel;		// CR_MATTE_LUMA / CR_MATTE_ALPHA
	A_long		matteInvert;		// 0/1
	PF_LRect	matteRect;			// the matte layer's checked-out rect (for offset)
	// Rect bookkeeping so SmartRender can map output pixels -> input pixels.
	PF_LRect	inRect;				// what we checked out
	PF_LRect	outRect;			// what we promised to fill
	A_long		pad;				// margin (px) the distance field must SEE
									// around the output rect
} CRInfo, *CRInfoP, **CRInfoH;


extern "C" {

	DllExport
	PF_Err
	EffectMain(
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);

}

#endif // CORNERROUNDER_H
