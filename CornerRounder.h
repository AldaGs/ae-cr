/*
	CornerRounder.h

	Third learning plugin, ported from python-proto/corner_rounder/
	(cr_step1..5). Stacked, individually-styled strokes grown outward (and/or
	inward) from the layer's ALPHA EDGE.

	This is the first port where the effect is NOT a point operation:
	  - output is LARGER than input  -> SmartFX buffer expansion is mandatory
	  - it is MULTI-PASS: build a signed distance field first, then band it

	Locked design (see the Python prototype for the derivations):
	  - stroke = a threshold BAND on a signed distance field; compute the field
	    ONCE per frame, band it once per stroke (adding a stroke is ~free)
	  - CORNER STYLE == the distance metric. L-inf miter / L2 round / L1 bevel,
	    exposed as a slider that blends the three EXACT fields, extrapolating
	    past bevel to get CONCAVE.
	    (REJECTED, measured: one L2 vector field + a different L^p norm per
	    style. Nearest feature under L2 != nearest under L1/L-inf -> up to 73px
	    error. Must use a real per-metric transform.)
	  - SUB-PIXEL EDGE is required or the AA does nothing: integer-offset fields
	    quantize to ~3 distinct values per 1px window. Correct by the nearest
	    feature pixel's own alpha coverage.
	  - AA: smoothstep the band edges over +-feather px; coverage = stroke alpha
	  - PARAM MODEL: user edits RELATIVE (gap, width) per stroke; we accumulate
	    into absolute bands, with a separate cursor per side.

	DYNAMIC STROKE COUNT - the honest constraint:
	  AE fixes an effect's parameter list at PF_Cmd_PARAMS_SETUP. You CANNOT
	  create params at runtime, so a literal "Add Stroke" that spawns new
	  params is impossible with standard params. The shipping idiom (SDK sample
	  UI/Supervisor) is: declare CR_MAX_STROKES groups up front, keep a count,
	  and SHOW/HIDE groups with PF_PUI_INVISIBLE + PF_UpdateParamUI. The user
	  sees exactly the "Add Stroke" UX; the ceiling is CR_MAX_STROKES.
	  We keep the count in a HIDDEN PARAM (not sequence data) so the effect
	  stays thread-safe and MFR-enabled - Supervisor gives up threading
	  precisely because it writes sequence data during UPDATE_PARAMS_UI.
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

/* How many stroke slots exist in the UI. Raising this after shipping is safe
   (new params append); LOWERING it or reordering breaks saved projects. */
#define	CR_MAX_STROKES		8

/* Slider ranges, in pixels at full resolution. */
#define	CR_DIST_MAX			1000
#define	CR_WIDTH_DFLT		10
#define	CR_FEATHER_DFLT		1.0		// half-width of the AA ramp, PX. 0.5 gives
										// only a 1px transition - correct at 100%
										// but visibly stepped under magnification.
										// 1.0 (a 2px ramp) reads clean at any zoom.

/* Position popup (1-based, as AE reports popups). */
#define	CR_SIDE_OUTER		1
#define	CR_SIDE_INNER		2
#define	CR_SIDE_CENTER		3

/* Stacking order popup: does this stroke sit under or over the source art? */
#define	CR_ORDER_BEHIND		1
#define	CR_ORDER_FRONT		2

/* Fill popup: flat color, or a linear gradient between two colors + endpoints. */
#define	CR_FILL_SOLID		1
#define	CR_FILL_LINEAR		2

/* Corner popup -> the corner slider value used by CR_CornerField. */
#define	CR_CORNER_MITER		1
#define	CR_CORNER_ROUND		2
#define	CR_CORNER_BEVEL		3
#define	CR_CORNER_CONCAVE	4

/* Global parameter order. MUST match the order of PF_ADD_* in ParamsSetup. */
enum {
	CR_INPUT = 0,			// index 0 is always the input layer
	CR_COUNT,				// hidden: how many stroke groups are visible
	CR_ADD,					// button: reveal the next group
	CR_REMOVE,				// button: hide the last group
	CR_THRESHOLD,			// alpha cutoff for "inside" (0..100 %)
	CR_FEATHER,				// AA ramp half-width, px
	CR_SUBPIXEL,			// use the alpha ramp to refine the edge
	CR_STROKE_BASE,			// first per-stroke param lives here
	CR_NUM_PARAMS = CR_STROKE_BASE + CR_MAX_STROKES * 13	// == CRS_PER_STROKE
};

/* Offsets WITHIN one stroke group, incl. the topic + its end. */
enum {
	CRS_TOPIC = 0,
	CRS_GAP,
	CRS_WIDTH,
	CRS_SIDE,
	CRS_CORNER,
	CRS_COLOR,			// solid color, and gradient START color
	CRS_FILL,			// Solid | Linear Gradient
	CRS_COLOR2,			// gradient END color
	CRS_GSTART,			// gradient start point (layer coords)
	CRS_GEND,			// gradient end point
	CRS_OPACITY,
	CRS_ORDER,
	CRS_TOPIC_END,
	CRS_PER_STROKE
};

/* Absolute param index of stroke i's field f. */
#define	CR_P(i, f)		(CR_STROKE_BASE + (i) * CRS_PER_STROKE + (f))

/* Stable IDs saved in the project file. Never reuse/renumber once shipped.
   Globals take 1..9; stroke i takes 1000 + i*100 + field (wide stride so a
   group can grow past 10 fields without colliding with the next stroke). */
enum {
	COUNT_DISK_ID = 1,
	ADD_DISK_ID,
	REMOVE_DISK_ID,
	THRESHOLD_DISK_ID,
	FEATHER_DISK_ID,
	SUBPIXEL_DISK_ID,
};
#define	CR_DISK_ID(i, f)	(1000 + (i) * 100 + (f))

/* One resolved stroke, ready to band. */
typedef struct {
	PF_FpLong	lo, hi;			// absolute band on the signed field, px
	PF_FpLong	corner;			// 0 miter .. 1 round .. 2 bevel .. 3 concave
	PF_FpLong	opacity;		// 0..1
	PF_FpLong	color[3];		// straight RGB 0..1 (solid, or gradient start)
	A_long		side;			// CR_SIDE_*
	A_long		order;			// CR_ORDER_BEHIND / CR_ORDER_FRONT
	// --- fill ------------------------------------------------------------
	A_long		fill;			// CR_FILL_SOLID / CR_FILL_LINEAR
	PF_FpLong	color2[3];		// gradient end color
	PF_FpLong	gsx, gsy;		// gradient start point, layer pixels (render res)
	PF_FpLong	gex, gey;		// gradient end point
} CRStroke;

/* Per-render snapshot. Built in PreRender, consumed in SmartRender. */
typedef struct CRInfo {
	A_long		count;					// active strokes
	PF_FpLong	threshold;				// 0..1
	PF_FpLong	feather;				// px, already downsample-scaled
	A_long		subpixel;				// 0/1
	CRStroke	stroke[CR_MAX_STROKES];
	PF_FpLong	maxOuter;				// furthest outward extent, px (rect growth)
	// Rect bookkeeping so SmartRender can map output pixels -> input pixels.
	PF_LRect	inRect;					// what we checked out
	PF_LRect	outRect;				// what we promised to fill
	A_long		pad;					// margin, in px, around the output rect
										// that the distance field must SEE (see
										// SmartRender: the field needs shape
										// pixels from outside the rendered
										// region or strokes break at its edge)
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
