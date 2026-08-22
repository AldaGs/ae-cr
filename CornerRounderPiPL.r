#include "AEConfig.h"
#include "AE_EffectVers.h"

#ifndef AE_OS_WIN
	#include <AE_General.r>
#endif

resource 'PiPL' (16000) {
	{	/* array properties: 12 elements */
		/* [1] */
		Kind {
			AEEffect
		},
		/* [2] */
		Name {
			"Corner Rounder"
		},
		/* [3] */
		Category {
			"Learning"
		},
#ifdef AE_OS_WIN
    #if defined(AE_PROC_INTELx64)
		CodeWin64X86 {"EffectMain"},
    #elif defined(AE_PROC_ARM64)
		CodeWinARM64 {"EffectMain"},
    #endif
#elif defined(AE_OS_MAC)
		CodeMacIntel64 {"EffectMain"},
		CodeMacARM64 {"EffectMain"},
#endif
		/* [6] */
		AE_PiPL_Version {
			2,
			0
		},
		/* [7] */
		AE_Effect_Spec_Version {
			PF_PLUG_IN_VERSION,
			PF_PLUG_IN_SUBVERS
		},
		/* [8] */
		AE_Effect_Version {
			524289	/* PF_VERSION(1,0,0,DEVELOP,1): 524288 + build 1.
					   MUST match PF_VERSION in the code EXACTLY, build
					   included, or AE refuses to load: "version mismatch". */
		},
		/* [9] */
		AE_Effect_Info_Flags {
			0
		},
		/* [10] */
		AE_Effect_Global_OutFlags {
		0x06000200	// DEEP_COLOR_AWARE (1<<25) | SEND_UPDATE_PARAMS_UI (1<<26)
					// | I_EXPAND_BUFFER (1<<9)
		},
		AE_Effect_Global_OutFlags_2 {
		0x0A001400	// THREADED (1<<27) | GPU_RENDER_F32 (1<<25) |
					// FLOAT_COLOR_AWARE (1<<12) | SMART_RENDER (1<<10)
		},
		/* [11] */
		AE_Effect_Match_Name {
			"aldai CornerRounder"
		},
		/* [12] */
		AE_Reserved_Info {
			0
		},
		/* [13] */
		AE_Effect_Support_URL {
			"https://www.adobe.com"
		}
	}
};
