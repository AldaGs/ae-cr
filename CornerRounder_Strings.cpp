/*
	CornerRounder_Strings.cpp
*/

#include "CornerRounder.h"

typedef struct {
	A_u_long	index;
	A_char		str[256];
} TableString;

TableString g_strs[StrID_NUMTYPES] = {
	StrID_NONE,						"",
	StrID_Name,						"Corner Rounder",
	StrID_Description,				"Rounds the sharp corners of a layer's alpha silhouette.\rPorted from python-proto/corner_rounder.",
	StrID_Radius_Param_Name,		"Radius",
	StrID_Link_Param_Name,			"Link Convex/Concave",
	StrID_Convex_Param_Name,		"Convex Radius",
	StrID_Concave_Param_Name,		"Concave Radius",
	StrID_Profile_Param_Name,		"Corner Profile",
	StrID_Feather_Param_Name,		"Edge Softness",
	StrID_Amount_Param_Name,		"Amount",
	StrID_Threshold_Param_Name,		"Alpha Threshold",
	StrID_Preserve_Param_Name,		"Preserve Source AA",
	StrID_Matte_Param_Name,			"Rounding Matte",
	StrID_MatteChannel_Param_Name,	"Matte Channel",
	StrID_MatteInvert_Param_Name,	"Invert Matte",
	StrID_MatteChannel_Choices,		"Luminance|Alpha",
	StrID_ConvexStyle_Param_Name,	"Convex Style",
	StrID_ConcaveStyle_Param_Name,	"Concave Style",
	StrID_Style_Choices,			"Round|Bevel|Miter",
};

char *GetStringPtr(int strNum)
{
	return g_strs[strNum].str;
}
