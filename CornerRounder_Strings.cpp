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
	StrID_Description,				"Stacked strokes grown from the layer's alpha edge.\rPorted from python-proto/corner_rounder.",
	StrID_Count_Param_Name,			"Stroke Count",
	StrID_Add_Param_Name,			"Add Stroke",
	StrID_Remove_Param_Name,		"Remove Stroke",
	StrID_Threshold_Param_Name,		"Alpha Threshold",
	StrID_Feather_Param_Name,		"Edge Softness",
	StrID_Subpixel_Param_Name,		"Sub-pixel Edge",
	StrID_Gap_Param_Name,			"Gap",
	StrID_Width_Param_Name,			"Width",
	StrID_Side_Param_Name,			"Position",
	StrID_Corner_Param_Name,		"Corners",
	StrID_Color_Param_Name,			"Color",
	StrID_Opacity_Param_Name,		"Opacity",
	StrID_Order_Param_Name,			"Stacking",
	StrID_Fill_Param_Name,			"Fill",
	StrID_Color2_Param_Name,		"Gradient End Color",
	StrID_GStart_Param_Name,		"Gradient Start",
	StrID_GEnd_Param_Name,			"Gradient End",
	StrID_Side_Choices,				"Outside|Inside|Center",
	StrID_Corner_Choices,			"Miter|Round|Bevel|Concave",
	StrID_Order_Choices,			"Behind Art|In Front",
	StrID_Fill_Choices,				"Solid|Linear Gradient",
};

char *GetStringPtr(int strNum)
{
	return g_strs[strNum].str;
}
