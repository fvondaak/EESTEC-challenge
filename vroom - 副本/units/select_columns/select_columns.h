#pragma IMAGINET_INCLUDES_BEGIN
#include <stdint.h>
#include <math.h>
#pragma IMAGINET_INCLUDES_END

#pragma IMAGINET_FRAGMENT_BEGIN "select_columns"

static inline void select_columnsf32(const float* restrict input, int start_index, int end_index, float* restrict output)
{
	for (int i = 0; i <= end_index - start_index; i++)
	{
		*(output++) = *(start_index + input++);
	}
}

#pragma IMAGINET_FRAGMENT_END