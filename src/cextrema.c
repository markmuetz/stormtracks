#include <stdio.h>
#include <stdbool.h>

void cextrema(const float *data, 
           size_t imax, size_t jmax, 
           float *extrema, 
	   size_t min_max_length,
	   int *maxima_x, int *maxima_y,
	   int *minima_x, int *minima_y, 
	   int *maxima_length, int *minima_length) 
{
    size_t i;
    size_t j;

    size_t inner_i;
    size_t inner_j;

    bool is_max;
    bool is_min;

    float data_val;

    int max_length = 0;
    int min_length = 0;

    for (i = 1; i < imax - 1; ++i)
    {
        for (j = 1; j < jmax - 1; ++j)
        {
            is_max = true;
            is_min = true;
            data_val = data[i * jmax + j];

            for (inner_i = i - 1; inner_i < i + 2; ++inner_i)
            {
                for (inner_j = j - 1; inner_j < j + 2; ++inner_j)
                {
                    if (data[inner_i * jmax + inner_j] > data_val)
                    {
                        is_max = false;
                    }
                    if (data[inner_i * jmax + inner_j] < data_val)
                    {
                        is_min = false;
                    }
                }
            }

            if (is_max)
            {
                extrema[i * jmax + j] = 1;
		if (max_length < min_max_length)
		{
		    maxima_x[max_length] = i;
		    maxima_y[max_length] = j;
		    max_length += 1;
		}
            }
            else if (is_min)
            {
		if (min_length < min_max_length)
		{
		    extrema[i * jmax + j] = -1;
		    minima_x[min_length] = i;
		    minima_y[min_length] = j;
		    min_length += 1;
		}
            }
        }
	*maxima_length = max_length;
	*minima_length = min_length;
    }
}
