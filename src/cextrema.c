#include <stdio.h>
#include <stdbool.h>

void cextrema(const float *data, 
           size_t imax, size_t jmax, 
           float *extrema) 
{
    size_t i;
    size_t j;

    size_t inner_i;
    size_t inner_j;

    bool is_max;
    bool is_min;

    float data_val;

    int max_count;
    int min_count;

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
		max_count++;
            }
            else if (is_min)
            {
                extrema[i * jmax + j] = -1;
		min_count++;
            }
        }
    }

    int* maxima_x = (int*)malloc(max_count * sizeof(int));
    int* maxima_y = (int*)malloc(max_count * sizeof(int));
    int* minima_x = (int*)malloc(min_count * sizeof(int));
    int* minima_y = (int*)malloc(min_count * sizeof(int));
    int max_index = 0;
    int min_index = 0;

    for (i = 1; i < imax - 1; ++i)
    {
        for (j = 1; j < jmax - 1; ++j)
        {
	    if (extrema[i * jmax + j] == 1)
	    {
		maxima_x[max_index] = i;
		maxima_x[max_index] = j;
		max_index++;
	    }
	    if (extrema[i * jmax + j] == -1)
	    {
		minima_x[min_index] = i;
		minima_x[min_index] = j;
		min_index++;
	    }
	}
    }
}
