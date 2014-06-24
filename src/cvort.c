#include <stdio.h>

/* 2nd order vorticity. */
void cvort(const float *u, const float *v,
	   size_t imax, size_t jmax, 
	   const float *dx, float dy, 
	   float *vort) 
{
    size_t i;
    size_t j;

    float du_dy;
    float dv_dx;

    for (i = 1; i < imax - 1; ++i)
    {
	for (j = 1; j < jmax - 1; ++j)
	{
	    du_dy = (u[(i + 1) * jmax + j] - u[(i - 1) * jmax + j]) / dy;
	    dv_dx = (v[i * jmax + (j + 1)] - v[i * jmax + (j - 1)]) / dx[i];
	    vort[i * jmax + j] = dv_dx - du_dy;
	}
    }
}

/* 4th order vorticity. */
void cvort4(const float *u, const float *v, 
	    size_t imax, size_t jmax, 
	    const float *dx, float dy, 
	    float *vort) 
{
    size_t i;
    size_t j;

    float du_dy1;
    float dv_dx1;

    float du_dy2;
    float dv_dx2;

    float du_dy;
    float dv_dx;

    for (i = 2; i < imax - 2; ++i)
    {
	for (j = 2; j < jmax - 2; ++j)
	{
	    du_dy1 = 2 * (u[(i + 1) * jmax + j] - u[(i - 1) * jmax + j]) / (3 * dy);
	    du_dy2 = (u[(i + 2) * jmax + j] - u[(i - 2) * jmax + j]) / (12 * dy);
	    du_dy = du_dy1 - du_dy2;

	    dv_dx1 = 2 * (v[i * jmax + (j + 1)] - v[i * jmax + (j - 1)]) / (3 * dx[i]);
	    dv_dx2 = (v[i * jmax + (j + 2)] - v[i * jmax + (j - 2)]) / (12 * dx[i]);
	    dv_dx = dv_dx1 - dv_dx2;

	    vort[i * jmax + j] = dv_dx - du_dy;
	}
    }
}
