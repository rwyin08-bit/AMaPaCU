/******************************************************************************
 * greens.h — Green's function component definitions
 *
 * Implements the separable Green's function:
 *   G(n_x, n_y) = G_x(n_x) * G_y(n_y)
 *
 * This separability is exact for rectangular reservoirs with no-flow
 * outer boundary conditions. The decomposition reduces the 2D summation
 * to two independent 1D evaluations, cutting FP operations by ~50%.
 *
 * Reference: Section 3, Eq. (24)–(26) in the manuscript.
 ******************************************************************************/

#ifndef GREENS_H
#define GREENS_H

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * x-direction Green's function component.
 *
 * Parameters:
 *   nx   — series index in x-direction (1, 2, ..., N_x)
 *   geom — geometry array [x, x_f, L_x, y, y_f, L_y]
 *   t    — time at evaluation point
 *
 * Returns: G_x(nx) contribution
 */
static inline double G_x(int nx, const double geom[6], double t)
{
    (void)t;  /* time-independent for no-flow boundaries */
    double x  = geom[0];
    double xf = geom[1];
    double Lx = geom[2];
    double arg = nx * M_PI / Lx;
    return cos(arg * x) * sin(arg * xf) / (arg * xf);
}

/**
 * y-direction Green's function component.
 *
 * Parameters:
 *   ny   — series index in y-direction (1, 2, ..., N_y)
 *   geom — geometry array [x, x_f, L_x, y, y_f, L_y]
 *   t    — time at evaluation point
 *
 * Returns: G_y(ny) contribution
 */
static inline double G_y(int ny, const double geom[6], double t)
{
    (void)t;  /* time-independent for no-flow boundaries */
    double y  = geom[3];
    double yf = geom[4];
    double Ly = geom[5];
    double arg = ny * M_PI / Ly;
    return cos(arg * y) * sin(arg * yf) / (arg * yf);
}

#endif /* GREENS_H */