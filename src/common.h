/******************************************************************************
 * common.h — Shared data structures and utility functions
 *
 * Companion code for: "A Massively Parallel Computational Framework for
 * Real-Time Transient Pressure Simulation in Fractured Reservoirs"
 * Yin, R. & Zhang, S. (2026)
 ******************************************************************************/

#ifndef COMMON_H
#define COMMON_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Number of geometry parameters per integration point */
#define N_GEOM_PARAMS 6

/* Geometry parameter indices */
#define GEOM_X  0   /* x-coordinate of evaluation point            */
#define GEOM_XF 1   /* fracture half-length in x-direction         */
#define GEOM_LX 2   /* reservoir length in x-direction            */
#define GEOM_Y  3   /* y-coordinate of evaluation point            */
#define GEOM_YF 4   /* fracture half-length in y-direction         */
#define GEOM_LY 5   /* reservoir length in y-direction            */

/* Default benchmark configuration */
#define DEFAULT_N      300
#define DEFAULT_NX     100
#define DEFAULT_NY     100
#define DEFAULT_NTHREADS 28

/* GPU kernel configuration */
#define BLOCK_SIZE       512
#define SHARED_MEM_KB    48

/**
 * Initialize synthetic geometry and time-step data for benchmarking.
 * Uses representative values from the western China shale gas well
 * validation case (Table 1 in the manuscript).
 */
static inline void init_benchmark_data(
    double* geom_params,
    double* time_steps,
    int N)
{
    for (int i = 0; i < N; i++) {
        geom_params[i * N_GEOM_PARAMS + GEOM_X]  = 50.0;
        geom_params[i * N_GEOM_PARAMS + GEOM_XF] = 60.0;
        geom_params[i * N_GEOM_PARAMS + GEOM_LX] = 1000.0;
        geom_params[i * N_GEOM_PARAMS + GEOM_Y]  = 250.0;
        geom_params[i * N_GEOM_PARAMS + GEOM_YF] = 0.01;
        geom_params[i * N_GEOM_PARAMS + GEOM_LY] = 500.0;
        time_steps[i] = 0.1 * (i + 1);
    }
}

/**
 * Print benchmark header with configuration details.
 */
static inline void print_header(const char* impl, int N, int N_x, int N_y, int extra)
{
    printf("=== %s Influence-Coefficient Assembly ===\n", impl);
    printf("Configuration: N=%d, N_x=%d, N_y=%d", N, N_x, N_y);
    if (extra > 0) printf(", threads=%d", extra);
    printf("\n");
}

#endif /* COMMON_H */