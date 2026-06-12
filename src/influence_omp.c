/******************************************************************************
 * influence_omp.c — OpenMP CPU Implementation (Listing S1)
 *
 * Companion code for: "A Massively Parallel Computational Framework for
 * Real-Time Transient Pressure Simulation in Fractured Reservoirs"
 * Yin, R. & Zhang, S. (2026). Submitted to Computers & Geosciences.
 *
 * Compilation:
 *   gcc -O3 -fopenmp -march=native -I../src -o influence_omp influence_omp.c -lm
 *
 * Target: Intel Xeon Gold 6248R (28 physical cores, hyperthreading disabled)
 *        or any x86-64 CPU with OpenMP support.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "greens.h"
#include "common.h"

/**
 * Compute the influence-coefficient matrix on CPU using OpenMP.
 *
 * The outermost loop over integration points i is parallelized with
 * static scheduling. The inner summations over n_x and n_y remain
 * serial to avoid false-sharing penalties on the accumulation variable.
 *
 * This configuration (static scheduling, 28 threads) yielded the best
 * performance among tested policies (static, dynamic, guided) and
 * thread counts (14–56) on the Xeon Gold 6248R.
 *
 * Parameters:
 *   geom_params — [N * 6] geometry parameters per integration point
 *   time_steps  — [N]     time value at each evaluation point
 *   G_matrix    — [N]     output influence coefficients (pre-zeroed)
 *   N           — number of integration points
 *   N_x         — number of x-direction series terms
 *   N_y         — number of y-direction series terms
 *   N_threads   — OpenMP thread count
 */
void compute_influence_coefficients_cpu(
    const double* geom_params,
    const double* time_steps,
    double*       G_matrix,
    const int     N,
    const int     N_x,
    const int     N_y,
    int           N_threads)
{
    #pragma omp parallel for schedule(static) num_threads(N_threads)
    for (int i = 0; i < N; i++) {
        double G_i = 0.0;
        const double* geom = &geom_params[i * N_GEOM_PARAMS];
        double t = time_steps[i];

        /* Serial inner loops to avoid false-sharing on G_i */
        for (int nx = 1; nx <= N_x; nx++) {
            G_i += G_x(nx, geom, t);
        }
        for (int ny = 1; ny <= N_y; ny++) {
            G_i += G_y(ny, geom, t);
        }

        G_matrix[i] = G_i;
    }
}

int main(int argc, char** argv)
{
    int N        = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    int N_x      = (argc > 2) ? atoi(argv[2]) : DEFAULT_NX;
    int N_y      = (argc > 3) ? atoi(argv[3]) : DEFAULT_NY;
    int nthreads = (argc > 4) ? atoi(argv[4]) : DEFAULT_NTHREADS;

    print_header("OpenMP CPU", N, N_x, N_y, nthreads);

    /* Allocate host memory */
    double* geom_params = (double*)malloc(N * N_GEOM_PARAMS * sizeof(double));
    double* time_steps  = (double*)malloc(N * sizeof(double));
    double* G_matrix    = (double*)calloc(N, sizeof(double));

    if (!geom_params || !time_steps || !G_matrix) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }

    init_benchmark_data(geom_params, time_steps, N);

    /* Warm-up */
    compute_influence_coefficients_cpu(
        geom_params, time_steps, G_matrix, N, N_x, N_y, nthreads);

    /* Timed run */
    double start = omp_get_wtime();
    compute_influence_coefficients_cpu(
        geom_params, time_steps, G_matrix, N, N_x, N_y, nthreads);
    double elapsed = omp_get_wtime() - start;

    printf("Elapsed:   %.4f s\n", elapsed);
    printf("Throughput: %.2f Meval/s\n",
           (double)(N * N_x * N_y) / elapsed / 1e6);
    printf("Sample G[0] = %.10e\n", G_matrix[0]);

    free(geom_params);
    free(time_steps);
    free(G_matrix);
    return 0;
}