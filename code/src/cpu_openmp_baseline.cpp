/*
 * cpu_openmp_baseline.cpp - Optimized OpenMP CPU Baseline
 *
 * Implements the influence-coefficient assembly using OpenMP parallelization
 * over the outermost integration-point loop (28 threads, static scheduling)
 * on Intel Xeon Gold 6248R.  The inner n_x and n_y summations remain serial
 * to avoid false-sharing penalties on the reduction variable.
 *
 * Reference: Yin & Zhang (2026), Section 3.1, Supplementary Listing S1
 *
 * Compilation:
 *   g++ -O3 -march=native -fopenmp -o cpu_baseline cpu_openmp_baseline.cpp
 */

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI  3.14159265358979323846

// ---------------------------------------------------------------------------
// Green's function helpers
// ---------------------------------------------------------------------------

static inline double green_x_component(int n_x, double x_i, double x_ref, double L_x)
{
    double alpha_n = (double)n_x * PI / L_x;
    double cos_term = cos(alpha_n * x_i) * cos(alpha_n * x_ref);
    return (2.0 / L_x) * cos_term / (alpha_n * alpha_n);
}

static inline double green_y_component(int n_y, double y_i, double y_ref, double L_y)
{
    double beta_m = (double)n_y * PI / L_y;
    double cos_term = cos(beta_m * y_i) * cos(beta_m * y_ref);
    return (2.0 / L_y) * cos_term / (beta_m * beta_m);
}

// ---------------------------------------------------------------------------
// Main assembly (OpenMP parallel)
// ---------------------------------------------------------------------------

/**
 * Assembles the influence-coefficient matrix G on the CPU.
 *
 * Parallelization: outermost loop over integration points i (OpenMP for).
 * Inner loops n_x, n_y remain serial to avoid false sharing on G[i][j].
 *
 * Complexity: O(N * N * N_x * N_y) = O(N^3) in the worst case.
 */
void assemble_influence_cpu(
    double **G,               // output matrix [N x N] (zero-initialized by caller)
    const double *x,          // x-coordinates [N]
    const double *y,          // y-coordinates [N]
    int N,
    int N_x,
    int N_y,
    double L_x,
    double L_y)
{
    #pragma omp parallel for schedule(static) num_threads(28)
    for (int i = 0; i < N; ++i) {
        double xi = x[i];
        double yi = y[i];

        // Private accumulator per thread to avoid false sharing
        double *G_i = (double *)calloc(N, sizeof(double));

        for (int n_x = 1; n_x <= N_x; ++n_x) {
            double Gx_i = green_x_component(n_x, xi, xi, L_x);

            for (int n_y = 1; n_y <= N_y; ++n_y) {
                double Gy_i = green_y_component(n_y, yi, yi, L_y);
                double G_contrib = Gx_i * Gy_i;
                G_i[i] += G_contrib;

                // Cross-terms
                for (int j = 0; j < N; ++j) {
                    if (j == i) continue;
                    double Gx_ij = green_x_component(n_x, xi, x[j], L_x);
                    double Gy_ij = green_y_component(n_y, yi, y[j], L_y);
                    G_i[j] += Gx_ij * Gy_ij;
                }
            }
        }

        // Write back to global matrix
        #pragma omp critical
        {
            for (int j = 0; j < N; ++j) {
                G[i][j] = G_i[j];
            }
        }
        free(G_i);
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    // Example: 3 fractures, 300 time steps, N_x=N_y=50
    int N = 3, N_x = 50, N_y = 50, N_t = 300;
    double L_x = 4000.0, L_y = 3000.0;

    if (argc > 1) N   = atoi(argv[1]);
    if (argc > 2) N_x = atoi(argv[2]);
    if (argc > 3) N_y = atoi(argv[3]);

    printf("CPU Baseline: N=%d, N_x=%d, N_y=%d\n", N, N_x, N_y);

    // Allocate
    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));
    double **G = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; ++i) {
        x[i] = (double)(i + 1) * L_x / (N + 1);
        y[i] = L_y / 2.0;
        G[i] = (double *)calloc(N, sizeof(double));
    }

    // Benchmark
    double t_start = omp_get_wtime();
    assemble_influence_cpu(G, x, y, N, N_x, N_y, L_x, L_y);
    double t_elapsed = omp_get_wtime() - t_start;

    printf("CPU time: %.6f s\n", t_elapsed);
    printf("G[0][0] = %.12e\n", G[0][0]);

    // Cleanup
    for (int i = 0; i < N; ++i) free(G[i]);
    free(G); free(x); free(y);
    return 0;
}
