/*
 * Listing S1: Optimized OpenMP-Based CPU Baseline
 *
 * Pseudo-code of the influence-coefficient matrix assembly procedure
 * using OpenMP.  The outermost loop over integration points i is
 * parallelized with static scheduling across 28 threads on the
 * Intel Xeon Gold 6248R.  Inner n_x, n_y summations remain serial
 * to avoid false-sharing penalties on the reduction variable G[i][j].
 *
 * Reference: Yin & Zhang (2026), Supplementary Material
 *
 * Compilation:
 *   g++ -O3 -march=native -fopenmp -o listing_s1 listing_s1_openmp.cpp
 *
 * Execution:
 *   ./listing_s1 [N] [N_x] [N_y]
 */

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI  3.14159265358979323846

// ---------------------------------------------------------------
// Pseudo-code listing for the OpenMP baseline
// ---------------------------------------------------------------

void assemble_influence_cpu_omp(
    double **G,              // Output: influence matrix [N x N]
    const double *x,         // Fracture segment x-coordinates [N]
    const double *y,         // Fracture segment y-coordinates [N]
    int N,                   // Number of fracture segments
    int N_x,                 // Fourier modes in x
    int N_y,                 // Fourier modes in y
    double L_x,              // Domain length in x
    double L_y               // Domain length in y
)
{
    /*
     * PSEUDO-CODE:
     *
     * #pragma omp parallel for schedule(static) num_threads(28)
     * for i = 0 to N-1:
     *     // Thread-private accumulator to avoid false sharing
     *     double *G_i = calloc(N, sizeof(double))
     *
     *     for n_x = 1 to N_x:
     *         alpha_n = n_x * PI / L_x
     *         Gx = (2.0/L_x) * cos(alpha_n * x_i)^2 / (alpha_n^2)
     *
     *         for n_y = 1 to N_y:
     *             beta_m = n_y * PI / L_y
     *             Gy = (2.0/L_y) * cos(beta_m * y_i)^2 / (beta_m^2)
     *             G_i[i] += Gx * Gy  // diagonal term
     *
     *             for j = 0 to N-1, j != i:
     *                 Gx_ij = (2.0/L_x) * cos(alpha_n*x_i)*cos(alpha_n*x_j)
     *                         / (alpha_n^2)
     *                 Gy_ij = (2.0/L_y) * cos(beta_m*y_i)*cos(beta_m*y_j)
     *                         / (beta_m^2)
     *                 G_i[j] += Gx_ij * Gy_ij  // off-diagonal term
     *
     *     // Critical section: write-back to global matrix
     *     #pragma omp critical
     *     for j = 0 to N-1:
     *         G[i][j] = G_i[j]
     *
     *     free(G_i)
     *
     * Complexity: O(N^3) when N_x, N_y ~ N
     * Best scheduling policy: static (tested against dynamic, guided)
     * Optimal thread count: 28 (matching physical cores, HT disabled)
     */

    #pragma omp parallel for schedule(static) num_threads(28)
    for (int i = 0; i < N; ++i) {
        double xi = x[i];
        double yi = y[i];
        double *G_i = (double *)calloc(N, sizeof(double));

        for (int n_x = 1; n_x <= N_x; ++n_x) {
            double alpha_n = (double)n_x * PI / L_x;
            double coeff_x = 2.0 / L_x;
            double cos_xi = cos(alpha_n * xi);
            double Gx_i = coeff_x * cos_xi * cos_xi / (alpha_n * alpha_n);

            for (int n_y = 1; n_y <= N_y; ++n_y) {
                double beta_m = (double)n_y * PI / L_y;
                double coeff_y = 2.0 / L_y;
                double cos_yi = cos(beta_m * yi);
                double Gy_i = coeff_y * cos_yi * cos_yi / (beta_m * beta_m);

                // Diagonal contribution
                G_i[i] += Gx_i * Gy_i;

                // Off-diagonal contributions
                for (int j = 0; j < N; ++j) {
                    if (j == i) continue;
                    double cos_xj = cos(alpha_n * x[j]);
                    double Gx_ij = coeff_x * cos_xi * cos_xj / (alpha_n * alpha_n);
                    double cos_yj = cos(beta_m * y[j]);
                    double Gy_ij = coeff_y * cos_yi * cos_yj / (beta_m * beta_m);
                    G_i[j] += Gx_ij * Gy_ij;
                }
            }
        }

        // Write-back (critical section to avoid race on G[j][i])
        #pragma omp critical
        {
            for (int j = 0; j < N; ++j) {
                G[i][j] = G_i[j];
            }
        }
        free(G_i);
    }
}

// ---------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------

int main(int argc, char **argv)
{
    int N   = (argc > 1) ? atoi(argv[1]) : 3;
    int N_x = (argc > 2) ? atoi(argv[2]) : 50;
    int N_y = (argc > 3) ? atoi(argv[3]) : 50;

    double L_x = 4000.0, L_y = 3000.0;

    printf("OpenMP Baseline (Listing S1)\n");
    printf("  N=%d, N_x=%d, N_y=%d\n", N, N_x, N_y);
    printf("  Threads: %d\n", omp_get_max_threads());

    // Allocate
    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));
    double **G = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; ++i) {
        x[i] = (i + 1.0) * L_x / (N + 1.0);
        y[i] = L_y / 2.0;
        G[i] = (double *)calloc(N, sizeof(double));
    }

    // Warm-up
    assemble_influence_cpu_omp(G, x, y, N, N_x, N_y, L_x, L_y);

    // Timed run
    double t0 = omp_get_wtime();
    for (int trial = 0; trial < 3; ++trial) {
        // Clear matrix
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                G[i][j] = 0.0;
        assemble_influence_cpu_omp(G, x, y, N, N_x, N_y, L_x, L_y);
    }
    double t_total = omp_get_wtime() - t0;
    printf("  Avg time: %.6f s\n", t_total / 3.0);
    printf("  G[0][0] = %.12e\n", G[0][0]);

    // Cleanup
    for (int i = 0; i < N; ++i) free(G[i]);
    free(G); free(x); free(y);
    return 0;
}
