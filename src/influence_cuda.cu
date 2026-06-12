/******************************************************************************
 * influence_cuda.cu — CUDA GPU Implementation (Listing S2)
 *
 * Companion code for: "A Massively Parallel Computational Framework for
 * Real-Time Transient Pressure Simulation in Fractured Reservoirs"
 * Yin, R. & Zhang, S. (2026). Submitted to Computers & Geosciences.
 *
 * Compilation:
 *   nvcc -arch=sm_80 -O3 -use_fast_math --restrict -I../src \
 *        -o influence_cuda influence_cuda.cu
 *
 * Requirements:
 *   CUDA Toolkit >= 11.0
 *   NVIDIA GPU with Compute Capability >= 8.0 (A100 or newer)
 *
 * Design strategies (see manuscript Section 3):
 *   1. Grid-stride loop — arbitrary problem sizes
 *   2. __restrict__ — A100 read-only data cache
 *   3. Shared memory — amortize global-memory latency
 *   4. Separable kernel G = G_x * G_y — ~50% FP ops reduction
 *   5. atomicAdd — per-i output accumulation
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "greens.h"
#include "common.h"

/* --------------------------------------------------------------------------
 * Device-side Green's function components
 * Identical mathematics to the CPU version; compiled for GPU execution.
 * -------------------------------------------------------------------------- */
__device__ static inline double dev_G_x(int nx, const double geom[6], double t)
{
    (void)t;
    double x  = geom[GEOM_X];
    double xf = geom[GEOM_XF];
    double Lx = geom[GEOM_LX];
    double arg = nx * M_PI / Lx;
    return cos(arg * x) * sin(arg * xf) / (arg * xf);
}

__device__ static inline double dev_G_y(int ny, const double geom[6], double t)
{
    (void)t;
    double y  = geom[GEOM_Y];
    double yf = geom[GEOM_YF];
    double Ly = geom[GEOM_LY];
    double arg = ny * M_PI / Ly;
    return cos(arg * y) * sin(arg * yf) / (arg * yf);
}

/* --------------------------------------------------------------------------
 * GPU Kernel
 *
 * Launch configuration:
 *   Grid:  ceil((N * N_x * N_y) / BLOCK_SIZE) blocks (1D)
 *   Block: BLOCK_SIZE threads (1D, 16 warps)
 *   Shared memory: SHARED_MEM_KB KB per block (via cudaFuncSetAttribute)
 * -------------------------------------------------------------------------- */
__global__ void compute_influence_coefficients(
    const double* __restrict__ geom_params,
    const double* __restrict__ time_steps,
    double* __restrict__ G_matrix,
    const int N,
    const int N_x,
    const int N_y)
{
    int tid           = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    /* Coalesced load: contiguous threads read contiguous geometry data */
    extern __shared__ double shared_geom[];
    if (threadIdx.x < N_GEOM_PARAMS) {
        shared_geom[threadIdx.x] = geom_params[threadIdx.x];
    }
    __syncthreads();

    /* Grid-stride loop: parallel evaluation over all (i, n_x, n_y) */
    for (int idx = tid; idx < N * N_x * N_y; idx += total_threads) {

        /* Linear index → 3D coordinate decomposition */
        int i  = idx / (N_x * N_y);
        int ny = (idx / N_x) % N_y;
        int nx = idx % N_x;

        /* Separable Green's function: G = G_x * G_y */
        double G_x   = dev_G_x(nx, shared_geom, time_steps[i]);
        double G_y   = dev_G_y(ny, shared_geom, time_steps[i]);
        double G_val = G_x * G_y;

        /* Atomic accumulation (G_matrix zeroed via cudaMemset) */
        atomicAdd(&G_matrix[i], G_val);
    }
}

/* --------------------------------------------------------------------------
 * Host-side launcher
 * Handles device memory allocation, H2D/D2H transfers, and kernel launch.
 * -------------------------------------------------------------------------- */
void launch_influence_kernel(
    const double* h_geom_params,
    const double* h_time_steps,
    double*       h_G_matrix,
    int N, int N_x, int N_y)
{
    size_t gbytes = N * N_GEOM_PARAMS * sizeof(double);
    size_t tbytes = N * sizeof(double);
    size_t mbytes = N * sizeof(double);

    /* Device memory */
    double *d_geom, *d_time, *d_Gmat;
    cudaMalloc(&d_geom, gbytes);
    cudaMalloc(&d_time, tbytes);
    cudaMalloc(&d_Gmat, mbytes);

    /* Host → Device */
    cudaMemcpy(d_geom, h_geom_params, gbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_time, h_time_steps,  tbytes, cudaMemcpyHostToDevice);
    cudaMemset(d_Gmat, 0, mbytes);

    /* Kernel launch */
    int tasks = N * N_x * N_y;
    int grid  = (tasks + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaFuncSetAttribute(compute_influence_coefficients,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SHARED_MEM_KB * 1024);

    compute_influence_coefficients<<<grid, BLOCK_SIZE,
                                     SHARED_MEM_KB * 1024>>>(
        d_geom, d_time, d_Gmat, N, N_x, N_y);
    cudaDeviceSynchronize();

    /* Device → Host */
    cudaMemcpy(h_G_matrix, d_Gmat, mbytes, cudaMemcpyDeviceToHost);

    cudaFree(d_geom);
    cudaFree(d_time);
    cudaFree(d_Gmat);
}

/* --------------------------------------------------------------------------
 * Benchmark harness
 * -------------------------------------------------------------------------- */
int main(int argc, char** argv)
{
    int N   = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    int N_x = (argc > 2) ? atoi(argv[2]) : DEFAULT_NX;
    int N_y = (argc > 3) ? atoi(argv[3]) : DEFAULT_NY;

    print_header("CUDA GPU", N, N_x, N_y, 0);

    /* Device query */
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("GPU: %s (CC %d.%d, %.0f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    /* Host memory */
    double* h_geom = (double*)malloc(N * N_GEOM_PARAMS * sizeof(double));
    double* h_time = (double*)malloc(N * sizeof(double));
    double* h_Gmat = (double*)malloc(N * sizeof(double));

    init_benchmark_data(h_geom, h_time, N);

    /* Warm-up */
    launch_influence_kernel(h_geom, h_time, h_Gmat, N, N_x, N_y);

    /* Timed run (CUDA events for accurate GPU timing) */
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch_influence_kernel(h_geom, h_time, h_Gmat, N, N_x, N_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Elapsed:   %.2f ms\n", ms);
    printf("Throughput: %.2f Meval/s\n",
           (double)(N * N_x * N_y) / (ms / 1000.0) / 1e6);
    printf("Sample G[0] = %.10e\n", h_Gmat[0]);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_geom); free(h_time); free(h_Gmat);
    return 0;
}