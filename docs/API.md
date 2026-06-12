# API Reference

## CPU Functions

### `compute_influence_coefficients_cpu()`

```c
void compute_influence_coefficients_cpu(
    const double* geom_params,  // [N * 6] geometry parameters
    const double* time_steps,   // [N] time values
    double*       G_matrix,     // [N] output (pre-zeroed)
    const int     N,            // integration points
    const int     N_x,          // x-series terms
    const int     N_y,          // y-series terms
    int           N_threads     // OpenMP threads
);
```

## GPU Functions

### `compute_influence_coefficients()` (kernel)

```cuda
__global__ void compute_influence_coefficients(
    const double* __restrict__ geom_params,
    const double* __restrict__ time_steps,
    double* __restrict__ G_matrix,
    const int N, const int N_x, const int N_y
);
```

Launch: `grid = ceil(N * N_x * N_y / 512), block = 512`
Shared memory: 48 KB per block

### `launch_influence_kernel()` (host)

```cuda
void launch_influence_kernel(
    const double* h_geom_params,  // host geometry data
    const double* h_time_steps,   // host time data
    double*       h_G_matrix,     // host output
    int N, int N_x, int N_y
);
```

Handles device allocation, H2D/D2H transfers, kernel launch, and cleanup.

## Green's Function Components

### `G_x()`

x-direction component: G_x(nx) = cos(π·nx·x/Lx) · sin(π·nx·xf/Lx) / (π·nx·xf/Lx)

### `G_y()`

y-direction component: G_y(ny) = cos(π·ny·y/Ly) · sin(π·ny·yf/Ly) / (π·ny·yf/Ly)

### Separable Kernel

Influence coefficient: G(nx, ny) = G_x(nx) · G_y(ny)

This separability is exact for rectangular reservoirs with no-flow outer boundaries.