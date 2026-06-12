# Build Instructions

## Prerequisites

| Component | Minimum Version | Notes |
|-----------|----------------|-------|
| GCC | 7.5 | Or Clang 10.0+ |
| CUDA Toolkit | 11.0 | Required for GPU target only |
| NVIDIA Driver | 450.80+ | For CUDA 11.0 |
| GPU | Compute Capability 8.0+ | NVIDIA A100 or newer |
| CMake | 3.18 | Optional; alternative build method |

## Quick Build

### CPU Only

```bash
cd src
gcc -O3 -fopenmp -march=native -I. -o influence_omp influence_omp.c -lm
```

### GPU

```bash
cd src
nvcc -arch=sm_80 -O3 -use_fast_math --restrict -I. \
     -o influence_cuda influence_cuda.cu
```

## Build Options

### CPU

| Flag | Purpose |
|------|---------|
| `-O3` | Aggressive optimization |
| `-fopenmp` | Enable OpenMP parallelism |
| `-march=native` | Optimize for host CPU architecture |
| `-I.` | Include current directory for headers |

### GPU

| Flag | Purpose |
|------|---------|
| `-arch=sm_80` | Target A100 (CC 8.0) |
| `-O3` | Maximum optimization |
| `-use_fast_math` | Use fast math intrinsics |
| `--restrict` | Enable restrict pointer optimization |

For other GPU architectures, change `-arch=sm_80` to the appropriate value:
- H100: `-arch=sm_90`
- A40: `-arch=sm_86`
- V100: `-arch=sm_70` (note: FP64 atomicAdd performance will differ)

## Troubleshooting

**"cuda.h not found"** — Ensure CUDA Toolkit is installed and `nvcc` is on PATH.

**"omp.h not found"** — GCC must be compiled with OpenMP support. On macOS, use `brew install libomp`.

**"atomicAdd not supported"** — If targeting CC < 6.0, FP64 atomics are not natively supported. Use the `--gpu-architecture=sm_80` flag.