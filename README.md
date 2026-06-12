# A Massively Parallel CUDA Framework for Real-Time Transient Pressure Simulation in Fractured Reservoirs

Rongwang Yin¹, Shaowei Zhang²,*

¹ Basic Experiment and Training Center, Hefei University, Hefei 230601, P.R. China  
² School of Computer Engineering, Anhui Wenda University of Information Engineering, Hefei 231201, P.R. China

---

## Repository Structure

```
├── README.md                                    # This file
├── manuscript/
│   ├── manuscript_revised.docx                  # Revised manuscript (final version)
│   └── manuscript_original.docx                 # Original submission
├── figures/
│   └── figure_01-13.*                           # Figure files (13 figures)
├── code/
│   ├── README.md                                # Code documentation
│   ├── src/                                     # Source code
│   │   ├── gpu_influence_kernel.cu              # CUDA kernel (Sections 3.1-3.3)
│   │   ├── cpu_openmp_baseline.cpp              # OpenMP CPU baseline (Section 3.1)
│   │   ├── green_function.py                    # Green's function evaluation (Eqs. 23-26)
│   │   ├── stehfest_inversion.py                # Stehfest Laplace inversion
│   │   ├── pressure_solver.py                   # Pressure-transient solver
│   │   └── benchmark_runner.py                  # Benchmark harness (Section 4)
│   ├── config/
│   │   ├── base_case.yaml                       # Base case (3 fractures, 300 steps)
│   │   └── parameters.yaml                      # Benchmark parameter sweep
│   ├── docker/
│   │   ├── Dockerfile                           # CUDA container definition
│   │   └── docker-compose.yml                   # Multi-GPU deployment
│   └── scripts/
│       └── run_benchmark.py                     # Entry point
├── supplementary/
│   ├── listing_s1_openmp.cpp                    # Listing S1: OpenMP pseudo-code
│   ├── listing_s2_cuda.cu                       # Listing S2: CUDA kernel pseudo-code
│   ├── highlights.docx                          # Research highlights
│   ├── cover_letter.docx                        # Cover letter
│   └── declaration_competing_interests.docx     # Declaration of competing interests
└── data/
    └── README.md                                # Benchmark dataset description
```

## Abstract

Semi-analytical models for transient pressure simulation in fractured reservoirs typically involve repeated evaluations of space-time convolution integrals or the inversion of dense influence matrices. In multi-well systems subject to real-time operational constraints, these operations become computationally prohibitive. This paper presents a computational framework that reformulates integral-dominated pressure-transient models for massively parallel execution on graphics processing units. By exploiting the intrinsic independence among fracture segments and temporal steps, a hierarchical data-to-thread mapping strategy maximizes on-chip memory reuse and parallel throughput. The resulting kernel implementation integrates structured memory access patterns with on-the-fly evaluation of Green's functions in a single optimized computational pathway. Across a comprehensive benchmark suite spanning problem sizes from 3 to 10,000 fracture segments and 100 to 3000 time steps, the GPU framework consistently delivers speedups of 50x-341x over an optimized 28-core OpenMP-based CPU implementation (Intel Xeon Gold 6248R). The solution maintains a relative error below 0.4% across all configurations. The framework exhibits robust strong scaling up to 10,000 fracture segments on an NVIDIA A100 GPU.

## Key Contributions

1. **Semi-analytical model reformulation for parallel computation**: Dense, convolution-dominated pressure-transient formulation reinterpreted as a boundary-element-like problem, exposing fine-grained parallelism
2. **Domain-decomposed task partitioning**: Fracture-segment-time-step decomposition transforming nested superposition loops into independent GPU-friendly tasks
3. **Hierarchical thread-to-data mapping**: CUDA thread-block mapping scheme aligned with GPU warp architecture (512 threads/block, 16 warps)
4. **Separable-kernel exploit**: Green's function decomposition G(n_x, n_y) = G_x(n_x) x G_y(n_y) reducing floating-point operations by approximately 50%
5. **Structure-aware memory management**: Shared-memory tiling, read-only cache utilization, and warp-centric execution for bandwidth-efficient computation
6. **Systematic ablation analysis**: Quantitative decomposition of individual optimization contributions with separable-kernel exploit identified as the single most impactful optimization (approximately 46% of cumulative gain)

## Software Requirements

- CUDA Toolkit 12.3+ (https://developer.nvidia.com/cuda-toolkit)
- NVIDIA HPC SDK 24.1+
- GCC 11+ / MSVC 2022+
- Python 3.8+ (benchmark orchestration)
- NumPy, SciPy, Matplotlib, PyYAML

## Hardware Requirements

- NVIDIA GPU with Compute Capability 8.0+ (A100, H100 recommended)
- CUDA-enabled GPU with FP64 support
- Minimum 16 GB GPU memory for small-scale benchmarks; 80 GB recommended for large-scale (N_f > 1000)

## License

MIT License

## Citation

```bibtex
@article{yin2026cuda,
  title   = {A Massively Parallel {CUDA} Framework for Real-Time Transient
             Pressure Simulation in Fractured Reservoirs},
  author  = {Yin, Rongwang and Zhang, Shaowei},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {Under review}
}
```
