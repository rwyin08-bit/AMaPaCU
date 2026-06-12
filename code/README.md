# CUDA-Accelerated Transient Pressure Simulation Framework

Open-source implementation of the massively parallel GPU framework for semi-analytical
transient pressure simulation described in **Yin & Zhang (2026)**.

## Architecture

```
code/
├── src/
│   ├── __init__.py                       # Package entry
│   ├── gpu_influence_kernel.cu           # CUDA influence-coefficient kernel (Section 3.2)
│   ├── cpu_openmp_baseline.cpp           # OpenMP CPU baseline (Section 3.1)
│   ├── green_function.py                 # Separable Green's function (Eqs. 23-26)
│   ├── stehfest_inversion.py             # Stehfest numerical Laplace inversion
│   ├── pressure_solver.py                # Pressure-transient assembly and solve
│   └── benchmark_runner.py               # Automated benchmark suite (Section 4)
├── config/
│   ├── base_case.yaml                    # Base case configuration (3 fractures, 300 steps)
│   └── parameters.yaml                   # Sensitivity and scaling parameter sweeps
├── docker/
│   ├── Dockerfile                        # Container with CUDA + Python
│   └── docker-compose.yml                # Multi-GPU orchestration
└── scripts/
    └── run_benchmark.py                  # Main entry point
```

## Quick Start

### Local Execution

```bash
# Install Python dependencies
pip install numpy scipy matplotlib pyyaml

# Compile CUDA kernel (requires CUDA Toolkit 12.3+)
nvcc -O3 -arch=sm_80 -Xcompiler -fopenmp -o influence_kernel src/gpu_influence_kernel.cu

# Compile CPU baseline (requires GCC 11+ with OpenMP)
g++ -O3 -march=native -fopenmp -o cpu_baseline src/cpu_openmp_baseline.cpp

# Run base case benchmark
python scripts/run_benchmark.py -c config/base_case.yaml

# Run full parameter sweep (Sections 4.1-4.6)
python scripts/run_benchmark.py -c config/parameters.yaml --full-sweep

# Run ablation study (Section 4.6)
python scripts/run_benchmark.py -c config/base_case.yaml --ablation

# Run scaling study (Section 4.4)
python scripts/run_benchmark.py -c config/parameters.yaml --scaling
```

### Docker Execution

```bash
# Build image
docker build -t cuda-pressure-sim -f docker/Dockerfile .

# Run base case
docker run --gpus all --rm -v $(pwd)/results:/results \
    cuda-pressure-sim -c config/base_case.yaml -o /results

# Run with specific GPU
docker run --gpus device=0 --rm -v $(pwd)/results:/results \
    cuda-pressure-sim -c config/base_case.yaml -o /results
```

## Key Results

| Metric | Value | Reference |
|--------|-------|-----------|
| Speedup (3 fractures, 300 steps) | 80x vs. 28-core CPU | Section 4.1, Table 2 |
| Speedup (10 fractures, 1000 steps) | 122x vs. 28-core CPU | Section 4.1, Table 2 |
| Maximum speedup (large-scale) | 341x vs. 28-core CPU | Section 4.4, Fig. 12 |
| Relative error | < 0.4% | Section 4.1 |
| cuSPARSE speedup | 3.2x vs. CPU | Section 3.3 |
| Custom kernel vs. cuSPARSE | 25x improvement | Section 3.3 |
| Parallel efficiency (N_t >= 500) | > 92% | Section 4.4, Fig. 12 |
| Energy reduction | 70-85% vs. CPU | Section 4.5, Fig. 13 |
| Separable-kernel contribution | ~46% of cumulative gain | Section 4.6, Table 3 |
| Shared-memory tiling contribution | ~23% of cumulative gain | Section 4.6, Table 3 |

## Dependencies

- CUDA Toolkit 12.3+
- NVIDIA GPU Driver 545+
- GCC 11+ (Linux) or MSVC 2022+ (Windows)
- Python 3.8+
- NumPy, SciPy, Matplotlib, PyYAML
- NVIDIA Management Library (NVML) for power measurements

## Reproducibility

All numerical experiments in Sections 4.1-4.6 can be reproduced using:

```bash
# Section 4.1: Field data validation
python scripts/run_benchmark.py -c config/base_case.yaml --field-validation

# Section 4.2: Parametric sensitivity
python scripts/run_benchmark.py -c config/parameters.yaml --sensitivity

# Section 4.3: Fracture interference analysis
python scripts/run_benchmark.py -c config/parameters.yaml --fracture-density

# Section 4.4: Strong scaling
python scripts/run_benchmark.py -c config/parameters.yaml --scaling

# Section 4.5: Energy efficiency
python scripts/run_benchmark.py -c config/base_case.yaml --power-measurement

# Section 4.6: Ablation study
python scripts/run_benchmark.py -c config/base_case.yaml --ablation
```

## Performance Notes

- The separable-kernel optimization (G = G_x * G_y) is exact for no-flow rectangular boundaries. For other boundary types, the Green's function separability assumption must be verified.
- All floating-point computations use double precision (FP64) for numerical robustness.
- The grid-stride loop pattern enables transparent scaling to arbitrarily large problem sizes.
- Atomic addition is used only at the final reduction stage; intermediate computations are thread-private.

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
