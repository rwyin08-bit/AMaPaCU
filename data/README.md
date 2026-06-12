# Data Directory

This directory contains benchmark input data and field validation datasets for the CUDA-accelerated transient pressure simulation framework.

## Directory Structure

```
data/
├── README.md                              # This file
├── field_bhp_data.csv                     # Field BHP measurements (western China shale gas well)
└── benchmark_configs/                     # Pre-configured benchmark scenarios
    ├── small_3f_300t.json                 # 3 fractures, 300 time steps
    ├── medium_10f_1000t.json              # 10 fractures, 1000 time steps
    ├── large_1000f_3000t.json             # 1000 fractures, 3000 time steps
    └── extreme_10000f_3000t.json          # 10000 fractures, 3000 time steps
```

## Field Data

The field validation dataset (`field_bhp_data.csv`) contains bottomhole pressure (BHP) measurements from a multi-stage fractured horizontal shale gas well located in western China. Key reservoir properties are summarized in Table 1 of the manuscript.

### Data Format

```
time_days, BHP_MPa, gas_rate_sm3_d
0.01, 34.85, 98000
0.02, 34.72, 97500
...
```

## Benchmark Configurations

Pre-configured JSON files specifying fracture geometry and simulation parameters for reproducible benchmarking:

| Configuration | N_f | N_t | Approx. GPU time (A100) |
|--------------|-----|-----|-------------------------|
| Small | 3 | 300 | ~0.12 s |
| Medium | 10 | 1000 | ~0.14 s |
| Large | 1000 | 3000 | ~7.0 s |
| Extreme | 10000 | 3000 | ~72 s (est.) |

## Sensitivity Parameters

The parametric sensitivity analysis (Section 4.2) varies the following parameters by +/- 20% around their base-case values:

- Fracture half-length: 120-180 m (base: 150 m)
- Fracture conductivity: 400-600 mD-m (base: 500 mD-m)
- Fracture spacing: 48-72 m (base: 60 m)
- Matrix permeability: 0.0064-0.0096 mD (base: 0.008 mD)
- Skin factor: -2 to +5 (base: 0)
- Klinkenberg slippage coefficient: 0.5-5.0 (base: 1.0)
- Adsorption coefficient: 8-120 (base: 10, 100)

## GPU Memory Requirements

| Problem Size | Matrix Size | GPU Memory (FP64) | Recommended GPU |
|-------------|------------|-------------------|-----------------|
| N_f = 10 | 10 x 10 | ~1 KB | Any CUDA GPU |
| N_f = 1000 | 1000 x 1000 | ~8 MB | 16 GB+ |
| N_f = 10000 | 10000 x 10000 | ~800 MB | 80 GB (A100/H100) |
| N_f = 100000 | 100000 x 100000 | ~80 GB | Multi-GPU required |

For N_f > 10000, the dense matrix exceeds single-GPU memory capacity. Multi-GPU domain decomposition or matrix-free methods are recommended (see Limitations, Section 4.7).

## Reproducibility

All benchmark configurations reproduce the numerical experiments in Sections 4.1-4.6 of Yin & Zhang (2026). Exact parameter sets are specified in `../code/config/parameters.yaml`.
