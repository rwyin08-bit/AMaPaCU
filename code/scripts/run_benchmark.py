#!/usr/bin/env python3
"""
run_benchmark.py - Main Entry Point for CUDA Pressure-Transient Benchmarks

Orchestrates the complete benchmark suite described in Yin & Zhang (2026),
Sections 4.1-4.6.

Usage:
    # Base case (Section 4.1)
    python scripts/run_benchmark.py -c config/base_case.yaml

    # Full parameter sweep (Sections 4.1-4.6)
    python scripts/run_benchmark.py -c config/parameters.yaml --full-sweep

    # Strong scaling study (Section 4.4)
    python scripts/run_benchmark.py -c config/parameters.yaml --scaling

    # Ablation study (Section 4.6)
    python scripts/run_benchmark.py -c config/base_case.yaml --ablation

    # Energy efficiency (Section 4.5)
    python scripts/run_benchmark.py -c config/base_case.yaml --power-measurement

    # Field validation (Section 4.1)
    python scripts/run_benchmark.py -c config/base_case.yaml --field-validation

Reference: Yin & Zhang (2026)
"""

import sys
import os
import argparse
import yaml
import json
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_runner import BenchmarkRunner


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="CUDA-Accelerated Transient Pressure Simulation Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py -c config/base_case.yaml
  python run_benchmark.py -c config/parameters.yaml --scaling
  python run_benchmark.py -c config/base_case.yaml --ablation
  python run_benchmark.py -c config/base_case.yaml --full-sweep
        """,
    )

    parser.add_argument(
        "--config", "-c",
        default="config/base_case.yaml",
        help="Path to configuration file (YAML)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Run the complete benchmark suite (Sections 4.1-4.6)",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run strong scaling study (Section 4.4)",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study (Section 4.6)",
    )
    parser.add_argument(
        "--field-validation",
        action="store_true",
        help="Run field data validation (Section 4.1)",
    )
    parser.add_argument(
        "--power-measurement",
        action="store_true",
        help="Run energy efficiency measurement (Section 4.5)",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run parametric sensitivity analysis (Section 4.2)",
    )
    parser.add_argument(
        "--fracture-density",
        action="store_true",
        help="Run fracture interference analysis (Section 4.3)",
    )
    parser.add_argument(
        "--solver-comparison",
        action="store_true",
        help="Run GPU vs. cuSPARSE vs. CPU comparison (Section 3.3)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run CPU-only baseline (disable GPU acceleration)",
    )
    parser.add_argument(
        "--N-fractures", type=int, default=None,
        help="Override number of fracture segments",
    )
    parser.add_argument(
        "--N-timesteps", type=int, default=None,
        help="Override number of time steps",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    use_gpu = not args.cpu_only

    # Apply overrides
    if args.N_fractures is not None:
        config['N_fractures_list'] = [args.N_fractures]
    if args.N_timesteps is not None:
        config['N_timesteps_list'] = [args.N_timesteps]

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize benchmark runner
    runner = BenchmarkRunner(config, output_dir=args.output, use_gpu=use_gpu)

    print("=" * 60)
    print("CUDA Pressure-Transient Simulation Benchmark Suite")
    print("Reference: Yin & Zhang (2026)")
    print("=" * 60)
    print(f"  GPU: {'Enabled' if use_gpu else 'Disabled (CPU-only)'}")
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output}")
    print()

    t_start = time.time()

    # ------------------------------------------------------------------
    # Execute requested benchmarks
    # ------------------------------------------------------------------
    if args.full_sweep:
        runner.run_all()

    elif args.scaling:
        results = runner.benchmark_scaling()
        with open(os.path.join(args.output, "scaling_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    elif args.ablation:
        results = runner.benchmark_ablation()
        with open(os.path.join(args.output, "ablation_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    elif args.field_validation:
        results = runner.benchmark_field_validation()
        with open(os.path.join(args.output, "field_validation.json"), 'w') as f:
            json.dump(results, f, indent=2)

    elif args.power_measurement:
        results = runner.benchmark_energy()
        with open(os.path.join(args.output, "energy_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    else:
        # Default: run single speedup benchmark
        N_f = args.N_fractures or config.get('fractures', {}).get('n_fractures', 3)
        N_t = args.N_timesteps or config.get('time', {}).get('N_t', 300)
        results = runner.benchmark_speedup(N_f, N_t)
        print(f"\nBenchmark: N_f={N_f}, N_t={N_t}")
        print(f"  GPU time:    {results['t_gpu']:.6f} s")
        print(f"  CPU time:    {results['t_cpu']:.6f} s")
        print(f"  Speedup:     {results['speedup']:.1f}x")
        print(f"  Rel. error:  {results['relative_error']:.4%}")

    t_total = time.time() - t_start
    print(f"\nTotal benchmark time: {t_total:.1f} s")
    print(f"Results directory: {args.output}")


if __name__ == "__main__":
    main()
