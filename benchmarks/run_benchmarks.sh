#!/bin/bash
# ============================================================================
# run_benchmarks.sh ? Automated benchmark script for influence-coefficient
# assembly CPU and GPU implementations.
#
# Usage: bash benchmarks/run_benchmarks.sh
# ============================================================================

set -e

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$BENCH_DIR")"
SRC_DIR="$REPO_DIR/src"
RESULTS_DIR="$BENCH_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " Influence-Coefficient Assembly Benchmarks"
echo "============================================"
echo ""

# Compile CPU
echo "[1/4] Compiling CPU implementation..."
gcc -O3 -fopenmp -march=native -I"$SRC_DIR" \
    -o "$SRC_DIR/influence_omp" "$SRC_DIR/influence_omp.c" -lm
echo "  Done."

# Compile GPU (skip if nvcc not available)
echo "[2/4] Compiling GPU implementation..."
if command -v nvcc &> /dev/null; then
    nvcc -arch=sm_80 -O3 -use_fast_math --restrict -I"$SRC_DIR" \
         -o "$SRC_DIR/influence_cuda" "$SRC_DIR/influence_cuda.cu"
    echo "  Done."
else
    echo "  SKIPPED: nvcc not found."
fi

# Benchmark configurations (N, N_x, N_y)
CONFIGS=(
    "300 100 100"     # 3 fractures ? 100 segments, 300 time steps
    "1000 100 100"    # 10 fractures ? 100 segments, 1000 time steps
    "300 200 200"     # 3 fractures, 200?200 series terms
)

echo "[3/4] Running CPU benchmarks..."
for cfg in "${CONFIGS[@]}"; do
    read N Nx Ny <<< "$cfg"
    echo "  N=$N, Nx=$Nx, Ny=$Ny"
    "$SRC_DIR/influence_omp" "$N" "$Nx" "$Ny" 28 | \
        tee "$RESULTS_DIR/cpu_N${N}_Nx${Nx}_Ny${Ny}.txt"
done

echo "[4/4] Running GPU benchmarks..."
if [ -f "$SRC_DIR/influence_cuda" ]; then
    for cfg in "${CONFIGS[@]}"; do
        read N Nx Ny <<< "$cfg"
        echo "  N=$N, Nx=$Nx, Ny=$Ny"
        "$SRC_DIR/influence_cuda" "$N" "$Nx" "$Ny" | \
            tee "$RESULTS_DIR/gpu_N${N}_Nx${Nx}_Ny${Ny}.txt"
    done
else
    echo "  SKIPPED: GPU binary not found."
fi

echo ""
echo "Benchmarks complete. Results in: $RESULTS_DIR"