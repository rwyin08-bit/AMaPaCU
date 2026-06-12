# ============================================================================
# run_benchmarks.ps1 — Automated benchmark script (Windows PowerShell)
#
# Usage: powershell -ExecutionPolicy Bypass -File benchmarks/run_benchmarks.ps1
# ============================================================================

$ErrorActionPreference = "Stop"
$BENCH_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_DIR  = Split-Path -Parent $BENCH_DIR
$SRC_DIR   = Join-Path $REPO_DIR "src"
$RESULTS_DIR = Join-Path $BENCH_DIR "results"
New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null

Write-Host "============================================"
Write-Host " Influence-Coefficient Assembly Benchmarks"
Write-Host "============================================"
Write-Host ""

# Compile CPU
Write-Host "[1/4] Compiling CPU implementation..."
gcc -O3 -fopenmp -march=native -I"$SRC_DIR" `
    -o "$SRC_DIR\influence_omp.exe" "$SRC_DIR\influence_omp.c" -lm
Write-Host "  Done."

# Compile GPU
Write-Host "[2/4] Compiling GPU implementation..."
try {
    nvcc -arch=sm_80 -O3 -use_fast_math --restrict -I"$SRC_DIR" `
         -o "$SRC_DIR\influence_cuda.exe" "$SRC_DIR\influence_cuda.cu"
    Write-Host "  Done."
} catch {
    Write-Host "  SKIPPED: nvcc not found."
}

# Benchmark configurations
$configs = @(
    @(300, 100, 100),
    @(1000, 100, 100),
    @(300, 200, 200)
)

Write-Host "[3/4] Running CPU benchmarks..."
foreach ($cfg in $configs) {
    $N = $cfg[0]; $Nx = $cfg[1]; $Ny = $cfg[2]
    Write-Host "  N=$N, Nx=$Nx, Ny=$Ny"
    & "$SRC_DIR\influence_omp.exe" $N $Nx $Ny 28 | `
        Tee-Object -FilePath "$RESULTS_DIR\cpu_N${N}_Nx${Nx}_Ny${Ny}.txt"
}

Write-Host "[4/4] Running GPU benchmarks..."
if (Test-Path "$SRC_DIR\influence_cuda.exe") {
    foreach ($cfg in $configs) {
        $N = $cfg[0]; $Nx = $cfg[1]; $Ny = $cfg[2]
        Write-Host "  N=$N, Nx=$Nx, Ny=$Ny"
        & "$SRC_DIR\influence_cuda.exe" $N $Nx $Ny | `
            Tee-Object -FilePath "$RESULTS_DIR\gpu_N${N}_Nx${Nx}_Ny${Ny}.txt"
    }
} else {
    Write-Host "  SKIPPED: GPU binary not found."
}

Write-Host ""
Write-Host "Benchmarks complete. Results in: $RESULTS_DIR"