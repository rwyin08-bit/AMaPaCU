# Dockerfile for CUDA-Accelerated Pressure-Transient Simulation Framework
#
# Build:
#   docker build -t cuda-pressure-sim -f docker/Dockerfile .
#
# Run (single GPU):
#   docker run --gpus all --rm cuda-pressure-sim -c config/base_case.yaml
#
# Reference: Yin & Zhang (2026)

FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

LABEL maintainer="Yin & Zhang"
LABEL description="CUDA-Accelerated Transient Pressure Simulation Framework v1.0"
LABEL cuda_version="12.3"
LABEL gpu_arch="sm_80"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential gfortran g++ \
    cmake git wget curl \
    python3 python3-pip python3-dev \
    libopenblas-dev liblapack-dev \
    libomp-dev \
    nvidia-cuda-toolkit \
    nvidia-management-library \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    matplotlib==3.7.1 \
    pyyaml==6.0 \
    pytest==7.3.1

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
WORKDIR /app

COPY src/         /app/src/
COPY config/      /app/config/
COPY scripts/     /app/scripts/
COPY data/        /app/data/

ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# ---------------------------------------------------------------------------
# Compile CUDA kernel at build time
# ---------------------------------------------------------------------------
RUN nvcc -O3 -arch=sm_80 -Xcompiler -fopenmp \
    -o /app/build/influence_kernel \
    /app/src/gpu_influence_kernel.cu \
    && mkdir -p /app/build

# Compile CPU baseline
RUN g++ -O3 -march=native -fopenmp \
    -o /app/build/cpu_baseline \
    /app/src/cpu_openmp_baseline.cpp

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
ENTRYPOINT ["python3", "scripts/run_benchmark.py"]
CMD ["--help"]
