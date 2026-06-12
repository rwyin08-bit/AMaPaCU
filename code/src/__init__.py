"""
Massively Parallel CUDA Framework for Real-Time Transient Pressure Simulation
in Fractured Reservoirs.

Reference: Yin & Zhang (2026)
"""

from .green_function import GreenFunction, separable_green_2d
from .stehfest_inversion import stehfest_inversion, STEHFEST_COEFFS_N8
from .pressure_solver import PressureTransientSolver
from .benchmark_runner import BenchmarkRunner

__version__ = "1.0.0"
__all__ = [
    "GreenFunction",
    "separable_green_2d",
    "stehfest_inversion",
    "STEHFEST_COEFFS_N8",
    "PressureTransientSolver",
    "BenchmarkRunner",
]
