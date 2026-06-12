#!/usr/bin/env python3
"""
pressure_solver.py - Transient Pressure Solver for Fractured Reservoirs

Integrates the Green's function influence-coefficient assembly, Laplace-domain
linear system solution, and Stehfest numerical inversion to compute transient
bottomhole pressure responses for multi-stage fractured horizontal wells.

Supports both CPU (NumPy/SciPy) and GPU (CUDA) execution backends.

Reference: Yin & Zhang (2026), Sections 2.2.3 and 3
"""

import numpy as np
from scipy import linalg as spla
from typing import Tuple, Optional, Dict
import time

from .green_function import GreenFunction
from .stehfest_inversion import stehfest_inversion, STEHFEST_COEFFS_N8


class PressureTransientSolver:
    """
    Semi-analytical pressure-transient solver for multi-stage fractured
    horizontal wells in bounded rectangular reservoirs.

    The solver performs three stages:
      1. Laplace-domain influence-coefficient matrix assembly (Eqs. 22-26)
      2. Linear system solution to obtain P_bar_D(s)
      3. Stehfest numerical inversion to time domain

    Parameters
    ----------
    L_x : float
        Reservoir length in x-direction (m).
    L_y : float
        Reservoir length in y-direction (m).
    h : float
        Formation thickness (m).
    N_x, N_y : int
        Number of Fourier modes for Green's function.
    use_gpu : bool
        If True, offload matrix assembly to GPU via CUDA.
    """

    def __init__(
        self,
        L_x: float = 4000.0,
        L_y: float = 3000.0,
        h: float = 30.0,
        N_x: int = 50,
        N_y: int = 50,
        use_gpu: bool = False,
    ):
        self.L_x = L_x
        self.L_y = L_y
        self.h = h
        self.N_x = N_x
        self.N_y = N_y
        self.use_gpu = use_gpu

        # Green's function evaluator
        self.gf = GreenFunction(L_x, L_y, N_x, N_y)

        # Stehfest parameters
        self.coeffs_n = 8
        self.stehfest_coeffs = STEHFEST_COEFFS_N8

        # Timing
        self.t_assembly = 0.0
        self.t_solve = 0.0
        self.t_inversion = 0.0

    def set_fracture_geometry(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        half_lengths: np.ndarray,
        conductivities: np.ndarray,
    ):
        """
        Configure fracture geometry and properties.

        Parameters
        ----------
        x_coords : np.ndarray of shape (N_f,)
            Fracture segment x-coordinates (midpoints).
        y_coords : np.ndarray of shape (N_f,)
            Fracture segment y-coordinates (midpoints).
        half_lengths : np.ndarray of shape (N_f,)
            Fracture half-lengths (m).
        conductivities : np.ndarray of shape (N_f,)
            Fracture conductivities (mD-m).
        """
        self.N_f = len(x_coords)
        self.x_f = np.asarray(x_coords, dtype=np.float64)
        self.y_f = np.asarray(y_coords, dtype=np.float64)
        self.xf_half = np.asarray(half_lengths, dtype=np.float64)
        self.F_cD = np.asarray(conductivities, dtype=np.float64)

    def assemble_laplace_system(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble the linear system in the Laplace domain for a given s.

        A(s) * q_bar_D = b(s)

        where A is the influence matrix (N_f x N_f) and q_bar_D is the
        dimensionless flux vector in Laplace space.

        Parameters
        ----------
        s : float
            Laplace variable.

        Returns
        -------
        A : np.ndarray of shape (N_f, N_f)
            Coefficient matrix.
        b : np.ndarray of shape (N_f,)
            Right-hand side vector.
        """
        t_start = time.time()

        # Assemble influence coefficient matrix G
        G = self.gf.assemble_matrix(self.x_f, self.y_f)

        # Build system:  A = G + diag(skin_i / F_cD_i)
        A = G.copy()
        for i in range(self.N_f):
            # Add finite-conductivity correction
            A[i, i] += 1.0 / (self.F_cD[i] * self.xf_half[i])

        # RHS for constant-rate production
        b = np.ones(self.N_f, dtype=np.float64) / s

        self.t_assembly += time.time() - t_start
        return A, b

    def solve_laplace(self, s: float) -> np.ndarray:
        """
        Solve the linear system in the Laplace domain.

        Parameters
        ----------
        s : float
            Laplace variable.

        Returns
        -------
        q_bar_D : np.ndarray of shape (N_f,)
            Dimensionless flux solution in Laplace space.
        """
        t_start = time.time()
        A, b = self.assemble_laplace_system(s)

        # Solve dense linear system
        q_bar_D = spla.solve(A, b, assume_a='pos')
        self.t_solve += time.time() - t_start

        return q_bar_D

    def compute_bhp_laplace(self, s: np.ndarray) -> np.ndarray:
        """
        Compute bottomhole pressure in the Laplace domain.

        P_wD(s) = sum_{j} G_{ref,j} * q_bar_D_j(s)

        where G_{ref,j} is the influence coefficient from the reference
        point (wellbore location) to fracture segment j.

        Parameters
        ----------
        s : np.ndarray
            Laplace variable array.

        Returns
        -------
        P_wD_s : np.ndarray
            Dimensionless bottomhole pressure in Laplace domain.
        """
        P_wD_s = np.zeros_like(s, dtype=np.float64)
        x_ref = np.mean(self.x_f)
        y_ref = np.mean(self.y_f)

        for i, s_val in enumerate(s):
            q_bar_D = self.solve_laplace(s_val)
            # Reference-point influence
            G_ref = np.array([
                self.gf.evaluate(x_ref, y_ref, self.x_f[j], self.y_f[j])
                for j in range(self.N_f)
            ])
            P_wD_s[i] = np.dot(G_ref, q_bar_D)

        return P_wD_s

    def solve_transient(
        self,
        t: np.ndarray,
        verbose: bool = False,
    ) -> Dict:
        """
        Solve the full transient pressure response.

        Performs Laplace-domain solves at Stehfest collocation points
        and numerically inverts to the time domain.

        Parameters
        ----------
        t : np.ndarray
            Dimensionless time points.
        verbose : bool
            Print progress information.

        Returns
        -------
        results : dict
            Contains:
                - 't_D': dimensionless time array
                - 'P_wD': dimensionless bottomhole pressure
                - 'dP_wD_dlnt': logarithmic pressure derivative
                - 't_assembly': matrix assembly time (s)
                - 't_solve': linear solve time (s)
                - 't_inversion': inversion time (s)
        """
        if verbose:
            print(f"Solving transient pressure: N_f={self.N_f}, "
                  f"N_t={len(t)}, N_x={self.N_x}, N_y={self.N_y}")

        # Reset timers
        self.t_assembly = 0.0
        self.t_solve = 0.0
        self.t_inversion = 0.0

        t_start = time.time()

        # Laplace-domain function for Stehfest inversion
        def F_s(s: np.ndarray) -> np.ndarray:
            return self.compute_bhp_laplace(s)

        # Numerical Laplace inversion
        P_wD = stehfest_inversion(F_s, t, N=self.coeffs_n,
                                  coeffs=self.stehfest_coeffs)
        self.t_inversion = time.time() - t_start

        # Logarithmic pressure derivative (Bourdet derivative)
        dP_wD_dlnt = np.zeros_like(t)
        for i in range(1, len(t) - 1):
            dln_t = np.log(t[i+1]) - np.log(t[i-1])
            dP = P_wD[i+1] - P_wD[i-1]
            dP_wD_dlnt[i] = dP / dln_t

        results = {
            't_D': t,
            'P_wD': P_wD,
            'dP_wD_dlnt': dP_wD_dlnt,
            't_assembly': self.t_assembly,
            't_solve': self.t_solve,
            't_inversion': self.t_inversion,
            't_total': self.t_assembly + self.t_solve + self.t_inversion,
        }

        if verbose:
            print(f"  Assembly:  {self.t_assembly:.3f} s")
            print(f"  Solve:     {self.t_solve:.3f} s")
            print(f"  Inversion: {self.t_inversion:.3f} s")
            print(f"  Total:     {results['t_total']:.3f} s")

        return results


def generate_fracture_configuration(
    n_fractures: int,
    n_stages: int = 1,
    well_x: float = 2000.0,
    well_y: float = 1500.0,
    spacing: float = 60.0,
    half_length: float = 150.0,
    conductivity: float = 500.0,
) -> Dict[str, np.ndarray]:
    """
    Generate a standard multi-stage fractured horizontal well configuration.

    Parameters
    ----------
    n_fractures : int
        Total number of hydraulic fractures.
    n_stages : int
        Number of fracture stages (clusters).
    well_x, well_y : float
        Wellbore center coordinates (m).
    spacing : float
        Spacing between fracture stages (m).
    half_length : float
        Fracture half-length (m).
    conductivity : float
        Dimensionless fracture conductivity, F_cD.

    Returns
    -------
    config : dict
        Fracture geometry arrays.
    """
    fractures_per_stage = n_fractures // n_stages
    x_coords = []
    y_coords = []

    for stage in range(n_stages):
        stage_x = well_x + (stage - (n_stages - 1) / 2) * spacing
        for f in range(fractures_per_stage):
            x_coords.append(stage_x)
            y_coords.append(well_y + (f - (fractures_per_stage - 1) / 2) * 10.0)

    N_f = len(x_coords)
    return {
        'x_coords': np.array(x_coords, dtype=np.float64),
        'y_coords': np.array(y_coords, dtype=np.float64),
        'half_lengths': np.full(N_f, half_length, dtype=np.float64),
        'conductivities': np.full(N_f, conductivity, dtype=np.float64),
    }


if __name__ == "__main__":
    # Quick test: 3-fracture well, 300 time steps
    config = generate_fracture_configuration(n_fractures=3, n_stages=1)
    solver = PressureTransientSolver(use_gpu=False)
    solver.set_fracture_geometry(**config)

    t_D = np.logspace(-3, 4, 300)
    results = solver.solve_transient(t_D, verbose=True)

    print(f"\nP_wD at t=1:   {results['P_wD'][100]:.6f}")
    print(f"P_wD at t=100: {results['P_wD'][200]:.6f}")
