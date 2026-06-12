#!/usr/bin/env python3
"""
green_function.py - Separable Green's Function Evaluation

Implements the separable Green's function G(n_x, n_y) = G_x(n_x) * G_y(n_y)
for no-flow rectangular boundaries (Eqs. 23-26).

Reference: Yin & Zhang (2026), Section 3.2
"""

import numpy as np
from scipy.special import exp1 as scipy_exp1
from typing import Tuple


PI = np.pi


class GreenFunction:
    """
    Separable 2D Green's function for a bounded rectangular reservoir
    with no-flow (Neumann) boundary conditions on all four sides.

    The influence coefficient G is expressed as the product of independent
    x- and y-direction Fourier series (Eq. 23):

        G(x_i, y_i; x_j, y_j) = sum_nx sum_ny G_x(n_x) * G_y(n_y)

    where:

        G_x(n_x) = (2/L_x) * cos(alpha_n * x_i) * cos(alpha_n * x_j) / alpha_n^2
        G_y(n_y) = (2/L_y) * cos(beta_m  * y_i) * cos(beta_m  * y_j) / beta_m^2

        alpha_n = n_x * pi / L_x
        beta_m  = n_y * pi / L_y

    The separability reduces the 2D evaluation to two independent 1D
    evaluations, cutting floating-point operations by approximately 50%.

    Parameters
    ----------
    L_x : float
        Reservoir length in the x-direction (m).
    L_y : float
        Reservoir length in the y-direction (m).
    N_x : int
        Number of Fourier modes in the x-direction.
    N_y : int
        Number of Fourier modes in the y-direction.
    """

    def __init__(self, L_x: float, L_y: float, N_x: int = 50, N_y: int = 50):
        self.L_x = L_x
        self.L_y = L_y
        self.N_x = N_x
        self.N_y = N_y

        # Pre-compute mode-dependent constants
        self.nx_range = np.arange(1, N_x + 1, dtype=np.float64)
        self.ny_range = np.arange(1, N_y + 1, dtype=np.float64)
        self.alpha = self.nx_range * PI / L_x   # shape (N_x,)
        self.beta  = self.ny_range * PI / L_y   # shape (N_y,)
        self.coeff_x = 2.0 / L_x / (self.alpha ** 2)  # (N_x,)
        self.coeff_y = 2.0 / L_y / (self.beta  ** 2)  # (N_y,)

    def gx_component(self, x_i: float, x_j: float) -> np.ndarray:
        """
        Compute G_x contribution for a source-observation pair (Eq. 24).

        Parameters
        ----------
        x_i : float
            Observation point x-coordinate.
        x_j : float
            Source point x-coordinate.

        Returns
        -------
        Gx : np.ndarray of shape (N_x,)
            x-direction Fourier components.
        """
        cos_xi = np.cos(self.alpha * x_i)
        cos_xj = np.cos(self.alpha * x_j)
        return self.coeff_x * cos_xi * cos_xj

    def gy_component(self, y_i: float, y_j: float) -> np.ndarray:
        """
        Compute G_y contribution for a source-observation pair (Eq. 25).

        Parameters
        ----------
        y_i : float
            Observation point y-coordinate.
        y_j : float
            Source point y-coordinate.

        Returns
        -------
        Gy : np.ndarray of shape (N_y,)
            y-direction Fourier components.
        """
        cos_yi = np.cos(self.beta * y_i)
        cos_yj = np.cos(self.beta * y_j)
        return self.coeff_y * cos_yi * cos_yj

    def evaluate(self, x_i: float, y_i: float, x_j: float, y_j: float) -> float:
        """
        Evaluate the full influence coefficient G for a single pair.

        G = sum_{n_x=1}^{N_x} sum_{n_y=1}^{N_y} G_x(n_x) * G_y(n_y)

        Parameters
        ----------
        x_i, y_i : float
            Observation point coordinates.
        x_j, y_j : float
            Source point coordinates.

        Returns
        -------
        G : float
            Influence coefficient value.
        """
        Gx = self.gx_component(x_i, x_j)   # (N_x,)
        Gy = self.gy_component(y_i, y_j)   # (N_y,)
        # Outer product and sum:  Gx[:, None] * Gy[None, :] -> sum over all
        return float(np.sum(np.outer(Gx, Gy)))

    def assemble_matrix(self, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """
        Assemble the full N x N influence coefficient matrix.

        Uses vectorized outer products for efficiency on CPU.

        Parameters
        ----------
        x_coords : np.ndarray of shape (N,)
            Fracture segment x-coordinates.
        y_coords : np.ndarray of shape (N,)
            Fracture segment y-coordinates.

        Returns
        -------
        G : np.ndarray of shape (N, N)
            Influence coefficient matrix (dense).
        """
        N = len(x_coords)
        G = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            Gx_i = np.cos(self.alpha[:, None] * x_coords[None, :])   # (N_x, N)
            Gx_i *= np.cos(self.alpha * x_coords[i])[:, None]         # (N_x, N)
            Gx_i *= self.coeff_x[:, None]                              # (N_x, N)

            Gy_i = np.cos(self.beta[:, None] * y_coords[None, :])     # (N_y, N)
            Gy_i *= np.cos(self.beta * y_coords[i])[:, None]           # (N_y, N)
            Gy_i *= self.coeff_y[:, None]                              # (N_y, N)

            # Sum over n_x and n_y: G[i, :] = sum_nx sum_ny Gx * Gy
            G[i, :] = np.sum(Gx_i[:, None, :] * Gy_i[None, :, :],
                             axis=(0, 1))

        return G


def separable_green_2d(
    x_i: np.ndarray,
    y_i: np.ndarray,
    x_j: np.ndarray,
    y_j: np.ndarray,
    L_x: float,
    L_y: float,
    N_x: int = 50,
    N_y: int = 50,
) -> np.ndarray:
    """
    Vectorized separable Green's function for multiple pairs.

    G[i, j] = sum_{n_x,n_y} G_x(n_x, x_i, x_j) * G_y(n_y, y_i, y_j)

    This function is the direct Python analogue of the CUDA kernel logic
    and is used for numerical verification of the GPU results.

    Parameters
    ----------
    x_i, y_i : np.ndarray of shape (N,)
        Observation point coordinates.
    x_j, y_j : np.ndarray of shape (M,)
        Source point coordinates.
    L_x, L_y : float
        Domain dimensions.
    N_x, N_y : int
        Number of Fourier modes.

    Returns
    -------
    G : np.ndarray of shape (N, M)
        Influence coefficient matrix.
    """
    gf = GreenFunction(L_x, L_y, N_x, N_y)
    N, M = len(x_i), len(x_j)
    G = np.zeros((N, M), dtype=np.float64)

    nx = np.arange(1, N_x + 1, dtype=np.float64)
    ny = np.arange(1, N_y + 1, dtype=np.float64)
    alpha = nx * PI / L_x
    beta  = ny * PI / L_y
    cx = 2.0 / L_x / (alpha ** 2)
    cy = 2.0 / L_y / (beta  ** 2)

    for k, n_x in enumerate(nx):
        Gx_k = cx[k] * np.outer(np.cos(alpha[k] * x_i),
                                np.cos(alpha[k] * x_j))
        for l, n_y in enumerate(ny):
            Gy_l = cy[l] * np.outer(np.cos(beta[l] * y_i),
                                    np.cos(beta[l] * y_j))
            G += Gx_k * Gy_l

    return G


def exp_integral_e1(x: np.ndarray) -> np.ndarray:
    """
    Exponential integral E1(x) for transient pressure evaluation.

    Uses scipy.special.exp1 for reference values.  For GPU-ported
    computation, see the rational approximation in gpu_influence_kernel.cu.

    Parameters
    ----------
    x : np.ndarray
        Positive argument array.

    Returns
    -------
    E1 : np.ndarray
        Exponential integral values.
    """
    return scipy_exp1(np.maximum(x, 1e-300))
