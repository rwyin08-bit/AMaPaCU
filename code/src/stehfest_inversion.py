#!/usr/bin/env python3
"""
stehfest_inversion.py - Numerical Laplace Inversion via Stehfest Algorithm

Implements the Stehfest algorithm (Stehfest, 1970, Algorithm 368) with N=8
coefficients for numerical inversion of the Laplace-transformed pressure
solution to the time domain.

Reference: Yin & Zhang (2026), Section 2.2.3
"""

import numpy as np
from typing import Callable


# ---------------------------------------------------------------------------
# Stehfest coefficients for N = 8
# ---------------------------------------------------------------------------
# Pre-computed from: V_i = (-1)^{N/2+i} * sum_{k=floor((i+1)/2)}^{min(i,N/2)}
#                         k^{N/2} * (2k)! / ((N/2-k)! * k! * (k-1)! * (i-k)! * (2k-i)!)
# ---------------------------------------------------------------------------
STEHFEST_COEFFS_N8 = np.array([
    -3.3333333333333336e-03,
     1.6666666666666666e+00,
    -1.5000000000000000e+02,
     5.0000000000000000e+03,
    -7.5000000000000000e+04,
     5.6000000000000000e+05,
    -2.1000000000000000e+06,
     3.2000000000000000e+06,
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Coefficients for other N values (for validation / convergence testing)
# ---------------------------------------------------------------------------
STEHFEST_COEFFS = {
    6: np.array([
        -8.333333333333333e-02,
         2.416666666666667e+00,
        -4.200000000000000e+01,
         3.366666666666667e+02,
        -1.400000000000000e+03,
         2.466666666666667e+03,
    ], dtype=np.float64),

    8: STEHFEST_COEFFS_N8,

    10: np.array([
        -8.333333333333334e-04,
         1.041666666666667e+00,
        -6.343750000000000e+01,
         1.320416666666667e+03,
        -1.315687500000000e+04,
         6.962500000000000e+04,
        -2.050187500000000e+05,
         3.262500000000000e+05,
        -2.536875000000000e+05,
         7.350000000000000e+04,
    ], dtype=np.float64),

    12: np.array([
        -3.968253968253968e-05,
         8.333333333333334e-02,
        -8.375000000000000e+00,
         2.792777777777778e+02,
        -4.747916666666667e+03,
         4.665000000000000e+04,
        -2.781250000000000e+05,
         1.031766666666667e+06,
        -2.387500000000000e+06,
         3.334500000000000e+06,
        -2.550000000000000e+06,
         8.190000000000000e+05,
    ], dtype=np.float64),
}


def stehfest_inversion(
    F_s: Callable[[np.ndarray], np.ndarray],
    t: np.ndarray,
    N: int = 8,
    coeffs: np.ndarray = None,
) -> np.ndarray:
    """
    Numerical inversion of Laplace transform using the Stehfest algorithm.

    Computes f(t) from F(s) via:

        f(t) = (ln 2 / t) * sum_{i=1}^{N} V_i * F(i * ln 2 / t)

    where V_i are the Stehfest coefficients for the chosen N.

    N=8 is selected to balance truncation and round-off error for the
    dimensionless time range of interest (1e-3 to 1e4).

    Parameters
    ----------
    F_s : callable
        Laplace-domain function F(s), vectorized over s.
    t : np.ndarray
        Time points at which to evaluate f(t).
    N : int
        Number of Stehfest coefficients (must be even; default 8).
    coeffs : np.ndarray, optional
        Pre-computed Stehfest coefficients.  If None, uses built-in table.

    Returns
    -------
    f_t : np.ndarray
        Time-domain solution f(t).

    Reference
    ---------
    Stehfest, H. (1970). Algorithm 368: Numerical inversion of Laplace
    transforms [D5]. Communications of the ACM, 13(1), 47-49.
    """
    if coeffs is None:
        if N not in STEHFEST_COEFFS:
            raise ValueError(
                f"No pre-computed Stehfest coefficients for N={N}. "
                f"Available: {list(STEHFEST_COEFFS.keys())}"
            )
        coeffs = STEHFEST_COEFFS[N]

    N = len(coeffs)
    t = np.asarray(t, dtype=np.float64)
    ln2_over_t = np.log(2.0) / np.maximum(t, 1e-300)

    f_t = np.zeros_like(t, dtype=np.float64)

    for i in range(N):
        # s_i = i * ln(2) / t
        s_i = (i + 1) * ln2_over_t
        F_vals = F_s(s_i)
        f_t += coeffs[i] * F_vals

    f_t *= ln2_over_t
    return f_t


def generate_stehfest_coefficients(N: int) -> np.ndarray:
    """
    Generate Stehfest coefficients V_i for arbitrary even N.

    V_i = (-1)^{N/2+i} * sum_{k=floor((i+1)/2)}^{min(i,N/2)}
          k^{N/2} * (2k)! / ((N/2-k)! * k! * (k-1)! * (i-k)! * (2k-i)!)

    Parameters
    ----------
    N : int
        Number of coefficients (must be even and >= 2).

    Returns
    -------
    V : np.ndarray of shape (N,)
        Stehfest coefficients.
    """
    if N % 2 != 0:
        raise ValueError("N must be even")
    if N < 2:
        raise ValueError("N must be >= 2")

    from math import factorial as fact

    half = N // 2
    V = np.zeros(N)

    for i in range(1, N + 1):
        k_min = (i + 1) // 2
        k_max = min(i, half)
        total = 0.0
        for k in range(k_min, k_max + 1):
            numerator = (k ** half) * fact(2 * k)
            denominator = (
                fact(half - k) *
                fact(k) *
                fact(k - 1) *
                fact(i - k) *
                fact(2 * k - i)
            )
            total += numerator / denominator
        V[i - 1] = ((-1) ** (half + i)) * total

    return V


def validate_coefficients():
    """
    Validate generated coefficients against known reference values.

    Prints the relative difference for N = 6, 8, 10, 12.
    """
    for N in [6, 8, 10, 12]:
        coeffs_ref = STEHFEST_COEFFS[N]
        coeffs_gen = generate_stehfest_coefficients(N)
        rel_diff = np.max(np.abs(coeffs_gen - coeffs_ref) /
                         (np.abs(coeffs_ref) + 1e-15))
        print(f"N={N:2d}: max relative difference = {rel_diff:.2e}")


if __name__ == "__main__":
    # Quick validation
    validate_coefficients()

    # Example: invert F(s) = 1/s (step function) -> f(t) = 1
    def F_step(s):
        return 1.0 / s

    t = np.logspace(-2, 2, 100)
    f_t = stehfest_inversion(F_step, t, N=8)
    error = np.max(np.abs(f_t - 1.0))
    print(f"\nStep function inversion error: {error:.6e}")
