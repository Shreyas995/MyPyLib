"""
compact_derivatives.py
======================
Compact (Padé) finite-difference schemes for first and second derivatives
on a 2D Cartesian grid suited to DNS of the atmospheric boundary layer.

Grid assumptions
----------------
* X direction : uniform spacing dx (optionally periodic).
* Y direction : non-uniform / stretched.  Derivatives are evaluated in a
                uniform computational space η = 0, 1, …, N_y−1 and then
                mapped to physical space via the chain rule.

Numerical scheme
----------------
* Interior points (j = 2 … N−3) : 6th-order symmetric tridiagonal Padé
  (Lele 1992, Tables 1 & 2).
* Boundary/near-boundary rows (j = 0, 1, N−2, N−1) : 4th-order explicit
  one-sided finite differences — no extra unknowns, no ghost cells.

This gives a non-symmetric but still tridiagonal LHS, solved efficiently
with scipy's banded-matrix solver.  Passing all columns (or rows) of the
2D field as a single multi-RHS call keeps the Python overhead minimal.

Coordinate transform for Y
--------------------------
With η the uniform index and y = y(η) the physical coordinate,

    ∂f/∂y        =  (∂f/∂η) / (∂y/∂η)

    ∂²f/∂y²      = [∂²f/∂η²  −  (∂f/∂η)(∂²y/∂η²)/(∂y/∂η)] / (∂y/∂η)²

Metrics ∂y/∂η and ∂²y/∂η² are computed once with the same 4th-order
explicit stencils used at the physical boundaries.

IBM / solid region
------------------
Within the solid region the velocity field is already set to zero (or
interpolated) before calling these routines.  The compact scheme then acts
as a pure linear operator on whatever field values are supplied — no
special treatment is needed inside the solver itself.  Near the fluid–solid
interface the interpolation your code already applies will prevent
spurious oscillations in the RHS.

Array convention
----------------
All 2D field arrays have shape (Ny, Nx), i.e. rows = Y, columns = X.

Reference
---------
Lele, S. K. (1992). Compact finite difference schemes with spectral-like
resolution. Journal of Computational Physics, 103(1), 16–42.
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve


# ============================================================
#  Low-level banded helpers
# ============================================================

def _pack_banded(lower, diag, upper):
    """
    Pack three diagonal arrays into scipy's (3, n) banded-matrix format.

        ab[0, j] = M[j-1, j]   (super-diagonal)
        ab[1, j] = M[j,   j]   (main diagonal)
        ab[2, j] = M[j+1, j]   (sub-diagonal)
    """
    n = len(diag)
    ab = np.zeros((3, n), dtype=np.float64)
    ab[0, 1:]  = upper[:-1]   # super-diagonal stored shifted left
    ab[1, :]   = diag
    ab[2, :-1] = lower[1:]    # sub-diagonal stored shifted right
    return ab


def _solve_banded_system(ab, rhs):
    """
    Solve A x = rhs where A is tridiagonal in banded format *ab*.
    *rhs* may be 1-D (n,) or 2-D (n, m) for m simultaneous RHS vectors.
    """
    return solve_banded((1, 1), ab, rhs)


def _solve_periodic_tridiag(alpha, n, rhs):
    """
    Solve the circulant tridiagonal system arising from periodic compact
    schemes:

        α f'_{i-1}  +  f'_i  +  α f'_{i+1}  =  rhs_i   (indices mod n)

    Uses scipy's sparse direct solver (CSC format).
    *rhs* may be 1-D (n,) or 2-D (n, m).
    """
    diag_vals  = np.ones(n)
    off_vals   = alpha * np.ones(n - 1)
    corner_val = alpha  # wrapping corners

    # Build the circulant tridiagonal as a sparse matrix
    mat = sp_diags(
        [off_vals, diag_vals, off_vals, [corner_val], [corner_val]],
        offsets=[-1, 0, 1, -(n - 1), (n - 1)],
        shape=(n, n), format='csc', dtype=np.float64
    )
    return spsolve(mat, rhs)


# ============================================================
#  LHS matrix builders
# ============================================================

def _lhs_1st_nonperiodic(n):
    """
    Tridiagonal LHS for the non-periodic 6th-order compact 1st-derivative
    scheme (Lele 1992, Table 1):

        α = 1/3   at interior points j = 2 … n-3
        α = 0     at boundary / near-boundary rows (explicit treatment)

    Returns the pre-packed (3, n) banded matrix.
    """
    alpha = 1.0 / 3.0
    lower = np.zeros(n)
    diag  = np.ones(n)
    upper = np.zeros(n)
    # Only interior points carry the off-diagonal coupling
    lower[2:n-2] = alpha
    upper[2:n-2] = alpha
    return _pack_banded(lower, diag, upper)


def _lhs_2nd_nonperiodic(n):
    """
    Tridiagonal LHS for the non-periodic 6th-order compact 2nd-derivative
    scheme (Lele 1992, Table 2):

        α = 2/11  at interior points j = 2 … n-3
        α = 0     at boundary / near-boundary rows
    """
    alpha = 2.0 / 11.0
    lower = np.zeros(n)
    diag  = np.ones(n)
    upper = np.zeros(n)
    lower[2:n-2] = alpha
    upper[2:n-2] = alpha
    return _pack_banded(lower, diag, upper)


# ============================================================
#  RHS builders  (differentiation along axis 0)
# ============================================================

def _rhs_1st(f, h):
    """
    Build the compact-scheme RHS for df/d(axis-0 coord) on a uniform
    grid with spacing *h*.

    Works for 1-D arrays (n,) or 2-D arrays (n, m) — the m columns
    are processed simultaneously.

    Stencils
    --------
    j = 0           : 4th-order forward one-sided
    j = 1           : 4th-order forward one-sided (shifted)
    j = 2 … n-3     : 6th-order compact Padé  (α = 1/3)
    j = n-2         : 4th-order backward one-sided (shifted)
    j = n-1         : 4th-order backward one-sided
    """
    n = f.shape[0]
    rhs = np.empty_like(f, dtype=np.float64)
    ih = 1.0 / h

    # --- j = 0  (4th-order forward) ---
    # coefficients: (-25, 48, -36, 16, -3) / 12
    rhs[0] = ih * (-25*f[0] + 48*f[1] - 36*f[2] + 16*f[3] - 3*f[4]) / 12.0

    # --- j = 1  (4th-order, one-sided, shifted by -1) ---
    # coefficients: (-3, -10, 18, -6, 1) / 12
    rhs[1] = ih * (-3*f[0] - 10*f[1] + 18*f[2] - 6*f[3] + f[4]) / 12.0

    # --- j = 2 … n-3  (6th-order Padé, α = 1/3, a = 14/9, b = 1/9) ---
    a = 14.0 / 9.0
    b =  1.0 / 9.0
    rhs[2:n-2] = ih * (
          a * (f[3:n-1] - f[1:n-3]) / 2.0
        + b * (f[4:n]   - f[0:n-4]) / 4.0
    )

    # --- j = n-2  (4th-order, one-sided backward, shifted by +1) ---
    # coefficients: (-1, 6, -18, 10, 3) / 12  [offsets -3,-2,-1,0,+1]
    rhs[n-2] = ih * (
        -f[n-5] + 6*f[n-4] - 18*f[n-3] + 10*f[n-2] + 3*f[n-1]
    ) / 12.0

    # --- j = n-1  (4th-order backward) ---
    # coefficients: (3, -16, 36, -48, 25) / 12
    rhs[n-1] = ih * (
        25*f[n-1] - 48*f[n-2] + 36*f[n-3] - 16*f[n-4] + 3*f[n-5]
    ) / 12.0

    return rhs


def _rhs_2nd(f, h):
    """
    Build the compact-scheme RHS for d²f/d(axis-0 coord)² on a uniform
    grid with spacing *h*.

    Stencils
    --------
    j = 0       : 4th-order forward one-sided
    j = 1       : 4th-order forward one-sided (shifted)
    j = 2…n-3   : 6th-order compact Padé  (α = 2/11)
    j = n-2     : 4th-order backward one-sided (shifted)
    j = n-1     : 4th-order backward one-sided
    """
    n = f.shape[0]
    rhs = np.empty_like(f, dtype=np.float64)
    ih2 = 1.0 / (h * h)

    # --- j = 0  ---
    # (35, -104, 114, -56, 11) / 12  [Fornberg 4th-order one-sided 2nd deriv]
    rhs[0] = ih2 * (
         35*f[0] - 104*f[1] + 114*f[2] - 56*f[3] + 11*f[4]
    ) / 12.0

    # --- j = 1  ---
    # (11, -20, 6, 4, -1) / 12  [offsets -1,0,1,2,3]
    rhs[1] = ih2 * (
        11*f[0] - 20*f[1] + 6*f[2] + 4*f[3] - f[4]
    ) / 12.0

    # --- j = 2 … n-3  (6th-order Padé, α = 2/11, a = 12/11, b = 3/44) ---
    a = 12.0 / 11.0
    b =  3.0 / 44.0
    rhs[2:n-2] = ih2 * (
          a * (f[3:n-1] - 2*f[2:n-2] + f[1:n-3])
        + b * (f[4:n]   - 2*f[2:n-2] + f[0:n-4])
    )

    # --- j = n-2  ---
    # (-1, 4, 6, -20, 11) / 12  [offsets -3,-2,-1,0,+1]
    rhs[n-2] = ih2 * (
        -f[n-5] + 4*f[n-4] + 6*f[n-3] - 20*f[n-2] + 11*f[n-1]
    ) / 12.0

    # --- j = n-1  ---
    # (11, -56, 114, -104, 35) / 12
    rhs[n-1] = ih2 * (
        11*f[n-5] - 56*f[n-4] + 114*f[n-3] - 104*f[n-2] + 35*f[n-1]
    ) / 12.0

    return rhs


def _rhs_1st_periodic(f, h):
    """
    Build the compact-scheme RHS for the 1st derivative assuming periodic
    boundary conditions (all indices wrapped mod n).

    All rows use the interior 6th-order Padé stencil.
    """
    n = f.shape[0]
    a = 14.0 / 9.0
    b =  1.0 / 9.0
    ih = 1.0 / h

    # Wrap indices using np.roll for clarity and correctness at wrap-around
    rhs = ih * (
          a * (np.roll(f, -1, axis=0) - np.roll(f, +1, axis=0)) / 2.0
        + b * (np.roll(f, -2, axis=0) - np.roll(f, +2, axis=0)) / 4.0
    )
    return rhs


def _rhs_2nd_periodic(f, h):
    """
    Build the compact-scheme RHS for the 2nd derivative with periodic BC.
    """
    a = 12.0 / 11.0
    b =  3.0 / 44.0
    ih2 = 1.0 / (h * h)

    rhs = ih2 * (
          a * (np.roll(f, -1, axis=0) - 2*f + np.roll(f, +1, axis=0))
        + b * (np.roll(f, -2, axis=0) - 2*f + np.roll(f, +2, axis=0))
    )
    return rhs


# ============================================================
#  Grid metric computation
# ============================================================

def compute_y_metrics(y):
    """
    Compute the coordinate-transform metrics on the uniform computational
    grid η = 0, 1, …, n−1 using 4th-order finite differences (dη = 1).

    Parameters
    ----------
    y : array_like, shape (Ny,)
        Physical (stretched) Y grid.

    Returns
    -------
    dydeta   : ndarray, shape (Ny,)  — ∂y/∂η
    d2ydeta2 : ndarray, shape (Ny,)  — ∂²y/∂η²
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    assert n >= 5, "Y grid must have at least 5 points."

    dydeta   = np.empty(n)
    d2ydeta2 = np.empty(n)

    # -------- 1st metric  dy/dη  --------
    # Interior (4th-order centred, dη = 1):
    dydeta[2:n-2] = (-y[4:n] + 8*y[3:n-1] - 8*y[1:n-3] + y[0:n-4]) / 12.0
    # Boundary j = 0 (4th-order forward):
    dydeta[0]   = (-25*y[0] + 48*y[1] - 36*y[2] + 16*y[3] -  3*y[4]) / 12.0
    # Near-boundary j = 1:
    dydeta[1]   = ( -3*y[0] - 10*y[1] + 18*y[2] -  6*y[3] +    y[4]) / 12.0
    # Near-boundary j = n-2:
    dydeta[n-2] = (-y[n-5] +  6*y[n-4] - 18*y[n-3] + 10*y[n-2] + 3*y[n-1]) / 12.0
    # Boundary j = n-1 (4th-order backward):
    dydeta[n-1] = (25*y[n-1] - 48*y[n-2] + 36*y[n-3] - 16*y[n-4] + 3*y[n-5]) / 12.0

    # -------- 2nd metric  d²y/dη²  --------
    # Interior (4th-order centred, dη = 1):
    d2ydeta2[2:n-2] = (
        -y[4:n] + 16*y[3:n-1] - 30*y[2:n-2] + 16*y[1:n-3] - y[0:n-4]
    ) / 12.0
    # Boundary j = 0:
    d2ydeta2[0]   = ( 35*y[0] - 104*y[1] + 114*y[2] -  56*y[3] + 11*y[4]) / 12.0
    # Near-boundary j = 1:
    d2ydeta2[1]   = ( 11*y[0] -  20*y[1] +   6*y[2] +   4*y[3] -    y[4]) / 12.0
    # Near-boundary j = n-2:
    d2ydeta2[n-2] = (-y[n-5] +   4*y[n-4] +  6*y[n-3] - 20*y[n-2] + 11*y[n-1]) / 12.0
    # Boundary j = n-1:
    d2ydeta2[n-1] = (11*y[n-5] - 56*y[n-4] + 114*y[n-3] - 104*y[n-2] + 35*y[n-1]) / 12.0

    return dydeta, d2ydeta2


def _fornberg_weights(x_stencil, x0, m):
    """
    Fornberg (1988) finite-difference weights for the m-th derivative at x0
    using the (arbitrarily spaced) nodes in *x_stencil*.  Returns the weights
    for derivative order *m* only, shape (len(x_stencil),).

    Self-contained copy kept here so this module has no import dependency on
    functions.py; identical algorithm to functions.fornberg_weights.
    """
    x_stencil = np.asarray(x_stencil, dtype=np.float64)
    n = len(x_stencil)
    c = np.zeros((n, m + 1))
    c1 = 1.0
    c4 = x_stencil[0] - x0
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x_stencil[i] - x0
        for j in range(i):
            c3 = x_stencil[i] - x_stencil[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, m]


def _first_deriv_matrix_nonuniform(y, stencil=7):
    """
    Build the (Ny, Ny) first-derivative operator on the physical, possibly
    non-uniform grid *y* using *stencil*-point Fornberg weights.

    Each row j uses a stencil centred on j where possible, shifting to a
    fully forward/backward stencil near the two boundaries so every node
    stays in range.  The weights are computed from the actual y values, so an
    abrupt change in spacing is handled exactly (no kink artifact).

    Returns
    -------
    D : ndarray, shape (Ny, Ny)   such that  df/dy ≈ D @ f
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < stencil:
        raise ValueError(f"y must have at least {stencil} points; got {n}.")
    half = stencil // 2
    D = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        start = min(max(j - half, 0), n - stencil)   # keep stencil in [0, n)
        idx = np.arange(start, start + stencil)
        D[j, idx] = _fornberg_weights(y[idx], y[j], 1)
    return D


def _nonuniform_compact_operators(y, bnd_stencil=5):
    """
    Build the implicit (Padé) operators for a 4th-order **non-uniform**
    compact first derivative, derived directly in physical space.

    For each interior node i the tridiagonal relation

        α_i f'_{i-1} + f'_i + β_i f'_{i+1}
            = a_i f_{i-1} + b_i f_i + c_i f_{i+1}

    has its five coefficients (α, β, a, b, c) chosen to cancel Taylor terms
    up to 4th order on the *actual* spacings h_- = y_i−y_{i-1},
    h_+ = y_{i+1}−y_i — so a change in spacing is built into the coefficients
    rather than smeared by a uniform-η metric.  Boundary rows use explicit
    one-sided Fornberg weights (α = β = 0).

    Returns
    -------
    ab : ndarray, shape (3, n)   — packed tridiagonal LHS for solve_banded
    R  : ndarray, shape (n, n)   — RHS operator, so  f' = solve(ab, R @ f)
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    lower = np.zeros(n)
    diag  = np.ones(n)
    upper = np.zeros(n)
    R = np.zeros((n, n), dtype=np.float64)

    for i in range(1, n - 1):
        dm = y[i-1] - y[i]      # = -h_-   (offset of left node)
        dp = y[i+1] - y[i]      # = +h_+   (offset of right node)
        # Match f^{(1..4)} : unknowns [α, β, a, c]; b fixed by f^{(0)}: a+b+c=0
        A = np.array([
            [1.0,      1.0,      -dm,         -dp        ],   # f'
            [dm,       dp,       -dm**2/2.0,  -dp**2/2.0  ],   # f''
            [dm**2,    dp**2,    -dm**3/3.0,  -dp**3/3.0  ],   # f'''
            [dm**3,    dp**3,    -dm**4/4.0,  -dp**4/4.0  ],   # f''''
        ])
        rhs = np.array([-1.0, 0.0, 0.0, 0.0])
        alpha, beta, a, c = np.linalg.solve(A, rhs)
        b = -(a + c)
        lower[i] = alpha
        upper[i] = beta
        R[i, i-1] = a
        R[i, i]   = b
        R[i, i+1] = c

    # Boundary rows: explicit one-sided Fornberg (no implicit coupling)
    for i, idx in ((0, np.arange(0, bnd_stencil)),
                   (n - 1, np.arange(n - bnd_stencil, n))):
        R[i, idx] = _fornberg_weights(y[idx], y[i], 1)

    ab = _pack_banded(lower, diag, upper)
    return ab, R


def _second_deriv_matrix_nonuniform(y, stencil=7):
    """
    Build the (Ny, Ny) **second**-derivative operator on the physical,
    possibly non-uniform grid *y* using *stencil*-point Fornberg weights
    (Fornberg order m=2).  Direct analogue of
    :func:`_first_deriv_matrix_nonuniform`; an abrupt change in spacing is
    handled exactly because the weights use the actual y values.

    Returns
    -------
    D2 : ndarray, shape (Ny, Ny)   such that  d²f/dy² ≈ D2 @ f
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < stencil:
        raise ValueError(f"y must have at least {stencil} points; got {n}.")
    half = stencil // 2
    D2 = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        start = min(max(j - half, 0), n - stencil)
        idx = np.arange(start, start + stencil)
        D2[j, idx] = _fornberg_weights(y[idx], y[j], 2)
    return D2


def _nonuniform_compact_operators2(y, bnd_stencil=6):
    """
    Implicit (Padé) operators for a 4th-order **non-uniform** compact *second*
    derivative, derived directly in physical space.

    For each interior node i the tridiagonal relation

        α_i f''_{i-1} + f''_i + β_i f''_{i+1}
            = a_i f_{i-1} + b_i f_i + c_i f_{i+1}

    has its coefficients chosen to cancel Taylor terms through 4th order on the
    actual spacings.  Matching f^{(0)}, f^{(1)} fixes two constraints
    (a+b+c = 0 and a·dm + c·dp = 0, i.e. no spurious value/first-derivative
    leakage); f^{(2..4)} give the operator.  Boundary rows use explicit
    one-sided Fornberg second-derivative weights (α = β = 0).

    Returns
    -------
    ab : ndarray, shape (3, n)   — packed tridiagonal LHS for solve_banded
    R  : ndarray, shape (n, n)   — RHS operator, so  f'' = solve(ab, R @ f)
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    lower = np.zeros(n)
    diag  = np.ones(n)
    upper = np.zeros(n)
    R = np.zeros((n, n), dtype=np.float64)

    for i in range(1, n - 1):
        dm = y[i-1] - y[i]      # = -h_-
        dp = y[i+1] - y[i]      # = +h_+
        # Unknowns [α, β, a, c]; b from f^{(0)}: a+b+c = 0.
        # f^{(1)} : a·dm + c·dp = 0
        # f^{(2)} : α + β + 1 = a·dm²/2 + c·dp²/2
        # f^{(3)} : α·dm + β·dp = a·dm³/6 + c·dp³/6
        # f^{(4)} : α·dm²/2 + β·dp²/2 = a·dm⁴/24 + c·dp⁴/24
        A = np.array([
            [0.0,        0.0,        dm,            dp           ],   # f'
            [1.0,        1.0,        -dm**2/2.0,    -dp**2/2.0    ],   # f''
            [dm,         dp,         -dm**3/6.0,    -dp**3/6.0    ],   # f'''
            [dm**2/2.0,  dp**2/2.0,  -dm**4/24.0,   -dp**4/24.0   ],   # f''''
        ])
        rhs = np.array([0.0, -1.0, 0.0, 0.0])
        alpha, beta, a, c = np.linalg.solve(A, rhs)
        b = -(a + c)
        lower[i] = alpha
        upper[i] = beta
        R[i, i-1] = a
        R[i, i]   = b
        R[i, i+1] = c

    for i, idx in ((0, np.arange(0, bnd_stencil)),
                   (n - 1, np.arange(n - bnd_stencil, n))):
        R[i, idx] = _fornberg_weights(y[idx], y[i], 2)

    ab = _pack_banded(lower, diag, upper)
    return ab, R


# ============================================================
#  Main class
# ============================================================

class CompactDerivatives2D:
    """
    Compact (Padé) first and second derivatives on a 2D Cartesian grid
    with uniform X spacing and stretched Y spacing.

    Parameters
    ----------
    x : array_like, shape (Nx,)
        Uniformly spaced horizontal (X) grid.
    y : array_like, shape (Ny,)
        Stretched vertical (Y) grid.
    periodic_x : bool, optional
        If True, apply periodic compact scheme along X (typical for DNS
        channel flow).  Default False.

    Array convention
    ----------------
    All field arrays have shape **(Ny, Nx)** — rows index Y, columns X.

    Quick-start example
    -------------------
    >>> cd = CompactDerivatives2D(x, y, periodic_x=True)
    >>> dfdx = cd.ddx(f)           # ∂f/∂x
    >>> dfdy = cd.ddy(f)           # ∂f/∂y
    >>> d2fdx2 = cd.d2dx2(f)      # ∂²f/∂x²
    >>> d2fdy2 = cd.d2dy2(f)      # ∂²f/∂y²
    >>> lapl  = cd.laplacian(f)   # ∂²f/∂x² + ∂²f/∂y²
    """

    def __init__(self, x, y, periodic_x=False):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.Nx = self.x.size
        self.Ny = self.y.size
        self.dx = float(self.x[1] - self.x[0])
        self.periodic_x = periodic_x

        # Computational Y spacing (index units, dη = 1)
        self._deta = 1.0

        # --- Y coordinate-transform metrics ---
        self._dydeta, self._d2ydeta2 = compute_y_metrics(self.y)

        # Lazily built physical-space y-derivative operators (built on first use):
        #   _Dy_fornberg[stencil]  : explicit Fornberg 1st-derivative matrix
        #   _nucompact             : (ab, R) for the non-uniform compact 1st deriv
        #   _D2y_fornberg[stencil] : explicit Fornberg 2nd-derivative matrix
        #   _nucompact2            : (ab, R) for the non-uniform compact 2nd deriv
        self._Dy_fornberg  = {}
        self._nucompact    = None
        self._D2y_fornberg = {}
        self._nucompact2   = None

        # --- Pre-build banded LHS matrices ---
        # Y direction: always non-periodic (wall-bounded)
        self._ab_y1 = _lhs_1st_nonperiodic(self.Ny)
        self._ab_y2 = _lhs_2nd_nonperiodic(self.Ny)
        # X direction: periodic or non-periodic
        if not periodic_x:
            self._ab_x1 = _lhs_1st_nonperiodic(self.Nx)
            self._ab_x2 = _lhs_2nd_nonperiodic(self.Nx)

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _diff_x(self, field, order):
        """Apply compact derivative of *order* along X (axis 1)."""
        f = np.asarray(field, dtype=np.float64)
        # Transpose so X is along axis 0 → solve all Y-rows at once
        ft = np.ascontiguousarray(f.T)   # (Nx, Ny)

        if self.periodic_x:
            alpha = (1.0/3.0) if order == 1 else (2.0/11.0)
            rhs_fn = _rhs_1st_periodic if order == 1 else _rhs_2nd_periodic
            rhs = rhs_fn(ft, self.dx)
            # Solve each Y-column independently (periodic circulant)
            result = np.empty_like(ft)
            for j in range(self.Ny):
                result[:, j] = _solve_periodic_tridiag(alpha, self.Nx, rhs[:, j])
        else:
            ab     = self._ab_x1 if order == 1 else self._ab_x2
            rhs_fn = _rhs_1st    if order == 1 else _rhs_2nd
            rhs    = rhs_fn(ft, self.dx)
            result = _solve_banded_system(ab, rhs)

        return result.T   # back to (Ny, Nx)

    def _diff_eta(self, field, order):
        """
        Apply compact derivative of *order* along η (uniform Y-space, axis 0).
        Returns ∂^n f / ∂η^n in computational space, shape (Ny, Nx).
        """
        f = np.ascontiguousarray(
            np.asarray(field, dtype=np.float64)
        )  # (Ny, Nx) — differentiate along axis 0
        ab     = self._ab_y1 if order == 1 else self._ab_y2
        rhs_fn = _rhs_1st    if order == 1 else _rhs_2nd
        rhs    = rhs_fn(f, self._deta)
        return _solve_banded_system(ab, rhs)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def ddx(self, field):
        """
        First derivative ∂f/∂x.

        Parameters
        ----------
        field : array_like, shape (Ny, Nx)

        Returns
        -------
        ndarray, shape (Ny, Nx)
        """
        return self._diff_x(field, 1)

    def d2dx2(self, field):
        """
        Second derivative ∂²f/∂x².

        Parameters
        ----------
        field : array_like, shape (Ny, Nx)

        Returns
        -------
        ndarray, shape (Ny, Nx)
        """
        return self._diff_x(field, 2)

    # Accepted aliases for each first-y-derivative scheme
    _COMPACT_ALIASES   = ('compact', 'eta', 'pade')
    _FORNBERG_ALIASES  = ('physical', 'phys', 'fornberg', 'fornberg7',
                          'nonuniform')
    _FORNBERG5_ALIASES = ('fornberg5',)
    _FORNBERG9_ALIASES = ('fornberg9',)
    _NUCOMPACT_ALIASES = ('compact_nu', 'compact_nonuniform', 'nucompact')
    _SPLINE_ALIASES    = ('spline', 'cubic')

    def ddy(self, field, method='compact'):
        """
        First derivative ∂f/∂y, with a user-selectable scheme.

        Parameters
        ----------
        field : array_like, shape (Ny, Nx)
        method : str, optional
            Differentiation scheme:

            * ``'compact'`` (default; ``'eta'``, ``'pade'``) — 6th-order Padé
              in uniform η-space, mapped via ∂f/∂y = (∂f/∂η)/(∂y/∂η).
              Spectral-like accuracy on a *smooth* mapping y(η).
            * ``'fornberg'`` / ``'physical'`` (also ``'fornberg7'``) — explicit
              7-point Fornberg weights on the actual y (see :meth:`ddy_phys`).
            * ``'fornberg5'`` / ``'fornberg9'`` — 5- / 9-point Fornberg
              (narrower = more local at a kink; wider = higher formal order).
            * ``'compact_nu'`` (``'compact_nonuniform'``, ``'nucompact'``) —
              4th-order **non-uniform** Padé derived directly in physical space
              (see :meth:`ddy_compact_nu`): compact implicit accuracy with the
              spacing built into the coefficients.
            * ``'spline'`` (``'cubic'``) — derivative of a C² cubic spline
              through the actual y nodes (see :meth:`ddy_spline`).

        Returns
        -------
        ndarray, shape (Ny, Nx)

        Notes
        -----
        ``'compact'`` assumes a smooth mapping y(η).  Where the physical grid
        has an abrupt spacing change (e.g. the factor-2 dy step at the top of
        Zone 1 on the old 1024×832×1024 grid), its centred dy/dη metric is
        smeared and a spurious wiggle appears.  All other schemes use the real
        y coordinates and stay smooth there; away from any kink they agree with
        ``'compact'`` to ~0.1 %.
        """
        m = method.lower()
        if m in self._COMPACT_ALIASES:
            df_deta = self._diff_eta(field, 1)             # (Ny, Nx)
            return df_deta / self._dydeta[:, np.newaxis]
        if m in self._FORNBERG_ALIASES:
            return self.ddy_phys(field, stencil=7)
        if m in self._FORNBERG5_ALIASES:
            return self.ddy_phys(field, stencil=5)
        if m in self._FORNBERG9_ALIASES:
            return self.ddy_phys(field, stencil=9)
        if m in self._NUCOMPACT_ALIASES:
            return self.ddy_compact_nu(field)
        if m in self._SPLINE_ALIASES:
            return self.ddy_spline(field)
        valid = (self._COMPACT_ALIASES + self._FORNBERG_ALIASES
                 + self._FORNBERG5_ALIASES + self._FORNBERG9_ALIASES
                 + self._NUCOMPACT_ALIASES + self._SPLINE_ALIASES)
        raise ValueError(
            f"Unknown ddy method {method!r}; choose one of {valid}."
        )

    def ddy_phys(self, field, stencil=7):
        """
        First derivative ∂f/∂y from **explicit** non-uniform Fornberg weights.

        Uses an *stencil*-point finite-difference stencil built from the actual
        (possibly non-uniform) y coordinates — no computational-space mapping
        and no dy/dη metric.  Because the weights use the real local spacing,
        an abrupt change in dy (a grid kink) is differentiated exactly rather
        than smeared, so the result stays smooth where the η-space ``'compact'``
        scheme trembles.

        Parameters
        ----------
        field   : array_like, shape (Ny, Nx)
        stencil : int, optional — number of points (default 7).

        Returns
        -------
        ndarray, shape (Ny, Nx)
        """
        if stencil not in self._Dy_fornberg:
            self._Dy_fornberg[stencil] = _first_deriv_matrix_nonuniform(
                self.y, stencil=stencil)
        f = np.asarray(field, dtype=np.float64)
        return self._Dy_fornberg[stencil] @ f

    def ddy_compact_nu(self, field):
        """
        First derivative ∂f/∂y from the 4th-order **non-uniform compact**
        (Padé) scheme — implicit/tridiagonal, with coefficients derived from
        the actual local spacings (see :func:`_nonuniform_compact_operators`).

        Combines compact implicit accuracy with native non-uniform handling,
        so it stays smooth across grid-spacing changes.

        Parameters
        ----------
        field : array_like, shape (Ny, Nx)

        Returns
        -------
        ndarray, shape (Ny, Nx)
        """
        if self._nucompact is None:
            self._nucompact = _nonuniform_compact_operators(self.y)
        ab, R = self._nucompact
        f = np.ascontiguousarray(np.asarray(field, dtype=np.float64))
        return _solve_banded_system(ab, R @ f)

    def ddy_spline(self, field):
        """
        First derivative ∂f/∂y from a C² cubic spline through the actual y
        nodes (``scipy.interpolate.CubicSpline``, applied column-wise).

        The natural cubic spline is itself an implicit/compact construction on
        non-uniform grids, so it handles spacing changes gracefully.

        Parameters
        ----------
        field : array_like, shape (Ny, Nx)

        Returns
        -------
        ndarray, shape (Ny, Nx)
        """
        from scipy.interpolate import CubicSpline
        f = np.asarray(field, dtype=np.float64)
        cs = CubicSpline(self.y, f, axis=0, bc_type='not-a-knot')
        return cs(self.y, 1)

    def d2dy2(self, field, method='compact'):
        """
        Second derivative ∂²f/∂y², with a user-selectable scheme.

        Parameters
        ----------
        field : array_like, shape (Ny, Nx)
        method : str, optional — same option set as :meth:`ddy`:
            ``'compact'`` (default; η-space Padé chain rule),
            ``'fornberg5/7/9'`` (explicit non-uniform, :meth:`ddy2_phys`),
            ``'compact_nu'`` (non-uniform Padé, :meth:`ddy2_compact_nu`),
            ``'spline'`` (cubic spline, :meth:`ddy2_spline`).

        Returns
        -------
        ndarray, shape (Ny, Nx)

        Notes
        -----
        Like the first derivative, ``'compact'`` assumes a smooth mapping
        y(η); at an abrupt spacing change the metric d²y/dη² spikes and the
        result trembles.  The physical-space schemes use the real y and stay
        smooth there.
        """
        m = method.lower()
        if m in self._COMPACT_ALIASES:
            df_deta   = self._diff_eta(field, 1)
            d2f_deta2 = self._diff_eta(field, 2)
            J  = self._dydeta[:, np.newaxis]     # ∂y/∂η
            J2 = self._d2ydeta2[:, np.newaxis]   # ∂²y/∂η²
            return (d2f_deta2 - df_deta * (J2 / J)) / J**2
        if m in self._FORNBERG_ALIASES:
            return self.ddy2_phys(field, stencil=7)
        if m in self._FORNBERG5_ALIASES:
            return self.ddy2_phys(field, stencil=5)
        if m in self._FORNBERG9_ALIASES:
            return self.ddy2_phys(field, stencil=9)
        if m in self._NUCOMPACT_ALIASES:
            return self.ddy2_compact_nu(field)
        if m in self._SPLINE_ALIASES:
            return self.ddy2_spline(field)
        valid = (self._COMPACT_ALIASES + self._FORNBERG_ALIASES
                 + self._FORNBERG5_ALIASES + self._FORNBERG9_ALIASES
                 + self._NUCOMPACT_ALIASES + self._SPLINE_ALIASES)
        raise ValueError(
            f"Unknown d2dy2 method {method!r}; choose one of {valid}.")

    def ddy2_phys(self, field, stencil=7):
        """∂²f/∂y² from explicit non-uniform Fornberg weights (order 2).

        Direct physical-space analogue of :meth:`ddy_phys`; robust to grid
        spacing kinks.  *stencil* = number of points (default 7).
        """
        if stencil not in self._D2y_fornberg:
            self._D2y_fornberg[stencil] = _second_deriv_matrix_nonuniform(
                self.y, stencil=stencil)
        f = np.asarray(field, dtype=np.float64)
        return self._D2y_fornberg[stencil] @ f

    def ddy2_compact_nu(self, field):
        """∂²f/∂y² from the 4th-order non-uniform compact (Padé) scheme
        (see :func:`_nonuniform_compact_operators2`)."""
        if self._nucompact2 is None:
            self._nucompact2 = _nonuniform_compact_operators2(self.y)
        ab, R = self._nucompact2
        f = np.ascontiguousarray(np.asarray(field, dtype=np.float64))
        return _solve_banded_system(ab, R @ f)

    def ddy2_spline(self, field):
        """∂²f/∂y² from the second derivative of a C² cubic spline through
        the actual y nodes (``scipy.interpolate.CubicSpline``)."""
        from scipy.interpolate import CubicSpline
        f = np.asarray(field, dtype=np.float64)
        cs = CubicSpline(self.y, f, axis=0, bc_type='not-a-knot')
        return cs(self.y, 2)

    def gradient(self, field):
        """
        Return (∂f/∂x, ∂f/∂y), both of shape (Ny, Nx).
        """
        return self.ddx(field), self.ddy(field)

    def laplacian(self, field):
        """
        Return the Laplacian ∂²f/∂x² + ∂²f/∂y², shape (Ny, Nx).
        """
        return self.d2dx2(field) + self.d2dy2(field)

    # ----------------------------------------------------------
    # Convenience properties
    # ----------------------------------------------------------

    @property
    def metrics(self):
        """
        Return a dict with the Y coordinate-transform metrics:
            'dydeta'   : ∂y/∂η, shape (Ny,)
            'd2ydeta2' : ∂²y/∂η², shape (Ny,)
        """
        return {"dydeta": self._dydeta.copy(), "d2ydeta2": self._d2ydeta2.copy()}


# ============================================================
#  Grid utility: typical DNS stretched Y grid
# ============================================================

def make_uniform_x(Nx, Lx, endpoint=False):
    """
    Uniformly spaced X grid on [0, Lx).

    Parameters
    ----------
    Nx       : int
    Lx       : float  — domain length
    endpoint : bool   — if True include Lx (non-periodic); default False (periodic)

    Returns
    -------
    x : ndarray, shape (Nx,)
    """
    return np.linspace(0.0, Lx, Nx, endpoint=endpoint)


def make_stretched_y(Ny, y_bot, y_top,
                     n_lin_bot=None, n_lin_top=None,
                     stretch_param=2.0,
                     clustering='symmetric'):
    """
    Build a smooth (C-∞) stretched Y grid for DNS of the atmospheric
    boundary layer.

    The grid uses a **global tanh-based mapping** with no piecewise
    junctions, so the spacing and its derivatives are continuous everywhere.
    Continuity is essential: any kink in y(η) corrupts the chain-rule
    metrics (dy/dη, d²y/dη²) and destroys the accuracy of the compact
    Padé scheme.

    Near-wall behaviour
    -------------------
    For small β|ξ|, tanh(β ξ) ≈ β ξ, so the spacing is nearly linear
    close to the clustered wall(s) — matching the qualitative character
    of the previous piecewise grid without the discontinuity.

    Clustering modes
    ----------------
    'symmetric' (default)
        Fine spacing near *both* walls, coarser in the interior.
        Uses the two-sided formula:

            y = y_bot + H * [1 + tanh(β(ξ−½)) / tanh(β/2)] / 2

    'bottom'
        Fine spacing near the bottom wall only.
        One-sided formula:  y = y_bot + H * tanh(β ξ) / tanh(β)

    'top'
        Fine spacing near the top wall only.
        One-sided formula:  y = y_top − H * tanh(β(1−ξ)) / tanh(β)

    Parameters
    ----------
    Ny            : int     — total number of grid points
    y_bot, y_top  : float   — physical domain extents
    n_lin_bot     : int     — *legacy, unused* (kept for API compatibility)
    n_lin_top     : int     — *legacy, unused*
    stretch_param : float   — tanh stretching parameter β.
                              Larger β → stronger wall clustering.
                              Typical range 2–6.
    clustering    : str     — 'symmetric' | 'bottom' | 'top'

    Returns
    -------
    y : ndarray, shape (Ny,)
    """
    H    = float(y_top - y_bot)
    beta = float(stretch_param)
    xi   = np.linspace(0.0, 1.0, Ny)          # uniform index ∈ [0, 1]

    if clustering == 'symmetric':
        # Two-sided tanh: clusters near ξ = 0 AND ξ = 1
        y = y_bot + H * (
            1.0 + np.tanh(beta * (xi - 0.5)) / np.tanh(0.5 * beta)
        ) / 2.0

    elif clustering == 'bottom':
        # One-sided: clusters near ξ = 0 (bottom / IBM wall)
        y = y_bot + H * np.tanh(beta * xi) / np.tanh(beta)

    elif clustering == 'top':
        # One-sided: clusters near ξ = 1 (top wall)
        y = y_top - H * np.tanh(beta * (1.0 - xi)) / np.tanh(beta)

    else:
        raise ValueError(
            f"clustering must be 'symmetric', 'bottom', or 'top'; got {clustering!r}"
        )

    return y


# ============================================================
#  Validation / accuracy test
# ============================================================

def validate(Nx=128, Ny=96, verbose=True):
    """
    Validate CompactDerivatives2D on the test function

        f(x, y) = sin(kx·x) · cos(ky·y)

    whose exact derivatives are known analytically.

    Parameters
    ----------
    Nx, Ny  : int   — grid resolution
    verbose : bool  — print per-derivative max error

    Returns
    -------
    errors : dict  — max absolute error for each derivative operator
    """
    Lx = 2 * np.pi
    kx, ky = 2.0, 3.0

    x = make_uniform_x(Nx, Lx, endpoint=False)   # periodic in X
    y = make_stretched_y(Ny, 0.0, np.pi)

    X, Y = np.meshgrid(x, y)

    f              =  np.sin(kx*X) * np.cos(ky*Y)
    dfdx_exact     =  kx * np.cos(kx*X) * np.cos(ky*Y)
    dfdy_exact     = -ky * np.sin(kx*X) * np.sin(ky*Y)
    d2fdx2_exact   = -kx**2 * np.sin(kx*X) * np.cos(ky*Y)
    d2fdy2_exact   = -ky**2 * np.sin(kx*X) * np.cos(ky*Y)

    cd = CompactDerivatives2D(x, y, periodic_x=True)

    computed = {
        "ddx"   : cd.ddx(f),
        "ddy"   : cd.ddy(f),
        "d2dx2" : cd.d2dx2(f),
        "d2dy2" : cd.d2dy2(f),
    }
    exact = {
        "ddx"   : dfdx_exact,
        "ddy"   : dfdy_exact,
        "d2dx2" : d2fdx2_exact,
        "d2dy2" : d2fdy2_exact,
    }

    errors = {}
    if verbose:
        print(f"Validation  (Nx={Nx}, Ny={Ny})")
        print(f"  Grid: X uniform dx={x[1]-x[0]:.4f},  "
              f"Y stretched dy_min={np.diff(y).min():.4f}  "
              f"dy_max={np.diff(y).max():.4f}")
        print(f"  {'Operator':<10}  {'max|error|':>14}  {'rel. max|error|':>18}")
        print("  " + "-"*46)

    for key in ("ddx", "ddy", "d2dx2", "d2dy2"):
        err     = np.max(np.abs(exact[key] - computed[key]))
        rel_err = err / np.max(np.abs(exact[key]))
        errors[key] = err
        if verbose:
            print(f"  {key:<10}  {err:>14.3e}  {rel_err:>18.3e}")

    return errors


# ============================================================
#  Demo / usage example
# ============================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  Compact Padé Derivatives — DNS Boundary Layer Demo")
    print("=" * 60)

    # ---- Build grids representative of a DNS ABL simulation ----
    Nx, Ny = 256, 128
    Lx = 4 * np.pi          # streamwise domain length
    y_bot, y_top = 0.0, 1.0 # normalised wall-normal extent

    x = make_uniform_x(Nx, Lx, endpoint=False)
    y = make_stretched_y(Ny, y_bot, y_top, stretch_param=3.0,
                         clustering='symmetric')

    print(f"\nGrid: Nx={Nx}, Ny={Ny}")
    print(f"  X: uniform,   dx = {x[1]-x[0]:.5f}")
    print(f"  Y: stretched, dy_min = {np.diff(y).min():.5f}, "
          f"dy_max = {np.diff(y).max():.5f}")

    # ---- Initialise the derivative operator ----
    cd = CompactDerivatives2D(x, y, periodic_x=True)

    # ---- Synthetic velocity field (stand-in for DNS data) ----
    X, Y = np.meshgrid(x, y)
    u = np.sin(2*X) * np.cos(3*Y) * Y * (1 - Y)   # vanishes at walls

    # ---- Compute derivatives ----
    t0 = time.perf_counter()
    dudx   = cd.ddx(u)
    dudy   = cd.ddy(u)
    d2udx2 = cd.d2dx2(u)
    d2udy2 = cd.d2dy2(u)
    lapl_u = cd.laplacian(u)
    elapsed = time.perf_counter() - t0

    print(f"\nAll five derivative fields computed in {elapsed*1e3:.1f} ms")
    print(f"  dudx   shape={dudx.shape},   max|dudx|   = {np.max(np.abs(dudx)):.4f}")
    print(f"  dudy   shape={dudy.shape},   max|dudy|   = {np.max(np.abs(dudy)):.4f}")
    print(f"  lapl_u shape={lapl_u.shape}, max|lapl_u| = {np.max(np.abs(lapl_u)):.4f}")

    # ---- Accuracy check ----
    print("\n---- Accuracy validation ----")
    validate(Nx=128, Ny=96, verbose=True)
