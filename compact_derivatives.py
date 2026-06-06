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

    def ddy(self, field):
        """
        First derivative ∂f/∂y.

        Computed in uniform η-space and converted via the chain rule:
            ∂f/∂y = (∂f/∂η) / (∂y/∂η)

        Parameters
        ----------
        field : array_like, shape (Ny, Nx)

        Returns
        -------
        ndarray, shape (Ny, Nx)
        """
        df_deta = self._diff_eta(field, 1)                 # (Ny, Nx)
        J = self._dydeta[:, np.newaxis]                    # (Ny, 1)
        return df_deta / J

    def d2dy2(self, field):
        """
        Second derivative ∂²f/∂y².

        Chain rule for non-uniform grids:
            ∂²f/∂y² = [∂²f/∂η² − (∂f/∂η)(∂²y/∂η²)/(∂y/∂η)] / (∂y/∂η)²

        Parameters
        ----------
        field : array_like, shape (Ny, Nx)

        Returns
        -------
        ndarray, shape (Ny, Nx)
        """
        df_deta   = self._diff_eta(field, 1)
        d2f_deta2 = self._diff_eta(field, 2)

        J  = self._dydeta[:, np.newaxis]     # ∂y/∂η,  shape (Ny, 1)
        J2 = self._d2ydeta2[:, np.newaxis]   # ∂²y/∂η², shape (Ny, 1)

        return (d2f_deta2 - df_deta * (J2 / J)) / J**2

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
