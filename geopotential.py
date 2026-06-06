#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geopotential_v14.py
===================
Post-processing: potential function (phi) and stream function (psi)
from a DNS atmospheric boundary layer (Ekman) simulation.

Physics
-------
    ∇²φ =  ∇·u    (divergence  → potential function)
    ∇²ψ = −ω      (vorticity   → stream function)

═══════════════════════════════════════════════════════════════
CHANGES FROM v13  (two bugs fixed)
═══════════════════════════════════════════════════════════════

BUG 1 — φ ≈ e-8  (NOT a bug — it is correct physics)
──────────────────────────────────────────────────────
The DNS enforces ∇·u = 0 to machine precision via pressure
projection.  Therefore ∇²φ = ∇·u ≈ 0, so φ ≈ 0 exactly.
φ carries the compressible (irrotational) part of the flow;
for an incompressible field it must vanish.  The e-8 value
reported in v13 is physically correct and not a numerical
artefact.

BUG 2 — ψ ≈ e-3 instead of O(0.3)  ← real bug, now fixed
──────────────────────────────────────────────────────────────
The stream function is defined by  u = ∂ψ/∂y,  v = −∂ψ/∂x.
Its boundary values follow directly from integrating u:

    ψ(x, 0)   = 0                         (no-slip bottom)
    ψ(x, L_y) = Q(x) = ∫₀^{L_y} u(y,x) dy   (top BC)

Because the flow is incompressible and x-periodic,
Q(x) = const = 0.3187  (verified: std/mean < 1e-6).
The k=0 Fourier mode therefore requires an inhomogeneous
Dirichlet BC at the top:

    ψ̂_0(L_y) = Q_mean   (only for ki = 0)
    ψ̂_k(L_y) = 0        (all other wavenumbers, unchanged)

v13 imposed ψ = 0 at both boundaries for every wavenumber,
effectively subtracting the entire mean-flow stream function
and leaving only a residual of O(e-3).  The corrected v14
recovers the full physical stream function O(0.3), whose
contours are the flow streamlines arching over the hill.

Fix location: compute_phi_psi() — one extra argument
(psi_bc_top, default None) and four lines of code.

Method
------
1.  FFT along the periodic x direction.
    Each wavenumber k reduces the 2-D Poisson equation to a 1-D Helmholtz:
        (d²/dy² − k²) f̂  =  ĝ

2.  Build a (Ny × Ny) compact Padé d²/dy² matrix (D2) from
    CompactDerivatives2D.

3.  Two-pass capacitance matrix correction:

    Pass 1: solve ignoring solid geometry.
    Pass 2: build capacitance matrix C by placing unit sources at each
            interface point.  Solve C·λ = −r for correction strengths λ.
            Add correction to RHS and re-solve.

Boundary Conditions
-------------------
    φ (Neumann):   ∂φ/∂y = 0  at j=0 (bottom) and j=ny-1 (top)
                   Boundary rows of A replaced with D1 rows.
                   k=0 gauge fix: pin top node to 0.

    ψ (Dirichlet): ψ = 0         at j=0 (bottom)
                   ψ = Q_mean    at j=ny-1 (top), ki=0 only   ← FIXED
                   ψ = 0         at j=ny-1 (top), ki≠0
                   Boundary rows of A replaced with identity rows.
"""

import os
import numpy as np
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

from compact_derivatives import (
    CompactDerivatives2D,
    make_uniform_x,
    make_stretched_y,
)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Derivative matrix builder
# ═══════════════════════════════════════════════════════════════════════════

def build_derivative_matrices(cd: CompactDerivatives2D):
    """
    Build dense (Ny × Ny) D1, D2 matrices from the compact Padé operator.
    Built once and reused in all solvers and the capacitance matrix build.
    """
    ny = cd.Ny
    D1 = np.zeros((ny, ny))
    D2 = np.zeros((ny, ny))
    e  = np.zeros((ny, 1))
    for j in range(ny):
        e[:] = 0.0; e[j, 0] = 1.0
        D1[:, j] = cd.ddy(e)[:, 0]
        D2[:, j] = cd.d2dy2(e)[:, 0]
    return D1, D2


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Boundary condition enforcement
# ═══════════════════════════════════════════════════════════════════════════

def _apply_phi_bcs(A, rhs, D1, ki):
    """
    Neumann BCs for φ: replace boundary rows of A with D1 rows.
    Gauge fix at ki=0 (null space = constants under pure Neumann).
    Works for rhs shape (ny,) and (ny, M).
    """
    A[0,  :] = D1[0,  :]
    A[-1, :] = D1[-1, :]
    if rhs.ndim == 1:
        rhs[0] = 0.0;  rhs[-1] = 0.0
    else:
        rhs[0, :] = 0.0;  rhs[-1, :] = 0.0
    if ki == 0:                          # gauge fix: pin top node
        A[-1, :] = 0.0;  A[-1, -1] = 1.0
        if rhs.ndim == 1: rhs[-1] = 0.0
        else:             rhs[-1, :] = 0.0
    return A, rhs


def _apply_psi_bcs(A, rhs, psi_bc_top=0.0):
    """
    Dirichlet BCs for ψ:
        ψ(j=0)    = 0           bottom (no-slip wall)
        ψ(j=ny-1) = psi_bc_top  top

    For ki=0:  psi_bc_top = Q_mean = ∫ u dy  (volume flux per unit width)
    For ki≠0:  psi_bc_top = 0               (unchanged from v13)

    Works for rhs shape (ny,) and (ny, M).
    """
    A[0,  :] = 0.0;  A[0,  0]  = 1.0
    A[-1, :] = 0.0;  A[-1, -1] = 1.0
    if rhs.ndim == 1:
        rhs[0] = 0.0;  rhs[-1] = psi_bc_top
    else:
        rhs[0, :] = 0.0;  rhs[-1, :] = psi_bc_top
    return A, rhs


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Per-wavenumber solvers
# ═══════════════════════════════════════════════════════════════════════════

def _solve_phi_k(A, rhs, D1, ki):
    Aw = A.copy();  b = rhs.copy()
    Aw, b = _apply_phi_bcs(Aw, b, D1, ki)
    return np.linalg.solve(Aw, b)


def _solve_psi_k(A, rhs, psi_bc_top=0.0):
    """
    psi_bc_top: inhomogeneous Dirichlet value at j=ny-1.
    Pass Q_mean for ki=0, 0.0 for all other wavenumbers.
    """
    Aw = A.copy();  b = rhs.copy()
    Aw, b = _apply_psi_bcs(Aw, b, psi_bc_top=psi_bc_top)
    return np.linalg.solve(Aw, b)


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Poisson solver
# ═══════════════════════════════════════════════════════════════════════════

def compute_phi_psi(x, y, divergence, vorticity,
                    D1=None, D2=None,
                    psi_bc_top=None,
                    verbose=True):
    """
    Solve ∇²φ = div  and  ∇²ψ = −ω via FFT in x, compact Padé in y.

    Parameters
    ----------
    x, y       : 1-D coordinate arrays
    divergence : (ny, nx) field — solid region must be zeroed before calling
    vorticity  : (ny, nx) field — solid region must be zeroed before calling
    D1, D2     : prebuilt compact derivative matrices (built once, reused)
    psi_bc_top : float or None
        Top Dirichlet BC for ψ at y = y[-1].
        If None, it is computed here as  ∫ u dy  (requires the velocity
        field or an external call).  Passing it in avoids repeated
        integration.

        Physical meaning:
            ψ(x, 0)   = 0          (no-slip bottom wall)
            ψ(x, L_y) = Q_mean     (total volume flux per unit x-width)
        where Q_mean = (1/Lx) ∫∫ u dy dx = ∫ <u>(y) dy.

        For ki=0  → psi_bc_top  (the only non-zero Fourier component)
        For ki≠0  → 0           (flow is incompressible + periodic in x,
                                  so Q(x) = const → all harmonics zero)

        Set psi_bc_top=0.0 to reproduce v13 behaviour (wrong, for reference).
    verbose : bool

    Returns
    -------
    phi, psi : (ny, nx) real arrays
    """
    nx = x.size;  ny = y.size
    Lx = nx * (x[1] - x[0])
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    I  = np.eye(ny)

    if D1 is None or D2 is None:
        cd = CompactDerivatives2D(x, y, periodic_x=True)
        if verbose: print(f"  Building D1, D2  (Ny={ny}) …")
        D1, D2 = build_derivative_matrices(cd)

    if psi_bc_top is None:
        raise ValueError(
            "psi_bc_top must be supplied.  "
            "Compute it as: psi_bc_top = trapezoid(U_field, y, axis=0).mean() "
            "using the (interpolated) velocity field before zeroing the solid."
        )

    f_div  = np.fft.fft(divergence, axis=1)
    f_vort = np.fft.fft(vorticity,  axis=1)
    phi_hat = np.zeros((ny, nx), dtype=complex)
    psi_hat = np.zeros((ny, nx), dtype=complex)

    if verbose: print(f"  Solving Helmholtz BVP for {nx} wavenumbers …")

    for ki in range(nx):
        k = kx[ki];  A = D2 - k**2 * I

        # φ: Neumann BCs, gauge fix at ki=0 — unchanged
        phi_hat[:, ki] = _solve_phi_k(A, f_div[:, ki].copy(), D1, ki)

        # ψ: ki=0 gets the volume-flux top BC; all others keep 0
        bc_top = psi_bc_top if ki == 0 else 0.0
        psi_hat[:, ki] = _solve_psi_k(A, -f_vort[:, ki].copy(),
                                       psi_bc_top=bc_top)

    phi = np.real(np.fft.ifft(phi_hat, axis=1))
    psi = np.real(np.fft.ifft(psi_hat, axis=1))
    if verbose: print("  Done.")
    return phi, psi


# ═══════════════════════════════════════════════════════════════════════════
#  5.  Interface extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_interface_points(top_surface, x, y):
    """
    Extract M interface points from top_surface mask.
    Excludes j=0 (domain floor) and j=ny-1 (domain top) — handled by
    the outer wall BCs of the Poisson solver, not the capacitance correction.
    """
    ny = y.size
    mask = top_surface.copy()
    mask[0, :] = False;  mask[ny-1, :] = False
    j_idx, i_idx = np.where(mask)
    return j_idx, i_idx, x[i_idx], y[j_idx], j_idx.size


# ═══════════════════════════════════════════════════════════════════════════
#  6.  Capacitance matrix builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_C(D1, D2, kx, j_idx, i_idx, ny, nx, bc_type,
             psi_bc_top=0.0, verbose=True):
    """
    Build M×M capacitance matrix with specified outer wall BCs.

    bc_type = 'neumann'   → matches φ solver  (C_phi)
    bc_type = 'dirichlet' → matches ψ solver  (C_psi)

    Note: the capacitance matrix characterises the GREEN's function response
    to a unit source at each interface point.  The inhomogeneous top BC for ψ
    (psi_bc_top) is applied only to the main RHS, NOT to the Green's function
    columns here.  Each column of C is the response to a unit IBM source with
    HOMOGENEOUS outer BCs — this is correct because the capacitance correction
    is a superposition of such sources and the mean-flow contribution to the
    top BC is already in the first-pass solution.
    """
    M     = j_idx.size
    I_cap = np.eye(ny)
    C_acc = np.zeros((M, M), dtype=complex)

    for ki in range(nx):
        k  = kx[ki]
        A  = D2 - k**2 * I_cap

        src_phase = np.exp(-2j * np.pi * ki * i_idx / nx)
        rhs = np.zeros((ny, M), dtype=complex)
        rhs[j_idx, np.arange(M)] = src_phase

        if bc_type == 'neumann':
            A, rhs = _apply_phi_bcs(A, rhs, D1, ki)
        else:
            # Use homogeneous top BC for all Green's function columns
            A, rhs = _apply_psi_bcs(A, rhs, psi_bc_top=0.0)

        SOL = np.linalg.solve(A, rhs)

        eval_phase = np.exp(2j * np.pi * ki * i_idx / nx)
        C_acc     += eval_phase[:, None] * SOL[j_idx, :]

        if verbose and ki % 100 == 0:
            print(f"   wavenumber {ki}/{nx} done")

    return np.real(C_acc) / nx


def build_capacitance_matrices(D1, D2, kx, j_idx, i_idx,
                               ny, nx, verbose=True):
    """
    Build both C_phi (Neumann BCs) and C_psi (Dirichlet BCs).
    These must match their respective solvers exactly.
    """
    print("   Building C_phi (Neumann outer BCs) …")
    C_phi = _build_C(D1, D2, kx, j_idx, i_idx, ny, nx,
                     'neumann',   verbose=verbose)
    print("   Building C_psi (Dirichlet outer BCs) …")
    C_psi = _build_C(D1, D2, kx, j_idx, i_idx, ny, nx,
                     'dirichlet', verbose=verbose)
    return C_phi, C_psi

# def plot2D_equipotential(x, y, phi, title, xname, yname, savename,
#                           xfill, yfill, resolution=500, n_levels=25):
#     """
#     Plot equipotential lines of a scalar potential field phi.

#     Directly adapted from plot2D_streamlines_vorticityX:
#       - Same interpolation onto a uniform y-grid via RectBivariateSpline
#       - Same IBM solid fill (xfill / yfill polygon, black, on top)
#       - Same save / show pattern
#       - Replaces streamplot + vorticity contourf with:
#           contourf  → smooth background colour-fill of phi
#           contour   → black equipotential lines
#           clabel    → inline level labels (auto-scaled to O(1))

#     Scale management
#     ----------------
#     Automatically detects the order of magnitude of phi (e.g. 1e-6) and
#     multiplies by the inverse power of 10 before plotting.  All tick labels,
#     contour labels, and the colourbar label are expressed in those scaled
#     units (e.g. "phi [x10^-6]"), so the numbers on screen are always O(1)
#     regardless of the raw field magnitude.

#     Parameters
#     ----------
#     x          : 1-D array  (nx,)   x-coordinates (uniform or non-uniform)
#     y          : 1-D array  (ny,)   y-coordinates (possibly stretched)
#     phi        : 2-D array  (ny, nx)  scalar potential field
#     title      : str   figure title
#     xname      : str   x-axis label
#     yname      : str   y-axis label
#     savename   : str   full output path, e.g. 'fig/phi_equipotential.png'
#     xfill      : 1-D array   x-coordinates of the IBM / solid polygon
#     yfill      : 1-D array   y-coordinates of the IBM / solid polygon
#     resolution : int   number of uniform y-points for interpolation (default 500)
#     n_levels   : int   number of equipotential contour lines (default 25)

#     Example
#     -------
#     eps_hgt  = np.sum(eps, axis=0).astype(int)
#     hill_top = np.where(eps_hgt > 0, y[np.maximum(eps_hgt-1, 0)], 0.0)
#     xfill    = np.concatenate([[x[0]], x, [x[-1]]])
#     yfill    = np.concatenate([[0],    hill_top, [0]])

#     plot2D_equipotential(
#         x, y[:300], phi[:300, :],
#         title=r'Equipotential lines of $\\phi$',
#         xname='x', yname='y',
#         savename='fig/phi_equipotential.png',
#         xfill=xfill, yfill=yfill,
#         resolution=500, n_levels=25,
#     )
#     """
#     # ── 1. Interpolate phi onto a uniform y-grid ───────────────────────────
#     y_uniform  = np.linspace(y.min(), y.max(), resolution)
#     phi_interp = RectBivariateSpline(y, x, phi)
#     phi_uniform = phi_interp(y_uniform, x)
#     X, Y = np.meshgrid(x, y_uniform, indexing='xy')

#     # ── 2. Auto-scale to O(1) ──────────────────────────────────────────────
#     finite_vals = phi_uniform[np.isfinite(phi_uniform)]
#     mean_abs    = np.abs(finite_vals).mean()
#     if mean_abs > 0:
#         exp       = math.floor(math.log10(mean_abs))
#         scale     = 10 ** (-exp)          # e.g. 1e6 for phi ~ 1e-6
#     else:
#         scale, exp = 1.0, 0

#     phi_scaled = phi_uniform * scale
#     vmin_s = float(np.nanmin(phi_scaled))
#     vmax_s = float(np.nanmax(phi_scaled))

#     # ── 3. Contour levels: strictly increasing, inside data range ─────────
#     # Trim one step from each end so the outermost level never equals the
#     # data extremum (avoids a zero-width contour band at the boundary).
#     levels = np.linspace(vmin_s, vmax_s, n_levels + 2)[1:-1]

#     # ── 4. Choose colourmap ────────────────────────────────────────────────
#     # RdBu_r if phi has mixed sign (rare for potential), Blues_r if all negative,
#     # Reds if all positive.
#     if vmin_s < 0 < vmax_s:
#         cmap_fill = 'RdBu_r'
#     elif vmax_s <= 0:
#         cmap_fill = 'Blues_r'
#     else:
#         cmap_fill = 'Reds'

#     # ── 5. Figure ──────────────────────────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(10, 5))

#     # Background: smooth colour-fill (same as contourf in the original)
#     cf = ax.contourf(X, Y, phi_scaled,
#                      levels=np.linspace(vmin_s, vmax_s, 120),
#                      cmap=cmap_fill, alpha=1.0, zorder=2)

#     # Equipotential lines in black
#     cs = ax.contour(X, Y, phi_scaled,
#                     levels=levels,
#                     colors='black', linewidths=0.6, zorder=3)

#     # Inline labels every ~6 levels so the plot stays readable
#     label_levels = levels[::max(1, n_levels // 6)]
#     ax.clabel(cs, levels=label_levels,
#               fmt=lambda v: f'{v:.2f}',
#               fontsize=6, inline=True, inline_spacing=3, zorder=4)

#     # IBM / solid region on top — identical to the original function
#     ax.fill(xfill, yfill, facecolor='black', zorder=5)

#     # ── 6. Colourbar with scaled label ────────────────────────────────────
#     cbar = fig.colorbar(cf, ax=ax, pad=0.01, shrink=0.95)
#     if exp != 0:
#         cbar_label = fr'$\phi$ [$\times 10^{{{exp}}}$]'
#     else:
#         cbar_label = r'$\phi$'
#     cbar.set_label(cbar_label, fontsize=9)
#     cbar.ax.tick_params(labelsize=8)

#     # ── 7. Labels, save, show ─────────────────────────────────────────────
#     ax.set_title(title, fontsize=10)
#     ax.set_xlabel(xname, fontsize=9)
#     ax.set_ylabel(yname, fontsize=9)
#     ax.tick_params(labelsize=8)

#     plt.savefig(savename, dpi=300, format='png', transparent=False,
#                 bbox_inches='tight')
#     plt.show()

def plot2D_equipotential(x, y, phi, title, xname, yname, savename,
                         xfill, yfill, resolution=500, n_levels=25):
    """
    Plot equipotential lines of a scalar potential field phi.

    Colour scaling
    --------------
    vmin / vmax for the contourf background are set to the 2nd and 98th
    percentile of the scaled field rather than the global min/max.
    This prevents a handful of near-wall outliers from collapsing the
    visible colour range into a pale band around zero.

    Line style convention
    ---------------------
    Negative levels  ->  dashed lines
    Positive levels  ->  solid lines

    Label & Noise Filtering
    -----------------------
    Native contours are generated invisibly. The raw unaltered line segments 
    are extracted via contours.allsegs and checked against a spatial 
    bounding-box filter. Tiny near-wall noise loops are discarded entirely. 
    Surviving lines are plotted and labeled at their apex.

    Parameters
    ----------
    x          : 1-D array  (nx,)
    y          : 1-D array  (ny,)     possibly stretched
    phi        : 2-D array  (ny, nx)  scalar potential field
    title      : str
    xname      : str   x-axis label
    yname      : str   y-axis label
    savename   : str   full output path
    xfill      : 1-D array   IBM / solid polygon x-coordinates
    yfill      : 1-D array   IBM / solid polygon y-coordinates
    resolution : int   uniform y-points for interpolation (default 500)
    n_levels   : int   number of equipotential lines     (default 25)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline
    import matplotlib.patheffects as pe

    # -- 1. Interpolate phi onto a uniform y-grid ---------------------------
    y_uniform   = np.linspace(y.min(), y.max(), resolution)
    phi_interp  = RectBivariateSpline(y, x, phi)
    phi_uniform = phi_interp(y_uniform, x)
    X, Y = np.meshgrid(x, y_uniform, indexing='xy')

    # -- 2. Auto-scale to O(1) ----------------------------------------------
    finite_vals = phi_uniform[np.isfinite(phi_uniform)]
    mean_abs    = np.abs(finite_vals).mean()
    if mean_abs > 0:
        exp   = int(np.floor(np.log10(mean_abs)))
        scale = 10 ** (-exp)
    else:
        scale, exp = 1.0, 0

    phi_scaled = phi_uniform * scale

    vmin_full = float(np.nanmin(phi_scaled))
    vmax_full = float(np.nanmax(phi_scaled))

    vmin_p = float(np.nanpercentile(phi_scaled, 2))
    vmax_p = float(np.nanpercentile(phi_scaled, 98))

    # -- 3. Contour levels (full range) -------------------------------------
    levels = np.linspace(vmin_full, vmax_full, n_levels + 2)[1:-1]

    # -- 4. Choose colourmap ------------------------------------------------
    if vmin_p < 0 < vmax_p:
        cmap_fill = 'RdBu_r'
    elif vmax_p <= 0:
        cmap_fill = 'Blues_r'
    else:
        cmap_fill = 'Reds'

    # -- 5. Figure ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    cf = ax.contourf(X, Y, phi_scaled,
                     levels=np.linspace(vmin_p, vmax_p, 120),
                     cmap=cmap_fill, alpha=1.0,
                     vmin=vmin_p, vmax=vmax_p,
                     extend='both',
                     zorder=2)

    linestyles = ['dashed' if lv < 0 else 'solid' for lv in levels]

    # Generate contours INVISIBLY (alpha=0.0) to extract paths safely
    contours = ax.contour(X, Y, phi_scaled,
                          levels=levels,
                          alpha=0.0)

    # -- 6. Filter and Plot Apex labels using allsegs -----------------------
    label_effects = [pe.withStroke(linewidth=2.5, foreground='white')]

    x_span = x.max() - x.min()
    y_span = y.max() - y.min()

    # contours.allsegs returns a list of segments for each level.
    # This avoids polygon closing artifacts on open curves.
    for level_idx, (lev, ls) in enumerate(zip(levels, linestyles)):
        segments = contours.allsegs[level_idx]
        
        for seg in segments:
            if len(seg) < 3:
                continue
            
            # Calculate the spatial bounding box of the contour line
            width = np.ptp(seg[:, 0])
            height = np.ptp(seg[:, 1])

            # GEOMETRIC FILTER: 
            # If the loop is smaller than 3% of the domain in BOTH directions, 
            # consider it near-wall noise and skip plotting it entirely.
            if width < 0.03 * x_span and height < 0.03 * y_span:
                continue

            # Plot the filtered line manually
            ax.plot(seg[:, 0], seg[:, 1], color='black', linewidth=0.6, linestyle=ls, zorder=3)

            # Find apex and plot label
            apex_idx = np.argmax(seg[:, 1])
            x_apex = seg[apex_idx, 0]
            y_apex = seg[apex_idx, 1]

            ax.text(x_apex, y_apex,
                    f'{lev:.2f}',
                    fontsize=6.5,
                    ha='center', va='bottom',
                    color='black',
                    path_effects=label_effects,
                    zorder=5)

    # IBM / solid region on top
    ax.fill(xfill, yfill, facecolor='black', zorder=6)

    # -- 7. Colourbar — ticks reflect the clipped percentile range ----------
    cbar = fig.colorbar(cf, ax=ax, pad=0.01, shrink=0.95)
    if exp != 0:
        cbar_label = fr'$\phi$ [$\times 10^{{{exp}}}$]'
    else:
        cbar_label = r'$\phi$'
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # -- 8. Axes, save, show ------------------------------------------------
    # Added pad=20 to push the title up and away from the top-boundary labels
    ax.set_title(title, fontsize=10, pad=20)
    ax.set_xlabel(xname, fontsize=9)
    ax.set_ylabel(yname, fontsize=9)
    ax.tick_params(labelsize=8)

    plt.savefig(savename, dpi=300, format='png', transparent=False,
                bbox_inches='tight')
    plt.show()
    
def plot2D_streamline(x, y, phi, title, xname, yname, savename,
                      xfill, yfill, resolution=500, n_levels=25):
    """
    Plot streamlines (iso-contours of ψ) of a stream function field.

    Colour scaling
    --------------
    vmin / vmax for the contourf background are set to the 2nd and 98th
    percentile of the scaled field to prevent near-wall outliers from
    compressing the colour range into a pale band.

    Line style
    ----------
    All streamlines are drawn solid.  The dashed/solid sign convention used
    in plot2D_equipotential is dropped here because:
      (a) all values of ψ share the same sign, so the distinction carries
          no information,
      (b) streamlines are conventionally always solid regardless of sign.

    Colourmap & Centering
    ---------------------
    If the field spans zero, a 'RdBu_r' diverging colourmap is used.
    A TwoSlopeNorm is applied to force 0 to map precisely to white, ensuring
    negatives are blue and positives are red even when data is asymmetric.

    Label & Noise Filtering
    -----------------------
    Native contours are generated invisibly. The raw unaltered line segments 
    are extracted via contours.allsegs and checked against a spatial 
    bounding-box filter. Tiny near-wall noise loops are discarded entirely. 
    Surviving lines are plotted and labeled at their apex.

    Parameters
    ----------
    x          : 1-D array  (nx,)
    y          : 1-D array  (ny,)     possibly stretched
    phi        : 2-D array  (ny, nx)  stream function field ψ
    title      : str
    xname      : str   x-axis label
    yname      : str   y-axis label
    savename   : str   full output path
    xfill      : 1-D array   IBM / solid polygon x-coordinates
    yfill      : 1-D array   IBM / solid polygon y-coordinates
    resolution : int   uniform y-points for interpolation (default 500)
    n_levels   : int   number of streamlines                (default 25)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.interpolate import RectBivariateSpline
    import matplotlib.patheffects as pe

    # -- 1. Interpolate onto a uniform y-grid --------------------------------
    y_uniform   = np.linspace(y.min(), y.max(), resolution)
    phi_interp  = RectBivariateSpline(y, x, phi)
    phi_uniform = phi_interp(y_uniform, x)
    X, Y = np.meshgrid(x, y_uniform, indexing='xy')

    # -- 2. Auto-scale to O(1) -----------------------------------------------
    finite_vals = phi_uniform[np.isfinite(phi_uniform)]
    mean_abs    = np.abs(finite_vals).mean()
    if mean_abs > 0:
        exp   = int(np.floor(np.log10(mean_abs)))
        scale = 10 ** (-exp)
    else:
        scale, exp = 1.0, 0

    phi_scaled = phi_uniform * scale

    # Full data range — for contour lines, no iso-line is lost
    vmin_full = float(np.nanmin(phi_scaled))
    vmax_full = float(np.nanmax(phi_scaled))

    # Percentile range — for contourf background only
    vmin_p = float(np.nanpercentile(phi_scaled, 2))
    vmax_p = float(np.nanpercentile(phi_scaled, 98))

    # -- 3. Contour levels (full range) --------------------------------------
    levels = np.linspace(vmin_full, vmax_full, n_levels + 2)[1:-1]

    # -- 4. Colourmap & TwoSlopeNorm Balance ---------------------------------
    if vmin_p < 0 < vmax_p:
        cmap_fill = 'RdBu_r'
        # Forces 0 to be pure white, mapping blue/red smoothly to asymmetric bounds
        norm_fill = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin_p, vmax=vmax_p)
    elif vmax_p <= 0:
        cmap_fill = 'Blues_r'
        norm_fill = mcolors.Normalize(vmin=vmin_p, vmax=vmax_p)
    else:
        cmap_fill = 'Reds'
        norm_fill = mcolors.Normalize(vmin=vmin_p, vmax=vmax_p)

    # -- 5. Figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Background fill using the customized norm tracking
    cf = ax.contourf(X, Y, phi_scaled,
                     levels=np.linspace(vmin_p, vmax_p, 120),
                     cmap=cmap_fill, alpha=1.0,
                     norm=norm_fill,
                     extend='both',
                     zorder=2)

    # Generate contours INVISIBLY (alpha=0.0) to extract paths safely
    contours = ax.contour(X, Y, phi_scaled,
                          levels=levels,
                          alpha=0.0)

    # -- 6. Filter and Plot Apex labels using allsegs -----------------------
    label_effects = [pe.withStroke(linewidth=2.5, foreground='white')]

    x_span = x.max() - x.min()
    y_span = y.max() - y.min()

    # contours.allsegs returns a list of segments for each level.
    for level_idx, lev in enumerate(levels):
        segments = contours.allsegs[level_idx]
        
        for seg in segments:
            if len(seg) < 3:
                continue
            
            # Calculate the spatial bounding box of the contour line
            width = np.ptp(seg[:, 0])
            height = np.ptp(seg[:, 1])

            # GEOMETRIC FILTER: 
            # If the loop is smaller than 3% of the domain in BOTH directions, 
            # consider it near-wall noise and skip plotting it entirely.
            if width < 0.03 * x_span and height < 0.03 * y_span:
                continue

            # Plot the filtered line manually (Streamlines are always solid)
            ax.plot(seg[:, 0], seg[:, 1], color='black', linewidth=0.6, linestyle='solid', zorder=3)

            # Find apex and plot label
            apex_idx = np.argmax(seg[:, 1])
            x_apex = seg[apex_idx, 0]
            y_apex = seg[apex_idx, 1]

            ax.text(x_apex, y_apex,
                    f'{lev:.2f}',
                    fontsize=6.5,
                    ha='center', va='bottom',
                    color='black',
                    path_effects=label_effects,
                    zorder=5)

    # IBM / solid region on top
    ax.fill(xfill, yfill, facecolor='black', zorder=6)

    # -- 7. Colourbar --------------------------------------------------------
    cbar = fig.colorbar(cf, ax=ax, pad=0.01, shrink=0.95)
    if exp != 0:
        cbar_label = fr'$\psi$ [$\times 10^{{{exp}}}$]'
    else:
        cbar_label = r'$\psi$'
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # -- 8. Axes, save, show -------------------------------------------------
    ax.set_title(title, fontsize=10, pad=20)
    ax.set_xlabel(xname, fontsize=9)
    ax.set_ylabel(yname, fontsize=9)
    ax.tick_params(labelsize=8)

    plt.savefig(savename, dpi=300, format='png', transparent=False,
                bbox_inches='tight')
    plt.show()
    
# ═══════════════════════════════════════════════════════════════════════════
#  Main driver
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    from config import *
    from PlotField import *
    from functions import *
    from functions import read_grid, epsfield, interpolate_component

    cwd = str(os.path.dirname(os.path.abspath(__file__)) + "/")
    os.makedirs(cwd + "/fig", exist_ok=True)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 — Grid
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 1: Load grid ──")
    x, y, z = read_grid(cwd)
    nx, ny, nz = x.size, y.size, z.size
    print(f"   nx={nx}  ny={ny}  nz={nz}")
    dx_val = x[1] - x[0]

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 — IBM epsilon field
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 2: Load / build eps field ──")
    eps_path = os.path.join(cwd, "eps_save.npy")
    if os.path.exists(eps_path):
        eps = np.load(eps_path);  print("   eps loaded from cache.")
    else:
        eps = epsfield();  np.save(eps_path, eps)

    eps_hgt  = np.sum(eps, axis=0).astype(int)
    hill_hgt = np.max(eps_hgt) - 1
    eps_vol  = epsVolume(eps, ny, nx, hill_hgt)
    eps_s    = np.mean(eps_vol, axis=1)
    eps_f    = 1 - eps_s

    eps_lf  = int(nx / 4)
    flk_hgt = eps_hgt[eps_lf]
    flk_wdt = np.where(eps_hgt == flk_hgt)[0]
    lf_ind  = flk_wdt[:len(flk_wdt) // 2]
    rf_ind  = flk_wdt[len(flk_wdt) // 2:]

    dx_oro = 2 * np.pi / x[-1]
    y_oro  = np.round((hill_hgt / 2) * (1 + np.cos(dx_oro * x)))
    y_oro  = y[y_oro.astype(int)]
    x_oro  = np.concatenate([[0], x,     [x[-1]]])
    y_oro  = np.concatenate([[0], y_oro, [0]])
    x_oro_in = x_oro/l_in
    y_oro_in = y_oro/l_in
    x_in = x/l_in
    y_in = y/l_in

    solid = (eps == 1)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 — Velocity fields + IBM interpolation
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 3: Load and interpolate velocity fields ──")
    AvgPhU = np.load(os.path.join(cwd, "AvgPhU.npy"))
    AvgPhV = np.load(os.path.join(cwd, "AvgPhV.npy"))
    print(f"   AvgPhU: {AvgPhU.shape}   AvgPhV: {AvgPhV.shape}")
    print("   Interpolating U …")
    AvgPhU_i, AvgPhU_j = interpolate_component(
        x, y, nx, ny, eps, AvgPhU, ghost_depth=5, n_anchor=4, smooth_width=5)
    print("   Interpolating V …")
    AvgPhV_i, AvgPhV_j = interpolate_component(
        x, y, nx, ny, eps, AvgPhV, ghost_depth=5, n_anchor=4, smooth_width=5)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3b — Compute psi_bc_top BEFORE zeroing solid
    # ══════════════════════════════════════════════════════════════════
    #
    # The top Dirichlet BC for ψ is the volume flux per unit x-width:
    #     Q(x) = ∫₀^{Ly} U(y,x) dy
    #
    # For incompressible flow in a periodic domain Q(x) = const, so we
    # use the spatial mean.  We integrate the INTERPOLATED field
    # (AvgPhU_j, used for y-derivatives) because that is the field
    # whose vorticity feeds the ψ Poisson equation.
    #
    # This MUST be done BEFORE zeroing the solid in Step 8.
    # ──────────────────────────────────────────────────────────────────
    psi_bc_top = trapezoid(AvgPhU_j, y, axis=0).mean()
    print(f"\n   psi_bc_top (Q_mean = ∫U dy) = {psi_bc_top:.6f}")
    print(f"   (This replaces the wrong ψ(top)=0 from v13)")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 — Build D1, D2, kx once
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 4: Build compact derivative matrices ──")
    cd = CompactDerivatives2D(x, y, periodic_x=True)
    print(f"  Building D1, D2  (Ny={ny}) …")
    D1, D2 = build_derivative_matrices(cd)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx_val)

    # ══════════════════════════════════════════════════════════════════
    # STEP 5 — Interface identification
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 5: Identify solid-fluid interface ──")
    eps_above = np.roll(eps, -1, axis=0);  eps_above[-1, :] = 0
    eps_below = np.roll(eps,  1, axis=0);  eps_below[ 0, :] = 0
    eps_right = np.roll(eps, -1, axis=1)
    eps_left  = np.roll(eps,  1, axis=1)
    top_surface = solid & (eps_above == 0)
    bottom_surf = solid & (eps_below == 0)
    right_face  = solid & (eps_right == 0)
    left_face   = solid & (eps_left  == 0)
    ibm_surface = top_surface | bottom_surf | right_face | left_face
    print(f"   IBM surface: {np.sum(ibm_surface)}  "
          f"(top={np.sum(top_surface)}  right={np.sum(right_face)}  left={np.sum(left_face)})")

    # ══════════════════════════════════════════════════════════════════
    # STEP 6 — Interface points
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 6: Extract interface points ──")
    j_idx, i_idx, x_pts, y_pts, M = extract_interface_points(
        top_surface, x, y)
    print(f"   M={M}  y=[{y_pts.min():.6f}, {y_pts.max():.6f}]  "
          f"x=[{x_pts.min():.4f}, {x_pts.max():.4f}]")

    # ══════════════════════════════════════════════════════════════════
    # STEP 7 — Velocity gradients
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 7: Velocity gradients ──")
    du_dx = cd.ddx(AvgPhU_i);  du_dy = cd.ddy(AvgPhU_j)
    dv_dx = cd.ddx(AvgPhV_i);  dv_dy = cd.ddy(AvgPhV_j)

    # ══════════════════════════════════════════════════════════════════
    # STEP 8 — Divergence and vorticity; zero solid
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 8: Divergence and vorticity ──")
    divergence = du_dx + dv_dy
    vorticity  = dv_dx - du_dy
    divergence[solid] = 0.0
    vorticity[solid]  = 0.0
    print(f"   max|div|  = {np.max(np.abs(divergence)):.4e}  "
          f"(expected ~0 for incompressible DNS → φ ≈ 0 is correct)")
    print(f"   max|vort| = {np.max(np.abs(vorticity)):.4e}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 9 — Pass 1: first-pass solve
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 9: First-pass Poisson solve ──")
    phi, psi = compute_phi_psi(x, y, divergence, vorticity,
                               D1=D1, D2=D2,
                               psi_bc_top=psi_bc_top,
                               verbose=True)
    print(f"   max|phi|={np.max(np.abs(phi)):.4e}  "
          f"(~0 is correct: incompressible DNS)")
    print(f"   max|psi|={np.max(np.abs(psi)):.4e}  "
          f"(expected O({psi_bc_top:.3f}))")

    # ══════════════════════════════════════════════════════════════════
    # STEP 10 — Residuals at interface (ψ=0, φ=0 at solid surface)
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 10: Residuals ──")
    r_phi = phi[j_idx, i_idx]
    r_psi = psi[j_idx, i_idx]
    print(f"   max|r_phi|={np.max(np.abs(r_phi)):.4e}  mean={np.mean(np.abs(r_phi)):.4e}")
    print(f"   max|r_psi|={np.max(np.abs(r_psi)):.4e}  mean={np.mean(np.abs(r_psi)):.4e}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 11 — Build capacitance matrices
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 11: Build capacitance matrices ──")
    C_phi, C_psi = build_capacitance_matrices(
        D1, D2, kx, j_idx, i_idx, ny, nx, verbose=True)
    print(f"   C_phi: cond={np.linalg.cond(C_phi):.4e}  "
          f"sym={np.max(np.abs(C_phi-C_phi.T)):.4e}")
    print(f"   C_psi: cond={np.linalg.cond(C_psi):.4e}  "
          f"sym={np.max(np.abs(C_psi-C_psi.T)):.4e}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 12 — Solve C·λ = −r
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 12: Solve for λ ──")
    lam_phi, _, rank_phi, _ = np.linalg.lstsq(C_phi, -r_phi, rcond=1e-10)
    lam_psi, _, rank_psi, _ = np.linalg.lstsq(C_psi, -r_psi, rcond=1e-10)
    print(f"   rank phi={rank_phi}/{M}  max|λ_phi|={np.max(np.abs(lam_phi)):.4e}")
    print(f"   rank psi={rank_psi}/{M}  max|λ_psi|={np.max(np.abs(lam_psi)):.4e}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 13 — Spread corrections onto grid
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 13: Spread corrections ──")
    F_IBM = np.zeros((ny, nx))
    G_IBM = np.zeros((ny, nx))
    np.add.at(F_IBM, (j_idx, i_idx), lam_phi)
    np.add.at(G_IBM, (j_idx, i_idx), lam_psi)
    print(f"   max|F_IBM|={np.max(np.abs(F_IBM)):.4e}  max|G_IBM|={np.max(np.abs(G_IBM)):.4e}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 14 — Pass 2: corrected solve
    #
    # φ RHS: div + F_IBM           (F_IBM added directly)
    # ψ RHS: −(ω − G_IBM) = −ω + G_IBM
    #        G_IBM SUBTRACTED from vorticity because ψ RHS is −ω
    #
    # NOTE: psi_bc_top is passed again — the top BC does not change
    # between passes because the correction sources λ only alter the
    # interior; the mean flow flux Q is unchanged.
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 14: Final corrected Poisson solve ──")
    phi_final, psi_final = compute_phi_psi(
        x, y,
        divergence + F_IBM,
        vorticity  - G_IBM,
        D1=D1, D2=D2,
        psi_bc_top=psi_bc_top,   # same top BC as pass 1
        verbose=True)

    r_phi_f = phi_final[j_idx, i_idx]
    r_psi_f = psi_final[j_idx, i_idx]
    print(f"\n   φ residual: max={np.max(np.abs(r_phi_f)):.4e}  "
          f"reduction={np.max(np.abs(r_phi_f))/(np.max(np.abs(r_phi))+1e-30):.4e}")
    print(f"   ψ residual: max={np.max(np.abs(r_psi_f)):.4e}  "
          f"reduction={np.max(np.abs(r_psi_f))/(np.max(np.abs(r_psi))+1e-30):.4e}")

    print(f"\n   ψ range (fluid):     [{psi_final[~solid].min():.4e}, {psi_final[~solid].max():.4e}]")
    print(f"   ψ at top boundary:   [{psi_final[-1,:].min():.4e}, {psi_final[-1,:].max():.4e}]  "
          f"(expected ≈ {psi_bc_top:.4f})")
    print(f"   ψ at IBM interface:  [{r_psi_f.min():.4e}, {r_psi_f.max():.4e}]  (target = 0)")

    # ══════════════════════════════════════════════════════════════════
    # STEP 15 — Recover velocities from phi and psi
    # ══════════════════════════════════════════════════════════════════
    #
    # Helmholtz decomposition (2-D):
    #     u =  ∂φ/∂x + ∂ψ/∂y      v =  ∂φ/∂y − ∂ψ/∂x
    #
    # φ carries the irrotational/compressible part  (≈ 0 for DNS)
    # ψ carries the solenoidal/rotational  part     (dominates)
    # ──────────────────────────────────────────────────────────────────
    print("\n── Step 15: Recover velocities ──")
    dphi_dx = cd.ddx(phi_final);   dphi_dy = cd.ddy(phi_final)
    dpsi_dx = cd.ddx(psi_final);   dpsi_dy = cd.ddy(psi_final)

    U_rec = dphi_dx + dpsi_dy     # recovered u
    V_rec = dphi_dy - dpsi_dx     # recovered v

    # zero solid region (IBM cells carry no physical velocity)
    U_rec[solid] = 0.0
    V_rec[solid] = 0.0

    # ── differences vs input ──
    U_interp = AvgPhU_i   # use the i-variant (used for ddx)
    V_interp = AvgPhV_i
    dU = U_interp - U_rec
    dV = V_interp - V_rec
    # zero solid in diff too (input was already 0 there)
    dU[solid] = 0.0
    dV[solid] = 0.0

    dU_fl = dU[~solid];  dV_fl = dV[~solid]
    print(f"   ΔU fluid: max={np.max(np.abs(dU_fl)):.4e}  "
          f"rms={np.sqrt(np.mean(dU_fl**2)):.4e}  "
          f"rel={np.sqrt(np.mean(dU_fl**2))/np.sqrt(np.mean(U_interp[~solid]**2))*100:.3f}%")
    print(f"   ΔV fluid: max={np.max(np.abs(dV_fl)):.4e}  "
          f"rms={np.sqrt(np.mean(dV_fl**2)):.4e}  "
          f"rel={np.sqrt(np.mean(dV_fl**2))/np.sqrt(np.mean(V_interp[~solid]**2))*100:.3f}%")

    # ══════════════════════════════════════════════════════════════════
    # STEP 16 — Velocity recovery plots
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 16: Velocity recovery plots ──")
    limity = 450

    plot2D_div(x, y[:limity], phi_final[:limity, :],
               '', 'φ — final (φ=0 at IBM interface)',
               r'$x$', r'$z$', cwd + '/fig/phi_final.png', x_oro, y_oro, 1000)

    plot2D_div(x, y[:limity], psi_final[:limity, :],
               '', 'ψ — final (ψ=0 at IBM interface, ψ=Q at top)',
               r'$x$', r'$z$', cwd + '/fig/psi_final.png', x_oro, y_oro, 1000)

    plot2D_div(x, y[:limity], U_rec[:limity, :],
               '', r'U_rec = ∂φ/∂x + ∂ψ/∂y',
               r'$x$', r'$z$', cwd + '/fig/U_rec.png', x_oro, y_oro, 1000)

    plot2D_div(x, y[:limity], V_rec[:limity, :],
               '', r'V_rec = ∂φ/∂y − ∂ψ/∂x',
               r'$x$', r'$z$', cwd + '/fig/V_rec.png', x_oro, y_oro, 1000)

    plot2D_div(x, y[:limity], dU[:limity, :],
               '', r'ΔU = U_input − U_rec',
               r'$x$', r'$z$', cwd + '/fig/dU.png', x_oro, y_oro, 1000)

    plot2D_div(x, y[:limity], dV[:limity, :],
               '', r'ΔV = V_input − V_rec',
               r'$x$', r'$z$', cwd + '/fig/dV.png', x_oro, y_oro, 1000)

    # ══════════════════════════════════════════════════════════════════
    # STEP 17 — ABL diagnostic plots
    #
    # Five figures derived from phi, psi, and the velocity fields:
    #
    #  Fig A  — Equipotential lines (phi)
    #  Fig B  — Streamlines (psi=const) overlaid on vorticity
    #  Fig C  — BL integral parameters along x
    #           (delta*, theta, shape factor H, wall shear, speed-up)
    #  Fig D  — Orographic effects: Cp, strain rate, flow angle
    #  Fig E  — x-averaged profiles + log-law check
    #  Fig F  — Helmholtz kinetic energy partition (phi vs psi)
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 17: ABL diagnostic plots ──")

    # ── shared geometry helpers ────────────────────────────────────────
    eps_hgt  = np.sum(eps, axis=0).astype(int)       # solid-cell count per column
    hill_top = np.where(eps_hgt > 0,
                        y[np.maximum(eps_hgt - 1, 0)], 0.0)  # hill surface y(x)
    hill_h   = y[eps_hgt.max() - 1]                  # peak hill height
    j_h      = eps_hgt.max()                          # row index just above crest

    # free-stream reference: mean of top 50 rows
    U_ref = np.mean(AvgPhU_i[-50:, :])

    # plot region: 200 rows captures full BL with room above
    j_top_plot = 200
    y_top_plot = y[j_top_plot]

    # compact Padé derivative operators (reuse cd already built in Step 4)
    def _ddx(f): return cd.ddx(f)
    def _ddy(f): return cd.ddy(f)

    def _fill_hill(ax, zorder=10):
        """Dark-fill solid region and draw white + black hill outline."""
        ax.fill_between(x, 0, hill_top, color='#2d2d2d', zorder=zorder)
        ax.plot(x, hill_top, 'w-', lw=0.9, zorder=zorder + 1)

    def _draw_solid_mesh(ax, j_top_idx, zorder=9):
        """Overlay dark pcolormesh on solid cells so they hide any field artefacts."""
        solid_show = np.where(solid[:j_top_idx, :], 1.0, np.nan)
        ax.pcolormesh(x, y[:j_top_idx], solid_show,
                      cmap=mcolors.ListedColormap(['#2d2d2d']),
                      shading='auto', zorder=zorder, rasterized=True)

    def _ax_fmt(ax, title, xlabel='x', ylabel='y', xlim=None, ylim=None):
        ax.set_xlim(*(xlim or (x[0], x[-1])))
        ax.set_ylim(*(ylim or (0, y_top_plot)))
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    # ── derived fields ─────────────────────────────────────────────────
    vorticity_plot = _ddx(AvgPhV_i) - _ddy(AvgPhU_j)
    vorticity_plot[solid] = np.nan

    # strain-rate magnitude |S| = sqrt(2 S_ij S_ij)
    dudx = _ddx(AvgPhU_i); dudy = _ddy(AvgPhU_j)
    dvdx = _ddx(AvgPhV_i); dvdy = _ddy(AvgPhV_j)
    S_mag = np.sqrt(2.0 * (dudx**2 + dvdy**2 + 2.0*(0.5*(dudy + dvdx))**2))
    S_mag[solid] = np.nan

    # pressure coefficient (Bernoulli, inviscid)
    q2  = AvgPhU_i**2 + AvgPhV_i**2
    Cp  = 1.0 - q2 / max(U_ref**2, 1e-10)
    Cp[solid] = np.nan

    # flow deflection angle (degrees)
    alpha = np.degrees(np.arctan2(AvgPhV_i,
                                   np.where(AvgPhU_i > 1e-6, AvgPhU_i, 1e-6)))
    alpha[solid] = np.nan

    # orographic speed-up ΔS at crest row
    U_flat  = np.mean(AvgPhU_i[j_h, eps_hgt == 0])
    delta_S = (AvgPhU_i[j_h, :] - U_flat) / max(U_flat, 1e-10)

    # BL integral parameters: integrate from local wall height up to j_top=250
    j_bl_top = 250
    delta_star = np.zeros(nx)
    theta_arr  = np.zeros(nx)
    tau_wall   = np.zeros(nx)
    for i in range(nx):
        j0 = eps_hgt[i]
        j1 = j_bl_top
        if j0 >= j1:
            continue
        y_l = y[j0:j1]
        u_l = AvgPhU_i[j0:j1, i]
        delta_star[i] = trapezoid(1.0 - u_l / U_ref, y_l)
        u_n = u_l / U_ref
        theta_arr[i]  = trapezoid(u_n * (1.0 - u_n), y_l)
        if j0 + 1 < ny:
            tau_wall[i] = (AvgPhU_i[j0 + 1, i] - AvgPhU_i[j0, i]) / (y[j0 + 1] - y[j0])

    H_arr = np.where(theta_arr > 5e-5, delta_star / theta_arr, np.nan)
    # clip outliers for clean plotting
    H_plot = np.where((H_arr > 1.0) & (H_arr < 5.0), H_arr, np.nan)

    # Helmholtz KE partition
    dphi_dx_f = _ddx(phi_final);  dphi_dy_f = _ddy(phi_final)
    dpsi_dx_f = _ddx(psi_final);  dpsi_dy_f = _ddy(psi_final)
    KE_phi = 0.5 * (dphi_dx_f**2 + dphi_dy_f**2)
    KE_psi = 0.5 * (dpsi_dy_f**2 + dpsi_dx_f**2)
    KE_tot = 0.5 * (AvgPhU_i**2 + AvgPhV_i**2)
    KE_phi[solid] = np.nan
    KE_psi[solid] = np.nan
    KE_tot[solid] = np.nan

    # ── Figure A: equipotential lines (phi) ───────────────────────────
    print("   Fig A: equipotential lines …")
    j_eq   = 300
    phi_s  = phi_final[:j_eq, :] * 1e6              # scale → units of 1e-6
    phi_s[solid[:j_eq, :]] = np.nan
    p_min  = float(np.nanmin(phi_s))
    p_max  = float(np.nanmax(phi_s))
    lev_eq = np.linspace(p_min, p_max, 30)

    fig_A, ax = plt.subplots(figsize=(13, 5), dpi=200)
    im = ax.contourf(x, y[:j_eq], phi_s,
                     levels=np.linspace(p_min, p_max, 120),
                     cmap='Blues_r', extend='neither', zorder=1)
    cs = ax.contour(x, y[:j_eq], phi_s,
                    levels=lev_eq, colors='black', linewidths=0.7, zorder=4)
    ax.clabel(cs, levels=lev_eq[::4],
              fmt=lambda v: f'{v:.2f}', fontsize=5.5,
              inline=True, inline_spacing=2, zorder=5)
    _draw_solid_mesh(ax, j_eq, zorder=8)
    _fill_hill(ax, zorder=9)
    cb = fig_A.colorbar(im, ax=ax, pad=0.01, shrink=0.92,
                        label=r'$\phi\ [\times 10^{-6}]$')
    cb.ax.tick_params(labelsize=7.5)
    _ax_fmt(ax,
            r'Equipotential lines of $\phi$   '
            r'($\nabla^2\!\phi=\nabla\!\cdot\!\mathbf{u}\approx0$, '
            r'Neumann walls, $\phi=0$ at IBM surface)',
            ylim=(0, y[j_eq]))
    ax.text(0.99, 0.97,
            f'Values ×10⁻⁶: [{p_min:.2f}, {p_max:.2f}]\n30 equipotential lines',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(fc='white', alpha=0.85, ec='lightgray', lw=0.5))
    fig_A.tight_layout()
    fig_A.savefig(cwd + '/fig/abl_A_equipotential.png',
                  dpi=200, bbox_inches='tight')
    print("      saved: abl_A_equipotential.png")

    # ── Figure B: streamlines (psi=const) on vorticity background ─────
    print("   Fig B: streamlines + vorticity …")
    om_p = vorticity_plot[:j_top_plot, :].copy()
    psi_p = psi_final[:j_top_plot, :].copy()
    psi_p[solid[:j_top_plot, :]] = np.nan
    vm_om = np.nanpercentile(np.abs(om_p), 98)
    Q_total = float(np.mean(psi_final[-1, :]))

    # Build strictly-increasing ψ levels within the actual data range.
    # Two bands: dense near the wall (streamlines curve most over the hill)
    # and coarser in the outer flow.  np.unique + clipping guarantees that
    # matplotlib never receives a non-increasing level sequence.
    psi_pos  = psi_p[psi_p > 1e-6]                           # exclude exact-zero wall
    psi_dmin = float(np.nanmin(psi_pos)) if psi_pos.size else 0.0001
    psi_dmax = float(np.nanmax(psi_p))
    near_lo  = max(psi_dmin, 0.0001)
    near_hi  = min(0.012, psi_dmax * 0.95)
    if near_lo >= near_hi:
        near_hi = psi_dmax * 0.5
    lev_near = np.linspace(near_lo, near_hi, 14)
    far_lo   = lev_near[-1] * 1.05
    far_hi   = psi_dmax * 0.97
    lev_far  = np.linspace(far_lo, far_hi, 8) if far_lo < far_hi else np.array([])
    lev_psi  = np.unique(np.concatenate([lev_near, lev_far]))   # sorted, no dups
    lev_psi  = lev_psi[(lev_psi > psi_dmin) & (lev_psi < psi_dmax)]
    lev_label = lev_psi[::max(1, len(lev_psi) // 5)]            # ~5 inline labels

    fig_B, ax = plt.subplots(figsize=(13, 4.8), dpi=200)
    im = ax.pcolormesh(x, y[:j_top_plot], om_p,
                       cmap='RdBu_r', vmin=-vm_om, vmax=vm_om,
                       shading='gouraud', rasterized=True, zorder=1)
    cs = ax.contour(x, y[:j_top_plot], psi_p,
                    levels=lev_psi, colors='k', linewidths=0.7, zorder=4)
    if lev_label.size:
        ax.clabel(cs, levels=lev_label,
                  fmt=lambda v: f'{v:.4f}', fontsize=5.5,
                  inline=True, zorder=5)
    _fill_hill(ax)
    cb = fig_B.colorbar(im, ax=ax, pad=0.01, shrink=0.92,
                        label=r'$\omega = \partial V/\partial x - \partial U/\partial y$')
    cb.ax.tick_params(labelsize=7.5)
    _ax_fmt(ax,
            r'Streamlines ($\psi$ = const) over vorticity  $\omega$')
    ax.text(0.99, 0.97,
            f'Black lines: streamlines (ψ contours)\nColour: vorticity ω\n$Q={Q_total:.4f}$',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(fc='white', alpha=0.85, ec='lightgray', lw=0.5))
    fig_B.tight_layout()
    fig_B.savefig(cwd + '/fig/abl_B_streamlines.png',
                  dpi=200, bbox_inches='tight')
    print("      saved: abl_B_streamlines.png")

    # ── Figure C: BL integral parameters along x ──────────────────────
    print("   Fig C: BL integral parameters …")
    fig_C, axes_C = plt.subplots(2, 2, figsize=(13, 7), dpi=200,
                                  constrained_layout=True)
    fig_C.suptitle('Boundary-layer integral parameters (function of x)',
                   fontsize=11)

    # displacement + momentum thickness
    ax = axes_C[0, 0]
    ax.plot(x, delta_star * 1e3, 'b-',  lw=1.6, label=r'$\delta^*$ (×10⁻³)')
    ax.plot(x, theta_arr  * 1e3, 'r--', lw=1.6, label=r'$\theta$ (×10⁻³)')
    ax2 = ax.twinx()
    ax2.fill_between(x, 0, hill_top * 1e3, color='#999', alpha=0.25)
    ax2.set_ylabel('Hill height (×10⁻³)', fontsize=7.5, color='#888')
    ax2.tick_params(labelsize=7, colors='#888')
    ax.set_xlabel('x', fontsize=8.5); ax.set_ylabel('(×10⁻³)', fontsize=8.5)
    ax.set_title(r'Displacement $\delta^*$ and momentum thickness $\theta$',
                 fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.tick_params(labelsize=7.5); ax.set_xlim(x[0], x[-1])

    # shape factor H
    ax = axes_C[0, 1]
    ax.plot(x, H_plot, color='purple', lw=1.8)
    ax.axhline(1.4,  color='green', lw=0.9, ls='--',
               label='H = 1.4  (turbulent flat plate)')
    ax.axhline(2.59, color='red',   lw=0.9, ls='--',
               label='H = 2.59 (Blasius laminar)')
    ax2 = ax.twinx()
    ax2.fill_between(x, 0, hill_top, color='#999', alpha=0.25)
    ax2.set_ylabel('Hill height', fontsize=7.5, color='#888')
    ax2.tick_params(labelsize=7, colors='#888')
    ax.set_xlabel('x', fontsize=8.5)
    ax.set_ylabel('H = δ*/θ', fontsize=8.5)
    ax.set_title('Shape factor  H = δ*/θ', fontsize=9)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.tick_params(labelsize=7.5); ax.set_xlim(x[0], x[-1])
    finite_H = H_plot[np.isfinite(H_plot)]
    if finite_H.size:
        ax.set_ylim(finite_H.min() - 0.05, finite_H.max() + 0.05)

    # wall shear proxy
    ax = axes_C[1, 0]
    ax.plot(x, tau_wall, color='darkorange', lw=1.6)
    ax2 = ax.twinx()
    ax2.fill_between(x, 0, hill_top, color='#999', alpha=0.25)
    ax2.set_ylabel('Hill height', fontsize=7.5, color='#888')
    ax2.tick_params(labelsize=7, colors='#888')
    ax.set_xlabel('x', fontsize=8.5)
    ax.set_ylabel(r'$\partial U/\partial y|_\mathrm{wall}$', fontsize=8.5)
    ax.set_title(r'Wall shear proxy  '
                 r'$\partial U/\partial y|_\mathrm{wall}$  (∝ skin-friction)',
                 fontsize=9)
    ax.tick_params(labelsize=7.5); ax.set_xlim(x[0], x[-1])

    # orographic speed-up
    ax = axes_C[1, 1]
    ax.plot(x, delta_S, color='teal', lw=1.8,
            label=r'$\Delta S=(U-U_\mathrm{flat})/U_\mathrm{flat}$')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax2 = ax.twinx()
    ax2.fill_between(x, 0, hill_top, color='#999', alpha=0.25)
    ax2.set_ylabel('Hill height', fontsize=7.5, color='#888')
    ax2.tick_params(labelsize=7, colors='#888')
    ax.set_xlabel('x', fontsize=8.5)
    ax.set_ylabel('Speed-up factor ΔS', fontsize=8.5)
    ax.set_title(r'Orographic speed-up  $\Delta S$  at $z = h_\mathrm{hill}$',
                 fontsize=9)
    ax.legend(fontsize=8); ax.tick_params(labelsize=7.5)
    ax.set_xlim(x[0], x[-1])

    fig_C.savefig(cwd + '/fig/abl_C_bl_params.png',
                  dpi=200, bbox_inches='tight')
    print("      saved: abl_C_bl_params.png")

    # ── Figure D: orographic effects — Cp, strain rate, flow angle ────
    print("   Fig D: orographic effects …")
    fig_D, axes_D = plt.subplots(1, 3, figsize=(15, 4.8), dpi=200,
                                  constrained_layout=True)
    fig_D.suptitle('Orographic effects on the ABL', fontsize=11)

    # Cp
    ax = axes_D[0]
    Cp_p = Cp[:j_top_plot, :].copy()
    vm_cp = max(abs(np.nanpercentile(Cp_p, 1)),
                abs(np.nanpercentile(Cp_p, 99)))
    im = ax.pcolormesh(x, y[:j_top_plot], Cp_p,
                       cmap='RdBu_r', vmin=-vm_cp, vmax=vm_cp,
                       shading='gouraud', rasterized=True, zorder=1)
    ax.contour(x, y[:j_top_plot], Cp_p, levels=[-0.1, 0, 0.1],
               colors=['b', 'k', 'r'], linewidths=0.6,
               linestyles=['--', '-', '--'], zorder=5)
    _draw_solid_mesh(ax, j_top_plot)
    _fill_hill(ax)
    cb = fig_D.colorbar(im, ax=ax, pad=0.01, shrink=0.88, label='$C_p$')
    cb.ax.tick_params(labelsize=7)
    _ax_fmt(ax, r'Pressure coefficient  $C_p = 1 - q^2/U_\mathrm{ref}^2$')

    # strain rate |S|
    ax = axes_D[1]
    Sm_p = S_mag[:j_top_plot, :].copy()
    vm_s = np.nanpercentile(Sm_p, 97)
    im = ax.pcolormesh(x, y[:j_top_plot], Sm_p,
                       cmap='hot_r', vmin=0, vmax=vm_s,
                       shading='gouraud', rasterized=True, zorder=1)
    _draw_solid_mesh(ax, j_top_plot)
    _fill_hill(ax)
    cb = fig_D.colorbar(im, ax=ax, pad=0.01, shrink=0.88, label='|S|')
    cb.ax.tick_params(labelsize=7)
    _ax_fmt(ax,
            r'Strain rate  $|S|=\sqrt{2S_{ij}S_{ij}}$  '
            r'(turbulence production proxy)',
            ylabel='')

    # flow deflection angle
    ax = axes_D[2]
    alph_p = alpha[:j_top_plot, :].copy()
    vm_a = np.nanpercentile(np.abs(alph_p), 98)
    im = ax.pcolormesh(x, y[:j_top_plot], alph_p,
                       cmap='PuOr', vmin=-vm_a, vmax=vm_a,
                       shading='gouraud', rasterized=True, zorder=1)
    ax.contour(x, y[:j_top_plot], alph_p, levels=[-1.5, 0, 1.5],
               colors=['b', 'k', 'r'], linewidths=0.5,
               linestyles=['--', '-', '--'], zorder=5)
    _draw_solid_mesh(ax, j_top_plot)
    _fill_hill(ax)
    cb = fig_D.colorbar(im, ax=ax, pad=0.01, shrink=0.88, label='α (°)')
    cb.ax.tick_params(labelsize=7)
    _ax_fmt(ax,
            r'Flow deflection angle  $\alpha=\arctan(V/U)$',
            ylabel='')

    fig_D.savefig(cwd + '/fig/abl_D_orography.png',
                  dpi=200, bbox_inches='tight')
    print("      saved: abl_D_orography.png")

    # ── Figure E: x-averaged profiles + log-law check ─────────────────
    print("   Fig E: x-averaged profiles …")
    U_xmean  = np.mean(AvgPhU_i, axis=1)
    om_xmean = np.mean(_ddx(AvgPhV_i) - _ddy(AvgPhU_j), axis=1)
    j_above_crest = j_h + 2    # start profiles just above hill crest

    fig_E, axes_E = plt.subplots(1, 3, figsize=(13, 5.5), dpi=200,
                                  constrained_layout=True)
    fig_E.suptitle('x-averaged boundary-layer profiles', fontsize=11)

    # mean U profile
    ax = axes_E[0]
    ax.plot(U_xmean, y, 'b-', lw=2, label=r'$\langle U\rangle_x$')
    ax.axvline(U_ref,  color='k',    lw=0.8, ls='--',
               label=f'$U_{{ref}}={U_ref:.3f}$')
    ax.axhline(hill_h, color='#888', lw=0.8, ls=':',
               label=f'Hill crest $h={hill_h:.4f}$')
    ax.set_xlabel(r'$\langle U\rangle_x$', fontsize=9)
    ax.set_ylabel('y', fontsize=9)
    ax.set_title('Mean streamwise velocity profile', fontsize=9)
    ax.legend(fontsize=8); ax.tick_params(labelsize=7.5)
    ax.set_xlim(0, U_ref * 1.05); ax.set_ylim(0, 0.05)

    # log-law check: U vs ln(y)
    ax = axes_E[1]
    y_log  = y[j_above_crest:]
    U_log  = U_xmean[j_above_crest:]
    mask   = y_log < 0.05
    lny    = np.log(y_log[mask])
    ax.plot(U_log[mask], lny, 'b-', lw=2)
    coeffs = np.polyfit(lny, U_log[mask], 1)
    ax.plot(np.polyval(coeffs, lny), lny, 'r--', lw=1.5,
            label=f'log fit  slope={coeffs[0]:.4f}\nκ_eff={1/coeffs[0]:.3f}')
    ax.set_xlabel(r'$\langle U\rangle_x$', fontsize=9)
    ax.set_ylabel(r'$\ln(y)$', fontsize=9)
    ax.set_title(r'Log-law check:  $U$ vs $\ln y$', fontsize=9)
    ax.legend(fontsize=8); ax.tick_params(labelsize=7.5)

    # x-averaged vorticity profile
    ax = axes_E[2]
    ax.plot(om_xmean[j_above_crest:], y[j_above_crest:], 'r-', lw=2)
    ax.axhline(hill_h, color='#888', lw=0.8, ls=':', label='Hill crest')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel(r'$\langle\omega\rangle_x$', fontsize=9)
    ax.set_ylabel('y', fontsize=9)
    ax.set_title(r'x-averaged vorticity profile  $\langle\omega\rangle_x$',
                 fontsize=9)
    ax.legend(fontsize=8); ax.tick_params(labelsize=7.5)
    ax.set_ylim(0, 0.04)

    fig_E.savefig(cwd + '/fig/abl_E_profiles.png',
                  dpi=200, bbox_inches='tight')
    print("      saved: abl_E_profiles.png")

    # ── Figure F: Helmholtz KE partition ──────────────────────────────
    print("   Fig F: Helmholtz KE partition …")
    irrot_frac = (np.nanmean(KE_phi[:j_top_plot, :]) /
                  max(np.nanmean(KE_tot[:j_top_plot, :]), 1e-30) * 100)

    fig_F, axes_F = plt.subplots(1, 3, figsize=(15, 4.8), dpi=200,
                                  constrained_layout=True)
    fig_F.suptitle(
        'Helmholtz KE partition: irrotational (φ) vs solenoidal (ψ)',
        fontsize=11)

    panels = [
        (KE_phi[:j_top_plot, :],
         r'Irrotational KE', 'Blues'),
        (KE_psi[:j_top_plot, :],
         r'Solenoidal KE ',   'Reds'),
        (KE_tot[:j_top_plot, :],
         r'Total KE ',             'viridis'),
    ]

    for ax, (field, title, cm) in zip(axes_F, panels):
        vm = np.nanpercentile(field, 99)
        im = ax.pcolormesh(x, y[:j_top_plot], field,
                           cmap=cm, vmin=0, vmax=max(vm, 1e-30),
                           shading='gouraud', rasterized=True, zorder=1)
        _draw_solid_mesh(ax, j_top_plot)
        _fill_hill(ax)
        cb = fig_F.colorbar(im, ax=ax, pad=0.01, shrink=0.88, label='KE')
        cb.ax.tick_params(labelsize=7)
        _ax_fmt(ax, title, ylabel='y' if ax is axes_F[0] else '')

    axes_F[0].text(
        0.03, 0.96,
        f'Irrotational fraction:\n{irrot_frac:.4f}% of total KE\n'
        r'→ flow is essentially pure solenoidal',
        transform=axes_F[0].transAxes, fontsize=7.5, va='top',
        bbox=dict(fc='white', alpha=0.85, ec='lightgray', lw=0.5))

    fig_F.savefig(cwd + '/fig/abl_F_energy.png',
                  dpi=200, bbox_inches='tight')
    print("      saved: abl_F_energy.png")

    print("\nDone.")
    
    eps_hgt  = np.sum(eps, axis=0).astype(int)
    hill_top = np.where(eps_hgt > 0, y[np.maximum(eps_hgt-1, 0)], 0.0)
    xfill    = np.concatenate([[x[0]], x, [x[-1]]])
    yfill    = np.concatenate([[0], hill_top, [0]])
    mask0 = 1 - eps
    phi_final = phi_final*mask0
    psi_final = psi_final*mask0
    plot2D_equipotential(
        x_in, y_in[:450], phi_final[:450, :],
        title=r'Equipotential lines of $\phi$',
        xname=r'$x^+$', yname=r'$z^+$',
        savename=cwd + '/fig/phi_equipotential.png',
        xfill=x_oro_in, yfill=y_oro_in,
        resolution=1000, n_levels=200,
    )
    
    plot2D_streamline(
        x_in, y_in[:300], psi_final[:300, :],
        title=r'Streamfunction $\psi$',
        xname=r'$x^+$', yname=r'$z^+$',
        savename=cwd + '/fig/psi_streamfunction_full.png',
        xfill=x_oro_in, yfill=y_oro_in,
        resolution=1000, n_levels=20,
    )

# %%
    

    # ═══════════════════════════════════════════════════════════════════════════
    #  EXTENSION TO geopotential_v14.py
    #  ─────────────────────────────────────────────────────────────────────────
    #  Append this block after the final print("\nDone.") of the main driver.
    #  All variables produced by the main driver are assumed in scope:
    #
    #    x, y, nx, ny, cwd, cd, D1, D2, kx, solid, eps, hill_top,
    #    AvgPhU_i, AvgPhV_i, phi_final, psi_final, psi_bc_top,
    #    KE_phi, KE_psi, KE_tot   (from Step 17)
    #
    #  Nothing in the original file is modified.
    # ═══════════════════════════════════════════════════════════════════════════
    
    # ── Configuration ──────────────────────────────────────────────────────────
    INST_FILE_U  = cwd + 'flow.264000.1'   # binary file — u-velocity component
    INST_FILE_V  = cwd + 'flow.264000.2'   # binary file — v-velocity component
    INST_PLANE   = 2                      # 1-based spanwise (z) plane index
    ENSTR_THRESH = 0.01                     # TNTI enstrophy threshold (fraction of peak)
    
    # Header size: read once from the u-file; both component files are identical
    # in format so the same hdr value applies to INST_FILE_V without re-reading.
    # read_header is imported via  from functions import *  at the top of the driver.
    HDR_BYTES, _, _, _, _, _ = read_header(INST_FILE_U)
    print(f"\n   Header size (from read_header): {HDR_BYTES} bytes")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP E2 — Load instantaneous velocity fields
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Step E2: Load instantaneous velocity fields ──")
    
    u_inst = readplane(INST_FILE_U, nx, ny, INST_PLANE, HDR_BYTES)
    v_inst = readplane(INST_FILE_V, nx, ny, INST_PLANE, HDR_BYTES)
    
    print(f"   u_inst shape : {u_inst.shape}")
    print(f"   u_inst range : [{u_inst.min():.4f}, {u_inst.max():.4f}]")
    print(f"   v_inst range : [{v_inst.min():.4f}, {v_inst.max():.4f}]")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP E3 — Turbulent fluctuations   u' = u_inst − ⟨U⟩,  v' = v_inst − ⟨V⟩
    #
    #  The phase-averaged IBM-interpolated fields AvgPhU_i / AvgPhV_i serve as
    #  the mean.  Solid cells are zeroed in both the instantaneous field and the
    #  fluctuation — consistent with the IBM treatment throughout the main solver.
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Step E3: Turbulent fluctuations ──")
    
    # Zero solid in instantaneous field first (IBM: no physical velocity there)
    u_inst_clean = u_inst.copy();  u_inst_clean[solid] = 0.0
    v_inst_clean = v_inst.copy();  v_inst_clean[solid] = 0.0
    
    # Subtract phase-averaged mean
    u_prime = u_inst_clean - AvgPhU_i
    v_prime = v_inst_clean - AvgPhV_i
    u_prime[solid] = 0.0
    v_prime[solid] = 0.0
    
    fluid_mask   = ~solid
    TKE_snapshot = 0.5 * np.mean(u_prime[fluid_mask]**2 + v_prime[fluid_mask]**2)
    
    print(f"   max|u'| = {np.max(np.abs(u_prime)):.4e}   "
          f"rms = {np.sqrt(np.mean(u_prime[fluid_mask]**2)):.4e}")
    print(f"   max|v'| = {np.max(np.abs(v_prime)):.4e}   "
          f"rms = {np.sqrt(np.mean(v_prime[fluid_mask]**2)):.4e}")
    print(f"   Snapshot TKE (fluid cells) : {TKE_snapshot:.6e}")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP E4 — Gradients and boundary condition for fluctuation fields
    #
    #  psi_bc_top for the fluctuation:
    #      Q'(x) = ∫₀^{Ly} u'(y,x) dy
    #
    #  For an incompressible flow with a periodic x-domain the fluctuation
    #  integrates to zero: Q'(x) = Q_inst(x) − Q_mean ≈ 0.
    #  We compute it explicitly as a sanity check.
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Step E4: Gradients and BCs for fluctuation fields ──")
    
    du_prime_dx = cd.ddx(u_prime)
    du_prime_dy = cd.ddy(u_prime)
    dv_prime_dx = cd.ddx(v_prime)
    dv_prime_dy = cd.ddy(v_prime)
    
    divergence_prime = du_prime_dx + dv_prime_dy
    vorticity_prime  = dv_prime_dx - du_prime_dy
    
    divergence_prime[solid] = 0.0
    vorticity_prime[solid]  = 0.0
    
    # Top Dirichlet BC for ψ' — should be ≈ 0
    psi_bc_top_prime = trapezoid(u_prime, y, axis=0).mean()
    
    print(f"   max|∇·u'|         = {np.max(np.abs(divergence_prime)):.4e}  (expected ~0, confirms incompressibility)")
    print(f"   max|ω'|           = {np.max(np.abs(vorticity_prime)):.4e}")
    print(f"   psi_bc_top_prime  = {psi_bc_top_prime:.4e}  (expected ~0; non-zero → instantaneous flux fluctuation)")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP E5 — Helmholtz decomposition of the turbulent fluctuation
    #
    #     ∇²φ' = ∇·u'   (≈ 0 for incompressible snapshot → φ' ≈ 0)
    #     ∇²ψ' = −ω'    (turbulent stream function)
    #
    #  Uses the same solver, BCs, and prebuilt matrices (D1, D2) as the mean
    #  field.  No modifications to compute_phi_psi are needed.
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Step E5: Helmholtz decomposition of turbulent fluctuation ──")
    
    phi_prime, psi_prime = compute_phi_psi(
        x, y,
        divergence_prime,
        vorticity_prime,
        D1=D1, D2=D2,
        psi_bc_top=psi_bc_top_prime,   # ≈ 0 for fluctuation
        verbose=True,
    )
    phi_prime[solid] = 0.0
    psi_prime[solid] = 0.0
    
    print(f"   max|φ'| = {np.max(np.abs(phi_prime)):.4e}  "
          f"(expected ~0; confirms snapshot incompressibility)")
    print(f"   max|ψ'| = {np.max(np.abs(psi_prime)):.4e}")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP E6 — Turbulent kinetic energy partition
    #
    #  Recover fluctuation velocities from the Helmholtz potentials:
    #      u'_φ = ∂φ'/∂x        u'_ψ = ∂ψ'/∂y
    #      v'_φ = ∂φ'/∂y        v'_ψ = −∂ψ'/∂x
    #
    #  The total TKE decomposes as:
    #      TKE = KE_irrot + KE_solenoid + KE_cross
    #
    #  KE_cross integrates to zero over a periodic domain, but its spatial
    #  pattern is informative near the hill (irrotational and solenoidal modes
    #  are not orthogonal locally).
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Step E6: TKE partition ──")
    
    dphi_p_dx = cd.ddx(phi_prime)
    dphi_p_dy = cd.ddy(phi_prime)
    dpsi_p_dx = cd.ddx(psi_prime)
    dpsi_p_dy = cd.ddy(psi_prime)
    
    # Velocity contributions from each Helmholtz component
    u_irrot    =  dphi_p_dx    # irrotational u'
    u_solenoid =  dpsi_p_dy    # solenoidal   u'
    v_irrot    =  dphi_p_dy    # irrotational v'
    v_solenoid = -dpsi_p_dx    # solenoidal   v'
    
    KE_irrot    = 0.5 * (u_irrot**2    + v_irrot**2)
    KE_solenoid = 0.5 * (u_solenoid**2 + v_solenoid**2)
    KE_cross    = u_irrot * u_solenoid + v_irrot * v_solenoid   # cross-term
    
    KE_irrot[solid]    = 0.0
    KE_solenoid[solid] = 0.0
    KE_cross[solid]    = 0.0
    
    TKE_irrot_mean    = np.mean(KE_irrot[fluid_mask])
    TKE_solenoid_mean = np.mean(KE_solenoid[fluid_mask])
    TKE_cross_mean    = np.mean(KE_cross[fluid_mask])
    TKE_check         = TKE_irrot_mean + TKE_solenoid_mean + TKE_cross_mean
    
    print(f"   KE irrotational  : {TKE_irrot_mean:.4e}  "
          f"({100*TKE_irrot_mean / max(TKE_snapshot,1e-30):.3f}% of snapshot TKE)")
    print(f"   KE solenoidal    : {TKE_solenoid_mean:.4e}  "
          f"({100*TKE_solenoid_mean / max(TKE_snapshot,1e-30):.3f}%)")
    print(f"   KE cross-term    : {TKE_cross_mean:.4e}")
    print(f"   KE sum check     : {TKE_check:.4e}  (target: {TKE_snapshot:.4e})")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP E7 — Q-criterion  (instantaneous velocity gradients)
    #
    #  In 2-D:
    #     ||S||²  = (∂u/∂x)² + (∂v/∂y)² + ½(∂u/∂y + ∂v/∂x)²
    #     ||Ω||²  = ½ ω²             where ω = ∂v/∂x − ∂u/∂y
    #     Q       = ½ (||Ω||² − ||S||²)
    #
    #  Q > 0  → rotation-dominated  (vortex cores, recirculation bubble)
    #  Q < 0  → strain-dominated    (impingement zones, saddle points)
    #  Q = 0  isoline → boundary of coherent vortices
    #
    #  Computed from the INSTANTANEOUS field so vortical structures are
    #  not smeared out by the phase-averaging.
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Step E7: Q-criterion (instantaneous) ──")
    
    du_i_dx = cd.ddx(u_inst_clean)
    du_i_dy = cd.ddy(u_inst_clean)
    dv_i_dx = cd.ddx(v_inst_clean)
    dv_i_dy = cd.ddy(v_inst_clean)
    
    omega_inst    = dv_i_dx - du_i_dy                     # instantaneous vorticity
    S_norm_sq     = (du_i_dx**2 + dv_i_dy**2
                     + 0.5 * (du_i_dy + dv_i_dx)**2)      # ||S||²
    Omega_norm_sq = 0.5 * omega_inst**2                   # ||Ω||²  (2-D)
    Q_crit        = 0.5 * (Omega_norm_sq - S_norm_sq)
    
    Q_crit[solid] = np.nan
    
    Q_fluid = Q_crit[fluid_mask]
    print(f"   Q range (fluid) : [{np.nanmin(Q_fluid):.4e}, {np.nanmax(Q_fluid):.4e}]")
    print(f"   Q > 0 fraction  : "
          f"{np.sum(Q_fluid > 0) / Q_fluid.size * 100:.2f}%  "
          f"(rotation-dominated cells)")
    print(f"   Q < 0 fraction  : "
          f"{np.sum(Q_fluid < 0) / Q_fluid.size * 100:.2f}%  "
          f"(strain-dominated cells)")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP E8 — Enstrophy, intermittency proxy, and TNTI
    #
    #  Enstrophy density:
    #      Ω(x,y) = ½ ω'²
    #
    #  Single-snapshot intermittency proxy:
    #      γ(x,y) = 1  if  Ω(x,y) > Ω_thresh
    #              = 0  otherwise
    #      Ω_thresh = ENSTR_THRESH × max(Ω)   (default 1%)
    #
    #  γ = 1 marks fluid that is locally turbulent (enstrophy-carrying).
    #  γ = 0 marks locally irrotational (non-turbulent) fluid.
    #
    #  TNTI height from enstrophy threshold:
    #      y_TNTI(x) = max{ y : γ(x,y) = 1 }  (topmost turbulent row per column)
    #
    #  TNTI height from solenoidal KE gradient:
    #      y_TNTI_grad(x) = argmax_y |∂KE_ψ/∂y|  per column
    #      This is purely mean-field derived and less noisy for a single snapshot.
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Step E8: Enstrophy and intermittency ──")
    
    enstrophy_prime = 0.5 * vorticity_prime**2
    enstrophy_prime[solid] = 0.0
    
    enstr_peak   = float(np.max(enstrophy_prime[fluid_mask]))
    enstr_thresh = ENSTR_THRESH * enstr_peak
    gamma_proxy  = (enstrophy_prime > enstr_thresh).astype(float)
    gamma_proxy[solid] = np.nan
    
    mean_gamma = float(np.nanmean(gamma_proxy))
    print(f"   Enstrophy peak      : {enstr_peak:.4e}")
    print(f"   Threshold (1%)      : {enstr_thresh:.4e}")
    print(f"   Mean γ (fluid)      : {mean_gamma:.4f}  "
          f"(fraction of fluid domain classified turbulent at this instant)")
    
    # TNTI height — enstrophy threshold method
    y_TNTI = np.zeros(nx)
    for i in range(nx):
        col      = gamma_proxy[:, i]
        turb_idx = np.where(col == 1.0)[0]
        y_TNTI[i] = y[turb_idx.max()] if turb_idx.size > 0 else y[0]
    
    print(f"\n   TNTI (enstrophy)  : mean={y_TNTI.mean():.6f}  "
          f"std={y_TNTI.std():.6f}  "
          f"range=[{y_TNTI.min():.6f}, {y_TNTI.max():.6f}]")
    
    # TNTI height — solenoidal KE gradient method (mean-field, less noisy)
    grad_KE_sol_dy      = cd.ddy(KE_solenoid)
    grad_KE_sol_dy[solid] = 0.0
    y_TNTI_grad = np.array([
        y[np.argmax(np.abs(grad_KE_sol_dy[:, i]))]
        for i in range(nx)
    ])
    
    print(f"   TNTI (|∇KE_ψ|)   : mean={y_TNTI_grad.mean():.6f}  "
          f"std={y_TNTI_grad.std():.6f}")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP E9 — Spectral energy partition  E_φ'(k),  E_ψ'(k)
    #
    #  1-D energy spectra obtained by:
    #    1. rfft of φ' and ψ' along the periodic x-direction.
    #    2. Integrate over y using local grid spacing as quadrature weights
    #       (stretched grid — trapezoidal rule via np.gradient).
    #       Fully-solid rows are excluded (weight set to zero).
    #
    #  Physical interpretation:
    #    E_ψ(k)   solenoidal (vortical) spectrum — expect k^{-5/3} inertial range
    #    E_φ(k)   irrotational spectrum          — expect E_φ << E_ψ for DNS
    #    ratio    E_φ/E_ψ → 0 confirms incompressibility at every scale
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Step E9: Spectral energy partition ──")
    
    Lx = nx * (x[1] - x[0])                         # periodic domain length
    
    phi_hat_p = np.fft.rfft(phi_prime, axis=1)       # (ny, nx//2+1)
    psi_hat_p = np.fft.rfft(psi_prime, axis=1)       # (ny, nx//2+1)
    
    # y-integration weights: local spacing, zeroed for fully-solid rows
    dy_vec = np.gradient(y)                          # (ny,) local Δy
    wt     = dy_vec.copy()
    fully_solid_rows = np.all(solid, axis=1)         # rows entirely inside IBM
    wt[fully_solid_rows] = 0.0
    
    E_phi_k = np.sum(wt[:, None] * np.abs(phi_hat_p)**2, axis=0) / Lx
    E_psi_k = np.sum(wt[:, None] * np.abs(psi_hat_p)**2, axis=0) / Lx
    
    k_rfft   = np.fft.rfftfreq(nx, d=Lx / nx) * 2.0 * np.pi   # physical wavenumbers
    # Drop k=0 (mean mode) for log-log plots
    k_pos     = k_rfft[1:]
    E_phi_pos = E_phi_k[1:]
    E_psi_pos = E_psi_k[1:]
    ratio_k   = E_phi_pos / np.maximum(E_psi_pos, 1e-30)
    
    print(f"   E_φ total : {float(np.sum(E_phi_k)):.4e}")
    print(f"   E_ψ total : {float(np.sum(E_psi_k)):.4e}")
    print(f"   Max E_φ/E_ψ ratio across all k : {ratio_k.max():.4e}  "
          f"(should be << 1 for incompressible DNS)")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    #  STEP E10 — Plots
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n── Step E10: Plots ──")
    
    # ── shared geometry ────────────────────────────────────────────────────────
    j_plt   = min(200, ny - 1)    # plot domain height (row index)
    xfill_e = np.concatenate([[x[0]], x, [x[-1]]])
    yfill_e = np.concatenate([[0],    hill_top, [0]])
    
    def _fill_solid(ax):
        """Fill IBM solid and draw hill outline."""
        ax.fill(xfill_e, yfill_e, color='#2d2d2d', zorder=6)
    
    def _fmt_ax(ax, title, ylabel='y'):
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, y[j_plt])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x', fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    
    
    # ── Fig T1: φ' and ψ' side-by-side ────────────────────────────────────────
    print("   Fig T1: phi' and psi' fields …")
    
    fig_T1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), dpi=200,
                                       constrained_layout=True)
    fig_T1.suptitle(
        f"Helmholtz decomposition of turbulent fluctuation  (spanwise plane {INST_PLANE})",
        fontsize=11)
    
    for ax, field, title in [
        (ax1, phi_prime,
         r"$\phi'$  — irrotational potential   ($\nabla^2\phi'=\nabla\cdot\mathbf{u}'\approx0$)"),
        (ax2, psi_prime,
         r"$\psi'$  — turbulent stream function  ($\nabla^2\psi'=-\omega'$)"),
    ]:
        f  = field[:j_plt, :].copy()
        f[solid[:j_plt, :]] = np.nan
        vm = float(np.nanpercentile(np.abs(f[np.isfinite(f)]), 98))
        im = ax.pcolormesh(x, y[:j_plt], f,
                           cmap='RdBu_r', vmin=-vm, vmax=vm,
                           shading='gouraud', rasterized=True, zorder=1)
        _fill_solid(ax)
        cb = fig_T1.colorbar(im, ax=ax, pad=0.01, shrink=0.92)
        cb.ax.tick_params(labelsize=7.5)
        _fmt_ax(ax, title)
    
    fig_T1.savefig(cwd + '/fig/turb_T1_phi_psi_prime.png',
                   dpi=200, bbox_inches='tight')
    print("      saved: turb_T1_phi_psi_prime.png")
    
    
    # ── Fig T2: TKE partition ──────────────────────────────────────────────────
    print("   Fig T2: TKE partition …")
    
    fig_T2, axes_T2 = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200,
                                    constrained_layout=True)
    fig_T2.suptitle(
        f"Turbulent KE partition: irrotational vs solenoidal  (plane {INST_PLANE})",
        fontsize=11)
    
    panels_T2 = [
        (KE_irrot,    'Blues',
         r"$K_\phi = \frac{1}{2}[(u'_\phi)^2+(v'_\phi)^2]$  irrotational",
         False),
        (KE_solenoid, 'Reds',
         r"$K_\psi = \frac{1}{2}[(u'_\psi)^2+(v'_\psi)^2]$  solenoidal",
         False),
        (KE_cross,    'PuOr',
         r"Cross-term  $u'_\phi u'_\psi + v'_\phi v'_\psi$",
         True),
    ]
    
    for ax, (field, cm, title, sym) in zip(axes_T2, panels_T2):
        f  = field[:j_plt, :].copy()
        f[solid[:j_plt, :]] = np.nan
        finite = f[np.isfinite(f)]
        vm = float(np.nanpercentile(np.abs(finite), 98))
        im = ax.pcolormesh(x, y[:j_plt], f,
                           cmap=cm,
                           vmin=(-vm if sym else 0), vmax=vm,
                           shading='gouraud', rasterized=True, zorder=1)
        _fill_solid(ax)
        cb = fig_T2.colorbar(im, ax=ax, pad=0.01, shrink=0.92)
        cb.ax.tick_params(labelsize=7.5)
        ylabel = 'y' if ax is axes_T2[0] else ''
        _fmt_ax(ax, title, ylabel=ylabel)
    
    axes_T2[0].text(
        0.03, 0.96,
        f"Mean KE irrot    : {TKE_irrot_mean:.3e}\n"
        f"Mean KE solenoid : {TKE_solenoid_mean:.3e}\n"
        f"Irrot fraction   : {100*TKE_irrot_mean/max(TKE_snapshot,1e-30):.3f}% of TKE",
        transform=axes_T2[0].transAxes, fontsize=7, va='top',
        bbox=dict(fc='white', alpha=0.85, ec='lightgray', lw=0.5))
    
    fig_T2.savefig(cwd + '/fig/turb_T2_TKE_partition.png',
                   dpi=200, bbox_inches='tight')
    print("      saved: turb_T2_TKE_partition.png")
    
    
    # ── Fig T3: Q-criterion and enstrophy + TNTI ──────────────────────────────
    print("   Fig T3: Q-criterion and enstrophy …")
    
    fig_T3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), dpi=200,
                                       constrained_layout=True)
    fig_T3.suptitle(
        f"Flow topology: Q-criterion and enstrophy  (plane {INST_PLANE})",
        fontsize=11)
    
    # Q-criterion
    Q_plt = Q_crit[:j_plt, :].copy()
    vm_Q  = float(np.nanpercentile(np.abs(Q_plt[np.isfinite(Q_plt)]), 97))
    im1   = ax1.pcolormesh(x, y[:j_plt], Q_plt,
                            cmap='RdBu_r', vmin=-vm_Q, vmax=vm_Q,
                            shading='gouraud', rasterized=True, zorder=1)
    ax1.contour(x, y[:j_plt], Q_plt, levels=[0.0],
                colors='k', linewidths=0.8, zorder=4)   # Q = 0 isoline
    _fill_solid(ax1)
    cb1 = fig_T3.colorbar(im1, ax=ax1, pad=0.01, shrink=0.92, label='Q')
    cb1.ax.tick_params(labelsize=7.5)
    _fmt_ax(ax1, r'Q-criterion  (black: Q = 0 isoline  ↔  vortex boundary)')
    ax1.text(0.03, 0.96,
             f"Q>0 (rotation): {np.sum(Q_plt[np.isfinite(Q_plt)]>0)/np.sum(np.isfinite(Q_plt))*100:.1f}%\n"
             f"Q<0 (strain):   {np.sum(Q_plt[np.isfinite(Q_plt)]<0)/np.sum(np.isfinite(Q_plt))*100:.1f}%",
             transform=ax1.transAxes, fontsize=7, va='top',
             bbox=dict(fc='white', alpha=0.85, ec='lightgray', lw=0.5))
    
    # Enstrophy + TNTI overlay
    enstr_plt = enstrophy_prime[:j_plt, :].copy()
    enstr_plt[solid[:j_plt, :]] = np.nan
    vm_e = float(np.nanpercentile(enstr_plt[np.isfinite(enstr_plt)], 98))
    im2  = ax2.pcolormesh(x, y[:j_plt], enstr_plt,
                           cmap='hot_r', vmin=0, vmax=vm_e,
                           shading='gouraud', rasterized=True, zorder=1)
    ax2.plot(x, y_TNTI,      color='cyan',  lw=1.4, label='TNTI (enstrophy threshold)',  zorder=5)
    ax2.plot(x, y_TNTI_grad, color='blue',  lw=1.0, ls='--',
             label=r'TNTI ($|\nabla K_\psi|$ method)', zorder=5)
    _fill_solid(ax2)
    cb2 = fig_T3.colorbar(im2, ax=ax2, pad=0.01, shrink=0.92,
                           label=r"$\frac{1}{2}\omega'^2$")
    cb2.ax.tick_params(labelsize=7.5)
    _fmt_ax(ax2, r"Enstrophy $\frac{1}{2}\omega'^2$ with TNTI estimates", ylabel='')
    ax2.legend(fontsize=7.5, loc='upper right')
    ax2.text(0.03, 0.96,
             f"Mean γ = {mean_gamma:.3f}\n"
             f"TNTI mean height = {y_TNTI.mean():.4f}",
             transform=ax2.transAxes, fontsize=7, va='top',
             bbox=dict(fc='white', alpha=0.85, ec='lightgray', lw=0.5))
    
    fig_T3.savefig(cwd + '/fig/turb_T3_Qcrit_enstrophy.png',
                   dpi=200, bbox_inches='tight')
    print("      saved: turb_T3_Qcrit_enstrophy.png")
    
    
    # ── Fig T4: Spectral energy partition ─────────────────────────────────────
    print("   Fig T4: spectral partition …")
    
    fig_T4, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200,
                                       constrained_layout=True)
    fig_T4.suptitle(
        f"Spectral energy partition of turbulent fluctuations  (plane {INST_PLANE})",
        fontsize=11)
    
    # Left panel: log-log spectra + k^{-5/3} reference
    ax1.loglog(k_pos, E_psi_pos, 'r-', lw=1.8, label=r"$E_\psi(k)$  solenoidal")
    ax1.loglog(k_pos, E_phi_pos, 'b-', lw=1.8, label=r"$E_\phi(k)$  irrotational")
    
    # k^{-5/3} reference anchored at k_pos[3]
    k_ref = np.array([k_pos[3], k_pos[len(k_pos) // 2]])
    E_ref = E_psi_pos[3] * (k_ref / k_pos[3])**(-5.0 / 3.0)
    ax1.loglog(k_ref, E_ref, 'k--', lw=1.2, label=r'$k^{-5/3}$ reference')
    
    ax1.set_xlabel(r'$k_x$  (rad m⁻¹)',    fontsize=9)
    ax1.set_ylabel(r'$E(k)$',              fontsize=9)
    ax1.set_title(r'Energy spectra:  $E_\phi(k)$ vs $E_\psi(k)$', fontsize=9)
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=7.5)
    ax1.grid(True, which='both', alpha=0.3, lw=0.5)
    
    # Right panel: irrotational/solenoidal ratio
    ax2.semilogx(k_pos, ratio_k, color='purple', lw=1.8)
    ax2.axhline(1.0, color='k', lw=0.8, ls='--', label='ratio = 1')
    ax2.set_xlabel(r'$k_x$  (rad m⁻¹)',              fontsize=9)
    ax2.set_ylabel(r'$E_\phi(k)\,/\,E_\psi(k)$',     fontsize=9)
    ax2.set_title(r'Irrotational / solenoidal ratio  ($\ll 1$ → incompressible)', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=7.5)
    ax2.grid(True, which='both', alpha=0.3, lw=0.5)
    ax2.set_ylim(bottom=0)
    
    fig_T4.savefig(cwd + '/fig/turb_T4_spectra.png',
                   dpi=200, bbox_inches='tight')
    print("      saved: turb_T4_spectra.png")
    
    
    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  Extension complete.  Files written to:  " + cwd + "/fig/")
    print("  ─────────────────────────────────────────────────────────")
    print("  turb_T1_phi_psi_prime.png   —  φ' and ψ' fields")
    print("  turb_T2_TKE_partition.png   —  irrot / solenoid / cross TKE")
    print("  turb_T3_Qcrit_enstrophy.png —  Q-criterion + enstrophy + TNTI")
    print("  turb_T4_spectra.png         —  spectral energy partition")
    print("═" * 65)
