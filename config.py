#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation configuration for DNS post-processing (Ekman layer, Re_D = 500).

All values are non-dimensional unless noted otherwise.
"""
import numpy as np

# ── 1. Physics / Reynolds number ──────────────────────────────────────────────
Re        = 500               # bulk Reynolds number Re_D = U_G * D / nu
Re_lambda = 0.5 * Re * Re     # tlab parameter: Re_lambda = 1/nu = 0.5*Re_D^2
nu        = 1 / Re_lambda     # kinematic viscosity

# ── 2. Geostrophic forcing ────────────────────────────────────────────────────
f     = 1                     # Coriolis parameter (non-dimensionalised to 1)
alpha = -0.430511             # geostrophic-wind angle (rad); sets Ekman spiral
Gx    =  np.cos(alpha)        # streamwise component of geostrophic wind vector
Gz    = -np.sin(alpha)        # spanwise component

# ── 3. Inner / outer scaling ──────────────────────────────────────────────────
u_star     = 0.076            # friction velocity (orography case, Re_D = 500)
kappa      = 0.42             # von Kármán constant
Re_tau     = (u_star**2) / nu # friction Reynolds number u★²/ν
l_visc     = nu / u_star      # viscous length scale = 1 wall unit
l_in       = l_visc           # alias used in EditGrid / GridShapeCheck / PhAvg
wall_units = l_visc           # alias used in Ekmangridcreation / make_grid scripts
                              #   (both names are live in callers — do not merge)
l_out      = u_star           # outer length scale (friction velocity)
time_scale = 2 * np.pi        # non-dimensional time unit (one inertial period)

# ── 4. Simulation numerics ────────────────────────────────────────────────────
dt      = 0.827E-04           # time step
index   = 1                   # first snapshot index to read
restart = 500                 # snapshots per phase-averaging cycle
counter = 0                   # running snapshot counter (reset each run)
scal    = 1                   # number of active scalar fields
dim     = 3                   # spatial dimensions

# ── 5. Geometry / grid ────────────────────────────────────────────────────────
limity            = 463       # wall-normal index cap for analysis arrays
limity_range      = 150       # y-index range for limited-height diagnostics
hill_hgt          = 94        # placeholder; overwritten from eps field at runtime
canopy_extra_cells = 20       # canopy top = hill_hgt + this many grid cells

# ── 6. Runtime control flags ──────────────────────────────────────────────────
# Each flag below takes only {0, 1}  (0 = off / skip, 1 = on / execute).
cal_Avg        = 0  # 1 → recompute phase-average from raw field files
verify_TimeAvg = 0  # 1 → run time-average verification checks
save_avg       = 0  # 1 → write averaged fields to disk
load_ncfiles   = 0  # (reserved) 1 → load from NetCDF instead of .npy arrays
load_arrays    = 1  # 1 → load pre-saved .npy arrays; 0 → recompute from fields
postprocess    = 1  # 1 → execute the post-processing block
plotRes        = 1  # 1 → generate result plots
animate        = 0  # 1 → produce animation frames

# ── 7. Derivative & interpolation settings ────────────────────────────────────
# DY_METHOD / D2Y_METHOD — wall-normal derivative scheme (CompactDerivatives2D).
# Allowed values (case-insensitive; accepted aliases in parentheses):
#   'compact'    (eta, pade)                 — 6th-order η-space Padé; best on a
#                                              smooth mapping, but trembles at an
#                                              abrupt spacing change (Zone-1 top).
#   'fornberg7'  (fornberg, physical, phys,
#                 nonuniform)                — 7-pt Fornberg on the actual y nodes;
#                                              stays smooth across a grid kink.
#   'fornberg5'                              — 5-pt Fornberg (more local at a kink).
#   'fornberg9'                              — 9-pt Fornberg (higher formal order).
#   'compact_nu' (compact_nonuniform,
#                 nucompact)                 — 4th-order non-uniform Padé.
#   'spline'     (cubic)                     — derivative of a C² cubic spline.
# 'fornberg7' wins the kink benchmark on this grid (test_ddy_schemes.py).
recompute_derivatives = True   # False → use cached .npy derivatives if available
DY_METHOD  = 'fornberg7'       # first y-derivative scheme (CompactDerivatives2D)
D2Y_METHOD = 'compact'         # second y-derivative scheme
ghost_depth  = 5               # IBM ghost-cell stencil depth (cells)
n_anchor     = 4               # anchor points for ghost-cell interpolation
smooth_width = 5               # smoothing half-width at the IBM interface (cells)

# ── 8. Log-law fit window ─────────────────────────────────────────────────────
loglaw_zmin  =  60.0           # lower bound of log-law fit region (wall units z⁺)
loglaw_zmax  = 200.0           # upper bound
kappa_bounds = (0.40, 0.44)   # acceptable range for the fitted von Kármán constant

# ── 9. Paths ──────────────────────────────────────────────────────────────────
smooth_nc_path = (
    '/home/shreyad95/postprocessing/Code/Re500/'
    'ri00.00_re0500_2048x0192x2048_20110615_avg_all.nc'
)
