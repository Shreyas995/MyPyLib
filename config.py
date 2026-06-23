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

# Rough reference case (Kostelecky & Ansorge r1, Re_D = 1000) — Method-2 u* test.
# Same tlab avg_all.nc variable names as the smooth case, but a different grid and
# NO stored FrictionVelocity (u* is computed from the momentum-integral balance).
rough_nc_path = (
    '/home/shreyad95/postprocessing/Code/RoughRe1000/'
    'ri00.00_re1000_3072x0656x3072_20230119_r1_avg_all.nc'
)
Re_rough        = 1000                   # bulk Reynolds number of the rough case
Re_lambda_rough = 0.5 * Re_rough**2      # = 5e5  (tlab 1/nu)
nu_rough        = 1 / Re_lambda_rough    # = 2e-6  kinematic viscosity

# ── 10. Reference-overlay master switches (testing vs publication) ────────────
#   Each: True → overlay that reference case on the smooth-vs-orographic plots.
#   Default = smooth only (publication mode); flip plot_ref_rough on for testing.
plot_ref_smooth = True    # overlay smooth (Re=500) reference
plot_ref_rough  = False   # overlay rough r1 (Re=1000) reference  (test only)

# ── 11. Stratification / research diagnostics (8-goal post-processing) ────────
# Research.md:536-550.  Buoyancy = the scalar directly: AvgScal IS the non-dim
# buoyancy b, so the surface buoyancy B_0 is its near-wall (Dirichlet) value.
# Every term degrades gracefully to neutral / N/A when the run is unstratified.
#
# delta_neutral : matched NEUTRAL boundary-layer depth used in the bulk
#   Richardson number  Ri_B = B_0 * delta_neutral / G**2.  Set it to the neutral
#   run's depth  δ = u*_neutral / f  for each Reynolds number.  None → fall back
#   to the current run's own δ (a warning is printed); correct for the neutral
#   case itself, a placeholder for the stratified runs until measured.
delta_neutral     = None      # Re_D = 500 neutral depth (set once measured)
delta_neutral_750 = None      # Re_D = 750 neutral depth (set once measured)

Pr_t   = 0.85                 # turbulent Prandtl number (φ_h / eddy-diffusivity closure)
beta_m = 5.0                  # MOST stable slope:  φ_m = 1    + beta_m*ζ
beta_h = 5.0                  # MOST stable slope:  φ_h = Pr_t + beta_h*ζ
Ri_B_bins      = (0.05, 0.15) # weak | intermediate | strong  (Ansorge 2017)
Lplus_collapse = 100.0        # Obukhov length (wall units) collapse threshold ~O(100)

sponge_frac = 0.8             # Rayleigh-sponge bottom ≈ sponge_frac*Ly; wave window below it

# Local-similarity stations over the valley (x-column fraction of nx):
#   windward = left flank, floor = valley bottom, lee = right flank.
station_fracs = {'windward': 0.25, 'floor': 0.50, 'lee': 0.75}

# Goal 6 intermittency γ(z): instantaneous-planesK enstrophy threshold.
#   Off by default (reads many planesK files); set to 1 to compute.
compute_intermittency = 0
enstrophy_thresh = 0.01       # γ = 1 where ½ω'² > enstrophy_thresh * max(½ω'²)

# Goal 2 flat-wall stratified reference: {(Re, Fr): nc_path}.  Empty until that
# data exists (all current .nc are ri00.00 = neutral); loaded via load_smooth_case.
stratified_ref_paths = {}

# ── 12. Stratification switch (Fr) + Obukhov stability-corrected wall law ──────
# Fr distinguishes the NEUTRAL run from the STRATIFIED runs and carries the
# stratification strength.  It is the single switch the wall-law fit keys on
# (PhAvg.py / PhAvg_rotated.py):
#     np.isfinite(Fr) is False  (Fr = np.inf)  → classical neutral log law
#     np.isfinite(Fr) is True   (finite Fr)    → Obukhov (1971) stratified law
# Per-case values (smaller Fr ⇒ stronger stratification).  Re=500 Froude ladder:
#     Ekman18                  neutral :  np.inf
#     1056x672x1056/EkRe500Fr1         :  1.0
#     1056x672x1056/EkRe500Fr0.1       :  0.1
#     1056x672x1056/EkRe500Fr0.01      :  0.01
Fr = np.inf                   # this run's Froude number (np.inf ⇒ neutral)

# Obukhov (1971) stability-corrected log law.  The mean-wind gradient follows
#   √φ(Ri)·κ·z·dū/dz = u★              (Obukhov eqs 17 & 22)
# with the energy-balance universal function
#   φ(Ri) = (1 − Ri/Ri_cr)^(1/2)       (Obukhov eq 38),
# so the wall-law fit integrand carries the factor (1 − Ri/Ri_cr)^(−1/4) and
# reduces to the neutral ln(z⁺ − d⁺) when Ri → 0.  Ri is the gradient Richardson
# number measured from the profiles:  Ri = (∂⟨b⟩/∂z)/(∂⟨ū⟩/∂z)²  (AvgScal is b).
# Ri_cr is the critical Richardson number at which turbulence ceases (φ → 0).
# Obukhov cites 1/11 … 1 (Sverdrup ⇒ Ri_cr = 1/α = K/K_T ≈ Pr_t); 0.25 is the
# standard Miles–Howard value used here — set it to Pr_t if that closure is wanted.
Ri_cr = 0.25                  # critical gradient Richardson number  [Obukhov eqs 36/38]
