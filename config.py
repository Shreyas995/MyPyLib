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
plot_spectra   = 1  # 1 → streamwise energy spectra from planesK.* (Kolmogorov -5/3 check)

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
loglaw_zmin  =  30.0           # lower bound of log-law fit region (wall units z⁺)
loglaw_zmax  = 100.0           # upper bound (log law holds z⁺∈[30,100] for this flow)
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

# Rough Re=1000 STABLE LADDER (Cedrick's ri00.00 → ri18.78 avg_all.nc files) — a
# LOG-LAW-ONLY overlay.  Each file is read minimally (mean rU + stored
# FrictionVelocity only; see functions.load_loglaw_nc / load_rough_ladder_loglaw)
# and drawn in its OWN inner units (Re=1000 → nu_rough).  The directory lives on
# the machine that holds the data (not in this code-prep repo); the loader returns
# an empty list when it is absent, so leaving this on here is harmless.
rough_ladder_dir     = '/home/shreyad95/Documents/PhD/Code/Re1000/'
rough_ladder_pattern = 'ri*_avg.nc'      # glob for the ladder avg files
# These stable files store NO FrictionVelocity; the log-law only needs a
# normalising scale, so u⁺ = ⟨ū⟩/rough_ladder_ustar, z⁺ = y·rough_ladder_ustar/ν.
rough_ladder_ustar   = 0.0618

# ── 10. Reference-overlay master switches (testing vs publication) ────────────
#   Each: True → overlay that reference case on the smooth-vs-orographic plots.
#   Default = smooth only (publication mode); flip plot_ref_rough on for testing.
plot_ref_smooth = True    # overlay smooth (Re=500) reference
plot_ref_rough  = False   # overlay rough r1 (Re=1000) reference  (test only)
# Overlay the FULL rough Re=1000 stable ladder (ri00.00 → ri18.78) on the LOG-LAW
# plots ONLY (PhAvg_rotated PLOT 36; results.py P25/P25b).  Off by default.
plot_ref_rough_ladder = False

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

# Goal 6 intermittency γ(z) + 2-D fields from instantaneous planesK
#   (Ansorge & Mellado 2016).  Reads many planesK files; set to 1 to compute.
compute_intermittency = 1
# Threshold ω₀ = omega_thresh_factor · e_ω, with e_ω ≡ ω_rms(δ) the rms
# fluctuation-vorticity at the BL edge (eq 4.2); γ(z)=⟨H(|ω'|−ω₀)⟩ (eq 4.1).
# Paper sweeps the factor over 1/8…3; default = 1 (ω₀ = e_ω).
omega_thresh_factor = 1.0

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

# ── 13. PhAvg_rotated.py internal constants ───────────────────────────────────
# Constants formerly hard-coded inside PhAvg_rotated.py, collected here so every
# tunable lives in config (project rule: all constants in config.py).  Values are
# byte-for-byte identical to the literals they replace, so pickled quantities that
# depend on them (y0, u_most, …) are unchanged.

# Phase-average read loop: first DNS iteration of the avg_* file series.  File
# names are  avg_flow{srt}_{end}.c  with srt/end stepping by `restart` from this
# base (see the cal_Avg / verify_TimeAvg read loops).
avg_iter_base = 234500

# planesK.* instantaneous-field layout — shared by the intermittency, animation
# and spectra blocks (was triplicated as N_KPLANES/NVARS/KPLANE_IDX, _NK…, _SP_*).
planesK_n_kplanes  = 1        # k-planes saved per file  (kplanes%n in TLAB)
planesK_nvars      = 5        # variables per k-plane    (u, v, w, s1, p)
planesK_kplane_idx = 0        # which k-plane to read/show (0-based)
planesK_vel_max    = 1.5      # reject a frame whose |u| or |v| exceeds this (diverged)

# Modified (Obukhov 1971) stability-corrected log-law fit window (wall units z⁺).
# The neutral classical fit uses loglaw_zmin/zmax (§8); the modified fit sits higher.
modlaw_zmin = 70.0
modlaw_zmax = 150.0

# Monin–Obukhov reference-profile (MOST) near-wall scales & additive constant:
#   d  = most_d_factor  · u★   (zero-plane displacement in u_most_v)
#   y0 = most_y0_factor · u★   (roughness length / first evaluation height)
#   u_most = (1/κ) ln(z⁺) + most_B     (classical log law, additive constant B)
most_d_factor  = 0.01
most_y0_factor = 5.0
most_B         = 4.5

# planesK animation block (animate==1): which snapshots and how to render.
anim_ny         = 430         # wall-normal points to include
anim_first_iter = 262510      # first planesK iteration
anim_last_iter  = 264500      # last  planesK iteration
anim_iter_step  = 10          # iteration stride between frames
anim_fps        = 10          # animation frames per second

# Streamwise-spectra block (plot_spectra==1): wall-normal target heights z⁺ at
# which the pre-multiplied spectrum is sampled (log/inertial region).
spectra_z_targets = [30.0, 60.0, 100.0, 150.0]

# Modified (Obukhov 1971) log-law fixed von Kármán constant.  The paper/Table-III
# pins it to 0.4 (NOT config.kappa = 0.42, which the neutral fit fits within
# kappa_bounds); v* scales as 1/κ, so a different κ only rescales v*, not L1.
# Used by the Obukhov wall-law helpers in functions.py.
obu_kappa = 0.4

# Neutral log-law OLS fit fallback defaults (used only if no valid κ is found in
# kappa_bounds): κ, zero-plane displacement d⁺, roughness z₀ₘ⁺.  All three are
# pickled (kappa_loglaw / d_m_loglaw / z0m_loglaw).
loglaw_kappa_default = 0.41
loglaw_d_default     = 0.0
loglaw_z0m_default   = 0.068

# Kostelecky & Ansorge (2024) fig-4 validation budget (reference .nc cases;
# PLOT 32r in PhAvg[_rotated].py): the geostrophic vector is read at
# fig4_top_frac·Ly (free stream, below the Rayleigh sponge — the avg files
# store no usable G), and the Method-2 u* plateau is averaged over
# fig4_plateau_lo·y_top < y < y_top.
fig4_top_frac   = 0.8
fig4_plateau_lo = 0.05
# Display handedness of the SPANWISE panel (τ_zy).  Our tlab runs carry the
# opposite sign-of-f to K&A's written eq. 4.2, so the physically-native closing
# budget gives C_zy<0 / R_zy>0 — the exact MIRROR of the paper.  With this True
# the τ_zy budget is negated for DISPLAY (τ_zy → −τ_zy) so the panel matches the
# paper and fig4_smooth_standalone.py (Coriolis reads positive); it leaves τ_zx,
# u*, and the closure untouched (only the plotted sign of the spanwise flips).
fig4_paper_spanwise_sign = True
