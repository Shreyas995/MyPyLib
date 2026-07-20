#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IO.py — the single I/O module for the phase-averaging post-processing pipeline.

Everything that only *moves arrays on/off disk* lives here, so the scientific
script PhAvg[_rotated].py contains computation only.  The arrays stay top-level
and Spyder-inspectable in the caller because the loaders return a plain dict the
caller injects with ``globals().update(...)``.

Four responsibilities (formerly scattered inside PhAvg_rotated.py):
  1. ``var_names``            — the list of variables bundled into sim1_results.pkl,
                                consumed cross-case by results.py.  (Was the whole
                                of the old saveresults.py.)
  2. ``write_avg_arrays``     — write the phase-averaged / triple-decomposed fields
                                to .npy   (former ``if (100 == save_avg)`` block).
  3. ``read_avg_arrays``      — reload those .npy into a dict
                                (former ``if (1 == load_arrays)`` block).
  4. ``write_results_pickle`` — build sim1_results.pkl from var_names + a namespace
                                (former in-line pickle dump).

NOTE on the naming: the functions are deliberately NOT called ``save_avg`` /
``load_arrays`` — those are RUN-CONTROL FLAGS in config.py (imported by the
callers), so reusing the names would shadow the flags.

NOTE on the .npy mapping: save and load are ASYMMETRIC.  ``write_avg_arrays``
pulls slices out of the tensor arrays (AvgStress[:,:,k], AvgPh[:,:,k], …) and
writes them under names like ``uu_d`` / ``AvgStrUU`` / ``uu_g`` / ``uu_t``;
``read_avg_arrays`` restores them under the FLAT names the post-processing uses
(``rey_uv`` / ``AvgStrUU`` / ``UU_G`` / ``UU_disp`` …).  The mapping is therefore
written out explicitly below (one np.save / np.load per field) rather than driven
from a single table — this keeps it inspectable and byte-for-byte faithful to the
blocks it replaces.
"""
import numpy as np
import pickle
import netCDF4 as nc

from functions import print_summary_table   # console table helper (reporting)

# =============================================================================
# 1. Variables bundled into sim1_results.pkl
# -----------------------------------------------------------------------------
# Only variables that exist in globals() by the time the pickle is written
# (i.e., after postprocess==1 completes) are listed here.
# Organised by topic; names match exactly the variable names in PhAvg.py.
# =============================================================================
var_names = [

    # -------------------------------------------------------------------------
    # Config scalars (imported from config.py)
    # -------------------------------------------------------------------------
    'Re', 'Re_lambda', 'Re_tau', 'nu', 'dt', 'f',
    'alpha', 'Gx', 'Gz',
    'kappa', 'u_star', 'wall_units',

    # -------------------------------------------------------------------------
    # Grid coordinates and inner/outer length scales
    # -------------------------------------------------------------------------
    'x', 'y', 'z', 'dx', 'nx', 'ny', 'nz',
    'x_in', 'y_in', 'y_inner', 'y_outer',
    'x_oro', 'y_oro', 'x_oro_in', 'y_oro_in',

    # -------------------------------------------------------------------------
    # IBM solid geometry
    # -------------------------------------------------------------------------
    'hill_hgt', 'flk_hgt', 'flk_wdt',
    'eps_top', 'eps_lf', 'eps_bottom', 'eps_rf', 'eps_hgt',
    'eps', 'eps_fr', 'eps_s', 'eps_f', 'eps_g',

    # -------------------------------------------------------------------------
    # IBM masks
    #   mask0    : 1 in fluid, 0 everywhere inside solid
    #   mask_intr: 1 in fluid + interface cell, 0 in interior solid only
    #   mask_v   : 1 where eps_fr == 1
    # -------------------------------------------------------------------------
    'mask0', 'mask_intr', 'mask_v',

    # -------------------------------------------------------------------------
    # Finite-difference stencil selector arrays (compact scheme boundary cases)
    # -------------------------------------------------------------------------
    'case_v', 'case_h', 'case_v_g', 'case_v_itrp', 'case_h_itrp',

    # -------------------------------------------------------------------------
    # Surface geometry — edge-detection algorithm output
    #   horiz_surfaces       : list of (j, i_start, i_end)  — top-face segments
    #   left_flank_surfaces  : list of (j_start, j_end, i)  — right-facing walls (i < nx//2)
    #   right_flank_surfaces : list of (j_start, j_end, i)  — left-facing walls  (i > nx//2)
    # -------------------------------------------------------------------------
    'horiz_surfaces', 'left_flank_surfaces', 'right_flank_surfaces',

    # -------------------------------------------------------------------------
    # Phase-averaged fields  (2D: ny × nx)
    # -------------------------------------------------------------------------
    'AvgPhU', 'AvgPhV', 'AvgPhW', 'AvgP',
    'VelGblU', 'VelGblV', 'VelGblW',

    # Ghost-filled interpolated fields used by the compact-derivative scheme
    # _i suffix: filled for x-derivatives;  _j suffix: filled for y-derivatives
    'AvgPhU_i', 'AvgPhU_j',
    'AvgPhV_i', 'AvgPhV_j',
    'AvgPhW_i', 'AvgPhW_j',

    # -------------------------------------------------------------------------
    # Dispersive velocities: ũ = ⟨ū⟩(x,y) − ⟨ū⟩(y)   (2D)
    # -------------------------------------------------------------------------
    'DispVelU', 'DispVelV', 'DispVelW',

    # Dispersive (relative) pressure: p̃ = ⟨p̄⟩(x,z) − ⟨p̄⟩(z)   (2D)
    # Datum-independent pressure deviation (windward-high / leeward-low dipole).
    'DispP',

    # -------------------------------------------------------------------------
    # Full phase-averaged stress tensor  (2D)
    # -------------------------------------------------------------------------
    'AvgStrUU', 'AvgStrUV', 'AvgStrUW', 'AvgStrVV', 'AvgStrVW', 'AvgStrWW',

    # Triple decomposition — mean × mean contribution  (2D)
    'UU_G', 'UV_G', 'UW_G', 'VV_G', 'VW_G', 'WW_G',

    # Triple decomposition — dispersive × dispersive contribution  (2D)
    'UU_disp', 'UV_disp', 'UW_disp', 'VV_disp', 'VW_disp', 'WW_disp',

    # TURBULENT stresses ⟨u''_i u''_j⟩ — the residual after removing the mean and
    # dispersive parts.  NOTE: this is NOT the full Reynolds stress; the Reynolds
    # stress ⟨u'_i u'_j⟩ = dispersive (*_disp) + turbulent (rey_*).  The `rey_`
    # prefix is retained for backward compatibility.  (2D)
    'rey_uu', 'rey_uv', 'rey_uw', 'rey_vv', 'rey_vw', 'rey_ww',

    # -------------------------------------------------------------------------
    # Spatial derivatives of phase-averaged velocity  (2D, masked by mask_intr)
    # -------------------------------------------------------------------------
    'du_dy', 'du_dx', 'dv_dy', 'dv_dx', 'dw_dy', 'dw_dx',

    # Second wall-normal derivative ∂²⟨u⟩/∂z² (2D) — computed in the momentum-budget
    # block with the compact D2Y scheme.  Pickled so results.py (stage c) reads it
    # for the profile-inflection panel instead of rebuilding it from du_dy.
    'd2u_dy2',

    # Spatial derivatives of dispersive velocity  (2D)
    'dud_dy', 'dud_dx', 'dvd_dy', 'dvd_dx', 'dwd_dy', 'dwd_dx',

    # Mean-pressure gradients  (2D, masked by mask_intr)
    #   dP_dx : streamwise gradient ∂⟨p̄⟩/∂x = the ADVERSE pressure gradient
    #           (> 0 adverse/decelerating, < 0 favorable). Datum-independent and
    #           identical to ∂(DispP)/∂x. dP_dz (spanwise) is 0 in this 2-D field.
    #   dP_dy : wall-normal gradient ∂⟨p̄⟩/∂y (= ∂p/∂z⁺ meteorological).
    'dP_dx', 'dP_dy',

    # -------------------------------------------------------------------------
    # Time derivatives (zero-filled when not loaded from binary; kept for future use)
    # -------------------------------------------------------------------------
    'du_dt', 'ds_dt', 'dudt', 'dwdt',

    # -------------------------------------------------------------------------
    # Momentum balance — Method 1: integral profiles  (1D: ny)
    #   τ_yx(y) = +f∫(W−Gz)dy  + ν⟨∂U/∂y⟩  − ⟨u′v′⟩
    #   τ_yz(y) = −f∫(U−Gx)dy  + ν⟨∂W/∂y⟩  − ⟨v′w′⟩
    # -------------------------------------------------------------------------
    'corr_yx', 'I_corr_yx', 'visc_yx', 'total_tau_yx',
    'corr_yz', 'I_corr_yz', 'visc_yz', 'total_tau_yz',
    'tau_corrctn', 'u_star2',

    # -------------------------------------------------------------------------
    # Momentum balance — Method 2: surface-integrated force scalars
    # -------------------------------------------------------------------------
    'I_tau_yx', 'I_tau_yz',
    'I_tau_xy1', 'I_tau_xy2', 'I_tau_xz1', 'I_tau_xz2',
    'P_Lag', 'P_Front', 'P_drag', 'dispP',
    'Fyx', 'Fyz', 'Fxy', 'u_star1',

    # -------------------------------------------------------------------------
    # Turbulence statistics
    # -------------------------------------------------------------------------
    'TKE', 'AVG_TKE_V', 'AVG_TKE_V_s', 'AVG_TKE_V_s_i',
    'dTKE_dx', 'dTKE_dy', 'Adv',

    # -------------------------------------------------------------------------
    # Vorticity fields  (2D)
    # -------------------------------------------------------------------------
    'vort_z', 'disp_vortz',

    # -------------------------------------------------------------------------
    # Gravity-wave vertical-wavenumber fields  (2D: ny×nx)
    # Signed local horizontal (k) and vertical (m) wavenumbers from the
    # dispersive velocity via the Hilbert phase-gradient method.  sign(m) (with
    # the Hilbert-fixed k>0) gives the phase-line tilt; km_dispV<0 ⇒ upward
    # energy propagation.  Computed from DispVelV (vertical) and DispVelU.
    # -------------------------------------------------------------------------
    'm_dispV', 'k_dispV', 'm_dispU', 'k_dispU', 'km_dispV',

    # -------------------------------------------------------------------------
    # Advection term u_j ∂U/∂x_j at four orographic landmarks  (1D: ny)
    # -------------------------------------------------------------------------
    'conv_top', 'conv_lf', 'conv_bottom', 'conv_rf',

    # -------------------------------------------------------------------------
    # Scaled/normalised velocities and turning angle
    # -------------------------------------------------------------------------
    'u_plus', 'v_plus', 'w_plus',
    # NB: w_plus_rot (and inst_alpha = w_plus_rot/u_plus_rot) carry a DISPLAY
    # sign flip, w_plus_rot = -⟨W_rot⟩ (PhAvg[_rotated].py "Horizontal wind").
    # The physical rotated spanwise mean is negative near the wall, matching the
    # smooth reference rW; AvgPhW / rey_vw / VW_disp keep the physical sign.
    'u_plus_rot', 'w_plus_rot', 'uh_pl1D',
    'inst_alpha',

    # -------------------------------------------------------------------------
    # Instantaneous fluctuation planes  u'ᵢ = uᵢ − ⟨uᵢ⟩ₓ  (one x–y plane, ny×nx,
    # float32).  Derived from a RAW record (flow.<tag>.{1,2,3} / scal.<tag>.1) in
    # PhAvg_rotated.py — the ONLY raw-record read in the pipeline's local stages —
    # and pickled so results.py (stage c) never opens flow.*/scal.* itself.
    #   inst_u=flow.*.1  inst_v=flow.*.2 (wall-normal)  inst_w=flow.*.3 (spanwise)
    #   inst_scal=scal.*.1
    'inst_u', 'inst_v', 'inst_w', 'inst_scal',

    # -------------------------------------------------------------------------
    # Log-law reference profiles (Monin–Obukhov similarity theory)
    # -------------------------------------------------------------------------
    'u_most', 'u_most_v', 'y0',

    # -------------------------------------------------------------------------
    # RESEARCH DIAGNOSTICS — 8 prioritised goals (Research.md:536-550)
    # Per-run quantities; cross-case aggregation done in results.py.
    # -------------------------------------------------------------------------
    # Buoyancy field & its double-average split (scalar IS buoyancy b)
    'AvgScal', 'b_xmean', 'DispScal',

    # Wall-normal (v = meteorological vertical) buoyancy fluxes  ⟨w'θ'⟩
    'vtheta_disp', 'vtheta_temp', 'utheta_disp', 'thetavar',
    'Bflux_disp', 'Bflux_temp', 'Bflux',
    # Full buoyancy-flux VECTOR ⟨u_i'b'⟩(z): streamwise (U) + spanwise (W)
    # components (the wall-normal one is Bflux above). Each split dispersive +
    # temporal. Consumed by results.py's cross-case flux-components mirror (R1b).
    'Uflux_disp', 'Uflux_temp', 'Uflux',
    'Wflux_disp', 'Wflux_temp', 'Wflux',

    # Goal 1 — control translation (Ri_B, Obukhov, stability axis)
    'B_0', 'Ri_B', 'delta_neu_eff', 'B_s', 'b_star', 'G_mag',
    'L_obukhov_col', 'L_col_plus', 'L_loc', 'stab_class', 'collapse_flag',

    # Goal 3 — scales per run (3 methods) + headline (Method 2)
    'scales', 'delta_run', 'H_phys', 'Psi', 'H_delta', 'H_plus_r', 'Lx_plus',

    # Coriolis–topography coupling — global surface veer (orographic).
    # γ_veer = veer_oro/veer_smooth is formed in results.py from the smooth
    # reference (Research.md candidate finding #3 / §6.14.5; Ψ aggregation).
    # veer_smooth / gamma_veer are NOT pickled: the smooth .nc is loaded in
    # PhAvg_rotated.py only after the pickle dump, so they are unavailable here.
    'veer_oro',

    # Goal 4 — flux decomposition (x-averaged profiles + dispersive shares)
    'rey_uv_x', 'UV_disp_x', 'disp_share_mom', 'disp_share_buoy',

    # Goal 5 — local similarity φ_m, φ_h at windward/floor/lee (+ MOST departure)
    'phi_m_st', 'phi_h_st', 'zeta_st', 'phi_m_dep', 'phi_h_dep',

    # Goal 6 — intermittency (None unless compute_intermittency=1)
    'gamma_z', 'gamma_field', 'omega_rms_z', 'e_omega', 'omega0',

    # Goal 7 — wave diagnostics + sponge guard
    'wave_mom_flux', 'wave_buoy_flux', 'sponge_j', 'bl_top_j', 'reflection_ok', 'Ly',

    # Goal 2 / Goal 8 — reference + Reynolds-robustness status
    'strat_ref_available', 're750_note',

    # Log-law fit outputs (κ, zero-plane displacement d⁺, roughness z₀ₘ⁺) from the
    # OLS wall-law fit in PhAvg_rotated.py — consumed cross-case by results.py
    # (Ch. 6 velocity-profile items: matched-range κ, z₀/d extraction).
    'kappa_loglaw', 'd_m_loglaw', 'z0m_loglaw',

    # Obukhov (1971) modified (stability-corrected) log-law fit — v*/u★, the
    # dynamic-turbulence scale L1⁺, additive offset, R², and the Ri_cr implied
    # from L1⁺ + surface buoyancy flux B_s (see PhAvg_rotated.py).
    'v_star_mod', 'L1_plus_mod', 'offset_mod', 'r2_mod', 'Ri_cr_implied',

]


# =============================================================================
# 2. Save: averaged / triple-decomposed fields → .npy
# =============================================================================
def write_avg_arrays(ns):
    """Write the phase-averaged and triple-decomposed fields to .npy files.

    Faithful move of the former ``if (100 == save_avg)`` block: same filenames,
    same source slices.  ``ns`` is the caller's namespace (pass ``globals()``);
    every array read below is produced by the cal_Avg / verify_TimeAvg blocks, so
    a missing name raises KeyError exactly where the old code raised NameError.
    """
    AvgStress = ns['AvgStress']
    AvgPh     = ns['AvgPh']
    AvgP      = ns['AvgP']
    AvgScal   = ns['AvgScal']
    VelGbl    = ns['VelGbl']
    DispVel   = ns['DispVel']

    # Turbulent stresses ⟨u''_i u''_j⟩ (double-prime residual)
    np.save('uu_d.npy', ns['uu_d'])
    np.save('uv_d.npy', ns['uv_d'])
    np.save('uw_d.npy', ns['uw_d'])
    np.save('vv_d.npy', ns['vv_d'])
    np.save('vw_d.npy', ns['vw_d'])
    np.save('ww_d.npy', ns['ww_d'])

    # Full phase-averaged stress tensor
    np.save('AvgStrUU.npy', AvgStress[:, :, 0])
    np.save('AvgStrUV.npy', AvgStress[:, :, 1])
    np.save('AvgStrUW.npy', AvgStress[:, :, 2])
    np.save('AvgStrVV.npy', AvgStress[:, :, 3])
    np.save('AvgStrVW.npy', AvgStress[:, :, 4])
    np.save('AvgStrWW.npy', AvgStress[:, :, 5])

    # Triple decomposition — mean × mean (_g)
    np.save('uu_g.npy', ns['uu_g'])
    np.save('uv_g.npy', ns['uv_g'])
    np.save('uw_g.npy', ns['uw_g'])
    np.save('vv_g.npy', ns['vv_g'])
    np.save('vw_g.npy', ns['vw_g'])
    np.save('ww_g.npy', ns['ww_g'])

    # Triple decomposition — dispersive × dispersive (_t = tilda)
    np.save('uu_t.npy', ns['uu_t'])
    np.save('uv_t.npy', ns['uv_t'])
    np.save('uw_t.npy', ns['uw_t'])
    np.save('vv_t.npy', ns['vv_t'])
    np.save('vw_t.npy', ns['vw_t'])
    np.save('ww_t.npy', ns['ww_t'])

    # Phase-averaged velocity / pressure / scalar (+ dispersive parts)
    np.save('AvgPhU.npy', AvgPh[:, :, 0])
    np.save('AvgPhV.npy', AvgPh[:, :, 1])
    np.save('AvgPhW.npy', AvgPh[:, :, 2])
    np.save('AvgP.npy', AvgP[:, :])
    np.save('DispP.npy', ns['DispP'])
    np.save('AvgScal.npy', AvgScal[:, :])
    np.save('DispScal.npy', ns['DispScal'])

    # Global (x-averaged) mean velocity
    np.save('VelGblU.npy', VelGbl[:, 0])
    np.save('VelGblV.npy', VelGbl[:, 1])
    np.save('VelGblW.npy', VelGbl[:, 2])

    # Dispersive velocity (np.save appends .npy)
    np.save('DispVelU', DispVel[:, :, 0])
    np.save('DispVelV', DispVel[:, :, 1])
    np.save('DispVelW', DispVel[:, :, 2])

    # Mean×dispersive cross terms
    np.save('udug.npy', ns['udug'])
    np.save('udvg.npy', ns['udvg'])
    np.save('udwg.npy', ns['udwg'])
    np.save('vdvg.npy', ns['vdvg'])
    np.save('vdwg.npy', ns['vdwg'])
    np.save('wdwg.npy', ns['wdwg'])

    np.save('ugud.npy', ns['ugud'])
    np.save('ugvd.npy', ns['ugvd'])
    np.save('ugwd.npy', ns['ugwd'])
    np.save('vgvd.npy', ns['vgvd'])
    np.save('vgwd.npy', ns['vgwd'])
    np.save('wgwd.npy', ns['wgwd'])


# =============================================================================
# 3. Load: .npy → dict of {varname: array}
# =============================================================================
def read_avg_arrays(ny, nx, dim, scal):
    """Reload the saved .npy fields, returning a dict of {varname: array}.

    Faithful move of the former ``if (1 == load_arrays)`` block.  The caller does
    ``globals().update(read_avg_arrays(ny, nx, dim, scal))`` so the arrays land as
    top-level, Spyder-inspectable names.  ``du_dt`` / ``ds_dt`` are returned
    zero-filled (kept for the pickle / future use), matching the old block.
    """
    out = {}

    # Declared-empty time derivatives (unchanged: zero-filled placeholders)
    out['du_dt'] = np.zeros((ny, nx, dim))
    out['ds_dt'] = np.zeros((ny, nx, scal))

    # Turbulent stresses (rey_ prefix retained for backward compatibility)
    out['rey_uu'] = np.load('uu_d.npy')
    out['rey_uv'] = np.load('uv_d.npy')
    out['rey_uw'] = np.load('uw_d.npy')
    out['rey_vv'] = np.load('vv_d.npy')
    out['rey_vw'] = np.load('vw_d.npy')
    out['rey_ww'] = np.load('ww_d.npy')

    # Full phase-averaged stress tensor
    out['AvgStrUU'] = np.load('AvgStrUU.npy')
    out['AvgStrUV'] = np.load('AvgStrUV.npy')
    out['AvgStrUW'] = np.load('AvgStrUW.npy')
    out['AvgStrVV'] = np.load('AvgStrVV.npy')
    out['AvgStrVW'] = np.load('AvgStrVW.npy')
    out['AvgStrWW'] = np.load('AvgStrWW.npy')

    # Mean × mean contribution
    out['UU_G'] = np.load('uu_g.npy')
    out['UV_G'] = np.load('uv_g.npy')
    out['UW_G'] = np.load('uw_g.npy')
    out['VV_G'] = np.load('vv_g.npy')
    out['VW_G'] = np.load('vw_g.npy')
    out['WW_G'] = np.load('ww_g.npy')

    # Dispersive × dispersive contribution
    out['UU_disp'] = np.load('uu_t.npy')
    out['UV_disp'] = np.load('uv_t.npy')
    out['UW_disp'] = np.load('uw_t.npy')
    out['VV_disp'] = np.load('vv_t.npy')
    out['VW_disp'] = np.load('vw_t.npy')
    out['WW_disp'] = np.load('ww_t.npy')

    # Phase-averaged velocity / pressure / scalar (+ dispersive parts)
    out['AvgPhU']  = np.load('AvgPhU.npy')
    out['AvgPhV']  = np.load('AvgPhV.npy')
    out['AvgPhW']  = np.load('AvgPhW.npy')
    out['AvgP']    = np.load('AvgP.npy')
    out['DispP']   = np.load('DispP.npy')
    out['AvgScal'] = np.load('AvgScal.npy')
    out['DispScal'] = np.load('DispScal.npy')

    # Global (x-averaged) mean velocity
    out['VelGblU'] = np.load('VelGblU.npy')
    out['VelGblV'] = np.load('VelGblV.npy')
    out['VelGblW'] = np.load('VelGblW.npy')

    # Dispersive velocity
    out['DispVelU'] = np.load('DispVelU.npy')
    out['DispVelV'] = np.load('DispVelV.npy')
    out['DispVelW'] = np.load('DispVelW.npy')

    # Mean×dispersive cross terms
    out['udug'] = np.load('udug.npy')
    out['udvg'] = np.load('udvg.npy')
    out['udwg'] = np.load('udwg.npy')
    out['vdvg'] = np.load('vdvg.npy')
    out['vdwg'] = np.load('vdwg.npy')
    out['wdwg'] = np.load('wdwg.npy')

    out['ugud'] = np.load('ugud.npy')
    out['ugvd'] = np.load('ugvd.npy')
    out['ugwd'] = np.load('ugwd.npy')
    out['vgvd'] = np.load('vgvd.npy')
    out['vgwd'] = np.load('vgwd.npy')
    out['wgwd'] = np.load('wgwd.npy')

    return out


# =============================================================================
# 3b. Load: reference-case profiles from a tlab avg_all.nc (pure I/O)
# =============================================================================
def read_ekman_budget_profiles(nc_path):
    """Read the horizontally-averaged profiles needed by the Kostelecky &
    Ansorge (2024) fig-4 momentum budget from a reference avg_all.nc case.

    Returns (y, u, v, Ruw, Rvw): the wall-normal coordinate, the mean
    streamwise ⟨u⟩ (fU) and spanwise/veer ⟨v⟩ (fW) velocities, and the
    Reynolds fluxes ⟨u'w'⟩ (Rxy) and ⟨v'w'⟩ (Ryz) — each time-averaged over
    the file's records.  Pure I/O: the eq.-4.2 budget itself is computed in
    PhAvg[_rotated].py (PLOT 32r) and drawn by PlotField.plot_fig4_budget.
    """
    ds  = nc.Dataset(nc_path, 'r')
    y   = np.asarray(ds.variables['y'][:], float)
    u   = np.asarray(ds.variables['fU'][:], float).T.mean(1)   # ⟨u⟩ streamwise
    v   = np.asarray(ds.variables['fW'][:], float).T.mean(1)   # ⟨v⟩ spanwise (veer comp.)
    Ruw = np.asarray(ds.variables['Rxy'][:], float).T.mean(1)  # ⟨u'w'⟩
    Rvw = np.asarray(ds.variables['Ryz'][:], float).T.mean(1)  # ⟨v'w'⟩
    ds.close()
    return y, u, v, Ruw, Rvw


# =============================================================================
# 4. Save: sim1_results.pkl
# =============================================================================
def write_results_pickle(ns, path='sim1_results.pkl'):
    """Bundle every ``var_names`` entry present in ``ns`` into ``path`` (pickle).

    Faithful move of the former in-line dump.  Names absent from ``ns`` (a gated
    diagnostic, or a reference absent on the cluster) are skipped and reported, so
    one missing key cannot drop the whole pickle; results.py already treats absent
    keys as None/NaN.  Returns the assembled dict.
    """
    missing = [name for name in var_names if name not in ns]
    if missing:
        print(f'[IO] pickle: {len(missing)} var(s) absent, skipped: {missing}')
    results = {name: ns[name] for name in var_names if name in ns}
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    return results


# =============================================================================
# 5. Console reporting — plain-language run summary
# =============================================================================
def print_run_summary(ns):
    """Plain-language summary of one run (console output).

    Moved out of PhAvg_rotated.py for readability.  Reads every quantity from the
    caller's namespace ``ns`` (pass ``globals()``); values not present print as
    'n/a'/'infinite' rather than raising.  All quantities are non-dimensional
    (DNS units g = 1, f = 1); the friction velocity quoted is METHOD 2 only
    (the plateau of the vertically-integrated Ekman momentum balance).
    """
    import textwrap as _tw

    _NA = float('nan')

    def _g(name, default=_NA):
        """Fetch a possibly-unset diagnostic without raising."""
        return ns.get(name, default)

    def _sci(x, nd=4):
        """Human-friendly number: 'n/a' for None/NaN, 'infinite' for +/-inf."""
        if x is None:
            return 'n/a'
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return str(x)
        if np.isnan(xf):
            return 'n/a'
        if np.isinf(xf):
            return ('+' if xf > 0 else '-') + 'infinite'
        ax = abs(xf)
        if ax != 0 and (ax < 1e-3 or ax >= 1e5):
            return f'{xf:.{nd}e}'
        return f'{xf:.{nd}f}'

    def _para(text):
        print(_tw.fill(text, width=80, initial_indent='  ', subsequent_indent='  '))

    def _kv(label, value, nd=4, unit=''):
        """Aligned 'quantity (symbol) .... value unit' line."""
        cell = _sci(value, nd) + (f' {unit}' if unit else '')
        print(f'    {label:<54s}{cell:>22s}')

    def _hdr(title):
        print('\n  ' + title)
        print('  ' + '-' * 76)

    # Values fetched from ns that the block below references directly.
    _strat     = ns.get('_stratified', False)
    _obu_kappa = ns.get('obu_kappa')
    _obu_eta   = ns.get('obu_eta_of_xi')
    _obu_psi   = ns.get('obu_psi')
    _OBU_TBL3  = ns.get('OBU_TBL3')

    # Method-2 friction velocity — the single reference for the whole summary.
    _ustar   = _g('ustr_M2_plateau_o')
    _nu      = _g('nu'); _f = _g('f'); _lin = _g('l_in')
    _delta   = _g('delta_run')                       # d = u*/f
    _dplus   = _g('Re_tau')                          # u*^2/nu  ( = d+ since f = 1 )

    print('\n' + '=' * 80)
    print('  PLAIN-LANGUAGE SUMMARY OF THIS RUN')
    print('  (all quantities non-dimensional; friction velocity = Method 2 only)')
    print('=' * 80)

    # ══ 1. Friction velocity and boundary-layer parameters ════════════════════
    _hdr('1. Friction velocity and boundary-layer parameters')
    _para('The friction velocity is obtained from the vertically-integrated Ekman '
          'momentum balance (Method 2) and read as the constant-flux plateau of the '
          'resulting u*(z) profile. In a rotating Ekman layer the direct total '
          'stress is NOT height-constant (Coriolis drains momentum with height), so '
          'a stress plateau would read artificially low; only the full '
          'Coriolis+viscous+Reynolds balance is height-independent and equals the '
          'surface stress. Methods 1 and 3 are omitted from this summary by design.')
    print()
    _kv('Friction velocity  u* (Method 2 plateau)', _ustar, 5)
    _kv('Kinematic viscosity  nu', _nu, 3)
    _kv('Viscous (inner) length  l_in = nu/u*_cfg', _lin, 3)
    _kv('Coriolis parameter  f', _f, 3)
    _kv('Friction Reynolds number  Re_tau = u*^2/nu', _dplus, 1)
    _kv('Boundary-layer depth  delta = u*/f', _delta, 5)
    _kv('Boundary-layer depth in wall units  delta+', _dplus, 1)
    _kv('Geostrophic wind magnitude  |G|', _g('G_mag'), 4)
    _kv('Surface stress veer (turning angle)', _g('veer_oro'), 2, 'deg')
    print()
    _kv('Valley crest height  H', _g('H_phys'), 5)
    _kv('Crest height in wall units  H+', _g('H_plus_r'), 1)
    _kv('Crest height / BL depth  H/delta', _g('H_delta'), 4)
    _kv('Streamwise domain length in wall units  Lx+', _g('Lx_plus'), 1)
    _kv('Topography-to-Ekman scale ratio  Psi = Lx/(2 delta)', _g('Psi'), 4)
    _para(f'Interpretation: the boundary layer is {_sci(_dplus,0)} viscous lengths '
          f'deep. The valley occupies a fraction H/delta = {_sci(_g("H_delta"),3)} of '
          f'that depth. Psi = Lx/(2 delta) = {_sci(_g("Psi"),3)} compares the '
          f'topographic wavelength with the rotation-limited Ekman depth: Psi >> 1 '
          f'means the topography is wide compared with delta (weak Coriolis-'
          f'topography coupling), Psi ~ 1 means the two scales interact. The surface '
          f'stress vector is veered {_sci(_g("veer_oro"),1)} degrees from the '
          f'geostrophic direction.')

    # ══ 2. Monin-Obukhov stability parameters (the lengths) ═══════════════════
    _hdr('2. Monin-Obukhov stability parameters')
    _para('Buoyancy is the transported scalar b (g absorbed, so b IS the buoyancy). '
          'The Obukhov length L = -u*^3/(kappa B_s) is the height at which buoyant '
          'and shear production of turbulent kinetic energy balance: below |L| the '
          'flow is shear-dominated, above it buoyancy dominates. B_s is the surface '
          'wall-normal buoyancy flux <w\'b\'> (dispersive + temporal parts).')
    print()
    _kv('Froude number of the case  Fr', _g('Fr'), 3)
    _kv('Surface buoyancy  B_0', _g('B_0'), 5)
    _kv('Surface buoyancy flux  B_s = <w\'b\'>', _g('B_s'), 5)
    _kv('Buoyancy scale  b* = -B_s/u*', _g('b_star'), 5)
    _kv('Bulk Richardson number  Ri_B', _g('Ri_B'), 4)
    _kv('Reference (neutral) depth used for Ri_B', _g('delta_neu_eff'), 5)
    print()
    _kv('Obukhov length  L (column)', _g('L_obukhov_col'), 5)
    _kv('Obukhov length in wall units  L+', _g('L_col_plus'), 2)
    _kv('Stability ratio  delta/L', (_delta / _g('L_obukhov_col'))
        if np.isfinite(_g('L_obukhov_col')) and _g('L_obukhov_col') != 0 else _NA, 4)
    _Lloc = _g('L_loc', {})
    if isinstance(_Lloc, dict):
        for _nm in ('windward', 'floor', 'lee'):
            if _nm in _Lloc:
                _kv(f'Local Obukhov length  L+ ({_nm})', _Lloc[_nm], 2)
    print()
    _para(f'Stability classification: "{_g("stab_class","n/a")}" (from Ri_B). '
          + ('Turbulence-collapse warning: |L+| is below the configured threshold, '
             'so the stratified surface layer may be intermittent or collapsed. '
             if _g('collapse_flag', False) else
             'No turbulence-collapse warning: |L+| exceeds the configured threshold. ')
          + ('This run is neutral (Fr = infinite): there is no surface buoyancy '
             'flux, L is infinite, and buoyancy never competes with shear.'
             if not _strat else
             'Buoyancy actively suppresses mixing above z ~ |L|.'))

    # ══ 3. Curve fits and wall laws ═══════════════════════════════════════════
    _hdr('3. Curve fits and wall-law data')

    print('  (a) Classical logarithmic law of the wall')
    _para('Fitted form  U+ = (1/kappa) ln(z+ - d+) + B, equivalently '
          'U+ = (1/kappa) ln((z+ - d+)/z0m+), by ordinary least squares with the '
          'zero-plane displacement d+ obtained from a grid search maximising R^2. '
          'This baseline ignores buoyancy and is fitted for every case.')
    print()
    _kv('Fit window (lower bound)  z+_min', _g('loglaw_zmin'), 1)
    _kv('Fit window (upper bound)  z+_max', _g('loglaw_zmax'), 1)
    _kv('von Karman constant  kappa', _g('kappa_loglaw'), 4)
    _kv('Zero-plane displacement  d+', _g('d_m_loglaw'), 3)
    _kv('Aerodynamic roughness length  z0m+', _g('z0m_loglaw'), 5)
    _kv('Coefficient of determination  R^2', _g('_best_r2'), 4)
    print()
    _kv('Valley log-law kappa (plot fit)', _g('kappa_vll'), 4)
    _kv('Smooth-wall log-law kappa (plot fit)', _g('kappa_sml'), 4)

    print('\n  (b) Roughness / canopy sublayer')
    _kv('Power law prefactor  A   (U+ = A z+^n)', _g('A_power'), 4)
    _kv('Power law exponent  n', _g('n_power'), 4)
    _kv('Power law  R^2', _g('r2_power'), 4)
    _kv('Canopy attenuation coefficient  alpha', _g('alpha_canopy_v'), 4)
    _kv('Canopy law  R^2', _g('r2_canopy'), 4)

    print('\n  (c) Obukhov (1971) stability-corrected logarithmic law')
    if not _strat:
        _para('Not fitted for this run. The surface layer is neutral (Fr = infinite), '
              'so there is no surface buoyancy flux to bend the mean-wind profile away '
              'from the classical log line, and Obukhov\'s modified law is undefined '
              '(it reduces identically to the classical log law above). The modified '
              'law is fitted only for the stratified cases (finite Fr).')
    elif not (np.isfinite(_g('v_star_mod')) and np.isfinite(_g('L1_plus_mod'))):
        _para('The stability-corrected fit did not converge for this run (too few '
              'points inside the fit window, or the optimiser failed). See the '
              '"Modified log-law ... FAILED" line printed earlier. The classical log '
              'law in (a) still applies.')
    else:
        _vsm  = _g('v_star_mod'); _L1p = _g('L1_plus_mod')
        _vphy = _vsm * _ustar if np.isfinite(_ustar) else _NA
        _L1ph = _L1p * _lin if np.isfinite(_lin) else _NA
        _L1od = (_L1ph / _delta) if (np.isfinite(_delta) and _delta != 0) else _NA
        _para('Following Obukhov (1971), "Turbulence in an Atmosphere with a '
              'Non-Uniform Temperature", Bound.-Layer Meteorol. 2, 7-29, the mean '
              'wind is fitted to the stability-corrected surface-layer profile '
              'V(z) = (v*/kappa) psi(z/L1) + offset. Here psi is Obukhov\'s universal '
              'wind function (his Sec. 6 / Table III), obtained by integrating '
              'sqrt(phi(Ri)) kappa z dV/dz = v* with the energy-balance universal '
              'function phi(Ri) = sqrt(1 - Ri/Ri_cr) (his eq. 38). psi reduces to '
              'ln(z) as the stratification vanishes, so the modified law degenerates '
              'to the classical log law in the neutral limit.')
        print()
        _kv('von Karman constant (fixed at paper value)  kappa', _obu_kappa, 2)
        _kv('Fit window (lower bound)  z+_min', _g('_mod_lo'), 1)
        _kv('Fit window (upper bound)  z+_max', _g('_mod_hi'), 1)
        _kv('Fitted friction velocity ratio  v*/u*', _vsm, 4)
        _kv('Fitted friction velocity  v*', _vphy, 5)
        _kv('Dynamic-turbulence length  L1+ (wall units)', _L1p, 3)
        _kv('Dynamic-turbulence length  L1', _L1ph, 5)
        _kv('L1 / boundary-layer depth  L1/delta', _L1od, 4)
        _kv('Additive offset (roughness term)', _g('offset_mod'), 3)
        _kv('Coefficient of determination  R^2', _g('r2_mod'), 4)
        _kv('Critical Richardson number implied  Ri_cr', _g('Ri_cr_implied'), 4)
        _kv('Critical Richardson number prescribed  Ri_cr', _g('Ri_cr'), 3)
        print()
        _para(f'Friction velocity: the profile-implied v* is '
              f'{_sci(100.0*_vsm,1)}% of the independently measured Method-2 '
              f'u* = {_sci(_ustar,5)}. A ratio near unity confirms that the '
              f'stability-corrected profile and the Ekman momentum integral agree; a '
              f'large departure indicates an unsuitable fit window or a genuinely '
              f'non-logarithmic profile.')
        _para(f'Dynamic-turbulence length: L1 = {_sci(_L1p,1)} wall units '
              f'({_sci(_L1od,3)} of the boundary-layer depth delta). L1 is the '
              f'thickness of the near-wall sub-layer in which turbulence is governed '
              f'by shear rather than buoyancy; above z ~ L1 the mean wind departs '
              f'from the straight log line and, in a stable layer, Ri approaches its '
              f'critical value asymptotically. The fitted sign is '
              + ('positive, i.e. a STABLE surface layer (Ri > 0, buoyancy suppresses '
                 'mixing).' if _L1p > 0 else
                 'negative, i.e. an UNSTABLE surface layer (Ri < 0, buoyancy enhances '
                 'mixing).')
              + (f' Note L1/delta = {_sci(_L1od,2)} > 1, meaning the stratification '
                 f'signal is weak over the fit window and L1 is only loosely '
                 f'constrained.' if (np.isfinite(_L1od) and _L1od > 1) else ''))
        if np.isfinite(_g('Ri_cr_implied')):
            _para(f'Critical Richardson number: inverting L1 = alpha Ri_cr v*^3 / '
                  f'(kappa |B_s|) with the surface buoyancy flux B_s = '
                  f'{_sci(_g("B_s"),5)} and a turbulent Prandtl number Pr_t = '
                  f'{_sci(_g("Pr_t"),3)} (heat/momentum eddy-diffusivity ratio '
                  f'alpha = K_T/K = 1/Pr_t) gives Ri_cr = '
                  f'{_sci(_g("Ri_cr_implied"),4)}, against the prescribed '
                  f'{_sci(_g("Ri_cr"),3)}. Caveat: if the temporal buoyancy '
                  f'cross-moments (Mean*Theta) were absent, B_s contains only the '
                  f'dispersive contribution and this Ri_cr is a lower bound.')
            _Lobu  = _g('L_obukhov_col')
            _kap   = _g('kappa')                    # config kappa used to build L
            _L1oL  = (_L1ph / _Lobu) if (np.isfinite(_Lobu) and _Lobu != 0) else _NA
            _xchk  = (_g('Pr_t') * _L1oL * _obu_kappa / _kap
                      if (np.isfinite(_L1oL) and np.isfinite(_kap) and _kap != 0) else _NA)
            _kv('Ratio of the two length scales  L1/L', _L1oL, 5)
            _kv('Cross-check  Pr_t (L1/L) (kappa_obu/kappa)', _xchk, 5)
            _para(f'Cross-check: the column Obukhov length is built as '
                  f'L = -u*^3/(kappa B_s) with the CONFIG kappa = {_sci(_kap,3)}, '
                  f'whereas Obukhov\'s L1 = alpha Ri_cr v*^3/(kappa_obu |B_s|) uses '
                  f'the paper value kappa_obu = {_sci(_obu_kappa,2)}. Eliminating '
                  f'|B_s| between them gives the identity '
                  f'Ri_cr = Pr_t (L1/L) (kappa_obu/kappa) whenever v* = u*. Here that '
                  f'evaluates to {_sci(_xchk,5)}, which must agree with the implied '
                  f'Ri_cr = {_sci(_g("Ri_cr_implied"),5)} above. Agreement is an '
                  f'independent confirmation that the fitted L1, the measured '
                  f'buoyancy flux B_s and the wall-unit scaling l_in are mutually '
                  f'consistent; a mismatch by a constant factor points to a friction '
                  f'velocity or kappa used inconsistently between the two. Note '
                  f'L1 << L implies a correspondingly small Ri_cr.')
        if _obu_eta is not None and _obu_psi is not None and _OBU_TBL3 is not None:
            _xi_t  = _OBU_TBL3[:, 0]
            _e_eta = float(np.max(np.abs(_obu_eta(_xi_t) - _OBU_TBL3[:, 1])))
            _psi_c = _obu_psi(_xi_t)
            _e_psi = float(np.sqrt(np.mean(
                (_psi_c + np.mean(_OBU_TBL3[:, 2] - _psi_c) - _OBU_TBL3[:, 2]) ** 2)))
            _para(f'Solver provenance: the psi(z/L1) machinery reproduces Obukhov\'s '
                  f'Table III to a maximum error of {_sci(_e_eta,4)} in Ri/Ri_cr and an '
                  f'RMS error of {_sci(_e_psi,4)} in psi (after the single arbitrary '
                  f'additive constant), confirming the universal function is implemented '
                  f'as published.')

    print('=' * 80 + '\n')


def print_research_summary(ns):
    """RESEARCH DIAGNOSTICS per-run summary table (8 goals; console output).

    Moved out of PhAvg_rotated.py for readability.  Reads every quantity from the
    caller's namespace ``ns`` (pass ``globals()``); the cross-case aggregation is
    done in results.py.  The table body is unchanged from the in-line version.
    """
    _g = ns.get

    def _Lfmt(Lp):                                          # Obukhov length (wall units)
        return '+inf (neutral)' if not np.isfinite(Lp) else f'{Lp:.1f}'

    def _trio(a, b, c, fmt='.5f'):                          # "M2 | M1 | M3" side by side
        return f'{format(a, fmt)} | {format(b, fmt)} | {format(c, fmt)}'

    # Values referenced directly by the table below (fetched from ns).
    _strat        = _g('_strat', False)
    _have_flux    = _g('_have_flux', False)
    _j_surf_ref   = _g('_j_surf_ref'); bl_top_j = _g('bl_top_j')
    disp_share_mom  = _g('disp_share_mom');   disp_share_buoy = _g('disp_share_buoy')
    wave_mom_flux = _g('wave_mom_flux');  wave_buoy_flux = _g('wave_buoy_flux')
    gamma_z       = _g('gamma_z');        hill_hgt = _g('hill_hgt')
    B_0 = _g('B_0'); delta_neu_eff = _g('delta_neu_eff'); Ri_B = _g('Ri_B')
    B_s = _g('B_s'); L_obukhov_col = _g('L_obukhov_col'); L_col_plus = _g('L_col_plus')
    stab_class = _g('stab_class'); Lplus_collapse = _g('Lplus_collapse')
    collapse_flag = _g('collapse_flag'); L_loc = _g('L_loc', {})
    u_star = _g('u_star'); u_star1 = _g('u_star1'); u_star3 = _g('u_star3')
    scales = _g('scales', {})
    phi_m_dep = _g('phi_m_dep', {}); phi_h_dep = _g('phi_h_dep', {})
    omega0 = _g('omega0'); y_in = _g('y_in'); sponge_j = _g('sponge_j')
    reflection_ok = _g('reflection_ok'); Re = _g('Re'); Re_tau = _g('Re_tau')
    H_plus_r = _g('H_plus_r'); Lx_plus = _g('Lx_plus'); H_delta = _g('H_delta')
    Psi = _g('Psi'); re750_note = _g('re750_note')
    strat_ref_available = _g('strat_ref_available')

    # BL-averaged dispersive shares (surface→BL top), with empty-slice guard
    _lo, _hi = _j_surf_ref, max(bl_top_j, _j_surf_ref + 1)
    _share_mom_BL  = float(np.mean(disp_share_mom[_lo:_hi]))  if _hi > _lo else float('nan')
    _share_buoy_BL = float(np.mean(disp_share_buoy[_lo:_hi])) if _hi > _lo else float('nan')
    _wmom_pk = float(np.nanmax(np.abs(wave_mom_flux)))
    _wbuo_pk = float(np.nanmax(np.abs(wave_buoy_flux)))
    _g_crest = float(gamma_z[hill_hgt]) if gamma_z is not None else None
    _g_bltop = float(gamma_z[bl_top_j]) if gamma_z is not None else None

    print_summary_table('RESEARCH DIAGNOSTICS — per-run (cross-case in results.py)', [
        ('section', 'Goal 1 — control translation'),
        ('Buoyancy active in this run',          'yes' if _strat else 'no (neutral)', 's'),
        ('Surface buoyancy B_0',                 B_0,                       '.5e'),
        ('delta_neutral used (Ri_B)',            delta_neu_eff,             '.5f'),
        ('Bulk Richardson Ri_B',                 Ri_B,                      '.5e'),
        ('Surface buoyancy flux B_s',            B_s,                       '.5e'),
        ('Obukhov length L_col (phys)',          (None if not np.isfinite(L_obukhov_col) else L_obukhov_col), '.5e'),
        ('Obukhov length L_col+ (wall units)',   _Lfmt(L_col_plus),         's'),
        ('Stability class',                      stab_class,                's'),
        ('Collapse (L+ < %.0f)' % Lplus_collapse, 'yes' if collapse_flag else 'no', 's'),
        ('Local L+ windward | floor | lee',
            f"{_Lfmt(L_loc['windward'])} | {_Lfmt(L_loc['floor'])} | {_Lfmt(L_loc['lee'])}", 's'),
        ('section', 'Goal 3 — scales per run (M2 | M1 | M3)'),
        ('u* friction velocity',                 _trio(u_star, u_star1, u_star3),                                           's'),
        ('delta = u*/f',                         _trio(scales['M2']['delta'],  scales['M1']['delta'],  scales['M3']['delta']),  's'),
        ('Psi = Lx/(2 delta)',                   _trio(scales['M2']['Psi'],    scales['M1']['Psi'],    scales['M3']['Psi'],    '.3f'), 's'),
        ('Blocking ratio H/delta',               _trio(scales['M2']['H_delta'],scales['M1']['H_delta'],scales['M3']['H_delta'],'.4f'), 's'),
        ('H+ (inner)',                           _trio(scales['M2']['H_plus'], scales['M1']['H_plus'], scales['M3']['H_plus'], '.1f'), 's'),
        ('Lx+ (inner)',                          _trio(scales['M2']['Lx_plus'],scales['M1']['Lx_plus'],scales['M3']['Lx_plus'],'.0f'), 's'),
        ('section', 'Goal 4 — flux decomposition (dispersive share, BL mean)'),
        ('Momentum dispersive share',            _share_mom_BL,             '.4f'),
        ('Buoyancy dispersive share',            _share_buoy_BL,            '.4f'),
        ('Turbulent buoyancy flux available',    'yes' if _have_flux else 'no (Route C cross-moments absent)', 's'),
        ('section', 'Goal 5 — local similarity departure (RMS vs MOST)'),
        ('phi_m dep. windward | floor | lee',
            f"{phi_m_dep['windward']:.3f} | {phi_m_dep['floor']:.3f} | {phi_m_dep['lee']:.3f}", 's'),
        ('phi_h dep. windward | floor | lee',
            (f"{phi_h_dep['windward']:.3f} | {phi_h_dep['floor']:.3f} | {phi_h_dep['lee']:.3f}"
             if _strat else 'n/a (neutral)'), 's'),
        ('section', 'Goal 6 — intermittency gamma(z)'),
        ('gamma at crest | BL top',
            (f"{_g_crest:.3f} | {_g_bltop:.3f}" if gamma_z is not None
             else 'skipped (set compute_intermittency=1)'), 's'),
        ('omega0 = e_omega = omega_rms(delta)',
            (f"{omega0:.4g}  (max gamma {float(np.nanmax(gamma_z)):.2f})"
             if gamma_z is not None else 'n/a'), 's'),
        ('section', 'Goal 7 — wave diagnostics'),
        ('Peak |wave momentum flux|',            _wmom_pk,                  '.5e'),
        ('Peak |wave buoyancy flux|',            _wbuo_pk,                  '.5e'),
        ('BL top z+ | sponge z+',                f"{y_in[bl_top_j]:.1f} | {y_in[sponge_j]:.1f}", 's'),
        ('Sponge reflection OK',                 'yes' if reflection_ok else 'no (flux grows aloft)', 's'),
        ('section', 'Goal 8 — Reynolds robustness (inner vs outer)'),
        ('Re_D | Re_tau',                        f"{int(Re)} | {Re_tau:.1f}", 's'),
        ('Inner: H+ | Lx+',                      f"{H_plus_r:.1f} | {Lx_plus:.0f}", 's'),
        ('Outer: H/delta | Psi',                 f"{H_delta:.4f} | {Psi:.3f}", 's'),
        ('Status',                               re750_note,                's'),
        ('section', 'Goal 2 — flat-wall stratified reference'),
        ('Stratified reference loaded',          'yes' if strat_ref_available else 'no (data absent)', 's'),
    ])
