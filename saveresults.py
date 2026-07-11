#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variables bundled into sim1_results.pkl by PhAvg.py.

Only variables that exist in globals() by the time the pickle is written
(i.e., after postprocess==1 completes) are listed here.
Organised by topic; names match exactly the variable names in PhAvg.py.

Changes from previous version
------------------------------
- Added  : DispVelU (was missing), mask_intr, inst_alpha, uh_pl1D,
           tau_corrctn, horiz_surfaces, left_flank_surfaces,
           right_flank_surfaces, nu, f
- Removed: epsi_e1/e2, epsi_s1/s2, epsj_e/s  (replaced by surface lists)
           I_P_Front, I_P_Lag                  (replaced by P_Lag, P_Front, P_drag)
           diff_cases, epsfield, epsVolume     (functions, not picklable data)
           pdud2D, pdvd2D, pdwd2D             (loading commented out)
           x_tmp, y_tmp1, y_tmp2              (loop temporaries)
"""

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
