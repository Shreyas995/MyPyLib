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

    # -------------------------------------------------------------------------
    # Full phase-averaged stress tensor  (2D)
    # -------------------------------------------------------------------------
    'AvgStrUU', 'AvgStrUV', 'AvgStrUW', 'AvgStrVV', 'AvgStrVW', 'AvgStrWW',

    # Triple decomposition — mean × mean contribution  (2D)
    'UU_G', 'UV_G', 'UW_G', 'VV_G', 'VW_G', 'WW_G',

    # Triple decomposition — dispersive × dispersive contribution  (2D)
    'UU_disp', 'UV_disp', 'UW_disp', 'VV_disp', 'VW_disp', 'WW_disp',

    # Turbulent (Reynolds) stresses — residual after decomposition  (2D)
    'rey_uu', 'rey_uv', 'rey_uw', 'rey_vv', 'rey_vw', 'rey_ww',

    # -------------------------------------------------------------------------
    # Spatial derivatives of phase-averaged velocity  (2D, masked by mask_intr)
    # -------------------------------------------------------------------------
    'du_dy', 'du_dx', 'dv_dy', 'dv_dx', 'dw_dy', 'dw_dx',

    # Spatial derivatives of dispersive velocity  (2D)
    'dud_dy', 'dud_dx', 'dvd_dy', 'dvd_dx', 'dwd_dy', 'dwd_dx',

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
    # Advection term u_j ∂U/∂x_j at four orographic landmarks  (1D: ny)
    # -------------------------------------------------------------------------
    'conv_top', 'conv_lf', 'conv_bottom', 'conv_rf',

    # -------------------------------------------------------------------------
    # Scaled/normalised velocities and turning angle
    # -------------------------------------------------------------------------
    'u_plus', 'v_plus', 'w_plus',
    'u_plus_rot', 'w_plus_rot', 'uh_pl1D',
    'inst_alpha',

    # -------------------------------------------------------------------------
    # Log-law reference profiles (Monin–Obukhov similarity theory)
    # -------------------------------------------------------------------------
    'u_most', 'u_most_v', 'y0',

]
