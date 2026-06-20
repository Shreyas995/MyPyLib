#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 16:38:21 2026

@author: shreyad95
"""
import numpy as np

###############################################################################
############################# Varaible decleration ############################
################################# EKMAN18 #####################################
limity = 700
hill_hgt = 94 # This value is dummy. True value is calculated after reading the eps field
step = 2 # Vertical step intervals of the valley in terms of grid points
Re = 500
Re_lambda = 0.5*Re*Re
nu = 1/Re_lambda
dt = 0.827E-04
index = 1
limity_range = 150
limity = 463
f = 1
alpha = -0.430511
Gx = np.cos(alpha)
Gz = -np.sin(alpha)
u_star = 0.076
kappa = 0.42
Re_tau = (u_star**2)/nu
l_visc = nu/u_star
l_in = l_visc
l_out = u_star
time_scale = 2*np.pi
restart = 500
counter = 0
wall_units = nu/u_star
scal = 1
dim = 3 

# Controls
cal_Avg = 0
verify_TimeAvg = 0
save_avg = 0
load_ncfiles = 0
load_arrays = 1
postprocess = 1
plotRes = 1
animate = 0

# Derivative cache control
#   True  -> always recompute derivatives (ignore any cached .npy)
#   False -> load cached .npy if all present, otherwise compute (and save)
recompute_derivatives = True

###############################################################################
########################### Post-processing constants #########################
###############################################################################
# Wall-normal (y) derivative schemes used in PhAvg.py (see CompactDerivatives2D).
#   DY_METHOD  : first  y-derivative  ('compact','fornberg5/7/9','compact_nu','spline')
#   D2Y_METHOD : second y-derivative  (same options)
# 'fornberg7' wins the kink benchmark on this grid (test_ddy_schemes.py); the
# second derivative defaults to the original 'compact' scheme.
DY_METHOD  = 'fornberg7'
D2Y_METHOD = 'compact'

# Ghost-cell interpolation parameters (interpolate_component)
ghost_depth  = 5
n_anchor     = 4
smooth_width = 5

# Log-law fit window (inner units z+) and physical bounds on kappa
loglaw_zmin  = 60.0
loglaw_zmax  = 200.0
kappa_bounds = (0.40, 0.44)

# Canopy layer extends to hill_hgt + this many cells
canopy_extra_cells = 20

# Smooth-wall reference case (flat, neutral, Re=500) NetCDF, relative to cwd
smooth_nc_path = 'Re500/ri00.00_re0500_2048x0192x2048_20110615_avg_all.nc'