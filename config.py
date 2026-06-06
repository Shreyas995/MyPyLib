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
plotRes = 0
animate = 0