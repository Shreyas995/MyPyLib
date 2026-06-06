#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:31:23 2026

@author: shreyad95
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from config import *
from functions import *
import os
import re
import sys
import csv
import struct
import math
import pickle
import netCDF4 as nc
import numpy as np
from PlotField import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import simpson
from scipy.integrate import trapezoid
from scipy.stats import linregress
import matplotlib.animation as animation
from matplotlib import cm
from PIL import Image
from saveresults import *
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d


def interpolate_component(x, y, Nx, Ny, eps, field,
                          ghost_depth=5, n_anchor=4, smooth_width=5):
    """
    Mirror-ghost strategy:
      s         = first solid cell  (BC, val=0, never overwritten)
      e         = last  solid cell  (BC, val=0, never overwritten)
      s+1       = -field[s-1]   (negative mirror of adjacent fluid)
      e-1       = -field[e+1]   (negative mirror of adjacent fluid)
      s+2..e-2  = PCHIP interior, anchored on [...fluid, BC, mirror, mirror, BC, fluid...]

    PCHIP now sees a smooth sign-changing sequence around each zero, so
    it does not collapse to a flat-zero segment.
    Mirrors are only set when the solid is wide enough (e-s >= 3).
    Interior is only filled when e-s >= 4.
    """
    dy_bottom = y[1] - y[0]
    # eps[0, :] = 0  # floor row always fluid

    # ------------------------------------------------------------------
    # 1. Ghost cells below the domain
    # ------------------------------------------------------------------
    ghost_fld = np.zeros((ghost_depth, field.shape[1]))
    ghost_eps = np.zeros((ghost_depth, eps.shape[1]))
    full_fld  = np.concatenate((ghost_fld, field),   axis=0)
    full_eps  = np.concatenate((ghost_eps, eps), axis=0)

    y_ghost = np.linspace(y[0] - ghost_depth * dy_bottom, y[0] - dy_bottom, ghost_depth)
    y_full  = np.concatenate((y_ghost, y))

    full_fld_x          = full_fld.copy()
    full_fld_y          = full_fld.copy()
    interior_solid_mask = np.zeros_like(full_eps, dtype=bool)

    # ------------------------------------------------------------------
    # 2. Horizontal interpolation (X-direction)
    # ------------------------------------------------------------------
    for j in range(full_fld.shape[0]):
        row_eps = full_eps[j, :].copy()
        row_fld = full_fld[j, :].copy()
        f_min, f_max = -row_fld.max(), row_fld.max()

        # s = first solid cell (+1 shift), e = last solid cell
        starts = np.where((row_eps[:-1] == 0) & (row_eps[1:] == 1))[0] 
        ends   = np.where((row_eps[:-1] == 1) & (row_eps[1:] == 0))[0]

        if starts.size == 0:
            continue

        needs_roll = (ends.size > 0) and (starts[0] > ends[0])
        offset = 0
        if needs_roll:
            offset  = int(Nx / 2)
            row_eps = np.roll(row_eps, -offset)
            row_fld = np.roll(row_fld, -offset)
            starts  = np.where((row_eps[:-1] == 0) & (row_eps[1:] == 1))[0] + 1
            ends    = np.where((row_eps[:-1] == 1) & (row_eps[1:] == 0))[0]

        for s, e in zip(starts, ends):
            # Bounds guard: need fluid neighbours outside solid for mirrors
            if s < 1 or e + 1 >= row_fld.shape[0]:
                continue

            width = e - s  # 0 → 1 cell, 1 → 2 cells, etc.

            # --- Step 1: place mirror ghost values ---
            # Only when solid is wide enough that s+1 and e-1 are distinct
            # and do not coincide with the BC cells themselves.
            if width >= 3:                              # at least 4 solid cells
                row_fld[s + 1] = -row_fld[s - 1]      # left mirror
                row_fld[e - 1] = -row_fld[e + 1]      # right mirror

            # --- Step 2: interpolate interior s+2 .. e-2 ---
            if width < 4:                              # nothing between the mirrors
                continue

            x_fill = x[s + 2 : e - 1]                # interior indices
            if x_fill.size == 0:
                continue

            left_idx  = np.where(row_eps[:s]   == 0)[0][-n_anchor:]
            right_idx = np.where(row_eps[e+1:] == 0)[0][:n_anchor] + (e + 1)

            if left_idx.size < 1 or right_idx.size < 1:
                continue

            # Anchor order: fluid | BC=0 | mirror | ... gap ... | mirror | BC=0 | fluid
            x_anc = np.concatenate([
                x[left_idx],
                x[[s, s + 1]],          # BC then mirror
                x[[e - 1, e]],          # mirror then BC
                x[right_idx],
            ])
            f_anc = np.concatenate([
                row_fld[left_idx],
                row_fld[[s, s + 1]],    # 0, -f[s-1]
                row_fld[[e - 1, e]],    # -f[e+1], 0
                row_fld[right_idx],
            ])

            try:
                row_fld[s + 2 : e - 1] = np.clip(
                    PchipInterpolator(x_anc, f_anc)(x_fill), f_min, f_max
                )
            except Exception:
                continue

        if needs_roll:
            row_fld = np.roll(row_fld, offset)
        full_fld_x[j, :] = row_fld

    # ------------------------------------------------------------------
    # 3. Vertical interpolation (Y-direction)
    # ------------------------------------------------------------------
    for i in range(full_fld.shape[1]):
        col_eps = full_eps[:, i]
        col_fld = full_fld[:, i].copy()
        f_min, f_max = -col_fld.max(), col_fld.max()

        # s_idx: last fluid cell before solid  (col_eps[s_idx]=0, [s_idx+1]=1)
        # e_idx: last solid cell               (col_eps[e_idx]=1, [e_idx+1]=0)
        starts = np.where((col_eps[:-1] == 0) & (col_eps[1:] == 1))[0]
        ends   = np.where((col_eps[:-1] == 1) & (col_eps[1:] == 0))[0]

        for s_idx in starts:
            valid_ends = ends[ends > s_idx]
            if valid_ends.size == 0:
                continue
            e_idx = valid_ends[0]

            s = s_idx + 1   # first solid cell  (BC, val=0)
            e = e_idx       # last  solid cell  (BC, val=0)

            if s_idx < 0 or e_idx + 1 >= col_fld.shape[0]:
                continue

            width = e - s

            # --- Step 1: mirror ghost values ---
            if width >= 3:
                col_fld[s + 1] = -col_fld[s_idx]       # mirror fluid below
                col_fld[e - 1] = -col_fld[e_idx + 1]   # mirror fluid above

            # --- Step 2: interpolate interior s+2 .. e-2 ---
            if width == 2:
                if (e == s + 1):
                    col_fld[e - 1] = -col_fld[e + 1]   # mirror fluid above
                continue
            
            if width == 1:
                continue

            gap    = slice(s + 2, e - 1)
            y_fill = y_full[s + 2 : e - 1]
            if y_fill.size == 0:
                continue

            interior_solid_mask[gap, i] = True

            bot_idx = np.where(col_eps[:s]   == 0)[0][-n_anchor:]
            top_idx = np.where(col_eps[e+1:] == 0)[0][:n_anchor] + (e + 1)

            if bot_idx.size < 1 or top_idx.size < 1:
                continue

            y_anc = np.concatenate([
                y_full[bot_idx],
                y_full[[s, s + 1]],     # BC then mirror
                y_full[[e - 1, e]],     # mirror then BC
                y_full[top_idx],
            ])
            f_anc = np.concatenate([
                col_fld[bot_idx],
                col_fld[[s, s + 1]],    # 0, -f[s_idx]
                col_fld[[e - 1, e]],    # -f[e_idx+1], 0
                col_fld[top_idx],
            ])

            try:
                col_fld[gap] = np.clip(
                    PchipInterpolator(y_anc, f_anc)(y_fill), f_min, f_max
                )
            except Exception:
                continue

        full_fld_y[:, i] = col_fld

    # ------------------------------------------------------------------
    # 4. Smooth only interior solid cells; restore fluid + BC + mirror cells
    # ------------------------------------------------------------------
    for j in range(full_fld_y.shape[0]):
        if interior_solid_mask[j, :].sum() < 3:
            continue
        backup = full_fld_y[j, ~interior_solid_mask[j, :]].copy()
        full_fld_y[j, :] = uniform_filter1d(
            full_fld_y[j, :], size=smooth_width, mode='wrap'
        )
        full_fld_y[j, ~interior_solid_mask[j, :]] = backup
    
    # ------------------------------------------------------------------
    # 5. Crop ghost cells
    # ------------------------------------------------------------------
    eps_sum = np.sum(eps, axis=0).astype(int);
    eps_sum[eps_sum > 0] -= 1
    fld_x, fld_y = full_fld_x[ghost_depth:, :], full_fld_y[ghost_depth:, :]
    fld_x[eps_sum, np.arange(nx)] = 0; fld_y[eps_sum, np.arange(nx)] = 0
    return fld_x, fld_y


# def interpolate_component(x, y, Nx, Ny, eps, field,  ghost_depth=5):
#     # --- 1. Setup & Padding ---
#     dy_bottom = y[1] - y[0]

#     f_min, f_max = np.min(field), np.max(field)
#     eps[0,:] = 0
#     # 1. Vertical Ghost Cells
#     ghost_fld_rows = np.zeros((ghost_depth, field.shape[1]))
#     ghost_eps_rows = np.zeros((ghost_depth, eps.shape[1])) 
    
#     full_fld = np.concatenate((ghost_fld_rows, field), axis=0)
#     full_eps = np.concatenate((ghost_eps_rows, eps), axis=0)
    
#     # 3. Coordinate Reconstruction
#     y_ghost = np.linspace(y[0] - ghost_depth * dy_bottom, y[0] - dy_bottom, ghost_depth)
#     y_full = np.concatenate((y_ghost, y))
    
#     full_fld_x = full_fld.copy()
#     full_fld_y = full_fld.copy()

#     # --- 4. Horizontal Interpolation (X-Direction) ---
#     # Loop through each row (j) to find transitions in (i)
#     for j in range(full_fld.shape[0]):
#         row_eps = full_eps[j, :]
#         row_fld = full_fld[j, :]
#         # Find 0->1 transitions (Start of solid)
#         starts = np.where((row_eps[:-1] == 0) & (row_eps[1:] == 1))[0]
#         # Find 1->0 transitions (End of solid)
#         ends = np.where((row_eps[:-1] == 1) & (row_eps[1:] == 0))[0]
#         if starts.size > 0:
#             if starts > ends:
#                 # offset = ends[0].copy()
#                 offset =  int(Nx/2)
#                 row_eps = np.roll(row_eps, -offset)
#                 row_fld = np.roll(row_fld, -offset)
#                 starts = np.where((row_eps[:-1] == 0) & (row_eps[1:] == 1))[0]+1
#                 # Find 1->0 transitions (End of solid)
#                 ends = np.where((row_eps[:-1] == 1) & (row_eps[1:] == 0))[0]
#                 x_known = np.concat((x[:starts[0]+1], x[ends[0]:]))
#                 x_itrp = x[starts[0]+1:ends[0]]
#                 f_known = np.concat((row_fld[:starts[0]+1], row_fld[ends[0]:]))
#                 spline_func = make_interp_spline(x_known, f_known, k=3)
#                 f_itrp = spline_func(x_itrp)
#                 row_fld[starts[0]+1:ends[0]] = f_itrp
#                 row_fld = np.roll(row_fld, offset)
#                 full_fld_x[j, :] = row_fld
                
#     # --- 5. Vertical Interpolation (Y-Direction) ---
#     # Loop through each column (i) to find transitions in (j)
#     for i in range(full_fld.shape[1]):
#         col_eps = full_eps[:, i]
#         starts = np.where((col_eps[:-1] == 0) & (col_eps[1:] == 1))[0]
#         ends = np.where((col_eps[:-1] == 1) & (col_eps[1:] == 0))[0]

#         for s_idx in starts:
#             valid_ends = ends[ends > s_idx]
#             if valid_ends.size > 0:
#                 e_idx = valid_ends[0]
                
#                 idx_btm = [s_idx - 1, s_idx, s_idx + 1]
#                 idx_top = [e_idx, e_idx + 1, e_idx + 2]
                
#                 gap = slice(s_idx + 2, e_idx)
#                 gap_size = e_idx - (s_idx + 2)

#                 if gap_size > 0 and idx_btm[0] >= 0 and idx_top[-1] < full_fld.shape[0]:
#                     y_anc = y_full[idx_btm + idx_top]
#                     f_anc = full_fld[idx_btm + idx_top, i]
                    
#                     k_val = 3 if gap_size <= 2 else 5
#                     try:
#                         spline = make_interp_spline(y_anc, f_anc, k=k_val)
#                         full_fld_y[gap, i] = np.clip(spline(y_full[gap]), f_min, f_max)
#                     except: continue

#     # --- 6. Crop and Return ---
#     res_x = full_fld_x[ghost_depth:, :]
#     res_y = full_fld_y[ghost_depth:, :]
#     return res_x, res_y
                
# %%
###############################################################################
############################# Initialize ######################################

# Parameter decleration
cwd = str(os.path.dirname(__file__) + '/' )

# Read grid
x, y, z = read_grid(cwd)

nx = np.size(x)
ny = np.size(y)
nz = np.size(z)
    
try:
    eps = np.load(cwd + 'eps_save.npy')
    print('eps loaded')
except:
    print('Needed to read eps field')
    eps = epsfield()
    np.save('eps_save.npy', eps)

eps_top = int(0)         # horizontal grid position at valley top
eps_lf = int(nx/4)       # horizontal grid position at valley left flank
eps_bottom = int(nx/2)   # horizontal grid position at valley bottom
eps_rf = int(nx*0.75)    # horizontal grid position at valley right flank

eps_hgt = np.sum(eps, axis=0).astype(int)
hill_hgt = np.max(eps_hgt) - 1 # Directly take hill height from the eps field. THe real height is value -1.
# If no geomtery is created, there is 1 row where velocity is zero so we have + 1 no of eps 
eps = epsVolume(eps,ny,nx,hill_hgt)
eps_s = np.mean(eps,axis=1)
eps_f = 1 - eps_s

flk_hgt = eps_hgt[int(eps_lf)]
flk_wdt = np.where(eps_hgt == flk_hgt)[0]
lf_ind = flk_wdt[:int((len(flk_wdt))/2)]
rf_ind = flk_wdt[int((len(flk_wdt))/2):]
x_oro = x 
x_oro = np.append(0, x_oro)
x_oro = np.append(x_oro, x[-1])
dx = (2*np.pi/x[-1])
y_oro = np.round((hill_hgt/(2**1))*(1 + np.cos(dx*(x))))
y_oro = y[y_oro.astype(int)]
y_oro = np.append(0,y_oro)
y_oro = np.append(y_oro, 0)

x_oro_in = x_oro/l_in
y_oro_in = y_oro/l_in

x_in = x/l_in
y_in = y/l_in

# Forcing values in solid zero. If not it will introduce error when calculating average in x direction.
mask0 = 1 - eps
eps_g = np.zeros((ny,nx)).astype(int)
eps_g[0,:] = 1
mask_v = (eps == 1).astype(int)

# initialize cases for derivatives
case_v_itrp, case_h_itrp = diff_cases(eps,nx,ny)
case_v = case_v_itrp
case_h = case_h_itrp
case_v_g = np.reshape(case_v[:,512].astype(int),((ny,1)))

AvgPhU = np.load('AvgPhU.npy')
AvgPhV = np.load('AvgPhV.npy')

AvgPhU_i, AvgPhU_j = interpolate_component(x, y, nx, ny, eps, AvgPhU, ghost_depth=5, n_anchor=4, smooth_width=5)

AvgPhV_i, AvgPhV_j = interpolate_component(x, y, nx, ny, eps, AvgPhV, ghost_depth=5, n_anchor=4, smooth_width=5)
# AvgPhW_i, AvgPhW_j = interpolate_component(x, y, nx, ny, eps, AvgPhW, ghost_depth=5)
    
du_dy = diffu_dy(AvgPhU_j, ny, nx, case_v, y, 1)
du_dx = diffu_dx(AvgPhU_i, ny, nx, case_h, x, 1)

dv_dy = diffu_dy(AvgPhV_j, ny, nx, case_v, y, 1)
dv_dx = diffu_dx(AvgPhV_i, ny, nx, case_h, x, 1)    

# %%
limity = 350
plot2D_div_smooth(x, y[:limity], AvgPhU_i[:limity,:],'','AvgPhU_i',r'$x$',r'$z$', cwd + '/fig/' + 'AvgPhU_i' + '.png', x_oro, y_oro, 1000) 
plot2D_div_smooth(x, y[:limity], AvgPhU_j[:limity,:],'','AvgPhU_j',r'$x$',r'$z$', cwd + '/fig/' + 'AvgPhU_j' + '.png', x_oro, y_oro, 1000) 

plot2D_div_smooth(x, y[:limity], du_dy[:limity,:],'','du_dy',r'$x$',r'$z$', cwd + '/fig/' + 'du_dy' + '.png', x_oro, y_oro, 1000) 
plot2D_div_smooth(x, y[:limity], du_dx[:limity,:],'','du_dx',r'$x$',r'$z$', cwd + '/fig/' + 'du_dx' + '.png', x_oro, y_oro, 1000) 

plot2D_div_smooth(x, y[:limity], dv_dy[:limity,:],'','dv_dy',r'$x$',r'$z$', cwd + '/fig/' + 'dv_dy' + '.png', x_oro, y_oro, 1000) 
plot2D_div_smooth(x, y[:limity], dv_dx[:limity,:],'','dv_dx',r'$x$',r'$z$', cwd + '/fig/' + 'dv_dx' + '.png', x_oro, y_oro, 1000) 

# %%