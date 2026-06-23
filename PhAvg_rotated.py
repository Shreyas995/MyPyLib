# %%
# ============================================================================
# ROTATED-FRAME variant of PhAvg.py.
# The horizontal mean / dispersive velocity fields and the wall-normal Reynolds
# momentum-flux pair are rotated by `alpha` (the ~25° geostrophic tilt) AFTER the
# ghost-cell interpolation and BEFORE the derivatives, so the geostrophic wind is
# aligned with x — matching the frame of the Kostelecky & Ansorge reference .nc
# cases.  Everything else (derivatives, Method-2 budget, plots) runs unchanged.
# Outputs go to fig_rotated/ and derivative caches use a *_rot suffix, so this
# script never clobbers the unrotated PhAvg.py run.  See the "FRAME ROTATION"
# block for the exact transformation.
# ============================================================================
# Phase-averaging postprocessor for DNS of rotating (Ekman layer) flow over a sinusoidal valley.
# Computes friction velocity via two independent methods:
#   Method 1: momentum-integral balance (u_star via total_tau_yx / total_tau_yz profiles)
#   Method 2: direct surface integration of shear and pressure over the IBM body (u_star1)
import os
import sys
# ── Per-simulation bootstrap ─────────────────────────────────────────────────
# This file is normally run as a SYMLINK that lives in the simulation/data
# directory but points to the master MyPyLib copy.  For a symlinked script
# Python sets sys.path[0] to the *resolved* master dir, so a per-simulation
# `config.py` placed next to the symlink would be ignored.  Prepend the data
# directory (= dir of the symlink, via the un-resolved __file__) so a LOCAL
# config.py / module overrides the master one; the master still supplies every
# other module.  Falls back to the master config when no local copy exists.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ─────────────────────────────────────────────────────────────────────────────
import re
import csv
import struct
import math
import pickle
import netCDF4 as nc
import numpy as np
from PlotField import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from scipy.integrate import simpson
from scipy.integrate import trapezoid
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib import cm
from config import *
from functions import *
from saveresults import *
from functions import (        # simulation helper routines
    read_grid,
    epsfield,
    interpolate_component,
)
from compact_derivatives import (
    CompactDerivatives2D,
    make_uniform_x,
    make_stretched_y,
)
# from geopotential import *


# ── Reference-overlay helpers (gated by the config master switches) ──────────
# Plot / mark a reference-case curve only when its switch (plot_ref_smooth /
# plot_ref_rough) is True, so the same plot code serves publication (smooth-only)
# and testing (smooth + rough) without duplicating every figure.
def ref_plot(flag, *args, **kwargs):
    if flag:
        plt.plot(*args, **kwargs)

def ref_mark(flag, fn, *args, **kwargs):
    if flag:
        fn(*args, **kwargs)


def plot_fig4_budget(nc_path, nu_case, label, fig_dir, top_frac=0.8):
    """Replicate Kostelecky & Ansorge (2024) figure 4 for a reference .nc case.

    Builds the vertically-integrated momentum budget (their eq. 4.2) DIRECTLY and
    transparently from the horizontally-averaged profiles — this is the paper-correct
    "Method 2", kept independent of the loader's tau_* sign conventions so it serves
    as a validation reference:

        ⟨τ⟩_zi(z) = C + V + R           (temporal tendency T≈0 for these avg files)
          C_zx = f ∫₀ᶻ (g₂ − ⟨v⟩) dz' ,   C_zy = f ∫₀ᶻ (g₁ − ⟨u⟩) dz'   (f = 1)
          V    = (1/Re_Λ) d⟨u_i⟩/dz ,      R = −⟨u_i' w'⟩

    g = (g₁, g₂) is the geostrophic UNIT vector read from the velocity at the BL top
    (⟨u⟩, ⟨v⟩ at `top_frac`·domain height, below the Rayleigh sponge).  The Total of
    each component is the (height-independent) surface stress; u* = (τ_zx² + τ_zy²)^¼.

    Panels mirror fig. 4: (a) τ_zx and (b) τ_zy near-wall in inner units (z⁺, /u*²);
    (c,d) the same in outer units (z⁻ = y/u*, ·10⁻³/G²).  Returns the Method-2 u*.
    """
    ds = nc.Dataset(nc_path, 'r')
    y  = np.asarray(ds.variables['y'][:], float)
    su = np.asarray(ds.variables['fU'][:], float).T.mean(1)   # ⟨u⟩ streamwise
    sv = np.asarray(ds.variables['fW'][:], float).T.mean(1)   # ⟨v⟩ spanwise (veer comp.)
    Ruw = np.asarray(ds.variables['Rxy'][:], float).T.mean(1)  # ⟨u'w'⟩
    Rvw = np.asarray(ds.variables['Ryz'][:], float).T.mean(1)  # ⟨v'w'⟩
    ds.close()

    def _cumtrap(fp):                                # cumulative ∫ from the wall
        out = np.zeros_like(fp)
        out[1:] = np.cumsum(0.5 * (fp[1:] + fp[:-1]) * np.diff(y))
        return out

    top = int(top_frac * y.size)
    g1, g2 = su[top], sv[top]
    G = float(np.hypot(g1, g2))
    C_zx = _cumtrap(g2 - sv); V_zx = nu_case * np.gradient(su, y); R_zx = -Ruw
    C_zy = _cumtrap(g1 - su); V_zy = nu_case * np.gradient(sv, y); R_zy = -Rvw
    Tx = C_zx + V_zx + R_zx
    Ty = C_zy + V_zy + R_zy
    _pl = (y > 0.05 * y[top]) & (y < y[top])         # plateau window (below sponge)
    ustar = float(((Tx[_pl].mean())**2 + (Ty[_pl].mean())**2) ** 0.25)
    u2 = ustar**2; G2 = G**2
    zin = y * ustar / nu_case                        # inner z+
    zout = y / ustar                                 # outer z-

    def _panel(ax, x, C, V, R, T, sc, xlim, title, ylab, xlab):
        ax.plot(x, C/sc, color='blue',   label='Coriolis C')
        ax.plot(x, V/sc, color='red',    label='Viscous V')
        ax.plot(x, R/sc, color='orange', label='Reynolds R')
        ax.plot(x, T/sc, color='black',  lw=2, label='Total ⟨τ⟩')
        ax.axhline(0, color='grey', lw=0.5); ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylab); ax.set_xlabel(xlab)
        ax.grid(alpha=0.3)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=200)
    _panel(axs[0, 0], zin, C_zx, V_zx, R_zx, Tx, u2, (0, 100),
           f'(a) $\\tau_{{zx}}$ inner — {label}', r'$\langle\tau\rangle^{+}_{zx}$', r'$z^{+}$')
    _panel(axs[0, 1], zin, C_zy, V_zy, R_zy, Ty, u2, (0, 100),
           '(b) $\\tau_{zy}$ inner', r'$\langle\tau\rangle^{+}_{zy}$', r'$z^{+}$')
    _panel(axs[1, 0], zout, C_zx, V_zx, R_zx, Tx, G2*1e-3, (0, 1.2),
           '(c) $\\tau_{zx}$ outer', r'$\langle\tau\rangle^{-}_{zx}\cdot10^{-3}$', r'$z^{-}$')
    _panel(axs[1, 1], zout, C_zy, V_zy, R_zy, Ty, G2*1e-3, (0, 1.2),
           '(d) $\\tau_{zy}$ outer', r'$\langle\tau\rangle^{-}_{zy}\cdot10^{-3}$', r'$z^{-}$')
    axs[0, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle(f'Integrated momentum budget (eq. 4.2) — {label}   '
                 f'[Method-2 $u_*$ = {ustar:.4f}, geostrophic veer = {np.degrees(np.arctan2(g2, g1)):.1f}°]')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Fig4_momentum_budget_{label}.png'), dpi=200)
    plt.show()
    print(f"  [Fig4 budget] {label}: Method-2 u* = {ustar:.4f}  (G={G:.3f}, "
          f"τ_zx plateau={Tx[_pl].mean():.3e}, τ_zy plateau={Ty[_pl].mean():.3e})")
    return ustar


# %%
###############################################################################
############################# Initialize ######################################
# Run controls (cal_Avg, postprocess, plotRes, ...), physical scalars
# (Re, nu, Gx, Gz, u_star, kappa, l_in, ...) and post-processing constants
# (DY_METHOD, D2Y_METHOD, ghost_depth, recompute_derivatives, smooth_nc_path, ...)
# all come from config.py via `from config import *` (see imports above).

# ─────────────────────────────────────────────────────────────────────────────
# Grid & differential operators
# ─────────────────────────────────────────────────────────────────────────────
cwd = str(os.path.dirname(__file__) + '/' )
# All figures/plots are written to <data dir>/fig_rotated/ (rotated-frame variant).
fig_dir = os.path.join(cwd, 'fig_rotated')
os.makedirs(fig_dir, exist_ok=True)
x, y, z = read_grid(cwd)               # x: streamwise (periodic), y: wall-normal, z: spanwise
nx = np.size(x)
ny = np.size(y)
nz = np.size(z)
# Wall-normal derivatives use the DY_METHOD / D2Y_METHOD schemes (config.py);
# x-derivatives stay compact (x is uniform, where Padé is extremely precise).
cd = CompactDerivatives2D(x, y, periodic_x=True)

# ─────────────────────────────────────────────────────────────────────────────
# IBM geometry: solid indicator, landmark columns, heights, volume fractions
# ─────────────────────────────────────────────────────────────────────────────
# eps is the IBM indicator function: eps[j,i] = 1 inside the solid body, 0 in the fluid.
# Shape is (ny, nx); cached to disk because epsfield() is expensive.
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
eps_fr = epsVolume(eps,ny,nx,hill_hgt)

# Grid spacings (dy non-uniform; dx, dz uniform)
dy = np.roll(np.append(np.diff(y), np.diff(y)[0]), 1)
dx = np.diff(x)[0]
dz = np.diff(z)[0]
# Per-cell fluid/solid areas: cell area = dx * dy[j] (dy is the wall-normal
# spacing, independent of x).  Vectorised over (ny, nx); dy[:, None] broadcasts.
eps_area_f = (1 - eps_fr) * dx * dy[:, None]
eps_area_s = eps_fr * dx * dy[:, None]
total_areafr = eps_area_f + eps_area_s
eps_s = np.mean(eps_fr,axis=1)   # solid volume fraction per y-level (averaged over x)
eps_f = 1 - eps_s                 # fluid volume fraction per y-level

# flk_hgt: wall-normal index of the flank top; used to mark left/right flank x-columns
flk_hgt = eps_hgt[int(eps_lf)]
flk_wdt = np.where(eps_hgt == flk_hgt)[0]
lf_ind = flk_wdt[:int((len(flk_wdt))/2)]
rf_ind = flk_wdt[int((len(flk_wdt))/2):]
# x_oro / y_oro: orography outline coordinates for overlaying on 2D plots
x_oro = x
x_oro = np.append(0, x_oro)
x_oro = np.append(x_oro, x[-1])
dx = (2*np.pi/x[-1])
y_oro = np.round((hill_hgt/(2**1))*(1 + np.cos(dx*(x))))
y_oro = y[y_oro.astype(int)]
y_oro = np.append(0,y_oro)
y_oro = np.append(y_oro, 0)

# ─────────────────────────────────────────────────────────────────────────────
# Inner-unit (wall) scalings
# ─────────────────────────────────────────────────────────────────────────────
x_oro_in = x_oro/l_in
y_oro_in = y_oro/l_in

x_in = x/l_in
y_in = y/l_in

# ─────────────────────────────────────────────────────────────────────────────
# Masks & finite-difference stencil cases
# ─────────────────────────────────────────────────────────────────────────────
# Forcing values in solid zero. If not it will introduce error when calculating average in x direction.
mask0 = 1 - eps
# New interface-aware mask: zeros only INTERIOR solid cells, preserves the
# interface cell (first solid cell adjacent to fluid) where du/dy is maximum.
#
# Step 1: sum eps vertically → number of solid cells per x-column
eps_col_sum = np.sum(eps, axis=0).astype(int)                       # shape (nx,)

# Step 2: subtract 1 wherever sum >= 1 → interior solid depth per column
#         (this excludes the topmost solid cell, i.e. the interface)
interior_depth = np.where(eps_col_sum >= 1, eps_col_sum - 1, 0)    # shape (nx,)

# Step 3: build 2D mask — 0 for interior solid cells, 1 for interface + fluid
#         mask_intr[j, i] = 0  if  j < interior_depth[i]   (interior solid)
#         mask_intr[j, i] = 1  if  j >= interior_depth[i]  (interface or fluid)
j_idx    = np.arange(ny)
mask_intr = (j_idx[:, np.newaxis] >= interior_depth[np.newaxis, :]).astype(float)

# eps_g flags the bottom boundary row (j=0) regardless of solid/fluid state
eps_g = np.zeros((ny,nx)).astype(int)
eps_g[0,:] = 1
mask_v = (eps_fr == 1).astype(int)

# diff_cases returns stencil-type indices for compact finite-difference schemes
# near walls (Dirichlet BCs) and at interior/periodic boundaries
# initialize cases for derivatives
case_v_itrp, case_h_itrp = diff_cases(eps,nx,ny)
case_v = case_v_itrp
case_h = case_h_itrp
case_v_g = np.reshape(case_v[:,512].astype(int),((ny,1)))

# epsi_s, epsi_e, epsj_s, epsj_e = gap(x, y, nx, ny, eps)

# intepolate(x, y, Nx, Ny, eps_s, eps_e, gapi, gapj, field):
# %%
############################# Main Code #######################################
# Phase averaging: read DNS binary output files (avg_flow* for velocity, avg_stress* for stress
# tensor) over 'restart'-sized intervals and accumulate into AvgPh and AvgStress.
# 'counter' tracks how many complete intervals were found; arrays are divided by it at the end.
if (1 == cal_Avg):
    # AvgPh[j,i,c]      : phase-averaged velocity (c=0:u, 1:v, 2:w)
    # AvgStress[j,i,c]  : phase-averaged stress tensor (c: uu,uv,uw,vv,vw,ww)
    # VelGbl[j,c]       : global (x-averaged) mean velocity  ⟨ū⟩(y)
    # DispVel[j,i,c]    : dispersive velocity  ũ = ⟨ū⟩(x,y) − ⟨ū⟩(y)  (zero x-mean by definition)
    AvgPh = np.zeros((ny,nx,dim))
    AvgStress = np.zeros((ny,nx,6))
    SpaceAvgStr = np.zeros((ny,6))
    VelGbl = np.zeros((ny,dim))
    VelGbl2D = np.zeros((ny,nx,dim))
    Turb = np.zeros((ny,nx,6))
    DispVel = np.zeros((ny,nx,dim))
    turb1D = np.zeros((ny,6))
    ugud = np.zeros((ny,nx))
    udug = np.zeros((ny,nx))
    udvg = np.zeros((ny,nx))
    ugvd = np.zeros((ny,nx))
    udwg = np.zeros((ny,nx))
    ugwd = np.zeros((ny,nx))
    vgvd = np.zeros((ny,nx))
    vdwg = np.zeros((ny,nx))
    vgwd = np.zeros((ny,nx))
    wdwg = np.zeros((ny,nx))
    AvgP = np.zeros((ny,nx))
    AvgScal = np.zeros((ny,nx))

    for i in range(30):
        files = 0
        FilePath = []
        base = 234500
        srt = base + 1 + restart * i
        end = base + restart * (i + 1)
        for j in range(11):
            if (j <= 2):
                path = cwd + 'avg_flow' + str(srt) + '_' + str(end) + '.' + str(j+1)
            elif ((j > 2) and (j <= 8)):
                path = cwd + 'avg_stress' + str(srt) + '_' + str(end) + '.' + str(j-2)
            elif (j == 9):
                path = cwd + 'avg_p' + str(srt) + '_' + str(end) + '.' + str(1)
            elif (j == 10):
                path = cwd + 'avg_scal' + str(srt) + '_' + str(end) + '.' + str(1)
            if (os.path.exists(path)):
                FilePath.append([path])
                files += 1
        if (files == 11):
            counter += 1
            for k in range(11):
                hdr, _, _, _, _, _ = read_header(FilePath[k][0])
                if (k <= 2):
                    AvgPh[:, :, k] += readplane(FilePath[k][0], nx, ny, restart + 1, hdr)
                elif ((k > 2) and (k <= 8)):
                    AvgStress[:, :, k-3] += readplane((FilePath[k][0]), nx, ny, restart + 1, hdr)
                elif (k == 9):
                    AvgP[:, :] += readplane((FilePath[k][0]), nx, ny, restart + 1, hdr)
                elif (k == 10):
                    AvgScal[:, :] += readplane((FilePath[k][0]), nx, ny, restart + 1, hdr)

    for i in range (9):
        if (i <= 2):
            AvgPh[:,:,i] = AvgPh[:,:,i]/counter
            VelGbl[:,i] = np.mean((AvgPh[:,:,i]), axis = 1)
            DispVel[:,:,i] = (AvgPh[:,:,i] - VelGbl[:,i][:,np.newaxis])*mask0
        else:
            AvgStress[:, :, i-3] = AvgStress[:, :, i-3]/counter
            SpaceAvgStr[:,i-3] = np.mean((AvgStress[:,:,i-3]), axis = 1)
    AvgP[:,:] = (AvgP[:,:]*mask0)/counter
    PGbl      = np.mean(AvgP, axis=1)
    DispP     = (AvgP - PGbl[:,np.newaxis]) * mask0
    AvgScal[:,:] = (AvgScal[:,:]*mask0)/counter
    
    for i in range (dim):
        VelGbl2D[:,:,i] = (np.tile(VelGbl[:,i].reshape(ny,1), nx).reshape(ny,nx))*mask0
    
    
    for i in range(6):
        turb1D[:,i] = np.mean(Turb[:,:,i], axis=1)

    # Triple decomposition of the Reynolds stress tensor (Raupach & Shaw 1982):
    # <u_i u_j> = <u_i><u_j>  (mean×mean, _g)
    #           + ũ_i ũ_j      (dispersive×dispersive, _t for tilda)
    #           + <u_i' u_j'>  (turbulent, _d for double-prime, computed later)
    uu_t = DispVel[:,:,0]*DispVel[:,:,0]        # t here is tilda not turbulent
    uv_t = DispVel[:,:,0]*DispVel[:,:,1]
    uw_t = DispVel[:,:,0]*DispVel[:,:,2]
    vv_t = DispVel[:,:,1]*DispVel[:,:,1]
    vw_t = DispVel[:,:,1]*DispVel[:,:,2]
    ww_t = DispVel[:,:,2]*DispVel[:,:,2]

    # Intrinsic (fluid-only) x-average: divide by fluid fraction eps_f to exclude solid cells
    space_uu_t = (np.mean(uu_t, axis =1))/eps_f
    space_uv_t = (np.mean(uv_t, axis =1))/eps_f
    space_uw_t = (np.mean(uw_t, axis =1))/eps_f
    space_vv_t = (np.mean(vv_t, axis =1))/eps_f
    space_vw_t = (np.mean(vw_t, axis =1))/eps_f
    space_ww_t = (np.mean(ww_t, axis =1))/eps_f
    
    uu_g = VelGbl2D[:,:,0]*VelGbl2D[:,:,0]  
    uv_g = VelGbl2D[:,:,0]*VelGbl2D[:,:,1]
    uw_g = VelGbl2D[:,:,0]*VelGbl2D[:,:,2]
    vv_g = VelGbl2D[:,:,1]*VelGbl2D[:,:,1]
    vw_g = VelGbl2D[:,:,1]*VelGbl2D[:,:,2]
    ww_g = VelGbl2D[:,:,2]*VelGbl2D[:,:,2]

# uu_d = SpaceAvgStr[:,0] - uu_g - space_uu_t
# uv_d = SpaceAvgStr[:,1] - uv_g - space_uv_t
# uw_d = SpaceAvgStr[:,2] - uw_g - space_uw_t
# vv_d = SpaceAvgStr[:,3] - vv_g - space_vv_t
# vw_d = SpaceAvgStr[:,4] - vw_g - space_vw_t
# ww_d = SpaceAvgStr[:,5] - ww_g - space_ww_t

# verify_TimeAvg: recover the true turbulent stress _d by subtracting the mean×mean (_g),
# dispersive (_t), and cross-term (mean×dispersive) contributions from the total AvgStress.
# Cross terms u_d*u_g = u_g*u_d = 0 by construction but are included for completeness.
if(1 == verify_TimeAvg):
    for i in range(30):
        files = 0
        FilePath = []
        base = 234500
        srt = base + 1 + restart * i 
        end = base + restart * (i + 1)
        pathi = cwd + 'avg_flow' + str(srt) + '_' + str(end) + '.' + str(1)
        pathj = cwd + 'avg_flow' + str(srt) + '_' + str(end) + '.' + str(2)
        pathk = cwd + 'avg_flow' + str(srt) + '_' + str(end) + '.' + str(3)
    
    udug = DispVel[:,:,0]*VelGbl2D[:,:,0]
    ugud = udug
    udvg = DispVel[:,:,0]*VelGbl2D[:,:,1]
    ugvd = VelGbl2D[:,:,0]*DispVel[:,:,1]
    udwg = DispVel[:,:,0]*VelGbl2D[:,:,2]
    ugwd = VelGbl2D[:,:,0]*DispVel[:,:,2]
    
    vgvd = DispVel[:,:,1]*VelGbl2D[:,:,1]
    vdvg = vgvd
    vdwg = DispVel[:,:,1]*VelGbl2D[:,:,2]
    vgwd = VelGbl2D[:,:,1]*DispVel[:,:,2]
    
    wdwg = DispVel[:,:,2]*VelGbl2D[:,:,2]
    wgwd = wdwg


    uu_d = AvgStress[:,:,0] - uu_g - uu_t - udug - ugud
    uv_d = AvgStress[:,:,1] - uv_g - uv_t - udvg - ugvd
    uw_d = AvgStress[:,:,2] - uw_g - uw_t - udwg - ugwd
    vv_d = AvgStress[:,:,3] - vv_g - vv_t - vdvg - vgvd
    vw_d = AvgStress[:,:,4] - vw_g - vw_t - vdwg - vgwd
    ww_d = AvgStress[:,:,5] - ww_g - ww_t - wdwg - wgwd

# Write the required varaible in a file

if (100 == save_avg):
    np.save('uu_d.npy', uu_d)
    np.save('uv_d.npy', uv_d)
    np.save('uw_d.npy', uw_d)
    np.save('vv_d.npy', vv_d)
    np.save('vw_d.npy', vw_d)
    np.save('ww_d.npy', ww_d)
    
    np.save('AvgStrUU.npy', AvgStress[:,:,0])
    np.save('AvgStrUV.npy', AvgStress[:,:,1])
    np.save('AvgStrUW.npy', AvgStress[:,:,2])
    np.save('AvgStrVV.npy', AvgStress[:,:,3])
    np.save('AvgStrVW.npy', AvgStress[:,:,4])
    np.save('AvgStrWW.npy', AvgStress[:,:,5])
    
    np.save('uu_g.npy', uu_g)
    np.save('uv_g.npy', uv_g)
    np.save('uw_g.npy', uw_g)
    np.save('vv_g.npy', vv_g)
    np.save('vw_g.npy', vw_g)
    np.save('ww_g.npy', ww_g)
    
    np.save('uu_t.npy', uu_t)
    np.save('uv_t.npy', uv_t)
    np.save('uw_t.npy', uw_t)
    np.save('vv_t.npy', vv_t)
    np.save('vw_t.npy', vw_t)
    np.save('ww_t.npy', ww_t)
    
    np.save('AvgPhU.npy', AvgPh[:,:,0])
    np.save('AvgPhV.npy', AvgPh[:,:,1])
    np.save('AvgPhW.npy', AvgPh[:,:,2])
    np.save('AvgP.npy', AvgP[:,:])
    np.save('DispP.npy', DispP)
    np.save('AvgScal.npy', AvgScal[:,:])

    np.save('VelGblU.npy', VelGbl[:,0])
    np.save('VelGblV.npy', VelGbl[:,1])
    np.save('VelGblW.npy', VelGbl[:,2])
    
    np.save('DispVelU', DispVel[:,:,0])
    np.save('DispVelV', DispVel[:,:,1])
    np.save('DispVelW', DispVel[:,:,2])
    
    np.save('udug.npy', udug)
    np.save('udvg.npy', udvg)
    np.save('udwg.npy', udwg)
    np.save('vdvg.npy', vdvg)
    np.save('vdwg.npy', vdwg)
    np.save('wdwg.npy', wdwg)
    
    np.save('ugud.npy', ugud)
    np.save('ugvd.npy', ugvd)
    np.save('ugwd.npy', ugwd)
    np.save('vgvd.npy', vgvd)
    np.save('vgwd.npy', vgwd)
    np.save('wgwd.npy', wgwd)

# load_arrays: restore pre-computed fields from .npy files (avoids rerunning averaging)
# and load reference smooth-wall / rough-wall DNS data from NetCDF for comparison.
if (1 == load_arrays):
    # declares arrays to load
    du_dt = np.zeros((ny,nx,dim))
    ds_dt = np.zeros((ny,nx,scal))
    
    rey_uu = np.load('uu_d.npy')
    rey_uv = np.load('uv_d.npy')
    rey_uw = np.load('uw_d.npy')
    rey_vv = np.load('vv_d.npy')
    rey_vw = np.load('vw_d.npy')
    rey_ww = np.load('ww_d.npy')
    
    AvgStrUU = np.load('AvgStrUU.npy')
    AvgStrUV = np.load('AvgStrUV.npy')
    AvgStrUW = np.load('AvgStrUW.npy')
    AvgStrVV = np.load('AvgStrVV.npy')
    AvgStrVW = np.load('AvgStrVW.npy')
    AvgStrWW = np.load('AvgStrWW.npy')
    
    UU_G = np.load('uu_g.npy')
    UV_G = np.load('uv_g.npy')
    UW_G = np.load('uw_g.npy')
    VV_G = np.load('vv_g.npy')
    VW_G = np.load('vw_g.npy')
    WW_G = np.load('ww_g.npy')
    
    UU_disp = np.load('uu_t.npy')
    UV_disp = np.load('uv_t.npy')
    UW_disp = np.load('uw_t.npy')
    VV_disp = np.load('vv_t.npy')
    VW_disp = np.load('vw_t.npy')
    WW_disp = np.load('ww_t.npy')
        
    AvgPhU = np.load('AvgPhU.npy')
    AvgPhV = np.load('AvgPhV.npy')
    AvgPhW = np.load('AvgPhW.npy')
    AvgP  = np.load('AvgP.npy')
    DispP = np.load('DispP.npy')
    AvgScal = np.load('AvgScal.npy')

    VelGblU = np.load('VelGblU.npy')
    VelGblV = np.load('VelGblV.npy')
    VelGblW = np.load('VelGblW.npy')
    
    DispVelU = np.load('DispVelU.npy')
    DispVelV = np.load('DispVelV.npy')
    DispVelW = np.load('DispVelW.npy')
    
    udug = np.load('udug.npy')
    udvg = np.load('udvg.npy')
    udwg = np.load('udwg.npy')
    vdvg = np.load('vdvg.npy')
    vdwg = np.load('vdwg.npy')
    wdwg = np.load('wdwg.npy')
    
    ugud = np.load('ugud.npy')
    ugvd = np.load('ugvd.npy')
    ugwd = np.load('ugwd.npy')
    vgvd = np.load('vgvd.npy')
    vgwd = np.load('vgwd.npy')
    wgwd = np.load('wgwd.npy')
    
    # pdud2D = np.load('pdud2D.npy')
    # pdvd2D = np.load('pdvd2D.npy')
    # pdwd2D = np.load('pdwd2D.npy')

    # dq_dt
    # du_dt[:,:,0] = np.load('du_dt1.npy')
    # du_dt[:,:,1] = np.load('du_dt2.npy')
    # du_dt[:,:,2] = np.load('du_dt3.npy')
    # ds_dt[:,:,0] = np.load('ds_dt.npy')

# Postprocess
# %%
if (1 == postprocess):
    # Velocity is zero inside the IBM solid, making raw gradients at the interface unreliable.
    # interpolate_component fills the solid ghost cells with a smooth extrapolation so that
    # compact-scheme derivatives computed with cd.ddy / cd.ddx are meaningful at the wall.
    # _i suffix: field interpolated for x-derivatives; _j suffix: for y-derivatives.
    # Ghost-filled fields and all compact-scheme derivatives are cached as .npy files.
    # On subsequent runs the expensive PCHIP interpolation and spectral derivatives
    # are skipped if the files already exist.  Delete the .npy files to force recomputation.

    # All ghost-filled fields and derivatives are cached as .npy via load_or_compute:
    # with recompute_derivatives=False they are loaded if every file in the group
    # exists, otherwise (re)computed and saved.  Set recompute_derivatives=True in
    # config.py (or delete the .npy files) to force fresh computation.

    # ── Ghost-cell interpolated fields ────────────────────────────────────────
    def _compute_interp():
        AvgPhU_i, AvgPhU_j = interpolate_component(x, y, nx, ny, eps, AvgPhU, ghost_depth=ghost_depth, n_anchor=n_anchor, smooth_width=smooth_width)
        AvgPhV_i, AvgPhV_j = interpolate_component(x, y, nx, ny, eps, AvgPhV, ghost_depth=ghost_depth, n_anchor=n_anchor, smooth_width=smooth_width)
        AvgPhW_i, AvgPhW_j = interpolate_component(x, y, nx, ny, eps, AvgPhW, ghost_depth=ghost_depth, n_anchor=n_anchor, smooth_width=smooth_width)
        AvgP_i,   AvgP_j   = interpolate_component(x, y, nx, ny, eps, AvgP,   ghost_depth=ghost_depth, n_anchor=n_anchor, smooth_width=smooth_width)
        return (AvgPhU_i, AvgPhU_j, AvgPhV_i, AvgPhV_j,
                AvgPhW_i, AvgPhW_j, AvgP_i,   AvgP_j)
    AvgPhU_i, AvgPhU_j, AvgPhV_i, AvgPhV_j, AvgPhW_i, AvgPhW_j, AvgP_i, AvgP_j = \
        load_or_compute(['AvgPhU_i', 'AvgPhU_j', 'AvgPhV_i', 'AvgPhV_j',
                         'AvgPhW_i', 'AvgPhW_j', 'AvgP_i',   'AvgP_j'],
                        recompute_derivatives, _compute_interp,
                        label='ghost-cell interpolated fields (PCHIP)')

    # ══════════════════════════════════════════════════════════════════════════
    # ░░  FRAME ROTATION  ░░  (this is the ONLY physics change vs PhAvg.py)
    # Rotate the horizontal components by `alpha` (config; the ~25° geostrophic
    # tilt) so the geostrophic wind aligns with x — the frame of the reference
    # .nc cases.  Applied AFTER interpolation and BEFORE the derivatives, so the
    # derivative/budget pipeline below operates on the rotated fields unchanged.
    #   proper rotation R(alpha):  u' = u·cosα − w·sinα,  w' = u·sinα + w·cosα
    # Rotated (vectors): mean U,W and their _i/_j interpolations; dispersive U,W;
    # and the wall-normal momentum-flux pair (rey_uv=⟨u'w'⟩, rey_vw=⟨v'w'⟩).
    # Unchanged: wall-normal V (axis of rotation); in-plane stresses rey_uu/uw/ww
    # (do not enter the τ_z· balance; TKE trace is rotation-invariant).
    # u* = ‖τ_w‖ is rotation-invariant — only the τ_zx / τ_zy split changes.
    _rc, _rs = np.cos(alpha), np.sin(alpha)
    def _rotate_pair(a, b):
        return a * _rc - b * _rs, a * _rs + b * _rc
    AvgPhU,   AvgPhW   = _rotate_pair(AvgPhU,   AvgPhW)
    AvgPhU_i, AvgPhW_i = _rotate_pair(AvgPhU_i, AvgPhW_i)
    AvgPhU_j, AvgPhW_j = _rotate_pair(AvgPhU_j, AvgPhW_j)
    DispVelU, DispVelW = _rotate_pair(DispVelU, DispVelW)
    rey_uv,   rey_vw   = _rotate_pair(rey_uv,   rey_vw)
    # geostrophic vector in the rotated frame: aligned with x → spanwise comp = 0
    Gx, Gz = float(np.hypot(Gx, Gz)), 0.0
    print(f"[ROTATED] fields rotated by alpha={alpha:.4f} rad "
          f"({np.degrees(alpha):.1f}°); geostrophic now (Gx,Gz)=({Gx:.3f},{Gz:.3f})")
    # ══════════════════════════════════════════════════════════════════════════

    # ── Velocity gradients ∂u/∂y, ∂u/∂x, ∂v/∂y, ∂v/∂x, ∂w/∂y, ∂w/∂x ───────
    # First y-derivatives use the DY_METHOD scheme (config.py).  'fornberg7'
    # differentiates the accidental factor-2 dy step at the top of Zone 1 exactly,
    # avoiding the spurious tremble in the viscous shear stress (τ_zx) that the
    # η-space 'compact' metric produces there.
    def _compute_vel_deriv():
        return (cd.ddy(AvgPhU_j, method=DY_METHOD) * mask_intr,
                cd.ddx(AvgPhU_i) * mask_intr,
                cd.ddy(AvgPhV_j, method=DY_METHOD) * mask_intr,
                cd.ddx(AvgPhV_i) * mask_intr,
                cd.ddy(AvgPhW_j, method=DY_METHOD) * mask_intr,
                cd.ddx(AvgPhW_i) * mask_intr)
    du_dy, du_dx, dv_dy, dv_dx, dw_dy, dw_dx = \
        load_or_compute(['du_dy_rot', 'du_dx_rot', 'dv_dy_rot', 'dv_dx_rot', 'dw_dy_rot', 'dw_dx_rot'],
                        recompute_derivatives, _compute_vel_deriv,
                        label='velocity derivatives (rotated frame)')

    # ── Dispersive velocity gradients ─────────────────────────────────────────
    def _compute_disp_deriv():
        return (cd.ddy(DispVelU, method=DY_METHOD) * mask_intr,
                cd.ddy(DispVelV, method=DY_METHOD) * mask_intr,
                cd.ddy(DispVelW, method=DY_METHOD) * mask_intr,
                cd.ddx(DispVelU) * mask_intr,
                cd.ddx(DispVelV) * mask_intr,
                cd.ddx(DispVelW) * mask_intr)
    dud_dy, dvd_dy, dwd_dy, dud_dx, dvd_dx, dwd_dx = \
        load_or_compute(['dud_dy_rot', 'dvd_dy_rot', 'dwd_dy_rot', 'dud_dx_rot', 'dvd_dx_rot', 'dwd_dx_rot'],
                        recompute_derivatives, _compute_disp_deriv,
                        label='dispersive velocity derivatives (rotated frame)')

    # ── Second-order velocity derivatives and Reynolds/pressure gradients ─────
    # d2u_dy2 uses the D2Y_METHOD scheme (config.py; default 'compact').
    def _compute_misc_deriv():
        return (cd.d2dx2(AvgPhU_i) * mask_intr,                       # ∂²ū/∂x²
                cd.d2dy2(AvgPhU_j, method=D2Y_METHOD) * mask_intr,    # ∂²ū/∂y²
                cd.ddx(rey_uu) * mask_intr,                           # turbulent advection
                cd.ddy(rey_uv, method=DY_METHOD) * mask_intr,
                cd.ddx(AvgP_i) * mask_intr,                           # pressure gradients
                cd.ddy(AvgP_j, method=DY_METHOD) * mask_intr)
    d2u_dx2, d2u_dy2, dreyuu_dx, dreyuv_dy, dP_dx, dP_dy = \
        load_or_compute(['d2u_dx2_rot', 'd2u_dy2_rot', 'dreyuu_dx_rot', 'dreyuv_dy_rot', 'dP_dx_rot', 'dP_dy_rot'],
                        recompute_derivatives, _compute_misc_deriv,
                        label='second-order and stress/pressure derivatives')

    # Method 2 — friction velocity from the Ekman momentum-integral balance.
    # Steady-state (∂/∂t = 0) intrinsic-averaged momentum equations:
    #   τ_yx(y) = f∫₀ʸ ⟨w̃ − G_z⟩ dy'  +  (1/Re_Λ) ∂⟨ū⟩/∂y  − ⟨u'v'⟩
    #   τ_yz(y) = −f∫₀ʸ ⟨ũ − G_x⟩ dy'  +  (1/Re_Λ) ∂⟨w̄⟩/∂y  − ⟨v'w'⟩
    # u* = |τ_wall|^0.5 evaluated at y→0 (where τ profiles collapse to the surface stress).
    # Momentum Balance to find u*
    # Time derivative is zero
    # $f \int_0^y \epsilon_{1 2 3}\left(\langle\bar{v}\rangle_k-g_v\right) \mathrm{d} y + \frac{1}{\operatorname{Re} e_{\Lambda}} \frac{\partial\langle\bar{u}\rangle}{\partial y}-\left\langle\overline{u^{\prime} w^{\prime}}\right\rangle $
    # Turining angle is 23.29 degrees
    corr_yx = (AvgPhW - Gz)*mask0
    I_corr_yx = vIntegral(np.mean(corr_yx, axis=1), ny, y)
    visc_yx = (1/Re_lambda) * (avg_c(eps, du_dy, axis=1))
    turb_yx = (avg_c(eps, rey_uv, axis=1))
    # Reynolds stresses given by 'rey_uv'
    # Tau_yz(z) = - Temporal - Coriolis + Viscous - Reynolds
    total_tau_yx = - I_corr_yx + visc_yx - turb_yx

    # $f \int_0^z \epsilon_{2 1 3}\left(\langle\bar{u}\rangle_k-g_u\right) \mathrm{d} z + \frac{1}{\operatorname{Re} e_{\Lambda}} \frac{\partial\langle\bar{v}\rangle}{\partial z}-\left\langle\overline{v^{\prime} w^{\prime}}\right\rangle $
    corr_yz = (AvgPhU - Gx)*mask0
    I_corr_yz = vIntegral(np.mean(corr_yz, axis=1), ny, y) # Coriolis is positive
    visc_yz = (1/Re_lambda) * (avg_c(eps, dw_dy, axis=1))
    turb_yz = avg_c(eps, rey_vw, axis=1)
    # Reynolds stresses given by 'rey_vw'
    # Tau_yz(z) = - Temporal + Coriolis + Viscous - Reynolds
    # ROTATED-FRAME / Fig-4 convention: Reynolds enters with a MINUS sign,
    #   Ty = C_zy + V_zy + R_zy   with   C_zy = -I_corr_yz,  V_zy = visc_yz,  R_zy = -turb_yz.
    # This matches the validated loader (functions.py: tau_yz = -I_corr_yz + visc_yz - Ryz)
    # and plot_fig4_budget.  The unrotated PhAvg.py keeps the old +turb_yz plotting
    # convention; here we correct it so u_star2 below is paper-consistent.
    total_tau_yz = -I_corr_yz + visc_yz - turb_yz
    # tau_disp_yz retained as an alias of the (now Fig-4-consistent) total for the τ_zy plots.
    tau_disp_yz = total_tau_yz

    # ── Alternative Coriolis integral: integrate per-column, THEN intrinsic avg ──
    # The existing I_corr_y* x-average (np.mean, EXTRINSIC over all nx columns)
    # BEFORE the vertical integral — over orography this mixes solid (zero) and
    # fluid columns in the nx denominator.  Here each column is integrated first
    # with vIntegral_2d (solid cells pre-zeroed by mask0 add nothing → 0 in the
    # body), giving a 2-D integral; the intrinsic fluid-only average avg_c is then
    # applied.  Kept side-by-side for comparison ONLY — the budget above still
    # uses the original I_corr_yx / I_corr_yz.
    # (vIntegral_2d uses the trapezoidal rule = vIntegral2; the simpson-vs-trapz
    #  difference is negligible next to the averaging-order effect studied here.)
    I_corr_yx_2d = vIntegral_2d(corr_yx, ny, y)
    I_corr_yx_c  = avg_c(eps, I_corr_yx_2d, axis=1)
    I_corr_yz_2d = vIntegral_2d(corr_yz, ny, y)
    I_corr_yz_c  = avg_c(eps, I_corr_yz_2d, axis=1)

    print("\n══ Coriolis integral: mean→integrate (old)  vs  integrate→cavg (new) ══")
    print(f"{'z+':>8} | {'I_yx old':>11} {'I_yx new':>11} {'Δ%':>7} | "
          f"{'I_yz old':>11} {'I_yz new':>11} {'Δ%':>7}")
    def _pct(a, b):
        return 100.0 * (b - a) / a if abs(a) > 1e-12 else np.nan
    for _zt in (5, 15, 30, 60, 100, 200, 500):
        _j = int(np.argmin(np.abs(y_in - _zt)))
        print(f"{y_in[_j]:8.1f} | {I_corr_yx[_j]:11.4e} {I_corr_yx_c[_j]:11.4e} "
              f"{_pct(I_corr_yx[_j], I_corr_yx_c[_j]):7.1f} | "
              f"{I_corr_yz[_j]:11.4e} {I_corr_yz_c[_j]:11.4e} "
              f"{_pct(I_corr_yz[_j], I_corr_yz_c[_j]):7.1f}")
    print(f"  max|Δ| over profile:  yx = {np.max(np.abs(I_corr_yx_c - I_corr_yx)):.4e}"
          f"   yz = {np.max(np.abs(I_corr_yz_c - I_corr_yz)):.4e}")

    tau_corrctn = ((1/Re_lambda) * (avg_c(eps, dv_dx, axis=1)))
        
    # Resultant surface stress magnitude; square-root twice because stress ~ u*²
    # u_star2 = ((total_tau_yx**2 + total_tau_yz**2 + tau_corrctn**2)**0.5)**0.5
    u_star2 = ((total_tau_yx**2 + total_tau_yz**2)**0.5)**0.5
    # u_star: domain-averaged friction velocity used for inner scaling throughout
    u_star = np.mean(u_star2)

    # ── Friction velocity from the alternative (integrate→cavg) Coriolis term ──
    # Same momentum-balance formula, but using I_corr_*_c (per-column vertical
    # integral, THEN intrinsic fluid-only average) in place of I_corr_* (x-mean
    # THEN integral).  Compared in the Friction-Velocity comparison plot; inner
    # scaling (u_star) above is unchanged and still uses the original u_star2.
    total_tau_yx_c = -I_corr_yx_c + visc_yx - turb_yx
    total_tau_yz_c = -I_corr_yz_c + visc_yz + turb_yz
    u_star2_c = ((total_tau_yx_c**2 + total_tau_yz_c**2)**0.5)**0.5
    u_star_c  = np.mean(u_star2_c)

    y_inner =  y*(u_star/nu)
    y_outer = y/u_star
    
    # Turbulent Kinetic Energy
    TKE = 0.5*(rey_uu + rey_vv + rey_ww)

    dudt = np.mean(du_dt[:,:,0], axis=1)
    dwdt = np.mean(du_dt[:,:,2], axis=1)

    # Streamwise momentum budget — x-averaged 1D profiles
    # Equation: Temporal + MeanAdv + TurbAdv = Viscous + Coriolis
    mom_temporal  = dudt                                                         # ∂ū/∂t (≈ 0, steady)
    mom_mean_adv  = avg_c(eps, AvgPhU * du_dx + AvgPhV * du_dy, axis=1)         # ū ∂ū/∂x + v̄ ∂ū/∂y
    mom_turb_adv  = avg_c(eps, dreyuu_dx + dreyuv_dy, axis=1)                   # ∂(u'u')/∂x + ∂(u'v')/∂y
    mom_visc      = (1/Re_lambda) * avg_c(eps, d2u_dx2 + d2u_dy2, axis=1)       # (1/Re)(∂²ū/∂x² + ∂²ū/∂y²)
    mom_coriolis  = -avg_c(eps, corr_yx, axis=1)                                 # -(w̄ − Gz)

    # Horizontal wind
    u_plus = AvgPhU/u_star
    w_plus = AvgPhW/u_star
    v_plus = AvgPhV/u_star
    
    alphacos = np.cos(alpha)
    alphasin = np.sin(alpha)
    # ROTATED frame: the fields are ALREADY rotated to the geostrophic-aligned x-axis
    # (see FRAME ROTATION block above, Gz=0), so the hodograph/velocity-profile
    # quantities use them directly — no further rotation, and no division by Gz.
    # Normalise by the geostrophic magnitude Gx (= |G|; spanwise geo = 0).
    u_pl_rot2D = AvgPhU / Gx
    w_pl_rot2D = AvgPhW / Gx
    # Cache intrinsic x-averages once; reused for both components to avoid 4 redundant calls
    _avgU_1d = avg_c(eps, AvgPhU, axis=1)
    _avgW_1d = avg_c(eps, AvgPhW, axis=1)
    u_plus_rot = _avgU_1d
    w_plus_rot = -_avgW_1d
    
    # Turning angle
    # inst_alpha = ((avg_c(eps,AvgPhW,axis=1))/(avg_c(eps,AvgPhU,axis=1)))
    inst_alpha = w_plus_rot/u_plus_rot
    
    uh_plus = np.sqrt(u_plus**2 + w_plus**2)
    uh_pl1D = avg_c(eps, uh_plus, axis=1)

###############################################################################
##################### ABL log-law fit (z+ ∈ [60, 200]) ######################
###############################################################################
    # Log-law (law of the wall with displacement and roughness):
    #
    #   u⁺ = (1/κ) · ln( (z⁺ − d⁺) / z₀ₘ⁺ )
    #
    # where:
    #   u⁺  = u / u★          (velocity in inner/wall units)
    #   z⁺  = z · u★ / ν      (wall-normal distance in inner units)
    #   κ   = von Kármán constant  (≈ 0.40–0.44)
    #   d⁺  = zero-plane displacement height in inner units
    #   z₀ₘ⁺= aerodynamic roughness length in inner units
    #
    # Equivalently written as:  u⁺ = (1/κ) · ln(z⁺ − d⁺) + B
    #   where  B = −(1/κ) · ln(z₀ₘ⁺)
    #
    # Fitting procedure: OLS linear regression of u⁺ vs ln(z⁺ − d⁺).
    #   slope    = 1/κ          →  κ = 1/slope
    #   intercept = B           →  z₀ₘ⁺ = exp(−B · κ)
    # κ is constrained to [0.40, 0.44]; d⁺ is grid-searched in [0, 0.9·z⁺_min].
    #
    # NOTE: u_h_plus here uses u_star (simulation friction velocity, ≈ 0.0699)
    # NOT the 0.0617 reference used in the velocity-profile comparison plot below.
    # This estimate is only used for the early α_canopy scalar printed to console.
    u_h_plus    = u_plus_rot / u_star        # streamwise (rotated) velocity in inner units

    _fit_lo, _fit_hi = loglaw_zmin, loglaw_zmax
    _fit_mask   = (y_in >= _fit_lo) & (y_in <= _fit_hi)
    _z_fit      = y_in[_fit_mask]
    _u_fit      = u_h_plus[_fit_mask]

    # Defaults (fallback if no valid κ found in constrained range)
    kappa_loglaw = 0.41
    d_m_loglaw   = 0.0
    z0m_loglaw   = 0.068
    _best_r2     = -np.inf

    if _u_fit.size >= 3:
        for _d in np.linspace(0.0, 0.9 * _fit_lo, 1001):
            _zs = _z_fit - _d
            if np.any(_zs <= 0):
                break
            _slope, _intercept, _r, *_ = linregress(np.log(_zs), _u_fit)
            if _slope <= 0:
                continue
            _k = 1.0 / _slope
            if not (kappa_bounds[0] <= _k <= kappa_bounds[1]):
                continue
            if _r**2 > _best_r2:
                _best_r2     = _r**2
                kappa_loglaw = _k
                d_m_loglaw   = _d
                z0m_loglaw   = np.exp(-_intercept / _slope)   # z_{0m}⁺ = exp(−B·κ)

    print(f"Log-law fit (z+ ∈ [{_fit_lo:.0f},{_fit_hi:.0f}]):  "
          f"κ_m={kappa_loglaw:.4f}  d_m+={d_m_loglaw:.2f}  "
          f"z_0m+={z0m_loglaw:.5f}  R²={_best_r2:.4f}")

###############################################################################
################ Canopy exponential law (z+ ∈ [0, h⁺+20pts]) ################
###############################################################################
    # Canopy (exponential attenuation) law:
    #
    #   u(z) = u(h) · exp( α · (z/h − 1) )     for  0 ≤ z ≤ h
    #
    # where:
    #   u(h)  = streamwise velocity at hill-crest height h  (anchor point)
    #   h     = hill height in wall units (h⁺ = h · u★ / ν ≈ 28.6)
    #   α     = canopy attenuation coefficient  (dimensionless, > 0)
    #           larger α → steeper velocity decrease toward the wall
    #
    # At z = h:  u = u(h) · exp(0) = u(h)  ✓ (model is anchored at the crest)
    # At z → 0:  u → u(h) · exp(−α)        (exponential decay into the canopy)
    #
    # Fitting procedure: OLS linear regression of  ln(u/u(h))  vs  (z/h − 1).
    #   model:  ln(u/u(h)) = α · (z/h − 1)  (passes through origin by definition)
    #   slope   = α   (free-intercept OLS; same slope as regressing ln(u) vs z/h)
    #
    # Fitting range: indices 0 … hill_hgt+20  (z⁺ ∈ [0, ≈34.4])
    # Only fluid-occupied cells (u_h+ > 1e-6) are included to exclude IBM ghost cells.
    #
    # Early estimate uses u_h_plus = u_plus_rot / u_star (simulation u★ ≈ 0.0699).
    # The comparison plot below re-fits with u_star_ref = 0.0617; the α value is
    # identical because the normalization cancels in the ratio u/u(h).
    # Typical result for the neutral valley case: α ≈ 1.75
    # Fit α over full canopy region (indices 0 … hill_hgt+20) using only
    # fluid-occupied cells (u_h+ > 0) to exclude IBM solid ghost cells.
    h_inner_plus = float(y_in[hill_hgt])
    _can_end     = min(hill_hgt + canopy_extra_cells, ny)
    _z_can       = y_in[:_can_end]
    _u_can       = u_h_plus[:_can_end]
    _can_valid   = _u_can > 1e-6

    alpha_canopy = 3.0    # default attenuation coefficient
    if np.sum(_can_valid) >= 3:
        _slope_c, *_ = linregress(_z_can[_can_valid] / h_inner_plus,
                                np.log(_u_can[_can_valid]))
        alpha_canopy = float(_slope_c)

    print(f"Canopy law fit (z+ ∈ [0,{y_in[_can_end-1]:.1f}]):  α={alpha_canopy:.4f}")

###############################################################################
##################### Compute Friction velocity method 1 ######################
#####################  Calcualte horizontal surfaces ##########################
    '''
    Our simulation box is periodic in horzontal.
    The solid valley sits on this periodic boundary. Hence even if the Immersed
    Boundary Method creates a single solid, it appears as two distinct solids
    divided by the boundary.
    In this piece of code, we find out the start and end of fluid region.
    We consider the first solid grid point next to fluid as the interface.
    The derivative at this point point in solid is considered as the gradient is
    sharpest here.

    epsj_s / epsj_e : [j, i] pairs marking start/end of each horizontal surface segment
    epsi_s1/e1      : left-flank vertical surface segments (i < nx/2)
    epsi_s2/e2      : right-flank vertical surface segments (i > nx/2)
    '''
    # horiz_surfaces       : list of (j, i_start, i_end) — top-face solid/fluid interfaces
    # left_flank_surfaces  : list of (j_start, j_end, i) — right-facing walls (i < nx//2)
    # right_flank_surfaces : list of (j_start, j_end, i) — left-facing walls  (i > nx//2)

#####################  Calculating horizontal surfaces ##########################
    # Scan each row j. Within that row, scan i to find contiguous spans where the
    # solid top face is exposed to fluid above: eps[j,i]==1 and eps[j+1,i]==0.
    # Each contiguous span is recorded as (j, i_start, i_end).
    # Works for any number of solid objects without geometry-specific hard-coding.
    horiz_surfaces = []
    for j in range(hill_hgt + 1):
        if j + 1 >= ny:
            continue
        in_seg = False
        i_start = 0
        for i in range(nx):
            on_top = (eps[j, i] == 1) and (eps[j + 1, i] == 0)
            if on_top and not in_seg:
                i_start = i
                in_seg = True
            elif not on_top and in_seg:
                horiz_surfaces.append((j, i_start, i))
                in_seg = False
        if in_seg:
            horiz_surfaces.append((j, i_start, nx))

#####################  Calculating vertical surfaces ###########################
    # Scan each column i. Within that column, scan j to find contiguous spans where
    # the solid vertical face is exposed to fluid on the adjacent horizontal side.
    # Left flank  (i < nx//2): solid at i, fluid to the right — eps[j,i]==1, eps[j,i+1]==0.
    # Right flank (i > nx//2): solid at i, fluid to the left  — eps[j,i]==1, eps[j,i-1]==0.
    # Each contiguous span is recorded as (j_start, j_end, i).
    # Left and right flanks are collected independently — no forced pairing by index.
    left_flank_surfaces  = []
    right_flank_surfaces = []

    for i in range(nx - 1):
        if i < nx // 2:
            in_seg = False
            j_start = 0
            for j in range(hill_hgt + 1):
                on_wall = (eps[j, i] == 1) and (eps[j, i + 1] == 0)
                if on_wall and not in_seg:
                    j_start = j
                    in_seg = True
                elif not on_wall and in_seg:
                    left_flank_surfaces.append((j_start, j, i))
                    in_seg = False
            if in_seg:
                left_flank_surfaces.append((j_start, hill_hgt + 1, i))

        elif i > nx // 2:
            in_seg = False
            j_start = 0
            for j in range(hill_hgt + 1):
                on_wall = (eps[j, i + 1] == 1) and (eps[j, i] == 0)
                if on_wall and not in_seg:
                    j_start = j
                    in_seg = True
                elif not on_wall and in_seg:
                    right_flank_surfaces.append((j_start, j, i + 1))
                    in_seg = False
            if in_seg:
                right_flank_surfaces.append((j_start, hill_hgt + 1, i + 1))

    print(f"Horizontal surface segments   : {len(horiz_surfaces)}")
    print(f"Left  flank vertical segments : {len(left_flank_surfaces)}")
    print(f"Right flank vertical segments : {len(right_flank_surfaces)}")

###############################################################################
##################### Cp, orographic form drag, grad-P ratio #################
###############################################################################
    # Geostrophic speed (non-dim reference velocity, = 1 in this run)
    G_inf = np.sqrt(Gx**2 + Gz**2)

    # IBM surface height and surface pressure per x-column
    # eps_hgt[i] = number of solid cells in column i = index of first fluid cell
    y_w    = y[eps_hgt]                          # wall-normal position of surface, shape (nx,)
    P_surf = AvgP[eps_hgt, np.arange(nx)]        # ⟨P_y⟩ at first fluid cell, shape (nx,)

    # Pressure coefficient: Cp(x+) = P_w / (0.5 G^2)
    Cp = P_surf / (0.5 * G_inf**2)

    # Orographic form drag: D_form = -∮ ⟨P_y⟩(x, y_w) (dy_w/dx) dx
    # Surface slope via centred differences (handles periodic boundary)
    dy_w_dx   = np.gradient(y_w, x)
    dx_uni    = x[1] - x[0]
    D_form_oro = -np.sum(P_surf * dy_w_dx) * dx_uni
    print(f"  Orographic form drag D_form : {D_form_oro:.6f}")

    # Along-surface pressure gradient ratio |∂y P| / |∂x P|
    dP_dy_surf  = dP_dy[eps_hgt, np.arange(nx)]
    dP_dx_surf  = dP_dx[eps_hgt, np.arange(nx)]
    ratio_dP    = np.abs(dP_dy_surf) / (np.abs(dP_dx_surf) + 1e-12)

#####################  Integrating shear stress over surface ##################
# Method 1 — friction velocity from direct surface integration.
# Integrates viscous shear stress (τ = ν ∂u/∂n) and pressure drag over all
# IBM surface facets, then converts total force to u* via force balance:
#   u*² = F_resultant / L_x
##### Horizontal Integration surface #####
    I_tau_yx = 0
    I_tau_yz = 0
    for (j, i_srt, i_end) in horiz_surfaces:
        if i_end - i_srt < 2:
            continue
        tmp_dudy = du_dy[j, i_srt:i_end]; x_tmp = x[i_srt:i_end]
        tmp_dwdy = dw_dy[j, i_srt:i_end]
        tmp_dvdx = dv_dx[j, i_srt:i_end]
        # Full viscous traction on horizontal face (normal = ŷ):
        # t_x = ν(∂u/∂y + ∂v/∂x); previously only ∂u/∂y was included
        tau_yx = nu * simpson(y=(tmp_dudy + tmp_dvdx), x=x_tmp)
        tau_yz = nu * simpson(y=tmp_dwdy, x=x_tmp)
        # Summing up integrals of all horizontal surfaces
        I_tau_yx += tau_yx
        I_tau_yz += tau_yz

##### Integration over vertical left flank surfaces #####
    I_tau_xy1 = 0
    I_tau_xz1 = 0
    for (j_srt, j_end, i) in left_flank_surfaces:
        if j_end - j_srt < 2:
            continue
        y_tmp = y[j_srt:j_end]
        # tau_xy: lift/sink force on valley wall (diagnostic)
        # tau_xz: spanwise viscous drag on the vertical face (genuine horizontal drag)
        tau_xy1 = nu * trapezoid(y=dv_dx[j_srt:j_end, i], x=y_tmp)
        tau_xz1 = nu * trapezoid(y=dw_dx[j_srt:j_end, i], x=y_tmp)
        I_tau_xy1 += tau_xy1
        I_tau_xz1 += tau_xz1

##### Integration over vertical right flank surfaces #####
    I_tau_xy2 = 0
    I_tau_xz2 = 0
    for (j_srt, j_end, i) in right_flank_surfaces:
        if j_end - j_srt < 2:
            continue
        y_tmp = y[j_srt:j_end]
        tau_xy2 = nu * trapezoid(y=dv_dx[j_srt:j_end, i], x=y_tmp)
        tau_xz2 = nu * trapezoid(y=dw_dx[j_srt:j_end, i], x=y_tmp)
        I_tau_xy2 += tau_xy2
        I_tau_xz2 += tau_xz2

    # Form (pressure) drag: integrate the dispersive pressure deviation over left (lag)
    # and right (front) vertical wall faces, then take the net retarding force.
    # dispP = local phase-avg pressure minus its intrinsic x-mean at each height.
    P_Lag = []
    P_Front = []
    dispP = (AvgP - avg_c(eps, AvgP, axis=1)[:,np.newaxis]) * mask_intr
    for (j_srt, j_end, i) in left_flank_surfaces:
        if j_end - j_srt < 2:
            continue
        y_tmp = y[j_srt:j_end]
        # Pressure sampled at the first fluid cell to the right of the left wall face
        P_Lag.append(trapezoid(dispP[j_srt:j_end, i + 1], y_tmp))
    for (j_srt, j_end, i) in right_flank_surfaces:
        if j_end - j_srt < 2:
            continue
        y_tmp = y[j_srt:j_end]
        # Pressure sampled at the first fluid cell to the left of the right wall face
        P_Front.append(trapezoid(dispP[j_srt:j_end, i - 1], y_tmp))

    P_Lag = np.array(P_Lag)
    P_Front = np.array(P_Front)

    # FIX (Bug 2): Remove np.abs() — sign carries the physics of form drag.
    # P_Lag = windward face (high pressure), P_Front = leeward face (low pressure).
    # Net retarding form drag = sum(P_Lag) - sum(P_Front); scalar so np.sum() still works.
    P_drag = np.sum(P_Lag) - np.sum(P_Front)
    
    # Fyx: total streamwise drag = skin friction on horizontal surfaces + pressure form drag
    # Fyz: total spanwise drag   = skin friction on horizontal + vertical surfaces
    # Fxy: wall-normal (lift) force — kept for diagnostics, excluded from u_star1
    Fyx = I_tau_yx + P_drag                   # Streamwise: horizontal skin friction + form drag
    Fyz = I_tau_yz + I_tau_xz2 + I_tau_xz1  # Spanwise:   horizontal skin friction + vertical-wall spanwise skin friction
    Fxy = I_tau_xy1 + I_tau_xy2             # Vertical (lift) force on valley walls — diagnostic only

    # FIX (Bug 3): Fxy is a vertical force, not horizontal drag. Remove from u_star1.
    dx_grid = x[1] - x[0]
    L_x = x[-1] + dx_grid                # true periodic domain length
    u_star1 = (((Fyx**2 + Fyz**2)**0.5) / L_x)**0.5   # resultant of the horizontal components

###############################################################################
##################### Method 3 — shifted-column flat-surface drag ############
###############################################################################
    # For each column i the IBM solid occupies j = 0 … n_solid[i]-1.
    # Shifting the column upward by (n_solid[i] - 1) maps:
    #   j = 0  →  last solid cell (U ≈ 0, no-slip satisfied)
    #   j = 1  →  first fluid cell (U > 0)
    # After the shift every column has its no-slip surface at j = 0,
    # so there is no orography and the drag is a plain x-average of ν∂U/∂y|_{j=0}.
    # The Coriolis integral from 0 to the surface is identically zero at every column.

    n_solid_col = np.sum(eps, axis=0).astype(int)   # terrain height [nx]: index of first fluid cell

    U_flat = np.zeros((ny, nx))
    W_flat = np.zeros((ny, nx))
    dy_col = np.zeros(nx)   # original Δy at each column's surface

    for i in range(nx):
        hs    = n_solid_col[i]          # first fluid cell index (= number of solid cells)
        shift = max(hs - 1, 0)          # bring last solid cell (index hs-1) to j=0
        n_cp  = ny - shift
        U_flat[:n_cp, i] = AvgPhU[shift:, i]
        W_flat[:n_cp, i] = AvgPhW[shift:, i]
        U_flat[n_cp:, i] = AvgPhU[-1, i]   # pad top with outermost value
        W_flat[n_cp:, i] = AvgPhW[-1, i]
        hi = max(hs, 1)
        dy_col[i] = y[hi] - y[hi - 1]      # Δy at the physical surface of this column

    # Surface stress per column (first-order gradient at j=0 of shifted arrays)
    # U_flat[0] = last solid cell ≈ 0;  U_flat[1] = first fluid cell
    tau_yx_col = nu * (U_flat[1, :] - U_flat[0, :]) / dy_col
    tau_yz_col = nu * (W_flat[1, :] - W_flat[0, :]) / dy_col

    # Average over all columns — columns with hs==0 (flat floor) use the bottom cell gradient
    tau_yx_m3  = np.mean(tau_yx_col)
    tau_yz_m3  = np.mean(tau_yz_col)
    u_star3    = ((tau_yx_m3**2 + tau_yz_m3**2)**0.5)**0.5

    # ── Diagnostic: break down both methods component-by-component ───────────
    jc = 94  # hill-top index
    rey_uv_avg = avg_c(eps, rey_uv, axis=1)
    rey_vw_avg = avg_c(eps, rey_vw, axis=1)
    disp_yx    = avg_c(eps, DispVelU * DispVelV, axis=1)   # dispersive ⟨ũṽ⟩
    disp_yz    = avg_c(eps, DispVelW * DispVelV, axis=1)   # dispersive ⟨w̃ṽ⟩

    # ── Normal viscous stress on vertical flanks (candidate missing term) ──
    I_tau_xx1 = 0
    I_tau_xx2 = 0
    for (j_srt, j_end, i) in left_flank_surfaces:
        if j_end - j_srt < 2:
            continue
        I_tau_xx1 += nu * 2 * trapezoid(y=du_dx[j_srt:j_end, i], x=y[j_srt:j_end])
    for (j_srt, j_end, i) in right_flank_surfaces:
        if j_end - j_srt < 2:
            continue
        I_tau_xx2 += nu * 2 * trapezoid(y=du_dx[j_srt:j_end, i], x=y[j_srt:j_end])
    I_tau_xx_net = I_tau_xx1 - I_tau_xx2   # net streamwise normal-viscous on flanks

    visc_yz_avg = visc_yz   # visc_yz is already the intrinsic x-averaged 1-D profile

    print("\n══ Method 2  components at z[94] ══════════════════════════════")
    print(f"  Coriolis-yx  −I_corr_yx : {-I_corr_yx[jc]:+.6f}")
    print(f"  Viscous-yx    visc_yx   : {visc_yx[jc]:+.6f}")
    print(f"  Reynolds-yx  −rey_uv    : {-rey_uv_avg[jc]:+.6f}")
    print(f"  total_tau_yx            : {total_tau_yx[jc]:+.6f}")
    print("  ---")
    print(f"  Coriolis-yz  +I_corr_yz : {I_corr_yz[jc]:+.6f}")
    print(f"  Viscous-yz    visc_yz   : {visc_yz_avg[jc]:+.6f}")
    print(f"  Reynolds-yz  +rey_vw    : {rey_vw_avg[jc]:+.6f}")
    print(f"  total_tau_yz            : {total_tau_yz[jc]:+.6f}")
    print("  ---")
    print(f"  tau_corrctn (ν∂v/∂x)   : {tau_corrctn[jc]:+.6f}")
    print(f"  dispersive yx  ⟨ũṽ⟩    : {disp_yx[jc]:+.6f}")
    print(f"  dispersive yz  ⟨w̃ṽ⟩    : {disp_yz[jc]:+.6f}")
    print(f"  u_star2[94]             : {u_star2[jc]:.6f}")

    print("\n══ Method 1  components ════════════════════════════════════════")
    print(f"  Horiz skin-friction yx  I_tau_yx          : {I_tau_yx:+.6f}")
    print(f"  Horiz skin-friction yz  I_tau_yz          : {I_tau_yz:+.6f}")
    print(f"  Form drag               P_drag            : {P_drag:+.6f}")
    print(f"  Flank spanwise viscous  I_tau_xz1+xz2     : {I_tau_xz1+I_tau_xz2:+.6f}")
    print(f"  Flank normal viscous    I_tau_xx_net      : {I_tau_xx_net:+.6f}  ← candidate")
    print(f"  Domain length           L_x               : {L_x:.6f}")
    print(f"  Fyx / L_x                                 : {Fyx/L_x:+.6f}")
    print(f"  Fyz / L_x                                 : {Fyz/L_x:+.6f}")
    print(f"  u_star1                                   : {u_star1:.6f}")

    print("\n══ Method 3 (shifted-column flat-surface) ══════════════════════")
    print(f"  tau_yx_m3 (streamwise x-avg)  : {tau_yx_m3:+.6f}")
    print(f"  tau_yz_m3 (spanwise  x-avg)   : {tau_yz_m3:+.6f}")
    print(f"  u_star3                        : {u_star3:.6f}")
    print(f"  columns used                   : {nx} / {nx}")

    print("\n══ Three-method comparison (ref = u_star = mean of Method 2 profile) ══")
    ref = u_star
    for label, val in [("Method 2 mean (reference) ", u_star),
                        ("Method 1 (surface integ.) ", u_star1),
                        ("Method 2 [z=0]            ", u_star2[0]),
                        ("Method 3 (col-shift)      ", u_star3)]:
        d = (val - ref) / ref * 100
        print(f"  {label} : {val:.6f}  ({d:+.2f}% vs ref)")

    diff_pct_m1 = (u_star1 - u_star) / u_star * 100
    diff_pct_m3 = (u_star3 - u_star) / u_star * 100
    print(f"\n[u* check]  M2(mean)={u_star:.4f}  M1={u_star1:.4f} ({diff_pct_m1:+.2f}%)  M3={u_star3:.4f} ({diff_pct_m3:+.2f}%)")
    # import sys
    # sys.exit(0)

    # Advection term u_j ∂u/∂x_j sampled at four characteristic orographic locations.
    # Column slices are narrow averages around each landmark; indices depend on grid resolution.
    # Advection profile over valley
    ## Hill top x = 0
    conv_top = np.mean(AvgPhU[94:,0:5],axis=1) * np.mean(du_dx[94:,0:5],axis=1) + np.mean(AvgPhV[94:,0:5],axis=1) * np.mean(du_dy[94:,0:5],axis=1)                                           # hill top change values if eps is changed
    conv_lf = np.mean(AvgPhU[flk_hgt:,lf_ind],axis=1) * np.mean(du_dx[flk_hgt:,lf_ind],axis=1) + np.mean(AvgPhV[flk_hgt:,lf_ind],axis=1) * np.mean(du_dy[flk_hgt:,lf_ind],axis=1)            # left flank change values if eps is changed
    conv_bottom = np.mean(AvgPhU[:,507:517],axis=1) * np.mean(du_dx[:,507:517],axis=1) + np.mean(AvgPhV[:,507:517],axis=1) * np.mean(du_dy[:,507:517],axis=1)                                # bottom top change values if eps is changed
    conv_rf = np.mean(AvgPhU[flk_hgt:,rf_ind],axis=1) * np.mean(du_dx[flk_hgt:,rf_ind], axis=1) + np.mean(AvgPhV[flk_hgt:,rf_ind],axis=1) * np.mean(du_dy[flk_hgt:,rf_ind],axis=1)           # right flank change values if eps is changed
    
    # Spanwise vorticity of the phase-averaged field: ω_z = ∂v/∂x − ∂u/∂y
    vort_z = dv_dx - du_dy
    # Dispersive vorticity: computed from dispersive velocity only and zeroed inside solid
    # to isolate orographically-induced rotation and gravity waves in the fluid region
    disp_vortz = (dvd_dx - dud_dy)*(1-eps)
    res_dispz = np.sqrt(DispVelV**2+DispVelU**2)

    res_phavg_uv = np.sqrt(AvgPhU**2 + AvgPhV**2)
    
    # Monin-Obukhov Similarity Theory log-law: u⁺ = (1/κ) ln(z⁺) + B
    d = 0.01*u_star
    y0 = 5*u_star
    u_most = (1/kappa)*np.log(y_inner) + 4.5
    u_most[0] = 0
    u_most_v = (1/kappa)*np.log(((y-d)/(y0))/l_in)
    u_most_v[0] = 0
    
    # Tr_u = np.mean(AvgPhU, axis=1)*np.cos(-30*180/np.pi)+np.mean(AvgPhW, axis=1)*np.sin(-30*180/np.pi)
    # Tr_w = np.mean(AvgPhU, axis=1)*np.sin(-30*180/np.pi)-np.mean(AvgPhW, axis=1)*np.cos(-30*180/np.pi)

    # Horizontal (x-direction) profile of TKE, averaged over fluid cells only via avg_c
    # AVG_TKE_V[i]: streamwise-varying TKE profile at the surface level
    # TKE Vertical Profile
    AVG_TKE_V = avg_c(eps, TKE, axis=0)
    
    # AVG_TKE_V_s / AVG_TKE_V_s_i: smooth-case TKE — not available in PhAvg.py
    AVG_TKE_V_s   = np.zeros_like(AVG_TKE_V)
    AVG_TKE_V_s_i = np.zeros_like(AVG_TKE_V)
    
    # Advection of TKE
    dTKE_dx = cd.ddx(TKE)
    dTKE_dy = cd.ddy(TKE)
    Adv = AvgPhU*dTKE_dx + AvgPhV*dTKE_dy

    # IBM body-force magnitude proxy (Plot 1).
    # Approximates the traction the IBM solid exerts on the adjacent fluid as
    # ν * sqrt(|∂U/∂z|² + |∂W_y/∂z|² + |∂V_y/∂z|²) / u*² (met. coords)
    # at the first fluid cell above each column, spread over a 10-cell near-wall band.
    IBM_B_mag = np.zeros((ny, nx))
    for _ic_ibm in range(nx):
        _js_ibm = eps_hgt[_ic_ibm]
        _jt_ibm = min(_js_ibm + 10, ny)
        IBM_B_mag[_js_ibm:_jt_ibm, _ic_ibm] = (nu / u_star**2) * np.sqrt(
            du_dy[_js_ibm:_jt_ibm, _ic_ibm]**2
            + dv_dy[_js_ibm:_jt_ibm, _ic_ibm]**2
            + dw_dy[_js_ibm:_jt_ibm, _ic_ibm]**2
        )

    # Wall shear stress along the IBM surface, normalised by u*² (Plot 3).
    # τzx = ν ∂U/∂z  (streamwise component in met. coords)
    # τzy = ν ∂V_y/∂z  (spanwise component; V_y = engineering W in met. convention)
    # Evaluated at the first fluid cell above each IBM column (eps_hgt[i]).
    _j_surf_idx = np.minimum(eps_hgt, ny - 1)   # surface row index per column, shape (nx,)
    tau_wx = nu * du_dy[_j_surf_idx, np.arange(nx)] / u_star**2
    tau_wz = nu * dw_dy[_j_surf_idx, np.arange(nx)] / u_star**2
    tau_wm = np.sqrt(tau_wx**2 + tau_wz**2)

    # Bundle all post-processed fields listed in var_names (defined in config) into a dict
    # and pickle it so compile_results.py can assemble multi-case comparisons.
    # save varaibels in library for compiling results
    sim1_results = {name: globals()[name] for name in var_names}
    with open('sim1_results.pkl', 'wb') as f:
        pickle.dump(sim1_results, f)
        
    # delta_u_plus, B_s, B_r = calculate_roughness_function(y_s, U_s/u_star, y_in, u_plus_rot/u_star, 0.0618, 0.068, nu, 0.41)
    # phi = solve_compact_geopotential(PhAvgPU, PhAvgPV, x, y, nx, ny, eps, du_dx, dv_dy)
    
    # %%###########################################################################
    # ─── Smooth case (flat wall, neutral, Re=500) — loaded once, used in all plots ─
    # Centralised in functions.load_smooth_case so PhAvg.py and results.py share
    # exactly one implementation (single source of truth — cannot diverge).
    _sm = load_smooth_case(smooth_nc_path, x, nu, Re_lambda)
    sy = _sm['sy']; nys = _sm['nys']
    U_s = _sm['U_s']; V_s = _sm['V_s']; W_s = _sm['W_s']
    su = _sm['su']; sw = _sm['sw']; alpha_s = _sm['alpha_s']
    ustr_s1 = _sm['ustr_s1']; alpha_str_s = _sm['alpha_str_s']
    y_s = _sm['y_s']; y_s_p = _sm['y_s_p']
    rU_s = _sm['rU_s']; rV_s = _sm['rV_s']; rW_s = _sm['rW_s']
    G_x_s = _sm['G_x_s']; G_z_s = _sm['G_z_s']; G_s = _sm['G_s']
    U_s_p = _sm['U_s_p']; W_s_p = _sm['W_s_p']
    GblU_s = _sm['GblU_s']; GblW_s = _sm['GblW_s']
    Rxx_s = _sm['Rxx_s']; Rxy_s = _sm['Rxy_s']; Ryy_s = _sm['Ryy_s']
    Ryz_s = _sm['Ryz_s']; Rzz_s = _sm['Rzz_s']
    TKE_s = _sm['TKE_s']; case_v_s = _sm['case_v_s']
    cor_yx_s = _sm['cor_yx_s']; I_corr_yx_s = _sm['I_corr_yx_s']
    du_dy_s = _sm['du_dy_s']; visc_yx_s = _sm['visc_yx_s']; tau_yx_s = _sm['tau_yx_s']
    cor_yz_s = _sm['cor_yz_s']; I_corr_yz_s = _sm['I_corr_yz_s']
    dw_dy_s = _sm['dw_dy_s']; visc_yz_s = _sm['visc_yz_s']; tau_yz_s = _sm['tau_yz_s']
    AVG_TKE_V_s = _sm['AVG_TKE_V_s']; x_s = _sm['x_s']; AVG_TKE_V_s_i = _sm['AVG_TKE_V_s_i']
    ustr_M2_s = _sm['ustr_M2_s']; ustr_M2_plateau_s = _sm['ustr_M2_plateau_s']  # Method-2 (vs stored ustr_s1)
    SMOOTH_COLOR = 'grey'
    SMOOTH_LS = '--'

    # ─── Rough reference (Kostelecky r1, Re=1000) — same loader, Method-2 u* ──────
    # Loaded through the SAME shared core as the smooth case (functions.load_ekman_nc_case).
    # The .nc has identical variable names but no stored FrictionVelocity, so u* is the
    # momentum-integral (Method-2) plateau.  Overlaid on comparison plots only when
    # plot_ref_rough is True (config master switch).
    ROUGH_COLOR = 'green'
    ROUGH_LS = ':'
    _rg = load_ekman_nc_case(rough_nc_path, x, nu_rough, Re_lambda_rough)
    y_r = _rg['y']; y_r_p = _rg['y_p']; nys_r = _rg['nys']
    su_r = _rg['su']; sw_r = _rg['sw']; alpha_r = _rg['alpha']
    U_r_p = _rg['U_p']; W_r_p = _rg['W_p']; GblU_r = _rg['GblU']; GblW_r = _rg['GblW']
    Rxy_r = _rg['Rxy']; Ryz_r = _rg['Ryz']; TKE_r = _rg['TKE']
    I_corr_yx_r = _rg['I_corr_yx']; visc_yx_r = _rg['visc_yx']; tau_yx_r = _rg['tau_yx']
    I_corr_yz_r = _rg['I_corr_yz']; visc_yz_r = _rg['visc_yz']; tau_yz_r = _rg['tau_yz']
    AVG_TKE_V_r = _rg['AVG_TKE_V']
    ustr_M2_r = _rg['ustr_M2']                 # Method-2 u*(z) profile
    ustr_r1   = _rg['ustr_M2_plateau']         # representative (plateau) value

    # ── Method-2 friction velocity for ALL cases (orographic / smooth / rough) ───
    # u* read as the constant-flux plateau of the Method-2 u*(z) profile.
    ustr_M2_o          = u_star2                          # orographic profile (this run)
    ustr_M2_plateau_o  = plateau_value(u_star2, y_inner)  # orographic plateau
    print("\n══ Method-2 friction velocity (Ekman momentum-integral) — all cases ══")
    print(f"  {'case':<26}{'u* (Method 2)':>14}{'reference':>16}")
    print(f"  {'orographic Re=500':<26}{ustr_M2_plateau_o:>14.5f}{'(u_star2 plateau)':>16}")
    print(f"  {'smooth Re=500':<26}{ustr_M2_plateau_s:>14.5f}{ustr_s1:>16.5f}"
          f"   Δ={100*(ustr_M2_plateau_s-ustr_s1)/ustr_s1:+.1f}%  (stored FrictionVelocity)")
    print(f"  {'  smooth near-wall':<26}{float(ustr_M2_s[0]):>14.5f}{ustr_s1:>16.5f}"
          f"   Δ={100*(float(ustr_M2_s[0])-ustr_s1)/ustr_s1:+.1f}%")
    print(f"  {'rough r1 Re=1000':<26}{ustr_r1:>14.5f}{'(no stored u*)':>16}")

    # ══════════════════════════════════════════════════════════════════════════
    # ═══  KEY FLOW PARAMETERS — smooth vs orographic comparison  ═════════════
    # ══════════════════════════════════════════════════════════════════════════
    _fp_intg = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz  # numpy ≥2.0 renamed trapz

    # Reference friction velocities
    _fp_ustar_o  = float(u_star2[hill_hgt])   # Method-2 u* at hill-crest height
    _fp_ustar_o2 = _fp_ustar_o ** 2
    _fp_ustar_s  = float(ustr_s1)             # smooth-wall u* (FrictionVelocity mean)
    _fp_ustar_s2 = _fp_ustar_s ** 2

    # ── (2) Friction Re_τ = u*²/ν  (code convention from config.py) ──────────
    _fp_Retau_o = _fp_ustar_o2 / nu
    _fp_Retau_s = _fp_ustar_s2 / nu

    # ── (4) Peak Ekman cross-flow ratio: max|W_geo| / G_∞ ────────────────────
    # w_plus_rot: geo-frame spanwise dimensional velocity (1-D intrinsic avg)
    _fp_ekman_o = float(np.max(np.abs(w_plus_rot))) / float(G_inf)
    # Smooth: W_s shape (ny_s, nz_s) — average over span, then peak over height
    _fp_Ws_1d   = np.mean(W_s, axis=1)
    _fp_ekman_s = float(np.max(np.abs(_fp_Ws_1d))) / float(G_s)

    # ── Rotate orographic stress (Method 2) to geostrophic frame ─────────────
    #   tau_stream = tau_yx·cos α − tau_yz·sin α   [streamwise geo]
    #   tau_span   = tau_yx·sin α + tau_yz·cos α   [spanwise   geo]
    _fp_tyx_hh = float(total_tau_yx[hill_hgt])
    _fp_tyz_hh = float(total_tau_yz[hill_hgt])
    _fp_tyx_H = float(total_tau_yx[-1])
    _fp_tyz_H = float(total_tau_yz[-1])
    
    # Smooth: tau_yx_s / tau_yz_s are already in geo frame — evaluate at wall (j=0)
    _fp_tstr_s = float(tau_yx_s[0])
    _fp_tspn_s = float(tau_yz_s[0])

    # ── (5) τ_stream/u*² — flat / column-shift (Method 3) ────────────────────
    _fp_tau5_o  = _fp_tyx_H / _fp_ustar_o2
    _fp_tau5_s  = _fp_tstr_s  / _fp_ustar_s2

    # ── (6) τ_stream/u*² — total valley surface (skin + form, Method 1, geo) ─
    _fp_tau6_o   = (Fyx / L_x) / _fp_ustar_o2

    # ── (7a) Surface veer angle (deg) = arctan(τ_span / τ_stream) ────────────
    _fp_veer_ho  = np.abs(float(np.degrees(np.arctan2(_fp_tyz_hh, _fp_tyx_hh))))
    _fp_veer_Ho  = np.abs(float(np.degrees(np.arctan2(_fp_tyz_H, _fp_tyx_H))))
    _fp_veer_s  = np.abs(float(np.degrees(np.arctan2(_fp_tspn_s, _fp_tstr_s))))

    # ── (7b) Veer amplification α_oro / α_smooth ─────────────────────────────
    _fp_veer_amp = _fp_veer_Ho / _fp_veer_s if _fp_veer_s != 0.0 else np.nan

    # ── (9) Near-wall power law  u+ = a·z+^n  (valley, z+ < h+) ─────────────
    # Use u_h_plus = u_plus_rot/u_star and y_inner = y*u_star/nu (as in log-law fit)
    _fp_hplus   = float(y_inner[hill_hgt])
    _fp_pw_mask = (y_inner > 0) & (y_inner < _fp_hplus) & (u_h_plus > 1e-6)
    _fp_pw_z    = y_inner[_fp_pw_mask]
    _fp_pw_u    = u_h_plus[_fp_pw_mask]
    _fp_pw_a, _fp_pw_n = np.nan, np.nan
    if _fp_pw_z.size >= 3:
        try:
            _fp_popt, _ = curve_fit(power_law_model, _fp_pw_z, _fp_pw_u, p0=[0.65, 0.77])
            _fp_pw_a, _fp_pw_n = float(_fp_popt[0]), float(_fp_popt[1])
        except Exception:
            pass

    # ── (10,11) TKE peak value and peak height ────────────────────────────────
    _fp_TKE1d_o = avg_c(eps, TKE, axis=1)            # intrinsic x-avg TKE, (ny,)
    _fp_TKE1d_s = 0.5 * (np.mean(Rxx_s, axis=1) +   # standard TKE = 0.5*(Rxx+Ryy+Rzz)
                          np.mean(Ryy_s, axis=1) +
                          np.mean(Rzz_s, axis=1))
    _fp_jpk_o   = int(np.argmax(_fp_TKE1d_o))
    _fp_jpk_s   = int(np.argmax(_fp_TKE1d_s))
    _fp_TKEpk_o = float(_fp_TKE1d_o[_fp_jpk_o]) / _fp_ustar_s2
    _fp_TKEpk_s = float(_fp_TKE1d_s[_fp_jpk_s]) / _fp_ustar_s2
    _fp_TKEzp_o = float(y[_fp_jpk_o])    * _fp_ustar_s / nu
    _fp_TKEzp_s = float(y_s[_fp_jpk_s])  * _fp_ustar_s / nu

    # ── (12) Column-integrated TKE at valley centre (valley-floor column) ─────
    _fp_i_vbot  = int(np.argmin(eps_hgt))     # valley floor: fewest solid cells
    _fp_j0v     = int(eps_hgt[_fp_i_vbot])
    _fp_TKEvc_o = float(_fp_intg(TKE[_fp_j0v:, _fp_i_vbot], y[_fp_j0v:])) / _fp_ustar_s2
    # Full intrinsic x-averaged integral (for comparison with smooth)
    _fp_TKEint_o = float(_fp_intg(_fp_TKE1d_o, y))   / _fp_ustar_s2
    _fp_TKEint_s = float(_fp_intg(_fp_TKE1d_s, y_s)) / _fp_ustar_s2

    # ── (13) Column-integrated TKE at valley crest (hill-top column) ──────────
    _fp_i_crest = int(np.argmax(eps_hgt))     # hill crest: most solid cells
    _fp_j0c     = int(eps_hgt[_fp_i_crest])
    _fp_TKEcr_o = float(_fp_intg(TKE[_fp_j0c:, _fp_i_crest], y[_fp_j0c:])) / _fp_ustar_s2

    # ── (14–16) Dispersive W-peak amplitude and height (z+ < 200) ────────────
    _fp_zlim200  = 200.0 * nu / _fp_ustar_o
    _fp_jlim200  = max(1, int(np.searchsorted(y, _fp_zlim200)))
    _fp_DW       = DispVelW[:_fp_jlim200, :]          # near-surface dispersive W
    # Windward = column with max surface pressure; Lee = column with min
    _fp_i_wind   = int(np.argmax(P_surf))
    _fp_i_lee    = int(np.argmin(P_surf))
    _fp_Wwind    = float(np.max(np.abs(_fp_DW[:, _fp_i_wind]))) / _fp_ustar_o
    _fp_Wlee     = float(np.max(np.abs(_fp_DW[:, _fp_i_lee ]))) / _fp_ustar_o
    _fp_Wlee_j   = int(np.argmax(np.abs(_fp_DW[:, _fp_i_lee])))
    _fp_Wlee_zp  = float(y[_fp_Wlee_j]) * _fp_ustar_o / nu

    # ── (17) Pressure jump: windward Cp − lee Cp ─────────────────────────────
    _fp_dCp      = float(Cp[_fp_i_wind]) - float(Cp[_fp_i_lee])

    # ── (18) Surface pressure coefficient minimum (lee side) ─────────────────
    _fp_Cplee    = float(np.min(Cp))

    # ── (19) APG transition: first x where surface dP/dx changes sign to > 0 ─
    _fp_dPdx_sf  = dP_dx[eps_hgt, np.arange(nx)]
    _fp_apg_idx  = np.where(np.diff(np.sign(_fp_dPdx_sf)) > 0)[0]
    _fp_APGxp    = (float(x[_fp_apg_idx[0]]) * _fp_ustar_o / nu
                    if len(_fp_apg_idx) > 0 else np.nan)

    # ── (20) Orographic form drag (Cp-normalised) ─────────────────────────────
    _fp_Dform    = float(D_form_oro) / (0.5 * float(G_inf)**2)

    # ── (21) Drag coefficient CD = u*² / G²  ─────────────────────────────────
    _fp_CD_o     = float(u_star1)**2 / float(G_inf)**2
    _fp_CD_s     = _fp_ustar_s2      / float(G_s)**2

    # ── (22) Total spanwise stress / u*² (geo frame, Method 2 at hill_hgt) ───
    _fp_Fspn_geo = Fyx * alphasin + Fyz * alphacos
    _fp_tau22_o  = (_fp_Fspn_geo / L_x) / _fp_ustar_o2
    _fp_tau22_s  = _fp_tspn_s / _fp_ustar_s2

    # ── Print parameter table ──────────────────────────────────────────────────
    _fp_NA  = 'n/a'
    _fp_sep = '═' * 72
    _fp_div = '─' * 72
    print(f'\n{_fp_sep}')
    print(f'  KEY FLOW PARAMETERS         {"Smooth":>20s}  {"Orographic":>18s}')
    print(_fp_sep)
    print(f'  (1)  Friction velocity u*   {_fp_ustar_s:>20.5f}  {_fp_ustar_o:>18.5f}')
    print(f'  (2)  Re_τ = u*²/ν           {_fp_Retau_s:>20.1f}  {_fp_Retau_o:>18.1f}')
    print(f'  (3)  Geostrophic speed G_∞  {float(G_s):>20.5f}  {float(G_inf):>18.5f}')
    print(f'  (4)  Peak cross-flow W/G    {_fp_ekman_s:>20.5f}  {_fp_ekman_o:>18.5f}')
    print(_fp_div)
    print(f'  (5)  τ_str/u*² (col-shift)  {fmt_val(_fp_tau5_s):>20s}  {fmt_val(_fp_tau5_o):>18s}')
    print(f'  (6)  τ_str/u*² (total surf) {_fp_NA:>20s}  {fmt_val(_fp_tau6_o):>18s}')
    print(f'  (7a) Veer angle (deg)       {_fp_veer_s:>20.2f}  {_fp_veer_Ho:>18.2f}')
    print(f'  (7b) Veer amplification     {"1.00":>20s}  {fmt_val(_fp_veer_amp):>18s}')
    print(_fp_div)
    print(f'  (8)  Log-law κ              {0.41:>20.5f}  {fmt_val(float(0.43)):>18s}')
    print(f'  (9)  Power law u+=a·z+^n   {"n/a":>20s}  a={_fp_pw_a:.4f}  n={_fp_pw_n:.4f}')
    print(_fp_div)
    print(f'  (10) TKE peak / u*²         {fmt_val(_fp_TKEpk_s):>20s}  {fmt_val(_fp_TKEpk_o):>18s}')
    print(f'  (11) TKE peak height z+     {_fp_TKEzp_s:>20.2f}  {_fp_TKEzp_o:>18.2f}')
    print(f'  (12) ∫TKE dy/u*² valley ctr {_fp_NA:>19s}  {fmt_val(_fp_TKEvc_o):>18s}')
    print(f'       ∫TKE dy/u*² (x-avg)   {fmt_val(_fp_TKEint_s):>20s}  {fmt_val(_fp_TKEint_o):>18s}')
    print(f'  (13) ∫TKE dy/u*² valley top {_fp_NA:>19s}  {fmt_val(_fp_TKEcr_o):>18s}')
    print(_fp_div)
    print(f'  (14) Windward |ΔW_disp|/u*  {_fp_NA:>20s}  {fmt_val(_fp_Wwind):>18s}')
    print(f'  (15) Lee      |ΔW_disp|/u*  {_fp_NA:>20s}  {fmt_val(_fp_Wlee):>18s}')
    print(f'  (16) Lee W-peak height z+   {_fp_NA:>20s}  {_fp_Wlee_zp:>18.2f}')
    print(_fp_div)
    print(f'  (17) Pressure jump ΔCp      {_fp_NA:>20s}  {fmt_val(_fp_dCp):>18s}')
    print(f'  (18) Cp_lee (surf min)      {_fp_NA:>20s}  {fmt_val(_fp_Cplee):>18s}')
    print(f'  (19) APG transition x+      {_fp_NA:>20s}  {fmt_val(_fp_APGxp):>18s}')
    print(_fp_div)
    print(f'  (20) Form drag/(0.5 G²)     {0.0:>20.5f}  {fmt_val(_fp_Dform):>18s}')
    print(f'  (21) Drag coeff CD=u*²/G²   {fmt_val(_fp_CD_s):>20s}  {fmt_val(_fp_CD_o):>18s}')
    print(f'  (22) τ_span/u*² (geo)       {fmt_val(_fp_tau22_s):>20s}  {fmt_val(_fp_tau22_o):>18s}')
    print(f'{_fp_sep}\n')
    # ══════════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════════════
    # ═══  CONSOLIDATED RESULTS SUMMARY (orographic case)  ═════════════════════
    # ══════════════════════════════════════════════════════════════════════════
    # Single aligned table of the headline geometric and flow quantities, so the
    # user sees the important values at a glance without hunting through plots.
    _sum_CD        = float(u_star1)**2 / float(G_inf)**2          # drag coefficient
    _sum_turn_deg  = float(np.degrees(np.arctan(np.ravel(inst_alpha)[hill_hgt])))
    print_summary_table('RESULTS SUMMARY — orographic case', [
        ('section', 'Geometry'),
        ('Valley crest height h (grid cells)',   hill_hgt,                  'd'),
        ('Valley crest height h (physical)',     float(y[hill_hgt]),        '.5e'),
        ('Valley crest height h+ (wall units)',  float(y_in[hill_hgt]),     '.2f'),
        ('Flank-top index',                      int(flk_hgt),              'd'),
        ('Flank width (grid cells)',             int(len(flk_wdt)),         'd'),
        ('section', 'Flow parameters'),
        ('Reynolds number Re',                   Re,                        '.0f'),
        ('Friction Reynolds number Re_tau',      Re_tau,                    '.1f'),
        ('Kinematic viscosity nu',               nu,                        '.5e'),
        ('Geostrophic speed G_inf',              G_inf,                     '.5f'),
        ('section', 'Friction velocity (4 methods)'),
        ('u* - M2 (mean momentum balance)',      u_star,                    '.5f'),
        ('u* - M2 at crest  u_star2[h]',         float(u_star2[hill_hgt]),  '.5f'),
        ('u* - M1 (surface integral)',           u_star1,                   '.5f'),
        ('u* - M3 (shifted column)',             u_star3,                   '.5f'),
        ('Surface turning angle (deg)',          _sum_turn_deg,             '.2f'),
        ('section', 'Drag'),
        ('Orographic form drag D_form',          D_form_oro,                '.6f'),
        ('Pressure (form) drag P_drag',          P_drag,                    '.6f'),
        ('Streamwise force Fyx',                  Fyx,                       '.6f'),
        ('Spanwise force Fyz',                    Fyz,                       '.6f'),
        ('Drag coefficient CD = u*1^2 / G^2',    _sum_CD,                   '.5f'),
        ('section', 'Curve fits'),
        ('Log-law kappa',                        kappa_loglaw,              '.4f'),
        ('Log-law displacement d_m+',            d_m_loglaw,                '.2f'),
        ('Log-law roughness z0m+',               z0m_loglaw,                '.5f'),
        ('Log-law fit R^2',                      _best_r2,                  '.4f'),
        ('Canopy attenuation alpha',             alpha_canopy,              '.4f'),
    ])
    # ──────────────────────────────────────────────────────────────────────────


# %%
if (1 == plotRes):
    res_dispz    = np.sqrt(DispVelU**2 + DispVelV**2)
    res_phavg_uv = np.sqrt(AvgPhU**2 + AvgPhV**2)

    # ─────────────────────────────────────────────────────────────────────
    # Valley-crest height + boundary-layer sublayer levels (inner units z+).
    # The dashed 'h' line (mark_h) sits at y_in[h_idx] = y_in[94] (h_idx defined
    # below; the crest grid index, not the dynamic hill_hgt which here = 93).
    # Discrete on-curve markers (see functions.mark_layers / LAYER_MARKER_NAMES):
    #   'o' viscous top  z+≈5  |  's' canopy top / roughness start (oro)
    #   '^' log start (smooth z+≈30 / oro z+≈75)
    #   'D' log top   (smooth z+≈100 / oro z+≈200)
    # Layer structure (for future reference):
    #   Smooth     : viscous(≤5) | buffer(5–30)         | log(30–100)
    #   Orographic : viscous(≤5) | canopy(5–canopy_top) | roughness(canopy_top–75)
    #                | log(75–200)
    #   canopy_top = z+ where the x-averaged dispersive stress uv_t (= UV_disp)
    #                peaks and begins to dissipate (start of the roughness layer).
    # filled marker = valley curve ; hollow marker = smooth-wall curve.
    # ─────────────────────────────────────────────────────────────────────
    def _zidx(z_arr, zval):
        """Index of inner-unit height array z_arr closest to target z+ = zval."""
        return int(np.argmin(np.abs(np.asarray(z_arr) - zval)))

    # Valley-crest grid index for the 'h' line.  Fixed at 94 = y_in[94], the
    # project's crest-index convention (cf. conv_top = AvgPhU[94:,...]).  NOTE:
    # the dynamic hill_hgt = np.max(eps_hgt) - 1 evaluates to 93 for the eps in
    # this directory, so 'h' uses h_idx (=94) explicitly, not hill_hgt.
    h_idx = 94

    # Orographic (valley) sublayer indices on y_in  (= y_inner = y·u*/ν)
    _iv_visc   = _zidx(y_in,   5)    # viscous sublayer top      z+ ≈ 5
    _iv_buf    = _zidx(y_in,  30)    # buffer-height reference    z+ ≈ 30 (requested)
    _iv_logbeg = _zidx(y_in,  75)    # roughness top / log start  z+ ≈ 75
    _iv_logend = _zidx(y_in, 200)    # log-layer top              z+ ≈ 200
    # canopy top / roughness-layer start: the (positive) peak of the x-averaged
    # dispersive stress uv_t (loaded as UV_disp), where it stops rising and
    # begins to dissipate.  Solid cells are zeroed (UV_disp*mask0) before the
    # intrinsic (fluid) average, because the dispersive velocity is non-zero
    # inside the IBM body.  Signed argmax (not |·|): the profile reverses sign
    # higher up, so the first positive peak ≈ crest height is the canopy top.
    _disp_uv_prof = avg_c(eps, UV_disp*mask0, axis=1)
    _iv_canopy = int(np.argmax(_disp_uv_prof[:_iv_logbeg + 1]))

    # Smooth-wall sublayer indices on y_s_p  (smooth inner coordinate)
    _is_visc   = _zidx(y_s_p,   5)   # viscous sublayer top   z+ ≈ 5
    _is_buf    = _zidx(y_s_p,  30)   # buffer top / log start z+ ≈ 30
    _is_logend = _zidx(y_s_p, 100)   # log-layer top          z+ ≈ 100

    # symbol -> index marker dictionaries passed to mark_layers()
    _LYR_ORO = {'o': _iv_visc, 's': _iv_canopy, '^': _iv_logbeg, 'D': _iv_logend}
    _LYR_SMO = {'o': _is_visc, '^': _is_buf,    'D': _is_logend}

    print('Sublayer indices (oro): viscous z+=%.1f@%d, canopy z+=%.1f@%d, '
          'log-start z+=%.1f@%d, log-top z+=%.1f@%d'
          % (y_in[_iv_visc], _iv_visc, y_in[_iv_canopy], _iv_canopy,
             y_in[_iv_logbeg], _iv_logbeg, y_in[_iv_logend], _iv_logend))
    print('Sublayer indices (smooth): viscous z+=%.1f@%d, buffer z+=%.1f@%d, '
          'log-top z+=%.1f@%d'
          % (y_s_p[_is_visc], _is_visc, y_s_p[_is_buf], _is_buf,
             y_s_p[_is_logend], _is_logend))

    # %% ###########################################################################
    # All plots use inner-scaled coordinates (x_in = x/l_in, y_in = y/l_in) unless noted.
    # Orography outline (x_oro_in, y_oro_in) is overlaid on 2-D colour maps.
    # Phase Average
    # [PLOT 01] PhAvgU
    plot2D_div(x_in, y_in[:limity], AvgPhU[:limity,:],'',r'$\left\langle\overline{(U_y)}\right\rangle(x, z)$',r'$x^+$',r'$z^+$', cwd + '/fig/' + 'PhAvgU' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 02] PhAvgW
    plot2D_div(x_in, y_in[:limity], AvgPhV[:limity,:],'',r'$\left\langle\overline{(W_y)}\right\rangle(x, z)$',r'$x^+$',r'$z^+$', cwd + '/fig/' + 'PhAvgW' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 03] PhAvgV
    plot2D_div(x_in, y_in[:limity], AvgPhW[:limity,:],'',r'$\left\langle\overline{(V_y)}\right\rangle(x, z)$',r'$x^+$',r'$z^+$', cwd + '/fig/' + 'PhAvgV' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 04] Pressure
    plot2D_div(x_in, y_in[:limity], AvgP[:limity,:],'',r'$\left\langle\overline{(P_y)}\right\rangle(x, z)$',r'$x^+$',r'$z^+$', cwd + '/fig/' + 'Pressure' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 05] Potential Temperature
    plot2D_div(x_in, y_in[:limity], AvgScal[:limity,:],'',r'$\left\langle\overline{(\theta)}\right\rangle(x, z)$',r'$x^+$',r'$z^+$', cwd + '/fig/' + 'Potential Temperature' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 06] (field map)
    plot_phavg_velocity_3D(x_in, y_in[:limity],
                           AvgPhU[:limity,:], AvgPhV[:limity,:], AvgPhW[:limity,:],
                           eps[:limity,:], 1000,
                           x_oro_in, y_oro_in,
                           cwd + '/fig/' + 'PhAvg_3D_velocity.png')

    # %% ###########################################################################
    # Dispersive Velocity Component
    # [PLOT 07] DispU
    plot2D_div(x_in, y_in[:limity], DispVelU[:limity,:],'',r'$\widetilde{U}_y(x,z) = \left\langle\overline{(U_y)}\right\rangle(x, z) - (\langle \overline{U}\rangle) (z)$', r'$x^+$',r'$z^+$', cwd + '/fig/' + 'DispU' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 08] DispW
    plot2D_div(x_in, y_in[:limity], DispVelV[:limity,:],'',r'$\widetilde{W}_y(x,z) = \left\langle\overline{(W_y)}\right\rangle(x, z) - (\langle \overline{W}\rangle) (z)$', r'$x^+$',r'$z^+$', cwd + '/fig/' + 'DispW' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 09] DispV
    plot2D_div(x_in, y_in[:limity], DispVelW[:limity,:],'',r'$\widetilde{V}_y(x,z) = \left\langle\overline{(V_y)}\right\rangle(x, z) - (\langle \overline{V}\rangle) (z)$', r'$x^+$',r'$z^+$', cwd + '/fig/' + 'DispV' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 10] DispP
    plot2D_div(x_in, y_in[:limity], DispP[:limity,:],'',r'$\widetilde{P}(x,z) = \langle\overline{P}\rangle(x,z) - \langle\overline{P}\rangle(z)$',r'$x^+$',r'$z^+$', cwd + '/fig/' + 'DispP' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 11] (field map)
    plot_phavg_velocity_3D(x_in, y_in[:limity],
                           DispVelU[:limity,:], DispVelV[:limity,:], DispVelW[:limity,:],
                           eps[:limity,:], 1000,
                           x_oro_in, y_oro_in,
                           cwd + '/fig/' + 'Disp_3D_velocity.png')
    
    # %% ###########################################################################    
    # Streamlines and vorticity
    # [PLOT 12] Vorticity_Y
    plot2D_div(x_in, y_in[:limity], (vort_z[:limity,:]),'',r'$\langle\omega\rangle_\phi=\nabla \times\langle \overline{(U)}\rangle_\phi$',r'$x$',r'$z$', cwd + '/fig/' + 'Vorticity_Y' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 13] Disp_Vorticity_Y
    plot2D_div(x_in, y_in[:limity], (disp_vortz[:limity,:]),'',r'$\langle\omega\rangle_\phi=\nabla \times\langle \widetilde{(U)}\rangle_\phi$',r'$x$',r'$z$', cwd + '/fig/' + 'Disp_Vorticity_Y' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 14] Dispersive Resultant
    plot2D_streamlines_vorticity(x_in, y_in[:limity], DispVelU[:limity,:], DispVelV[:limity,:],res_dispz[:limity,:],eps[:limity,:],'Dispersive Resultant','',r'$x$',r'$z$', cwd + '/fig/' + 'Dispersive Resultant' + '.png', x_oro_in, y_oro_in ,1000)
    # [PLOT 15] Resultant flow
    plot2D_streamlines_vorticity(x_in, y_in[:limity], AvgPhU[:limity,:], AvgPhV[:limity,:],res_phavg_uv[:limity,:],eps[:limity,:],'Resultant flow','',r'$x$',r'$z$', cwd + '/fig/' + 'Resultant flow' + '.png', x_oro_in, y_oro_in,1000)

    # %% This cannnot be calculated unless one has 3D fields
    # plot2D_streamlines_vorticityX(x, y[:limity], DispVelV[:limity,:], DispVelW[:limity,:],disp_vortx[:limity,:],'','',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlineyz' + '.png', x_oro, y_oro,1000)
    # plot2D_streamlines_vorticityX(x, y[:limity], DispVelU[:limity,:], DispVelW[:limity,:],disp_vorty[:limity,:],'','',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlinezx' + '.png', x_oro, y_oro,1000)
    
    # %% ###########################################################################
    # Plot derivatives
    # [PLOT 16] dv_dx
    plot2D_div(x, y[:limity], dv_dx[:limity,:],'', 'dv_dx',r'$x^{+}$',r'$z^{+}$' , cwd + '/fig/' + 'dv_dx' + '.png', x_oro, y_oro ,1000) # quantity dv/dx where v is vertical component 
    # Streamlines of the phase average
    # [PLOT 17] Streamlinexy
    plot2D_streamlines_vorticityZ(x_in, y_in[:200], DispVelU[:200,:], DispVelV[:200,:], disp_vortz[:200,:],'Stream--vorticity',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlinexy' + '.png', x_oro_in, y_oro_in,1000)
    # [PLOT 18] ResPhXY
    plot2D_div(x, y[:limity], res_phavg_uv[:limity,:],'', 'ResPhXY',r'$x^{+}$',r'$z^{+}$' , cwd + '/fig/' + 'ResPhXY' + '.png', x_oro, y_oro ,1000)
    
    # %%###########################################################################
    # orographic wave drag
    # plot2D_div(x, y, AvgPhU,'','Phase Avg U',r'$x$',r'$z$', cwd + '/fig/' + 'PhAvgU_f' + '.png', x_oro, y_oro, 20)
    # plot2D_div(x, y, AvgPhV,'','Phase Avg W',r'$x$',r'$z$', cwd + '/fig/' + 'PhAvgW_f' + '.png', x_oro, y_oro, 20)
    # plot2D_div(x, y, AvgPhW,'','Phase Avg V',r'$x$',r'$z$', cwd + '/fig/' + 'PhAvgV_f' + '.png', x_oro, y_oro, 20)
    
    # %%###########################################################################
    # TKE
    # [PLOT 19] TKE
    plot2D_div(x_in, y_in[:limity], TKE[:limity,:], '', 'TKE', r'$x^+$',r'$z^+$', cwd + '/fig/' + 'TKE' + '.png', x_oro_in, y_oro_in, 1000)
    
    # %%###########################################################################
    # Reynolds stresses
    # [PLOT 20] Ruu
    plot2D_div(x, y[:limity], rey_uu[:limity,:], '', 'Reynolds stress (Ruu)', r'$x$',r'$z$', cwd + '/fig/' + 'Ruu' + '.png', x_oro, y_oro, 1000)
    # [PLOT 21] Ruw
    plot2D_div(x, y[:limity], rey_uv[:limity,:], '', 'Reynolds stress (Ruw)', r'$x$',r'$z$', cwd + '/fig/' + 'Ruw' + '.png', x_oro, y_oro, 1000)
    # [PLOT 22] Ruv
    plot2D_div(x, y[:limity], rey_uw[:limity,:], '', 'Reynolds stress (Ruv)', r'$x$',r'$z$', cwd + '/fig/' + 'Ruv' + '.png', x_oro, y_oro, 1000)
    # [PLOT 23] Rww
    plot2D_div(x, y[:limity], rey_vv[:limity,:], '', 'Reynolds stress (Rww)', r'$x$',r'$z$', cwd + '/fig/' + 'Rww' + '.png', x_oro, y_oro, 1000)
    # [PLOT 24] Rwv
    plot2D_div(x, y[:limity], rey_vw[:limity,:], '', 'Reynolds stress (Rwv)', r'$x$',r'$z$', cwd + '/fig/' + 'Rwv' + '.png', x_oro, y_oro, 1000)
    # [PLOT 25] Rvv
    plot2D_div(x, y[:limity], rey_ww[:limity,:], '', 'Reynolds stress (Rvv)', r'$x$',r'$z$', cwd + '/fig/' + 'Rvv' + '.png', x_oro, y_oro, 1000)
    
    # %%###########################################################################
    # Vorticity
    # plot2D_div(x, y[:limity], omega_x[:limity,:], '', 'Vorticity X', r'$x$',r'$z$', cwd + '/fig/' + 'VorticityX' + '.png', x_oro, y_oro, 50)
    # plot2D_div(x, y[:300], omega_y[:300,:], '', 'Vorticity Z', r'$x$',r'$z$', cwd + '/fig/' + 'VorticityZ' + '.png', x_oro, y_oro, 50)
    # plot2D_div(x, y[:200], omega_z[:200,:], '', 'Vorticity Y', r'$x$',r'$z$', cwd + '/fig/' + 'VorticityY' + '.png', x_oro, y_oro, 50)
    # plot2D_streamlines_vorticityX(x, y[:limity], AvgPhU[:limity,:], AvgPhV[:limity,:],omega_y[:limity,:],'','',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlinezx' + '.png', x_oro, y_oro,1000)
    
    # %%###########################################################################
    # Vorticity contour map
    # [PLOT 26] Dispersion velocity vorticity in XZ plane
    plt.figure(figsize=(8,6))
    plt.contourf(x, y[:limity], disp_vortz[:limity,:], levels=50, cmap='RdBu_r')  # transpose to match x-y orientation
    plt.colorbar(label='Vorticity (ωz)')
    plt.xlabel('X (streamwise)')
    plt.ylabel('Z (vertical)')
    plt.title('Dispersion velocity vorticity in XZ plane')
    # plt.savefig(savename, dpi=300)
    plt.show()
    

    # %%###########################################################################
    # Hodograph
    # [PLOT 27] Hodograph
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(u_plus_rot, w_plus_rot, label='valley', color='blue', linestyle='-')
    ref_plot(plot_ref_smooth, su, -sw, label='smooth', color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    ref_plot(plot_ref_rough, su_r, -sw_r, label='rough r1', color=ROUGH_COLOR, linestyle=ROUGH_LS)
    # z+ is not an axis here, so layer levels + crest h are placed ON the (u,w)
    # curve via their indices (oro filled, smooth hollow; 'X' = crest h@94).
    mark_layers(u_plus_rot, w_plus_rot, _LYR_ORO, filled=True)
    mark_layers(u_plus_rot, w_plus_rot, {'X': h_idx}, filled=True)
    ref_mark(plot_ref_smooth, mark_layers, su, -sw, _LYR_SMO, filled=False)
    plt.title('Hodograph')
    plt.ylabel(r'$\langle \bar{v} \rangle^{-} $')
    plt.xlabel(r'$\langle \bar{u} \rangle^{-} $')
    plt.legend()
    add_marker_legend()
    plt.grid(True)
    plt.show()
    
    # %%###########################################################################
    # Turning angle
    # [PLOT 28] Rotation angle
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(y_inner[1:], (inst_alpha[1:]*(180/np.pi)), label='valley', color='blue', linestyle='-')
    ref_plot(plot_ref_smooth, y_s_p[1:], -alpha_s[1:]*(180/np.pi), label='smooth', color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    ref_plot(plot_ref_rough, y_r_p[1:], -alpha_r[1:]*(180/np.pi), label='rough r1', color=ROUGH_COLOR, linestyle=ROUGH_LS)
    mark_layers(y_inner, inst_alpha*(180/np.pi), _LYR_ORO, filled=True)
    ref_mark(plot_ref_smooth, mark_layers, y_s_p, -alpha_s*(180/np.pi), _LYR_SMO, filled=False)
    mark_h(y_in[h_idx], 'v')
    plt.title('Rotation angle')
    plt.ylabel(r'$\alpha (\degree)$')
    plt.xlabel(r'$z^{+}$')
    plt.xscale("log")
    plt.legend()
    add_marker_legend()
    plt.grid(True)
    plt.show()
    
    # %%###########################################################################
    # ── Fig-4 convention assembly (single sign convention shared with plot_fig4_budget) ──
    # In the ROTATED frame the geostrophic wind is g ∥ x = (Gx,Gz) = (1,0) for ALL three
    # cases (orographic from config-rotation; smooth/rough from the loader, G_x=1,G_z=0).
    # Each component's curves are built identically (meteorological u=streamwise, v=spanwise,
    # z=wall-normal; Rxy=⟨u'w'⟩, Ryz=⟨v'w'⟩):
    #     C = ∫₀ᶻ(g⊥ − vel) dz' = −I_corr_*     (Coriolis; small near wall, grows outward)
    #     V = +ν d⟨vel⟩/dz                       (Viscous; ≈+1 at wall ÷u*², →0 by z+≈40–60)
    #     R = −⟨flux⟩                            (Reynolds; +hump ≈0.6–0.8 at z+≈30–50)
    #     T = C + V + R                          (Total; ≈const O(1) = surface-stress comp.)
    #   τ_zx:  C_zx=∫(g2−v), V_zx=ν du/dz, R_zx=−Rxy ;  τ_zy:  C_zy=∫(g1−u), V_zy=ν dv/dz, R_zy=−Ryz
    # This is the SAME convention validated against Kostelecky & Ansorge (2024) fig. 4
    # (plots 33a–d).  u* = (T_zx_plateau² + T_zy_plateau²)^¼ is rotation-invariant.
    # orographic (rotated; 1-D intrinsic profiles already)
    Czx_o = -I_corr_yx; Vzx_o = visc_yx; Rzx_o = -turb_yx; Tzx_o = total_tau_yx
    Czy_o = -I_corr_yz; Vzy_o = visc_yz; Rzy_o = -turb_yz; Tzy_o = total_tau_yz
    # smooth reference (loader; collapse the (ny,nt)/(ny,1) arrays to 1-D profiles)
    Czx_s = -I_corr_yx_s; Vzx_s = np.mean(visc_yx_s, axis=1); Rzx_s = -np.mean(Rxy_s, axis=1)
    Tzx_s = Czx_s + Vzx_s + Rzx_s
    Czy_s = -I_corr_yz_s; Vzy_s = np.mean(visc_yz_s, axis=1); Rzy_s = -np.mean(Ryz_s, axis=1)
    Tzy_s = Czy_s + Vzy_s + Rzy_s
    # rough r1 reference (loader)
    Czx_r = -I_corr_yx_r; Vzx_r = np.mean(visc_yx_r, axis=1); Rzx_r = -np.mean(Rxy_r, axis=1)
    Tzx_r = Czx_r + Vzx_r + Rzx_r
    Czy_r = -I_corr_yz_r; Vzy_r = np.mean(visc_yz_r, axis=1); Rzy_r = -np.mean(Ryz_r, axis=1)
    Tzy_r = Czy_r + Vzy_r + Rzy_r

    # %%###########################################################################
    # Shear Stress XY  (Fig-4 convention: Coriolis +C, Viscous +V, Reynolds −⟨flux⟩, Total C+V+R)
    # [PLOT 29] Shear stress $\tau_{zx}$
    plt.figure(figsize=(10, 6))
    plt.plot(y_inner[:], Czx_o[:], label='Coriolis', color='blue', linestyle='-')
    plt.plot(y_inner[:], Vzx_o[:], label='Viscous', color='red', linestyle='-')
    plt.plot(y_inner[:], Rzx_o[:], label='Rey Stress', color='orange', linestyle='-')
    plt.plot(y_inner[:], dudt, label='Temporal', color='saddlebrown', linestyle='-')
    plt.plot(y_inner[:], Tzx_o[:], label='Total', color='black', linestyle='-')
    ref_plot(plot_ref_smooth, y_s_p, Czx_s, color='blue', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Vzx_s, color='red', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Rzx_s, color='orange', linestyle=SMOOTH_LS)
    # rough r1 (Re=1000) overlay — Method-2 terms, own inner units (y_r_p)
    ref_plot(plot_ref_rough, y_r_p, Czx_r, color='blue', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Vzx_r, color='red', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Rzx_r, color='orange', linestyle=ROUGH_LS)
    mark_layers_multi(y_inner, [Czx_o, Vzx_o, Rzx_o,
                                dudt, Tzx_o], _LYR_ORO, filled=True)
    ref_mark(plot_ref_smooth, mark_layers_multi, y_s_p, [Czx_s, Vzx_s,
                              Rzx_s], _LYR_SMO, filled=False)
    mark_h(y_in[h_idx], 'v')
    plt.title(r'Shear stress $\tau_{zx}$')
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'${{\langle \bar{\tau} \rangle}^+}_{zx}$')
    plt.legend(handles=[
        mlines.Line2D([], [], color='blue',       linestyle='-',  label='Coriolis'),
        mlines.Line2D([], [], color='red',        linestyle='-',  label='Viscous'),
        mlines.Line2D([], [], color='orange',     linestyle='-',  label='Rey Stress'),
        mlines.Line2D([], [], color='saddlebrown',linestyle='-',  label='Temporal'),
        mlines.Line2D([], [], color='black',      linestyle='-',  label='Total'),
        mlines.Line2D([], [], color='black',      linestyle='-',  label='Valley'),
        mlines.Line2D([], [], color=SMOOTH_COLOR, linestyle=SMOOTH_LS, label='Smooth'),
    ])
    add_marker_legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Shear Stress XY.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # Zoomed plot
    # [PLOT 30] Shear stress $\tau_{zx}$
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Valley case (solid lines)
    plt.plot(y_inner[:limity], Czx_o[:limity]/u_star**2, color='blue', linestyle='-', label='Coriolis')
    plt.plot(y_inner[:limity], Vzx_o[:limity]/u_star**2, color='red', linestyle='-', label='Viscous')
    plt.plot(y_inner[:limity], Rzx_o[:limity]/u_star**2, color='orange', linestyle='-', label='Rey Stress')
    plt.plot(y_inner[:limity], dudt[:limity]/u_star**2, color='saddlebrown', linestyle='-', label='Temporal')
    plt.plot(y_inner[:limity], Tzx_o[:limity]/u_star**2, color='black', linestyle='-', label='Total')
    # Smooth case (dashed)
    ref_plot(plot_ref_smooth, y_s_p, Czx_s/ustr_s1**2, color='blue', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Vzx_s/ustr_s1**2, color='red', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Rzx_s/ustr_s1**2, color='orange', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, np.zeros(nys), color='saddlebrown', linestyle=SMOOTH_LS)
    # rough r1 — own inner units (ustr_r1)
    ref_plot(plot_ref_rough, y_r_p, Czx_r/ustr_r1**2, color='blue', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Vzx_r/ustr_r1**2, color='red', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Rzx_r/ustr_r1**2, color='orange', linestyle=ROUGH_LS)

    mark_layers_multi(y_inner, [Czx_o/u_star**2, Vzx_o/u_star**2,
                                Rzx_o/u_star**2, dudt/u_star**2,
                                Tzx_o/u_star**2],
                      _LYR_ORO, filled=True)
    ref_mark(plot_ref_smooth, mark_layers_multi, y_s_p, [Czx_s/ustr_s1**2, Vzx_s/ustr_s1**2,
                              Rzx_s/ustr_s1**2, np.zeros(nys)],
                      _LYR_SMO, filled=False)
    mark_h(y_in[h_idx], 'v')
    plt.title(r'Shear stress $\tau_{zx}$')
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'${{\langle \bar{\tau} \rangle}^+}_{zx}$')
    plt.legend(handles=[
        mlines.Line2D([], [], color='blue',        linestyle='-',       label='Coriolis'),
        mlines.Line2D([], [], color='red',         linestyle='-',       label='Viscous'),
        mlines.Line2D([], [], color='orange',      linestyle='-',       label='Rey Stress'),
        mlines.Line2D([], [], color='saddlebrown', linestyle='-',       label='Temporal'),
        mlines.Line2D([], [], color='black',       linestyle='-',       label='Total'),
        mlines.Line2D([], [], color='black',       linestyle='-',       label='Valley'),
        mlines.Line2D([], [], color=SMOOTH_COLOR,  linestyle=SMOOTH_LS, label='Smooth'),
    ])
    add_marker_legend()
    plt.grid(True)
    plt.xlim(0, 200)
    plt.ylim(-0.1, 1.0)
    plt.savefig(os.path.join(fig_dir, 'Zoomed Shear Stress XY.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # Shear Stress ZY  (Fig-4 convention: Coriolis +C, Viscous +V, Reynolds −⟨flux⟩, Total C+V+R)
    # [PLOT 31] Shear stress $\tau_{zy}$
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(y_inner[:], Czy_o[:], label='Coriolis', color='blue', linestyle='-')
    plt.plot(y_inner[:], Vzy_o[:], label='Viscous', color='red', linestyle='-')
    plt.plot(y_inner[:], Rzy_o[:], label='Rey Stress', color='orange', linestyle='-')
    plt.plot(y_inner[:], dwdt, label='Temporal', color='saddlebrown', linestyle='-')
    plt.plot(y_inner[:], Tzy_o[:], label='Total', color='black', linestyle='-')
    ref_plot(plot_ref_smooth, y_s_p, Czy_s, color='blue', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Vzy_s, color='red', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Rzy_s, color='orange', linestyle=SMOOTH_LS)
    # rough r1 (Re=1000) overlay — Method-2 terms, own inner units (y_r_p)
    ref_plot(plot_ref_rough, y_r_p, Czy_r, color='blue', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Vzy_r, color='red', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Rzy_r, color='orange', linestyle=ROUGH_LS)
    mark_layers_multi(y_inner, [Czy_o, Vzy_o,
                                Rzy_o, dwdt, Tzy_o],
                      _LYR_ORO, filled=True)
    ref_mark(plot_ref_smooth, mark_layers_multi, y_s_p, [Czy_s, Vzy_s,
                              Rzy_s], _LYR_SMO, filled=False)
    mark_h(y_in[h_idx], 'v')
    plt.title(r'Shear stress $\tau_{zy}$')
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'${{\langle \bar{\tau} \rangle}^+}_{zy}$')
    plt.legend(handles=[
        mlines.Line2D([], [], color='blue',        linestyle='-',       label='Coriolis'),
        mlines.Line2D([], [], color='red',         linestyle='-',       label='Viscous'),
        mlines.Line2D([], [], color='orange',      linestyle='-',       label='Rey Stress'),
        mlines.Line2D([], [], color='saddlebrown', linestyle='-',       label='Temporal'),
        mlines.Line2D([], [], color='black',       linestyle='-',       label='Total / Valley'),
        mlines.Line2D([], [], color=SMOOTH_COLOR,  linestyle=SMOOTH_LS, label='Smooth'),
    ])
    add_marker_legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Shear Stress ZY.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################    
    # Zoomed plot
    # [PLOT 32] Shear stress $\tau_{zy}$
    plt.figure(figsize=(8, 6), dpi=300)

    # Valley case (solid lines)
    plt.plot(y_inner[:limity], Czy_o[:limity]/u_star**2, color='blue', linestyle='-', label='Coriolis')
    plt.plot(y_inner[:limity], Vzy_o[:limity]/u_star**2, color='red', linestyle='-', label='Viscous')
    plt.plot(y_inner[:limity], Rzy_o[:limity]/u_star**2, color='orange', linestyle='-', label='Rey Stress')
    plt.plot(y_inner[:limity], dwdt[:limity]/u_star**2, color='saddlebrown', linestyle='-', label='Temporal')
    plt.plot(y_inner[:limity], Tzy_o[:limity]/u_star**2, color='black', linestyle='-', label='Total')
    # Smooth case (dashed)
    ref_plot(plot_ref_smooth, y_s_p, Czy_s/ustr_s1**2, color='blue', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Vzy_s/ustr_s1**2, color='red', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Rzy_s/ustr_s1**2, color='orange', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, np.zeros(nys), color='saddlebrown', linestyle=SMOOTH_LS)
    # rough r1 — own inner units (ustr_r1)
    ref_plot(plot_ref_rough, y_r_p, Czy_r/ustr_r1**2, color='blue', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Vzy_r/ustr_r1**2, color='red', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Rzy_r/ustr_r1**2, color='orange', linestyle=ROUGH_LS)

    mark_layers_multi(y_inner, [Czy_o/u_star**2, Vzy_o/u_star**2,
                                Rzy_o/u_star**2, dwdt/u_star**2,
                                Tzy_o/u_star**2],
                      _LYR_ORO, filled=True)
    ref_mark(plot_ref_smooth, mark_layers_multi, y_s_p, [Czy_s/ustr_s1**2, Vzy_s/ustr_s1**2,
                              Rzy_s/ustr_s1**2, np.zeros(nys)],
                      _LYR_SMO, filled=False)
    mark_h(y_in[h_idx], 'v')
    plt.title(r'Shear stress $\tau_{zy}$')
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'${{\langle \bar{\tau} \rangle}^+}_{zy}$')
    plt.legend(handles=[
        mlines.Line2D([], [], color='blue',        linestyle='-',       label='Coriolis'),
        mlines.Line2D([], [], color='red',         linestyle='-',       label='Viscous'),
        mlines.Line2D([], [], color='orange',      linestyle='-',       label='Rey Stress'),
        mlines.Line2D([], [], color='saddlebrown', linestyle='-',       label='Temporal'),
        mlines.Line2D([], [], color='black',       linestyle='-',       label='Total'),
        mlines.Line2D([], [], color='black',       linestyle='-',       label='Valley'),
        mlines.Line2D([], [], color=SMOOTH_COLOR,  linestyle=SMOOTH_LS, label='Smooth'),
    ])
    add_marker_legend()
    plt.grid(True)
    plt.xlim(0, 200)
    plt.ylim(-0.5, 1)
    plt.savefig(os.path.join(fig_dir, 'Zoomed Shear Stress ZY.png'), dpi=300)
    plt.show()

    # %%###########################################################################
    # Streamwise Momentum Budget
    # LHS: Temporal + MeanAdv + TurbAdv  =  RHS: Viscous + NetCoriolis
    # _ny_mb = 200
    # _us2   = u_star**2
    # plt.figure(figsize=(10, 6), dpi=300)
    # plt.plot(y_inner[:_ny_mb], mom_temporal[:_ny_mb] / _us2,  color='saddlebrown', linestyle='-', label='Temporal')
    # plt.plot(y_inner[:_ny_mb], mom_mean_adv[:_ny_mb] / _us2,  color='green',       linestyle='-', label='Mean Advection')
    # plt.plot(y_inner[:_ny_mb], mom_turb_adv[:_ny_mb] / _us2,  color='orange',      linestyle='-', label='Turbulent Advection')
    # plt.plot(y_inner[:_ny_mb], mom_visc[:_ny_mb]     / _us2,  color='red',         linestyle='-', label='Viscous')
    # plt.plot(y_inner[:_ny_mb], mom_coriolis[:_ny_mb] / _us2,  color='blue',        linestyle='-', label='Net Coriolis + Pressure')
    # plt.title(r'Streamwise Momentum Budget $\langle \bar{u} \rangle$')
    # plt.xlabel(r'$z^{+}$')
    # plt.ylabel(r'Terms / $u_*^2$')
    # plt.legend(handles=[
    #     mlines.Line2D([], [], color='saddlebrown', linestyle='-', label='Temporal'),
    #     mlines.Line2D([], [], color='green',       linestyle='-', label='Mean Advection'),
    #     mlines.Line2D([], [], color='orange',      linestyle='-', label='Turbulent Advection'),
    #     mlines.Line2D([], [], color='red',         linestyle='-', label='Viscous'),
    #     mlines.Line2D([], [], color='blue',        linestyle='-', label='Net Coriolis + Pressure'),
    # ])
    # plt.grid(True)
    # plt.savefig('fig/Momentum Budget', dpi=300)
    # plt.show()

    # %%###########################################################################
    # Wind profile
    # plt.figure(figsize=(8, 6), dpi=300)
    # plt.plot(avg_c(eps, corr_yx, axis=1), y, label='coriolis', color='blue', linestyle='-')
    # plt.title('Wind profile')
    # plt.ylabel(r'$z^{+}$')
    # plt.xlabel(r'$wind$')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('fig/Wind', dpi=300)
    # plt.show()
    
    # %%###########################################################################
    # Friction Velocity Profile
    # [PLOT 33] Friction Velocity
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(u_star2[:], y_in[:], label='u_{star}', color='blue', linestyle='-')
    mark_h(y_in[h_idx], 'h')
    plt.title('Friction Velocity')
    plt.ylabel(r'$z^+$')
    plt.xlabel(r'$u_{*}$')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Friction velocity.png'), dpi=300)
    plt.show()

    # %%###########################################################################
    # Friction Velocity — comparison of the two Coriolis-integral approaches
    # [PLOT 33b] Friction Velocity (Coriolis integral: mean→integrate vs integrate→cavg)
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(u_star2[:],   y_in[:], label=r'mean$\to$integrate (old)', color='blue', linestyle='-')
    plt.plot(u_star2_c[:], y_in[:], label=r'integrate$\to$cavg (new)', color='red',  linestyle='--')
    mark_h(y_in[h_idx], 'h')
    plt.title('Friction Velocity — Coriolis-integral approaches')
    plt.ylabel(r'$z^+$')
    plt.xlabel(r'$u_{*}$')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Friction velocity comparison.png'), dpi=300)
    plt.show()

    # %%###########################################################################
    # Friction Velocity — Method-2 for all reference cases (orographic / smooth / rough r1)
    # [PLOT 33c] Friction Velocity (Method-2: orographic vs smooth vs rough)
    # Each case in its own inner scaling; the dotted vertical lines mark the
    # constant-flux plateau values.  Smooth/rough shown per the config switches.
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(ustr_M2_o[:], y_in[:],
             label=f'orographic Re=500 (plateau {ustr_M2_plateau_o:.4f})',
             color='blue', linestyle='-')
    ref_plot(plot_ref_smooth, ustr_M2_s, y_s_p,
             label=f'smooth Re=500 M2 (plateau {ustr_M2_plateau_s:.4f}, stored {ustr_s1:.4f})',
             color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    ref_plot(plot_ref_rough, ustr_M2_r, y_r_p,
             label=f'rough r1 Re=1000 (plateau {ustr_r1:.4f})',
             color=ROUGH_COLOR, linestyle=ROUGH_LS)
    plt.axvline(ustr_M2_plateau_o, color='blue', linestyle=':', linewidth=1)
    if plot_ref_smooth:
        plt.axvline(ustr_s1, color=SMOOTH_COLOR, linestyle=':', linewidth=1)  # stored smooth u* (~0.0618)
    if plot_ref_rough:
        plt.axvline(ustr_r1, color=ROUGH_COLOR, linestyle=':', linewidth=1)
    mark_h(y_in[h_idx], 'h')
    plt.title('Friction Velocity — Method 2 (all cases)')
    plt.ylabel(r'$z^+$')
    plt.xlabel(r'$u_{*}$')
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Friction velocity all-cases.png'), dpi=300)
    plt.show()

    # %%###########################################################################
    # [PLOT 32r] Kostelecky & Ansorge (2024) figure-4 validation of Method 2:
    # integrated momentum budget (C, V, R, Total) for the smooth and rough r1
    # reference cases, in inner (a,b) and outer (c,d) units.  Built transparently
    # from eq. 4.2 (see plot_fig4_budget) so the Method-2 u* and the budget-term
    # shapes can be checked against the paper.  Produced for both cases regardless
    # of the overlay switches (these are dedicated validation figures).
    plot_fig4_budget(smooth_nc_path, nu,       'smooth_Re500',    fig_dir)
    plot_fig4_budget(rough_nc_path,  nu_rough, 'rough_r1_Re1000', fig_dir)

    STOP
    # %%###########################################################################
    # Velocity profile
    # [PLOT 34] Velocity Profile
    plt.figure(figsize=(8,6))
    plt.plot(y_in[(eps_hgt[0]-1):]  -  y_in[(eps_hgt[0]-1)]          ,u_pl_rot2D[(eps_hgt[0]-1):,0]/ustr_s1           , label='Valley top', color='blue', linestyle='-')
    plt.plot(y_in[(eps_hgt[eps_lf]-1):]  -  y_in[eps_hgt[eps_lf]]    ,u_pl_rot2D[(eps_hgt[eps_lf]-1):,eps_lf]/ustr_s1 , label='Left flank', color='saddlebrown', linestyle='-')
    plt.plot(y_in[eps_hgt[512]:]  -  y_in[eps_hgt[512]]              ,u_pl_rot2D[(eps_hgt[512]):,512]/ustr_s1         , label='Valley bottom', color='red', linestyle='-')
    plt.plot(y_in[(eps_hgt[eps_rf]-1):]-y_in[(eps_hgt[eps_rf]-1)]    ,u_pl_rot2D[(eps_hgt[eps_rf]-1):,eps_rf]/ustr_s1 , label='Right flank', color='magenta', linestyle='-')
    
    plt.plot(y_in[(eps_hgt[0]-1):]  -  y_in[(eps_hgt[0]-1)]           ,w_pl_rot2D[(eps_hgt[0]-1):,0]/ustr_s1           , label='Valley top', color='blue', linestyle='--')
    plt.plot(y_in[(eps_hgt[eps_lf]-1):] - y_in[(eps_hgt[eps_lf]-1)]   ,w_pl_rot2D[(eps_hgt[eps_lf]-1):,eps_lf]/ustr_s1 , label='Left flank', color='saddlebrown', linestyle='--')
    plt.plot(y_in[eps_hgt[512]:]  -  y_in[eps_hgt[512]]               ,w_pl_rot2D[eps_hgt[512]:,512]/ustr_s1           , label='Valley bottom', color='red', linestyle='--')
    plt.plot(y_in[(eps_hgt[eps_rf]-1):] - y_in[(eps_hgt[eps_rf] - 1)] ,w_pl_rot2D[(eps_hgt[eps_rf]-1):,eps_rf]/ustr_s1 , label='Right flank', color='magenta', linestyle='--')
    # Smooth case — single global profile (flat wall, no local shift)
    ref_plot(plot_ref_smooth, y_s_p, GblU_s/ustr_s1, color=SMOOTH_COLOR, linestyle='-')
    ref_plot(plot_ref_smooth, y_s_p, -GblW_s/ustr_s1, color=SMOOTH_COLOR, linestyle='--', alpha=0.6)
    # rough r1 (Re=1000) global profile — own inner units (ustr_r1)
    ref_plot(plot_ref_rough, y_r_p, GblU_r/ustr_r1, color=ROUGH_COLOR, linestyle='-')
    ref_plot(plot_ref_rough, y_r_p, -GblW_r/ustr_r1, color=ROUGH_COLOR, linestyle='--', alpha=0.6)

    custom_labels = ['Hill top', 'Left Flank', 'Valley Bottom', 'Right Flank',
                     r'$\langle \bar{u} \rangle$', r'$\langle \bar{v} \rangle$', 'Smooth']
    color_handles = [
        Line2D([0], [0], color='blue',        lw=4),
        Line2D([0], [0], color='saddlebrown', lw=4),
        Line2D([0], [0], color='red',         lw=4),
        Line2D([0], [0], color='magenta',     lw=4)]
    style_handles = [
        Line2D([0], [0], color='black',      linestyle='-',  lw=2),
        Line2D([0], [0], color='black',      linestyle='--', lw=2),
        Line2D([0], [0], color=SMOOTH_COLOR, linestyle='-',  lw=2)]
    custom_handles = color_handles + style_handles
    # Valley curves use surface-relative (shifted) z+, so absolute-z+ layer
    # markers are placed only on the smooth (unshifted) profile.
    ref_mark(plot_ref_smooth, mark_layers_multi, y_s_p, [GblU_s/ustr_s1, -GblW_s/ustr_s1], _LYR_SMO, filled=False)
    mark_h(y_in[h_idx], 'v')
    plt.title('Velocity Profile')
    plt.ylabel(r'$\langle \bar{u}_i \rangle ^+$')
    plt.xlabel(r'$z^{+}$')
    plt.xscale("log")
    plt.legend(custom_handles, custom_labels, loc='upper left')
    add_marker_legend(oro=False)
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'LogLaw.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # zoomed
    # [PLOT 35] Velocity Profile 
    plt.figure(figsize=(8,6))
    plt.plot(y_in[(eps_hgt[0]-1):limity]-y_in[(eps_hgt[0]-1)] ,u_plus[(eps_hgt[0]-1):limity,0], label='top', color='blue', linestyle='-')
    plt.plot(y_in[(eps_hgt[eps_lf]-1):limity]-y_in[eps_hgt[eps_lf]] ,u_plus[(eps_hgt[eps_lf]-1):limity,eps_lf], label='Flank left', color='saddlebrown', linestyle='-')
    plt.plot(y_in[eps_hgt[512]:limity]-y_in[eps_hgt[512]]     ,u_plus[(eps_hgt[512]):limity,512], label='Bottom', color='red', linestyle='-')
    plt.plot(y_in[(eps_hgt[eps_rf]-1):limity]-y_in[(eps_hgt[eps_rf]-1)] ,u_plus[(eps_hgt[eps_rf]-1):limity,eps_rf], label='Flank right', color='magenta', linestyle='-')
    
    plt.plot(y_in[(eps_hgt[0]-1):limity]-y_in[(eps_hgt[0]-1)],  w_plus[(eps_hgt[0]-1):limity,0], label='top', color='blue', linestyle='--')
    plt.plot(y_in[(eps_hgt[eps_lf]-1):limity]-y_in[(eps_hgt[eps_lf]-1)],w_plus[(eps_hgt[eps_lf]-1):limity,eps_lf], label='Flank left', color='saddlebrown', linestyle='--')
    plt.plot(y_in[eps_hgt[512]:limity]-y_in[eps_hgt[512]],w_plus[eps_hgt[512]:limity,512], label='Bottom', color='red', linestyle='--')
    plt.plot(y_in[(eps_hgt[eps_rf]-1):limity]-y_in[(eps_hgt[eps_rf]-1)],w_plus[(eps_hgt[eps_rf]-1):limity,eps_rf], label='Flank right', color='magenta', linestyle='--')
    
    # Valley curves are surface-relative (shifted) and there is no smooth curve
    # here, so only the crest line h is marked (absolute-z+ layer markers omitted).
    mark_h(y_in[h_idx], 'v')
    plt.axvline(x=(Re_tau), color='black', linestyle='--', linewidth=1)
    plt.text((Re_tau), 0.5, r'$\delta$', rotation=90, verticalalignment='center', horizontalalignment='right')
    
    custom_labels = ['Valley top', 'Left flank', 'Valley bottom', 'Right flank', r'$\langle \bar{u} \rangle$', r'$\langle \bar{v} \rangle$']
    color_handles = [
    Line2D([0], [0], color='blue', lw=4, label='Blue'),
    Line2D([0], [0], color='saddlebrown', lw=4, label='SaddleBrown'),
    Line2D([0], [0], color='red', lw=4, label='Red'),
    Line2D([0], [0], color='magenta', lw=4, label='Magenta')]
    style_handles = [
    Line2D([0], [0], color='black', linestyle='-', lw=2, label='(-)'),
    Line2D([0], [0], color='black', linestyle='--', lw=2, label='(--)')]
    custom_handles = color_handles + style_handles
    plt.title('Velocity Profile ')
    plt.ylabel(r'$\langle \bar{u}_i \rangle ^+$')
    plt.xlabel(r'$z^{+}$')
    plt.xscale("log")
    plt.legend(custom_handles, custom_labels, loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Zoomed_LogLaw.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # Monin Obukhov log layer — smooth-case comparison omitted (not available in PhAvg.py)
    # plt.figure(figsize=(8,6))
    # plt.plot(y_s_p, np.mean(U_s_p, axis=1),color='blue', linestyle='-' )
    # plt.plot(y_s_p, np.mean(V_s_p, axis=1),color='red', linestyle='-')
    # plt.plot(y_s_p, -np.mean(W_s_p, axis=1),color='black', linestyle='-')
    
    # %%###########################################################################
    # Velocity profile with and without orography.
    #
    # ── Normalisation ──────────────────────────────────────────────────────────
    # Both valley and smooth curves are non-dimensionalised with the SAME
    # reference friction velocity  ustr_ref = 0.0617  (= ustr_s1 from the smooth
    # flat-wall NetCDF).  Using a common reference makes the comparison consistent.
    #
    #   u⁺_valley = u_plus_rot  / ustr_ref        (x-avg over fluid cells)
    #   u⁺_smooth = (fU / ustr_s1) / G_s          (G_s = sqrt(Gx²+Gz²) ≈ 1.04)
    #   z⁺_valley = y_in   (already in wall units: y · u★_sim / ν)
    #   z⁺_smooth = y_s · ustr_s1 / ν
    #
    # ── Log-law equation ───────────────────────────────────────────────────────
    #
    #   u⁺ = (1/κ) · ln(z⁺) + B
    #
    # Fitted by OLS regression of u⁺ vs ln(z⁺):
    #   slope     = 1/κ     →   κ   = 1 / slope
    #   intercept = B
    #
    # Fitted ranges (linear region of the semi-log profile):
    #   Valley:  z⁺ ∈ [60, 200]   →  κ ≈ 0.450,  B ≈ 4.80,  R² ≈ 0.994
    #   Smooth:  z⁺ ∈ [23, 100]   →  κ ≈ 0.427,  B ≈ 4.69,  R² ≈ 0.988
    #
    # ── Log-law line extension ─────────────────────────────────────────────────
    # Each dotted log-law line is extended 5 u⁺ units below the fitting-range
    # start, extrapolating toward lower z⁺ (left on the log-x axis).
    #   z⁺_ext = exp( (u⁺_start − 5 − B) / slope )
    # This makes the extrapolated behaviour clearly visible on the plot.
    #
    # ── Canopy-region fit (auto-selected) ─────────────────────────────────────
    # Two physically motivated candidates are fitted over z⁺ ∈ (0, ≈34.4];
    # the one with higher R² (in log-linear or log-log space) is plotted.
    #
    # Candidate 1 — Exponential (canopy attenuation) law:
    #   u(z) = u(h) · exp( α · (z/h − 1) )     for  0 ≤ z ≤ h
    #   OLS of ln(u/u_h) vs (z/h − 1) → slope = α
    #   Physical basis: standard canopy-layer model; anchored at u(h) at z=h.
    #   Limitation: asymptotes to u(h)·exp(−α) ≈ 1.4 at z=0 (does not reach 0).
    #
    # Candidate 2 — Power law:
    #   u⁺ = A · (z⁺)ⁿ
    #   OLS of ln(u⁺) vs ln(z⁺) → slope = n, intercept = ln(A)
    #   Physical basis: power-law velocity profiles are standard for turbulent ABL
    #   and roughness/canopy sublayers; satisfies u⁺(0) = 0 naturally.
    #   Typical values: A ≈ 0.63, n ≈ 0.82 (neutral valley, Re_τ ≈ 722).
    #
    # Typical R² comparison (neutral valley):
    #   Canopy law:  R² ≈ 0.69  (poor near wall — asymptote mismatch)
    #   Power law:   R² ≈ 0.98  (better — naturally zero at wall)
    # → Power law selected automatically.
    #
    # Smooth-case variables (y_s_p, U_s_p, W_s_p, ustr_s1, SMOOTH_COLOR, SMOOTH_LS)
    # are already loaded above in the shared smooth-case block.

    # Plotted velocity profiles (consistent normalisation for fitting)
    _valley_u = u_plus_rot / 0.0617        # same as the valley curve on the plot
    _smooth_u = np.mean(U_s_p, axis=1)    # same as the smooth curve on the plot

    # --- Log-law fit for valley streamwise: z+ ∈ [60, 200] ---
    # u+ = (1/κ) ln(z+) + B  →  OLS of u+ vs ln(z+), slope=1/κ, intercept=B
    _vll_mask = (y_in >= 60.0) & (y_in <= 175.0)
    _z_vll, _u_vll = y_in[_vll_mask], _valley_u[_vll_mask]
    _slp_v, _int_v, *_ = linregress(np.log(_z_vll), _u_vll)
    kappa_vll = 1.0 / _slp_v
    print(f"Valley log-law fit (z+ ∈ [60,200]): κ={kappa_vll:.3f}, B={_int_v:.3f}")

    # Extend the plotted line 5 u+ units below the fit-range start (into the
    # (-z+, -u+) direction on the log-x plot) so the extrapolation is clearly visible.
    #   u+_start = slope·ln(z+_min) + B  ;  extend to u+_start − 5
    #   → z+_new = exp((u+_start − 5 − B) / slope)
    _u_vll_start   = _slp_v * np.log(_z_vll[0]) + _int_v
    _z_vll_ext_lo  = np.exp((_u_vll_start - 5.0 - _int_v) / _slp_v)
    _z_vll_ext_lo  = max(_z_vll_ext_lo, y_in[1])   # never below first valid grid point
    _z_vll_plot    = np.geomspace(_z_vll_ext_lo, _z_vll[-1], 300)
    u_loglaw_valley_plot = _slp_v * np.log(_z_vll_plot) + _int_v

    # --- Log-law fit for smooth streamwise: z+ ∈ [23, 100] ---
    # u+ = (1/κ) ln(z+) + B  →  OLS of u+ vs ln(z+), slope=1/κ, intercept=B
    _sml_mask = (y_s_p >= 23.0) & (y_s_p <= 100.0)
    _z_sml, _u_sml = y_s_p[_sml_mask], _smooth_u[_sml_mask]
    if _z_sml.size >= 3:
        _slp_s, _int_s, *_ = linregress(np.log(_z_sml), _u_sml)
        kappa_sml = 1.0 / _slp_s
        print(f"Smooth log-law fit (z+ ∈ [23,100]):  κ={kappa_sml:.3f}, B={_int_s:.3f}")
        # Same 5-u+ extension toward lower z+
        _u_sml_start  = _slp_s * np.log(_z_sml[0]) + _int_s
        _z_sml_ext_lo = np.exp((_u_sml_start - 5.0 - _int_s) / _slp_s)
        _z_sml_ext_lo = max(_z_sml_ext_lo, y_s_p[1])
        _z_sml_plot   = np.geomspace(_z_sml_ext_lo, _z_sml[-1], 300)
        u_loglaw_smooth_plot = _slp_s * np.log(_z_sml_plot) + _int_s
    else:
        kappa_sml = 0.41
        _z_sml_plot = np.array([])
        u_loglaw_smooth_plot = np.array([])

    # --- Canopy fit for valley (z+ ∈ [0, 20]) ---
    # Two candidates are evaluated; the one with higher R² in log-space is used.
    #
    # Candidate 1 — Exponential (canopy) law:
    #   u(z) = u(h) · exp(α·(z/h − 1))
    #   OLS of ln(u/u_h) vs (z/h − 1) → slope = α
    #   Physical basis: standard canopy-layer attenuation model.
    #   Limitation: does not go to 0 at the wall (asymptote u_h·exp(−α) ≈ 1.4).
    #
    # Candidate 2 — Power law:
    #   u+ = A · (z+)^n
    #   OLS of ln(u+) vs ln(z+) → slope = n, intercept = ln(A)
    #   Physical basis: turbulent-flow power law; widely used for roughness-sublayer
    #   and canopy-sublayer profiles; naturally satisfies u+(0) = 0.
    _u_at_h_v    = _valley_u[hill_hgt]
    _can_end_idx = min(hill_hgt + 20, ny)
    _z_fit_can   = y_in[1:_can_end_idx]        # skip index 0 (y≈0)
    _u_fit_can   = _valley_u[1:_can_end_idx]
    _can_valid   = _u_fit_can > 1e-6

    # Candidate 1: canopy (exponential)
    if np.sum(_can_valid) >= 3:
        _x_cv = _z_fit_can[_can_valid] / h_inner_plus - 1.0
        _y_cv = np.log(_u_fit_can[_can_valid] / _u_at_h_v)
        _slp_cv, _int_cv, _r_cv, *_ = linregress(_x_cv, _y_cv)
        alpha_canopy_v = float(_slp_cv)
        r2_canopy = float(_r_cv**2)
    else:
        alpha_canopy_v, r2_canopy = alpha_canopy, 0.0

    # Candidate 2: power law (ln(u+) vs ln(z+), fluid cells only)
    _pl_valid = _can_valid & (_z_fit_can > 0)
    if np.sum(_pl_valid) >= 3:
        _x_pl = np.log(_z_fit_can[_pl_valid])
        _y_pl = np.log(_u_fit_can[_pl_valid])
        _slp_pl, _int_pl, _r_pl, *_ = linregress(_x_pl, _y_pl)
        n_power  = float(_slp_pl)
        A_power  = float(np.exp(_int_pl))
        r2_power = float(_r_pl**2)
    else:
        n_power, A_power, r2_power = 0.8, 0.6, 0.0

    print(f"Canopy law  (z+∈[0,{y_in[_can_end_idx-1]:.1f}]): α={alpha_canopy_v:.4f}  R²={r2_canopy:.4f}")
    print(f"Power law   (z+∈[0,{y_in[_can_end_idx-1]:.1f}]): A={A_power:.4f}  n={n_power:.4f}  R²={r2_power:.4f}")

    # Select the better fit
    _z_can_v = y_in[(y_in > 0) & (y_in <= 20.0)]
    if r2_power >= r2_canopy:
        u_canopy_v    = A_power * _z_can_v**n_power
        canopy_legend = rf'Power law ($u^+\!=\!{A_power:.2f}\,z^{{+{n_power:.3f}}}$, $z^+\!\leq\!20$)'
        print(f"→ Using power law  (R²={r2_power:.4f} ≥ canopy R²={r2_canopy:.4f})")
    else:
        u_canopy_v    = _u_at_h_v * np.exp(alpha_canopy_v * (_z_can_v / h_inner_plus - 1.0))
        canopy_legend = rf'Canopy law ($\alpha$={alpha_canopy_v:.3f}, $z^+\!\leq\!20$)'
        print(f"→ Using canopy law (R²={r2_canopy:.4f} > power R²={r2_power:.4f})")

    # [PLOT 36] Velocity Profile with and without Orography
    plt.figure(figsize=(8, 6), dpi=300)
    # Valley case (solid lines)
    plt.plot(y_in, _valley_u,          color='red',  linestyle='-')
    plt.plot(y_in, w_plus_rot/0.0617,  color='blue', linestyle='-')
    # Smooth case (dashed grey)
    ref_plot(plot_ref_smooth, y_s_p, _smooth_u,                   color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, -np.mean(W_s_p, axis=1),     color=SMOOTH_COLOR, linestyle=SMOOTH_LS, alpha=0.4)
    # rough r1 (Re=1000) — own inner units (y_r_p), magnitude profiles
    ref_plot(plot_ref_rough, y_r_p, np.mean(U_r_p, axis=1),       color=ROUGH_COLOR, linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, -np.mean(W_r_p, axis=1),      color=ROUGH_COLOR, linestyle=ROUGH_LS, alpha=0.4)
    # Valley log-law — extended 5 u+ units below fit-range start
    plt.plot(_z_vll_plot, u_loglaw_valley_plot, color='red', linestyle='dotted', linewidth=2)
    # Smooth log-law — extended 5 u+ units below fit-range start
    if _z_sml_plot.size > 0:
        ref_plot(plot_ref_smooth, _z_sml_plot, u_loglaw_smooth_plot, color=SMOOTH_COLOR, linestyle='dotted', linewidth=2)
    # Canopy fit (best of exponential vs power law), z+ ∈ [0, 20]
    plt.plot(_z_can_v, u_canopy_v, color='green', linestyle='--')
    plt.axvline(x=(Re_tau), color='black', linestyle='-', linewidth=1)
    plt.text((Re_tau), 0.5, r'$\delta_{o}$', rotation=90, verticalalignment='center', horizontalalignment='right')
    mark_layers_multi(y_in, [_valley_u, w_plus_rot/0.0617], _LYR_ORO, filled=True)
    ref_mark(plot_ref_smooth, mark_layers_multi, y_s_p, [_smooth_u, -np.mean(W_s_p, axis=1)], _LYR_SMO, filled=False)
    mark_h(y_in[h_idx], 'v')

    plt.xscale('log')
    plt.xlabel(r'$z^+$')
    plt.ylabel(r'$\langle \bar{u_i} \rangle^+$')
    legend_elements = [
        Line2D([0], [0], color='red',        linestyle='-',       label='Streamwise (valley)'),
        Line2D([0], [0], color='blue',       linestyle='-',       label='Spanwise (valley)'),
        Line2D([0], [0], color=SMOOTH_COLOR, linestyle=SMOOTH_LS, label='Streamwise (smooth)'),
        Line2D([0], [0], color=SMOOTH_COLOR, linestyle=SMOOTH_LS, label='Spanwise (smooth)', alpha=0.4),
        Line2D([0], [0], color='red',        linestyle='dotted',  linewidth=2,
               label=rf'Log-law valley ($\kappa$={kappa_vll:.3f}, $z^+\!\in\![60,200]$)'),
        Line2D([0], [0], color=SMOOTH_COLOR, linestyle='dotted',  linewidth=2,
               label=rf'Log-law smooth ($\kappa$={kappa_sml:.3f}, $z^+\!\in\![23,100]$)'),
        Line2D([0], [0], color='green',      linestyle='--',      label=canopy_legend),
    ]
    # This is the reference plot for the layer-marker key (filled=oro, hollow=smooth).
    plt.legend(handles=legend_elements, fontsize=7)
    add_marker_legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title("Velocity Profile with and without Orography")
    plt.savefig(os.path.join(fig_dir, 'Velocity_Profile_with_and_without_Orography.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # Velocity Profile in Roughness layer
    # [PLOT 37] Velocity_profile_roughness_layer
    plt.figure(figsize=(8,6))
    plt.plot(y_in[:157], np.log(u_plus_rot/u_star)[:157], color='red', linestyle='-')
    mark_h(y_in[h_idx], 'v')
    plt.xscale('log')  # Logarithmic x-axis
    plt.xlabel(r'$z^+$')  # x-axis label
    # plt.yscale('log')  # Logarithmic x-axis
    plt.ylabel(r'$\langle \bar{u_i} \rangle^+$')  # y-axis label
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(fig_dir, 'Velocity_profile_roughness_layer.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # TKE Horizontal profile
    # [PLOT 38] TKE profile
    plt.figure(figsize=(8,6))
    
    plt.plot(y_in[:460], (avg_c(eps, TKE, axis=1)/ustr_s1**2)[:460], label='valley', color='blue', linestyle='-')
    ref_plot(plot_ref_smooth, y_s_p[:130], (np.mean(TKE_s, axis=1)/ustr_s1**2)[:130], label='smooth', color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    ref_plot(plot_ref_rough, y_r_p, np.mean(TKE_r, axis=1)/ustr_r1**2, label='rough r1', color=ROUGH_COLOR, linestyle=ROUGH_LS)
    plt.axvline(x=(Re_tau), color='black', linestyle='-', linewidth=1)
    
    plt.text((Re_tau), 0.5, r'$\delta_{v}$', rotation=90, verticalalignment='center', horizontalalignment='right')
    plt.axvline(x=ustr_s1**2/nu, color=SMOOTH_COLOR, linestyle=':', linewidth=1)
    plt.text(ustr_s1**2/nu, 0.5, r'$\delta_{s}$', rotation=90, verticalalignment='center', horizontalalignment='right',
             color=SMOOTH_COLOR)
    
    mark_layers(y_in, avg_c(eps, TKE, axis=1)/ustr_s1**2, _LYR_ORO, filled=True)
    ref_mark(plot_ref_smooth, mark_layers, y_s_p, np.mean(TKE_s, axis=1)/ustr_s1**2, _LYR_SMO, filled=False)
    mark_h(y_in[h_idx], 'v')
    plt.title('TKE profile')
    plt.xlabel(r'$z^{+}$')
    plt.ylabel(r'$TKE^+$')
    plt.xscale('log')
    plt.legend()
    add_marker_legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'TKE_profile.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # TKE distribution (streamwise variation of y-averaged TKE)
    # [PLOT 39] TKE distribution
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(x_in, AVG_TKE_V / (ustr_s1**2), label="valley", color="blue", linestyle="-")
    ref_plot(plot_ref_smooth, x_in, AVG_TKE_V_s_i / (ustr_s1**2), label="smooth", color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    # Orography outline
    black_line = (y[hill_hgt] / u_star) * (1 + np.cos(2 * x_in * np.pi / x_in[-1]))
    plt.fill_between(x_in, black_line, color="black", alpha=1.0, label="IBM solid")
    plt.plot(x_in, black_line, color="black", linestyle="-")
    plt.title("TKE distribution")
    plt.xlabel(r"$x^{+}$")
    plt.ylabel(r"$TKE^+$")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'TKE_Distribution.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # Streamwise convective momentum
    # [PLOT 40] Advection 
    plt.figure(figsize=(6, 5))
    plt.plot(conv_top[:450],    y_in[:450], label='Valley top', color="yellow")
    plt.plot(conv_lf[:450],     y_in[:450], label='Left flank', color="red")
    plt.plot(conv_bottom[:450], y_in[:450], label='Valley bottom', color="black")
    plt.plot(conv_rf[:450],     y_in[:450], label='Right flank', color="blue")
    for _cv in (conv_top, conv_lf, conv_bottom, conv_rf):    # oro layers on every flank curve
        mark_layers(_cv, y_in, _LYR_ORO, filled=True)
    mark_h(y_in[h_idx], 'h')
    plt.xlabel(r'$u_{j} \frac{\partial u_i}{\partial x_j}$')
    plt.ylabel('$z^{+}$')
    plt.title("Advection ")
    plt.legend()
    add_marker_legend(smooth=False)
    # Vertical grid lines only
    plt.grid(axis='x')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Streamwise_Advection.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # Streamwise convective momentum zoomed
    # [PLOT 41] Advection 
    plt.figure(figsize=(6, 5))
    plt.plot(conv_top[:200],    y_in[:200], label='Valley top', color="yellow")
    plt.plot(conv_lf[:200],     y_in[:200], label='Left flank', color="red")
    plt.plot(conv_bottom[:200], y_in[:200], label='Valley bottom', color="black")
    plt.plot(conv_rf[:200],     y_in[:200], label='Right flank', color="blue")
    for _cv in (conv_top, conv_lf, conv_bottom, conv_rf):    # oro layers on every flank curve
        mark_layers(_cv, y_in, _LYR_ORO, filled=True)
    mark_h(y_in[h_idx], 'h')
    plt.xlabel(r'$u_{j} \frac{\partial u_i}{\partial x_j}$')
    plt.ylabel('$z^{+}$')
    plt.title("Advection ")
    plt.legend()
    add_marker_legend(smooth=False)
    # Vertical grid lines only
    plt.grid(axis='x')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Advection.png'), dpi=300)
    plt.show()

    # %%###########################################################################
    # Plot 1: IBM body-force magnitude |B|(x,z) near the IBM surface.
    # IBM_B_mag computed in postprocess; shown here restricted to z+ < 200.
    _z_IBM_lim = 200
    _j_IBM_lim = np.argmin(np.abs(y_inner - _z_IBM_lim))
    _B_plot    = np.where(IBM_B_mag[:_j_IBM_lim, :] > 0,
                          IBM_B_mag[:_j_IBM_lim, :], np.nan)

    # [PLOT 42] IBM body-force magnitude $|\mathbf{B}|(x,z)$ near IBM surface
    fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
    _pcm1 = ax.contourf(x_in, y_in[:_j_IBM_lim], _B_plot, levels=50, cmap='hot_r')
    plt.colorbar(_pcm1, ax=ax, label=r'$|\mathbf{B}|\,\nu/u_*^2$')
    ax.fill(x_oro_in, y_oro_in, color='grey', zorder=3)
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$z^+$')
    ax.set_title(r'IBM body-force magnitude $|\mathbf{B}|(x,z)$ near IBM surface')
    ax.set_ylim(0, _z_IBM_lim)
    plt.tight_layout()
    plt.savefig(cwd + '/fig/IBM_body_force.png', dpi=300)
    plt.show()

    # %%###########################################################################
    # Plot 2: Spanwise vorticity ωz(x,z) — wall-normal vorticity in meteorological
    # convention — in the near-wall region.  Left panel: 2-D contourf; right panel:
    # 1-D profiles ωz(x+) extracted at three fixed low z+ levels.
    _z_vort_lim = 150
    _j_vort_lim = np.argmin(np.abs(y_inner - _z_vort_lim))
    _vz_norm = vort_z[:_j_vort_lim, :] * (nu / u_star**2)

    # [PLOT 43] Wall-normal vorticity $\omega_z(x,z)$ — near-wall region
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), dpi=300)

    _pcm2 = axes[0].contourf(x_in, y_in[:_j_vort_lim], _vz_norm,
                              levels=50, cmap='RdBu_r')
    plt.colorbar(_pcm2, ax=axes[0], label=r'$\omega_z\,\nu/u_*^2$')
    axes[0].fill(x_oro_in, y_oro_in, color='grey', zorder=3)
    axes[0].set_xlabel(r'$x^+$')
    axes[0].set_ylabel(r'$z^+$')
    axes[0].set_title(r'Wall-normal vorticity $\omega_z(x,z)$ — near-wall region')
    axes[0].set_ylim(0, _z_vort_lim)

    for _zp_fix, _col2 in zip([5, 15, 30], ['blue', 'red', 'green']):
        _jfix = np.argmin(np.abs(y_inner - _zp_fix))
        axes[1].plot(x_in, vort_z[_jfix, :] * (nu / u_star**2),
                     color=_col2, linewidth=1.2,
                     label=r'$z^+ = %d$' % _zp_fix)
    axes[1].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    axes[1].set_xlabel(r'$x^+$')
    axes[1].set_ylabel(r'$\omega_z\,\nu/u_*^2$')
    axes[1].set_title(r'$\omega_z(x^+)$ at fixed low $z^+$')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(cwd + '/fig/wallnormal_vorticity_near_wall.png', dpi=300)
    plt.show()

    # %%###########################################################################
    # Plot 3: Wall shear stress τw(x+) along the IBM surface.
    # tau_wx / tau_wz / tau_wm computed in postprocess.
    # τzx = ν ∂U/∂z  (streamwise); τzy = ν ∂V_y/∂z  (spanwise; V_y = eng. W, met. V).
    # τzx → 0 is the classical marker of marginal separation.

    # [PLOT 44] Wall shear stress $\tau_w(x^+)$ along IBM surface
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(x_in, tau_wx, color='blue',  linewidth=1.5,
            label=r'$\tau_{zx}/u_*^2$')
    ax.plot(x_in, tau_wz, color='red',   linewidth=1.5,
            label=r'$\tau_{zy}/u_*^2$')
    ax.plot(x_in, tau_wm, color='black', linewidth=1.5, linestyle='--',
            label=r'$|\tau_w|/u_*^2$')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
    ax.axvline(x=x_in[eps_top],    color='grey', linewidth=1, linestyle=':',  label='Hill top')
    ax.axvline(x=x_in[eps_lf],     color='grey', linewidth=1, linestyle='--', label='Left flank')
    ax.axvline(x=x_in[eps_bottom], color='grey', linewidth=1, linestyle='-',  label='Valley bottom')
    ax.axvline(x=x_in[eps_rf],     color='grey', linewidth=1, linestyle='-.', label='Right flank')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$\tau_w / u_*^2$')
    ax.set_title(r'Wall shear stress $\tau_w(x^+)$ along IBM surface')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(cwd + '/fig/wall_shear_stress.png', dpi=300)
    plt.show()

    # %%###########################################################################
    # Plot: Pressure coefficient Cp(x+) = ⟨P_y⟩(x+, z_w+) / (0.5 G^2)
    # [PLOT 45] Pressure coefficient $C_p(x^+)$
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(x_in, Cp, color='blue', linewidth=1.5)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
    ax.axvline(x=x_in[eps_top],    color='grey', linewidth=1, linestyle=':',  label='Hill top')
    ax.axvline(x=x_in[eps_lf],     color='grey', linewidth=1, linestyle='--', label='Left flank')
    ax.axvline(x=x_in[eps_bottom], color='grey', linewidth=1, linestyle='-',  label='Valley bottom')
    ax.axvline(x=x_in[eps_rf],     color='grey', linewidth=1, linestyle='-.', label='Right flank')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$C_p$')
    ax.set_title(r'Pressure coefficient $C_p(x^+)$')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(cwd + '/fig/Cp_surface.png', dpi=300)
    plt.show()

    # %%###########################################################################
    # Plot: 2-D field of wall-normal pressure gradient ∂⟨P_y⟩/∂z
    # [PLOT 46] Wall-normal pressure gradient $\partial\langle\overline{P_y}\rangle/\partial z$
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    _limity_P = min(limity, ny)
    cf = ax.contourf(x_in, y_in[:_limity_P],
                     dP_dy[:_limity_P, :] * (nu / u_star**3),
                     levels=100, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label=r'$(\partial \langle P \rangle / \partial z)\,\nu/u_*^3$')
    ax.fill(x_oro_in, y_oro_in, color='grey', zorder=3)
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$z^+$')
    ax.set_title(r'Wall-normal pressure gradient $\partial\langle\overline{P_y}\rangle/\partial z$')
    plt.tight_layout()
    plt.savefig(cwd + '/fig/dPdz_field.png', dpi=300)
    plt.show()

    # %%###########################################################################
    # Plot: along-surface ratio |∂P/∂z| / |∂P/∂x| — pressure-gradient anisotropy
    # [PLOT 47] Pressure gradient ratio $|\partial_z P|/|\partial_x P|$ along IBM surface
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    ax.plot(x_in, ratio_dP, color='black', linewidth=1.2)
    ax.axhline(1, color='grey', linewidth=0.8, linestyle='--', label='isotropic (ratio = 1)')
    ax.axvline(x=x_in[eps_top],    color='grey', linewidth=1, linestyle=':',  label='Hill top')
    ax.axvline(x=x_in[eps_lf],     color='grey', linewidth=1, linestyle='--', label='Left flank')
    ax.axvline(x=x_in[eps_bottom], color='grey', linewidth=1, linestyle='-',  label='Valley bottom')
    ax.axvline(x=x_in[eps_rf],     color='grey', linewidth=1, linestyle='-.', label='Right flank')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'$|\partial_z P| \,/\, |\partial_x P|$')
    ax.set_title(r'Pressure gradient ratio $|\partial_z P|/|\partial_x P|$ along IBM surface')
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(cwd + '/fig/grad_P_ratio_surface.png', dpi=300)
    plt.show()

    # %%###########################################################################
    # Plot 4: 1-D profiles of phase-averaged wall-normal velocity ⟨W̄y⟩(z+)
    # at three x+ locations on the same axes.
    # AvgPhV = W_y in met. convention (eng. v = met. vertical wind W).
    # z+ measured from the local surface height at each column.
    _x_tgt4 = [500, 1200, 2000]
    _col4   = ['blue', 'red', 'green']

    # [PLOT 48] Wall-normal (vertical) velocity $\langle\bar{W}_y\rangle(z^+)$ at $x^+$ = 500, 1200, 2000
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    for _xt4, _c4 in zip(_x_tgt4, _col4):
        _i4  = np.argmin(np.abs(x_in - _xt4))
        _js4 = eps_hgt[_i4]
        _npt = min(limity, ny) - _js4
        if _npt < 2:
            continue
        _z4 = y_inner[_js4:_js4 + _npt] - y_inner[_js4]
        _W4 = AvgPhV[_js4:_js4 + _npt, _i4] / u_star   # AvgPhV = W_y (met.)
        ax.plot(_W4, _z4, color=_c4, linewidth=1.5,
                label=r'$x^+ \approx %.0f$' % x_in[_i4])

    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xlabel(r'$\langle\bar{W}_y\rangle / u_*$')
    ax.set_ylabel(r'$z^+$ (from local surface)')
    ax.set_title(r'Wall-normal (vertical) velocity $\langle\bar{W}_y\rangle(z^+)$ at $x^+$ = 500, 1200, 2000')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(cwd + '/fig/Wy_profiles_3loc.png', dpi=300)
    plt.show()

    # %%###########################################################################
    # Plot 5: Streamwise deficit Δ_U = (Gx − ⟨Ū⟩)/u* at multiple x+ stations.
    # JH75 is a 2D streamwise theory only — spanwise is outside its scope.
    # Two fits per station: (a) power-law C·(z+)^α (empirical, log-log);
    #                       (b) log-law A·ln(z+)+B (JH75-consistent, semi-log).
    _Udef2D = (Gx - AvgPhU) * mask0

    # Station selection: crest, floor, 5 evenly-spaced points, + peak-deficit column
    _i_crest5 = int(np.argmax(eps_hgt))
    _i_floor5 = int(np.argmin(eps_hgt))
    _stations5 = np.linspace(min(_i_crest5, _i_floor5),
                              max(_i_crest5, _i_floor5), 5, dtype=int)
    _j_ns5   = min(hill_hgt + 5, ny)
    _i_peak5 = int(np.argmax(np.max(_Udef2D[:_j_ns5, :], axis=0)))
    if _i_peak5 not in _stations5:
        _stations5 = np.unique(np.append(_stations5, _i_peak5))

    # Fit window: rim height → 4× rim height (one decade, JH75 inner layer)
    _z_hill5  = y_inner[hill_hgt]
    _z_fit_hi = _z_hill5 * 4.0
    _yref5    = np.logspace(np.log10(_z_hill5 * 0.5), np.log10(_z_hill5 * 5.0), 200)

    # Extract profile and fit at each station
    _profiles5 = []
    for _ist5 in _stations5:
        _js5  = eps_hgt[_ist5]
        _z5   = y_inner[_js5:limity]
        _d5   = _Udef2D[_js5:limity, _ist5] / u_star
        _vld5 = (_z5 > 0.5) & (_d5 > 0)
        if np.sum(_vld5) < 5:
            continue
        _fm5  = _vld5 & (_z5 >= _z_hill5) & (_z5 <= _z_fit_hi)
        _rec5 = {'i': _ist5, 'z': _z5, 'd': _d5, 'vld': _vld5,
                 'power': None, 'log': None}
        if np.sum(_fm5) >= 4:
            _zf5 = _z5[_fm5]
            _df5 = _d5[_fm5]
            try:
                sl_p, ic_p, r_p, *_ = linregress(np.log(_zf5), np.log(_df5))
                _rec5['power'] = {'alpha': sl_p, 'C': np.exp(ic_p), 'r2': r_p**2}
            except Exception:
                pass
            try:
                sl_l, ic_l, r_l, *_ = linregress(np.log(_zf5), _df5)
                _rec5['log'] = {'A': sl_l, 'B': ic_l, 'r2': r_l**2}
            except Exception:
                pass
        _profiles5.append(_rec5)

    # --- Figure A: all stations, log-log, power-law fits overlaid ---
    _colors5 = plt.cm.viridis(np.linspace(0.1, 0.9, len(_profiles5)))
    # [PLOT 49] Streamwise deficit $\Delta_U$ — multiple $x^+$ stations (log–log)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    for k5, _pr5 in enumerate(_profiles5):
        _c5 = _colors5[k5]
        ax.loglog(_pr5['z'][_pr5['vld']], _pr5['d'][_pr5['vld']],
                  '-', color=_c5, linewidth=1.8,
                  label=r'$x^+ \approx %.0f$' % x_in[_pr5['i']])
        if _pr5['power'] is not None:
            _p5 = _pr5['power']
            ax.loglog(_yref5, _p5['C'] * _yref5**_p5['alpha'],
                      '--', color=_c5, linewidth=1.0, alpha=0.7)
    ax.axvline(_z_hill5,  color='grey', ls=':', lw=0.8)
    ax.axvline(_z_fit_hi, color='grey', ls=':', lw=0.8)
    ax.set_xlabel(r'$z^+$')
    ax.set_ylabel(r'$(G_x - \langle\bar{U}\rangle)\,/\,u_*$')
    ax.set_title(r'Streamwise deficit $\Delta_U$ — multiple $x^+$ stations (log–log)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(cwd + '/fig/U_deficit_multi_station.png', dpi=300)
    plt.show()

    # --- Figure B: power-law vs log-law at the peak-deficit station ---
    _prof_pk5 = next((p for p in _profiles5 if p['i'] == _i_peak5), None)
    if _prof_pk5 is None and _profiles5:
        _prof_pk5 = _profiles5[0]
    if _prof_pk5 is not None:
        _xp5 = x_in[_prof_pk5['i']]
        _zv5 = _prof_pk5['z'][_prof_pk5['vld']]
        _dv5 = _prof_pk5['d'][_prof_pk5['vld']]
        # [PLOT 50] Power-law fit (empirical, log–log)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

        # Left: log-log — power-law fit
        ax1.loglog(_zv5, _dv5, 'b-', linewidth=2,
                   label=r'$\Delta_U$, $x^+ \approx %.0f$' % _xp5)
        if _prof_pk5['power'] is not None:
            _p5 = _prof_pk5['power']
            ax1.loglog(_yref5, _p5['C'] * _yref5**_p5['alpha'], 'r--', linewidth=1.5,
                       label=r'$C\,(z^+)^{%.2f}$, $R^2=%.4f$' % (_p5['alpha'], _p5['r2']))
        ax1.axvline(_z_hill5,  color='grey', ls=':', lw=0.8)
        ax1.axvline(_z_fit_hi, color='grey', ls=':', lw=0.8)
        ax1.set_xlabel(r'$z^+$')
        ax1.set_ylabel(r'$(G_x - \langle\bar{U}\rangle)\,/\,u_*$')
        ax1.set_title(r'Power-law fit (empirical, log–log)')
        ax1.legend(fontsize=9)
        ax1.grid(True, which='both', ls='--', lw=0.5)

        # Right: semi-log — log-law fit (JH75-consistent)
        ax2.plot(np.log(_zv5), _dv5, 'b-', linewidth=2,
                 label=r'$\Delta_U$, $x^+ \approx %.0f$' % _xp5)
        if _prof_pk5['log'] is not None:
            _l5 = _prof_pk5['log']
            _ln5 = np.linspace(np.log(_z_hill5 * 0.5), np.log(_z_hill5 * 5.0), 200)
            ax2.plot(_ln5, _l5['A'] * _ln5 + _l5['B'], 'r--', linewidth=1.5,
                     label=r'$A\,\ln(z^+)+B$, $R^2=%.4f$' % _l5['r2'])
        ax2.axvline(np.log(_z_hill5),  color='grey', ls=':', lw=0.8)
        ax2.axvline(np.log(_z_fit_hi), color='grey', ls=':', lw=0.8)
        ax2.set_xlabel(r'$\ln(z^+)$')
        ax2.set_ylabel(r'$(G_x - \langle\bar{U}\rangle)\,/\,u_*$')
        ax2.set_title(r'Log-law fit (JH75-consistent, semi-log)')
        ax2.legend(fontsize=9)
        ax2.grid(True, which='both', ls='--', lw=0.5)

        plt.suptitle(r'Streamwise deficit $\Delta_U$ at peak station '
                     r'($x^+ \approx %.0f$)' % _xp5)
        plt.tight_layout()
        plt.savefig(cwd + '/fig/U_deficit_fit_comparison.png', dpi=300)
        plt.show()

    # --- Summary table ---
    print('\n' + '='*80)
    print('Streamwise deficit fit summary')
    print('Valley rim H+ = %.1f,  fit window: [%.1f, %.1f] z+'
          % (_z_hill5, _z_hill5, _z_fit_hi))
    print('='*80)
    print('%8s | %10s | %10s | %10s | %10s | %10s' %
          ('x+', 'alpha(pwr)', 'R2(power)', 'A (log)', 'B (log)', 'R2(log)'))
    print('-'*80)
    for _pr5 in _profiles5:
        _xp5 = x_in[_pr5['i']]
        _p5  = _pr5['power']
        _l5  = _pr5['log']
        if _p5 is not None and _l5 is not None:
            print('%8.1f | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f' %
                  (_xp5, _p5['alpha'], _p5['r2'], _l5['A'], _l5['B'], _l5['r2']))
        elif _p5 is not None:
            print('%8.1f | %10.4f | %10.4f | %10s | %10s | %10s' %
                  (_xp5, _p5['alpha'], _p5['r2'], '-', '-', '-'))
        else:
            print('%8.1f | %10s | %10s | %10s | %10s | %10s' %
                  (_xp5, '-', '-', '-', '-', '-'))
    print('='*80)

    # %%###########################################################################
    # Plot 6: ⟨Ūy⟩(z+) and ⟨W̄y⟩(z+) at x+=100 (inlet) vs x+=2400 (outlet),
    # compared with the Coleman et al. (1990) flat-wall Ekman DNS baseline.
    # AvgPhU = U_y (streamwise); AvgPhV = W_y (wall-normal; eng. v = met. W).
    # Coleman reference: log-law for U_y (κ=0.42, B=4.5); W_y = 0 on flat wall.
    # z+ is relative to the local surface height at each column.
    _i6_in  = np.argmin(np.abs(x_in - 100))
    _i6_out = np.argmin(np.abs(x_in - 2400))
    _js6_in  = eps_hgt[_i6_in]
    _js6_out = eps_hgt[_i6_out]

    _z6_in  = y_inner[_js6_in:limity]  - y_inner[_js6_in]
    _z6_out = y_inner[_js6_out:limity] - y_inner[_js6_out]
    _U6_in  = AvgPhU[_js6_in:limity,  _i6_in]  / u_star
    _U6_out = AvgPhU[_js6_out:limity, _i6_out] / u_star
    _W6_in  = AvgPhV[_js6_in:limity,  _i6_in]  / u_star   # AvgPhV = W_y (met.)
    _W6_out = AvgPhV[_js6_out:limity, _i6_out] / u_star

    # Coleman et al. log-law reference (valid for z+ > 30)
    _z6_ref = np.linspace(1, y_inner[limity - 1], 500)
    _U6_ref = np.where(_z6_ref > 30, (1/kappa) * np.log(_z6_ref) + 4.5, np.nan)

    # [PLOT 51] Streamwise $\langle\bar{U}_y\rangle(z^+)$
    fig, (ax6U, ax6W) = plt.subplots(1, 2, figsize=(12, 7), dpi=300, sharey=False)

    # --- Left panel: streamwise U_y ---
    ax6U.semilogx(_U6_in,  _z6_in,  'b-',  linewidth=1.5,
                  label=r'$x^+ \approx 100$ (inlet)')
    ax6U.semilogx(_U6_out, _z6_out, 'b--', linewidth=1.5,
                  label=r'$x^+ \approx 2400$ (outlet)')
    ax6U.semilogx(_U6_ref, _z6_ref, 'k:',  linewidth=1.5,
                  label=r'Coleman et al. (log law, $\kappa=0.42$, $B=4.5$)')
    ax6U.set_xlabel(r'$\langle\bar{U}_y\rangle / u_*$')
    ax6U.set_ylabel(r'$z^+$ (from local surface)')
    ax6U.set_title(r'Streamwise $\langle\bar{U}_y\rangle(z^+)$')
    ax6U.legend(fontsize=8)
    ax6U.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Right panel: wall-normal W_y (met.) ---
    ax6W.plot(_W6_in,  _z6_in,  'r-',  linewidth=1.5,
              label=r'$x^+ \approx 100$ (inlet)')
    ax6W.plot(_W6_out, _z6_out, 'r--', linewidth=1.5,
              label=r'$x^+ \approx 2400$ (outlet)')
    ax6W.axvline(0, color='k', linestyle=':', linewidth=1.5,
                 label=r'Coleman et al. ($\langle W_y\rangle = 0$, flat wall)')
    ax6W.set_xlabel(r'$\langle\bar{W}_y\rangle / u_*$')
    ax6W.set_ylabel(r'$z^+$ (from local surface)')
    ax6W.set_title(r'Wall-normal (vertical) $\langle\bar{W}_y\rangle(z^+)$')
    ax6W.legend(fontsize=8)
    ax6W.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.suptitle(r'Inlet ($x^+\!\approx\!100$) vs outlet ($x^+\!\approx\!2400$): '
                 r'comparison with Coleman et al.\ (1990)')
    plt.tight_layout()
    plt.savefig(cwd + '/fig/UWy_inlet_outlet_Coleman.png', dpi=300)
    plt.show()

    # %%###########################################################################
    # animate: read_plane / read_all_planes are defined in functions.py and imported
    # via 'from functions import *'.

if animate == 1:
    # ---- coordinate convention note -----------------------------------
    # tlab Fortran stores velocity components in this order:
    #   index 0 → u  : streamwise        (x-direction)
    #   index 1 → v  : wall-normal       (y-direction in tlab)
    #   index 2 → w  : spanwise          (z-direction in tlab)
    #
    # The meteorological community (convention used throughout this script) defines:
    #   u = streamwise  (along-wind)
    #   v = spanwise    (crosswind / North component)
    #   w = vertical    (wall-normal / up component)
    #
    # Mapping between the two systems:
    #   tlab u (index 0) = meteo u  →  streamwise          (both agree)
    #   tlab v (index 1) = meteo w  →  wall-normal/vertical (tlab's 'v' is meteo's 'w')
    #   tlab w (index 2) = meteo v  →  spanwise/crosswind   (tlab's 'w' is meteo's 'v')
    #
    # Labels below and throughout the script follow METEOROLOGICAL convention.
    # ---- user settings ------------------------------------------------
    N_KPLANES  = 1     # k-planes saved per file  (kplanes%n in TLAB)
    # File variable order in tlab: u(streamwise)=0, v(wall-normal)=1, w(spanwise)=2, s1=3, p=4
    NVARS      = 5
    KPLANE_IDX = 0     # which k-plane to show (0-based)
    NY_ANIM    = 430   # wall-normal points to include
    FIRST_ITER = 262510
    LAST_ITER  = 264500
    STEP       = 10
    FPS        = 10
    OUTPUT_MP4 = cwd + 'planesK_animation.mp4'
    # Frames where max|u| or max|v| (meteo: streamwise idx 0, spanwise idx 2) exceeds
    # this threshold are considered unphysical and are discarded before rendering.
    VEL_MAX_THRESHOLD = 1.5
    # Labels use meteorological convention:
    #   u' (tlab idx 0) = streamwise fluctuation
    #   w' (tlab idx 1) = wall-normal/vertical fluctuation  (tlab calls this v)
    #   v' (tlab idx 2) = spanwise/crosswind fluctuation    (tlab calls this w)
    #   θ' (tlab idx 3) = potential temperature fluctuation (scalar 1)
    #   p' (tlab idx 4) = pressure fluctuation
    ANIM_VARS  = [(0, r"$u'$ (streamwise)"),
                  (1, r"$w'$ (wall-normal) [tlab: $v$]"),
                  (2, r"$v'$ (spanwise) [tlab: $w$]"),
                  (3, r"$\theta'$ (pot. temperature)"),
                  (4, r"$p'$")]
    # -------------------------------------------------------------------

    # Collect available files
    filepaths_anim = []
    for it in range(FIRST_ITER, LAST_ITER + 1, STEP):
        fp = os.path.join(cwd, f'planesK.{it}')
        if os.path.isfile(fp):
            filepaths_anim.append((it, fp))
        else:
            print(f'WARNING: file not found – {fp}')

    if not filepaths_anim:
        raise FileNotFoundError(
            f'No planesK.* files found in {os.path.abspath(cwd)} '
            f'for iterations {FIRST_ITER}–{LAST_ITER}.'
        )
    n_iters = len(filepaths_anim)
    print(f'Found {n_iters} planesK files.')

    # Pass 1: scan every file to find color limits — only one frame lives in RAM at a time
    print('Pass 1: computing color limits …')
    var_abs_max = {vi: 0.0 for vi, _ in ANIM_VARS}
    p_solid_acc, p_fluid_acc, p_n = 0.0, 0.0, 0
    valid_filepaths = []
    for it, fp in filepaths_anim:
        all_planes = read_all_planes(fp, nx, ny, N_KPLANES, NVARS, KPLANE_IDX)
        # Velocity sanity check: tlab idx 0 = meteo u (streamwise), idx 2 = meteo v (spanwise)
        u_max = float(np.max(np.abs(all_planes[0])))
        v_max = float(np.max(np.abs(all_planes[2])))
        if u_max > VEL_MAX_THRESHOLD or v_max > VEL_MAX_THRESHOLD:
            print(f'  DISCARDING iter {it}: max|u|={u_max:.3f}  max|v|={v_max:.3f} '
                  f'(threshold {VEL_MAX_THRESHOLD})')
            del all_planes
            continue
        valid_filepaths.append((it, fp))
        # pressure sanity check (full ny, not cropped)
        p_full = all_planes[4]
        p_solid_acc += float(np.mean(np.abs(p_full[eps == 1])))
        p_fluid_acc += float(np.mean(np.abs(p_full[eps == 0])))
        p_n += 1
        for vi, _ in ANIM_VARS:
            plane  = all_planes[vi]
            x_mean = plane.mean(axis=1, keepdims=True)
            prime  = ((plane - x_mean) * mask0)[:NY_ANIM, :]
            frame_max = float(np.percentile(np.abs(prime), 99))
            var_abs_max[vi] = max(var_abs_max[vi], frame_max)
        del all_planes, plane, x_mean, prime   # free every frame before reading the next
    n_discarded = len(filepaths_anim) - len(valid_filepaths)
    print(f'Velocity check: {n_discarded} frame(s) discarded, {len(valid_filepaths)} kept.')
    if not valid_filepaths:
        raise RuntimeError('All frames were discarded by the velocity sanity check.')
    filepaths_anim = valid_filepaths
    n_iters = len(filepaths_anim)
    print(f'Pressure check — mean |p| solid: {p_solid_acc/p_n:.4e}  '
          f'fluid: {p_fluid_acc/p_n:.4e}  ratio: {(p_solid_acc/p_n)/(p_fluid_acc/p_n+1e-16):.3f}')
    var_limits = {vi: (-(var_abs_max[vi] or 1e-6), var_abs_max[vi] or 1e-6)
                  for vi, _ in ANIM_VARS}
    for vi, label in ANIM_VARS:
        print(f'  {label}: {var_limits[vi]}')

    # meshgrid — shape (NY_ANIM, nx) matches every field
    Gx_p, Gy_p = np.meshgrid(x_in, y_in[:NY_ANIM])

    GRID_KW = dict(linestyle=':', linewidth=0.3, alpha=0.25, color='grey')

    # Build 2×2 figure — one file read per frame serves all 4 panels simultaneously
    fig_p, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat   = axes.flatten()

    IBM_KW = dict(facecolor='black', edgecolor='white', linewidth=1.2)

    # Draw initial frame then free raw data
    # Order: contourf first, IBM fill second so it sits on top and covers solid cells.
    init_planes = read_all_planes(filepaths_anim[0][1], nx, ny, N_KPLANES, NVARS, KPLANE_IDX)
    for ax, (vi, label) in zip(axes_flat, ANIM_VARS):
        vmin, vmax = var_limits[vi]
        field0 = ((init_planes[vi] - init_planes[vi].mean(axis=1, keepdims=True))
                  * mask0)[:NY_ANIM, :]
        ax.contourf(Gx_p, Gy_p, field0, levels=500, cmap='Greys', vmin=vmin, vmax=vmax)
        del field0
        fig_p.colorbar(plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap='Greys'),
            ax=ax, label=label)
        ax.fill(x_oro_in, y_oro_in, **IBM_KW)   # drawn after contourf so it covers the solid
        ax.set_xlabel(r'$x^+$')
        ax.set_ylabel(r'$z^+$')
        ax.set_title(label)
        ax.grid(**GRID_KW)
    del init_planes

    axes_flat[0].legend(
        handles=[mpatches.FancyArrowPatch if False else
                 mpatches.Patch(facecolor='black', edgecolor='white',
                                linewidth=1.2, label='Solid IBM')],
        loc='upper right')
    suptitle_p = fig_p.suptitle('')
    fig_p.tight_layout()

    # update: clear both collections (contourf) AND patches (fill) each frame,
    # then redraw contourf first and IBM fill second so fill is always on top.
    def update_planes(frame_idx):
        it, fp  = filepaths_anim[frame_idx]
        planes  = read_all_planes(fp, nx, ny, N_KPLANES, NVARS, KPLANE_IDX)
        for ax, (vi, _) in zip(axes_flat, ANIM_VARS):
            vmin, vmax = var_limits[vi]
            field = ((planes[vi] - planes[vi].mean(axis=1, keepdims=True))
                     * mask0)[:NY_ANIM, :]
            for coll in list(ax.collections):
                coll.remove()
            for patch in list(ax.patches):
                patch.remove()
            ax.contourf(Gx_p, Gy_p, field, levels=500,
                        cmap='Greys', vmin=vmin, vmax=vmax)
            ax.fill(x_oro_in, y_oro_in, **IBM_KW)   # on top of contourf
            ax.grid(**GRID_KW)
            del field
        del planes
        suptitle_p.set_text(f'Iteration {it}   |   K-plane {KPLANE_IDX}')
        return []

    anim_p = animation.FuncAnimation(fig_p, update_planes, frames=n_iters,
                                     interval=1000 // FPS, blit=False)

    out_base = cwd + 'planesK_animation'
    if animation.FFMpegWriter.isAvailable():
        writer_p = animation.FFMpegWriter(fps=FPS, bitrate=2000,
                                          extra_args=['-vcodec', 'libx264',
                                                      '-pix_fmt', 'yuv420p'])
        out_path = out_base + '.mp4'
    else:
        writer_p = animation.PillowWriter(fps=FPS)
        out_path  = out_base + '.gif'

    print(f'Writing {out_path} …')
    anim_p.save(out_path, writer=writer_p)
    print(f'Animation saved as {out_path}')
    plt.close(fig_p)
    del anim_p, update_planes, fig_p, axes, axes_flat, suptitle_p

    # %%
