# %%
# ============================================================================
# ROTATED-FRAME variant of PhAvg.py.
# The horizontal mean / dispersive velocity fields and the wall-normal Reynolds
# momentum-flux pair are rotated by `alpha` (the ~25° geostrophic tilt) AFTER the
# ghost-cell interpolation and BEFORE the derivatives, so the geostrophic wind is
# aligned with x — matching the frame of the Kostelecky & Ansorge reference .nc
# cases.  Everything else (derivatives, Method-2 budget, plots) runs unchanged.
# Figures go to the shared fig/ folder; derivative caches use a *_rot suffix, so
# the caches never clobber the unrotated PhAvg.py run (the figures now merge into
# one folder by request).  See the "FRAME ROTATION" block for the transformation.
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
# Fallback: also expose the MyPyLib master dir (via realpath, following the
# symlink) so master-only modules such as IO.py resolve even when a per-sim data
# directory has not yet been re-linked by setup.sh.  Appended (not inserted at 0)
# so a LOCAL config.py / module still takes precedence.
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
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
from scipy.ndimage import uniform_filter
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib import cm
from config import *
# ── Master-config backfill (stale per-case config guard) ─────────────────────
# A per-simulation config.py (which SHADOWS the master via the sys.path
# bootstrap above — by design, it carries per-case values such as Fr) can
# predate constants later added to the master config (§13).  Backfill any
# names the local config lacks from the MASTER config (the one next to this
# script's realpath) so a stale case config prints a notice instead of
# NameError-ing mid-run.  Names the local config DOES define keep their local
# per-case values.
import config as _cfg_local
_cfg_master_dir = os.path.dirname(os.path.realpath(__file__))
if os.path.dirname(os.path.realpath(_cfg_local.__file__)) != _cfg_master_dir:
    import importlib.util as _ilu
    _cfg_spec = _ilu.spec_from_file_location(
        '_config_master', os.path.join(_cfg_master_dir, 'config.py'))
    _cfg_master = _ilu.module_from_spec(_cfg_spec)
    _cfg_spec.loader.exec_module(_cfg_master)
    _cfg_backfill = {_k: _v for _k, _v in vars(_cfg_master).items()
                     if not _k.startswith('_') and _k not in globals()}
    if _cfg_backfill:
        # Patch the LIVE config module (not just this script's globals) so that a
        # downstream direct `from config import <name>` — e.g. functions.py's
        # `from config import obu_kappa` — also sees the backfilled constants.
        for _k, _v in _cfg_backfill.items():
            setattr(_cfg_local, _k, _v)
        globals().update(_cfg_backfill)
        print(f'[config] local config.py is missing {len(_cfg_backfill)} newer '
              f'master constant(s) — backfilled from the master config: '
              f'{sorted(_cfg_backfill)}')
    del _ilu, _cfg_spec, _cfg_master, _cfg_backfill
del _cfg_local, _cfg_master_dir
# ──────────────────────────────────────────────────────────────────────────────
from functions import *
import IO   # var_names + array/pickle I/O (write_avg_arrays / read_avg_arrays / write_results_pickle)
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


# Reference-overlay helpers (ref_plot / ref_mark) are shared routines in
# functions.py (`from functions import *`).  The Kostelecky & Ansorge fig-4
# budget is split per module role: IO.read_ekman_budget_profiles loads the
# reference profiles, the eq.-4.2 budget is computed IN THIS SCRIPT (PLOT 32r),
# and PlotField.plot_fig4_budget only draws the passed-in terms.


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

# ─────────────────────────────────────────────────────────────────────────────
# Reynolds (→ Re_lambda) and Froude numbers — taken from the run's tlab.ini
# ─────────────────────────────────────────────────────────────────────────────
# The run's own tlab.ini is the source of truth for the two case-defining
# parameters, overriding the values imported via `from config import *` above:
#   • Reynolds → Re_lambda  (tlab's parameter = 1/nu = 0.5·Re_D²).  nu, Re (= Re_D)
#     and the inner scalings (Re_tau, l_visc/l_in/wall_units) are RE-DERIVED from it
#     here so the whole script is self-consistent.  These l_* are preliminary
#     (config's prescribed u_star); the per-case Method-2 u_star measured from the
#     data later re-refreshes l_in/y_in (see the "Refresh inner/outer scalings"
#     block), but nu is fixed HERE and is what that refresh uses.
#   • Froude → Fr.  A finite Fr is a STRATIFIED run (Obukhov wall law, the
#     `_stratified = np.isfinite(Fr)` gate); if 'Froude=' is absent the input
#     carries no buoyancy forcing ⇒ Fr = ∞ (neutral, classical log law).
# tlab.ini MUST be present (it defines the run) — its absence stops the script.
# Format (INI, one key=value per line, no spaces):
#     [Parameters]
#     Reynolds=125000
#     Schmidt=1.0
#     Rossby=1.0
#     Froude=1
_tlab_ini = os.path.join(cwd, 'tlab.ini')
if not os.path.isfile(_tlab_ini):
    raise SystemExit('[PhAvg_rotated] tlab.ini not found in %s — cannot determine '
                     'the Reynolds/Froude numbers; stopping execution.' % cwd)
with open(_tlab_ini, 'r') as _fh:
    _tlab_lines = _fh.readlines()

def _tlab_value(_key, _lines=_tlab_lines):
    """Token written after the literal '<key>' (e.g. 'Froude=') in tlab.ini; None
    if the key never appears.  Takes the first whitespace token and drops any
    inline #/; comment.  (_lines bound as a default at def-time so the later
    `del _tlab_lines` does not affect it.)"""
    for _ln in _lines:
        _i = _ln.find(_key)
        if _i == -1:
            continue
        _tok = _ln[_i + len(_key):].strip()
        _tok = _tok.split()[0] if _tok.split() else ''
        return _tok.split('#')[0].split(';')[0].strip()
    return None

# Reynolds → Re_lambda (+ the derived nu / Re / preliminary inner scalings)
_re_str = _tlab_value('Reynolds=')
if _re_str is None:
    print('[config] tlab.ini has no "Reynolds=" — keeping config Re_lambda = %s' % Re_lambda)
else:
    try:
        Re_lambda = float(_re_str)
    except ValueError:
        raise SystemExit('[PhAvg_rotated] could not parse the Reynolds value %r from '
                         'tlab.ini; stopping execution.' % _re_str)
    nu         = 1.0 / Re_lambda            # tlab Re_lambda = 1/nu
    Re         = np.sqrt(2.0 * Re_lambda)   # Re_D = sqrt(2·Re_lambda)
    Re_tau     = (u_star**2) / nu           # preliminary (config u_star); refreshed post-fit
    l_visc     = nu / u_star
    l_in       = l_visc
    wall_units = l_visc
    print('[config] Reynolds from tlab.ini: Re_lambda = %s  (nu = %.3e, Re_D = %.2f)'
          % (Re_lambda, nu, Re))

# Froude → Fr  (finite ⇒ stratified; 'Froude=' absent ⇒ neutral, Fr = inf)
_fr_str = _tlab_value('Froude=')
if _fr_str is None:
    Fr = np.inf
    print('[config] tlab.ini has no "Froude=" → neutral run (Fr = inf)')
else:
    try:
        Fr = float(_fr_str)
    except ValueError:
        raise SystemExit('[PhAvg_rotated] could not parse the Froude value %r from '
                         'tlab.ini; stopping execution.' % _fr_str)
    print('[config] Froude from tlab.ini: Fr = %s  (stratified)' % Fr)
del _tlab_ini, _tlab_lines, _tlab_value, _re_str, _fr_str

# All figures/plots are written to a SINGLE <data dir>/fig/ folder (shared with
# PhAvg.py and results.py) — the rotated-frame variant no longer isolates its
# output in fig_rotated/. Derivative caches still use the *_rot.npy suffix, so the
# rotated and unrotated runs remain cache-isolated even though the figures merge.
fig_dir = os.path.join(cwd, 'fig')
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
# Fundamental streamwise wavenumber of the sinusoidal valley (2π / L_x); used
# only to draw the orography outline y_oro.  NB: keep this SEPARATE from `dx`
# (the grid spacing set above) — `dx` is pickled and consumed by results.py as
# the streamwise grid spacing (its 2·dx⁺ Nyquist check), so it must NOT be
# repurposed as a wavenumber here.
kx0 = (2*np.pi/x[-1])
y_oro = np.round((hill_hgt/(2**1))*(1 + np.cos(kx0*(x))))
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
# ── Obukhov (1971) stability-corrected wall-law helpers ──────────────────────
# The parametric surface-layer profile + its lookup table were moved to
# functions.py (self-contained, arrive via `from functions import *`):
#   obu_up_of_xi, obu_eta_of_xi, obu_psi, obu_wind_profile, obu_K_unstable,
#   fit_modified_loglaw, validate_obukhov_tableIII.
# The paper's fixed κ = 0.4 is now config.obu_kappa.


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
        base = avg_iter_base
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
    ScalGbl   = np.mean(AvgScal, axis=1)
    DispScal  = (AvgScal - ScalGbl[:,np.newaxis]) * mask0
    
    for i in range (dim):
        VelGbl2D[:,:,i] = (np.tile(VelGbl[:,i].reshape(ny,1), nx).reshape(ny,nx))*mask0
    
    
    for i in range(6):
        turb1D[:,i] = np.mean(Turb[:,:,i], axis=1)

    # Triple decomposition of the Reynolds stress tensor (Raupach & Shaw 1982):
    # <u_i u_j> = <u_i><u_j>  (mean×mean, _g)
    #           + ũ_i ũ_j      (dispersive×dispersive, _t for tilda)
    #           + <u_i'' u_j''>  (turbulent, _d for double-prime, computed later)
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
        base = avg_iter_base
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
    IO.write_avg_arrays(globals())

# load_arrays: restore pre-computed fields from .npy files (avoids rerunning averaging)
# and load reference smooth-wall / rough-wall DNS data from NetCDF for comparison.
if (1 == load_arrays):
    # Reload the saved .npy fields; globals().update keeps every array as a
    # top-level, Spyder-inspectable name (rey_*, AvgStr*, UU_G/UU_disp, AvgPh*,
    # DispVel*, cross terms, and zero-filled du_dt/ds_dt).  See IO.read_avg_arrays.
    globals().update(IO.read_avg_arrays(ny, nx, dim, scal))

# Postprocess
# %%
if (1 == postprocess):
    # Tee all subsequent run statistics to sim_stats.log (rewritten each run)
    # while still printing them to the terminal. Recording starts here, at the
    # 'Computing ghost-cell interpolated fields (PCHIP) ...' message below.
    start_stats_log('sim_stats.log')

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
    # compute_ghost_interp lives in functions.py; the lambda defers it for the
    # load_or_compute .npy cache (recompute_derivatives / *_rot.npy).
    AvgPhU_i, AvgPhU_j, AvgPhV_i, AvgPhV_j, AvgPhW_i, AvgPhW_j, AvgP_i, AvgP_j = \
        load_or_compute(['AvgPhU_i', 'AvgPhU_j', 'AvgPhV_i', 'AvgPhV_j',
                         'AvgPhW_i', 'AvgPhW_j', 'AvgP_i',   'AvgP_j'],
                        recompute_derivatives,
                        lambda: compute_ghost_interp(x, y, nx, ny, eps,
                                                     [AvgPhU, AvgPhV, AvgPhW, AvgP],
                                                     ghost_depth, n_anchor, smooth_width),
                        label='ghost-cell interpolated fields (PCHIP)')

    # ══════════════════════════════════════════════════════════════════════════
    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║  🔒 LOCKED — VALIDATED DATA ORIENTATION.  DO NOT MODIFY THIS BLOCK.        ║
    # ║  (Verified against K&A 2024 fig. 4 + stored FrictionVelocity/Angle;        ║
    # ║   see CLAUDE.md "Standard shear-stress budget formulation".)               ║
    # ╠══════════════════════════════════════════════════════════════════════════╣
    # ║  WHAT THE PLOTTED DATA IS — rotation angle of the velocity & momentum      ║
    # ║  (stress) tensors:                                                         ║
    # ║  • Rotation angle:  α = config.alpha  (≈ −24.67° = −0.4305 rad, the        ║
    # ║    geostrophic tilt).  Fields are rotated by α so the GEOSTROPHIC WIND     ║
    # ║    aligns with +x (spanwise geostrophic → 0); u* is rotation-invariant.   ║
    # ║  • VELOCITY (rank-1) rotates by ONE contraction (proper rotation):        ║
    # ║        u' = u·cosα − w·sinα ,  w' = u·sinα + w·cosα   [functions.rotate_pair]║
    # ║    applied to ⟨U⟩,⟨W⟩ (+ _i/_j interps) and the dispersive ũ,w̃.            ║
    # ║  • MOMENTUM-FLUX pair (rey_uv,rey_vw)=(⟨u''v''⟩,⟨v''w''⟩): ONE index is on ║
    # ║    the wall-normal rotation axis v, so the single rotate_pair IS the full  ║
    # ║    tensor transform R_im R_jn τ_mn — rotates as a VECTOR.                  ║
    # ║  • IN-PLANE stress (rey_uu,rey_uw,rey_ww): BOTH indices in the u–w plane → ║
    # ║    full 2×2 rank-2 transform (trace-preserving; TKE & u* bit-unchanged).   ║
    # ║  • Wall-normal V and rey_vv (axis indices) and scalar b (rank-0) are       ║
    # ║    rotation-INVARIANT.  Reference .nc cases already carry their own frame  ║
    # ║    (rough r1 stored at ~18.7° off x; smooth Re500 geostrophic-aligned).   ║
    # ╚══════════════════════════════════════════════════════════════════════════╝
    # ░░  FRAME ROTATION  ░░  (this is the ONLY physics change vs PhAvg.py)
    # Rotate the horizontal components by `alpha` (config; the ~25° geostrophic
    # tilt) so the geostrophic wind aligns with x — the frame of the reference
    # .nc cases.  Applied AFTER interpolation and BEFORE the derivatives, so the
    # derivative/budget pipeline below operates on the rotated fields unchanged.
    #   proper rotation R(alpha):  u' = u·cosα − w·sinα,  w' = u·sinα + w·cosα
    # Rotated as VECTORS (one R contraction): mean U,W and their _i/_j interpolations;
    # dispersive U,W; and the wall-normal momentum-flux pair (rey_uv=⟨u''v''⟩,
    # rey_vw=⟨v''w''⟩; TURBULENT, not the full Reynolds stress).  The latter have ONE
    # index on the rotation axis v (R_v·=δ), so
    # the single rotate_pair already IS the full tensor transform R_im R_jn τ_mn.
    # Rotated as a rank-2 TENSOR (two R contractions): the in-plane stresses
    # rey_uu/uw/ww — both indices live in the rotated u–w plane, so they take the
    # full 2×2 transform (_rotate_inplane below).  It is a no-op at alpha=0 and
    # preserves the trace (uu+ww), so TKE = ½(rey_uu+rey_vv+rey_ww) and u* are
    # bit-for-bit unchanged; it makes the turbulent-advection term (∂rey_uu/∂x) and
    # the Ruu/Ruv/Rvv maps frame-consistent.
    # Unchanged: wall-normal V (rotation axis); rey_vv (both indices on the axis).
    # The scalar AvgScal (potential temperature / buoyancy b) is a rank-0 tensor and
    # is therefore rotation-INVARIANT — its value at each (x,z) point is identical in
    # both frames, so it is deliberately NOT passed through rotate_pair (there is no
    # second component to mix it with).  Rotation reaches buoyancy only THROUGH the
    # velocity: the horizontal heat-flux vector (e.g. ũb̃ = DispVelU·DispScal) rotates
    # because DispVelU does, while DispScal = AvgScal − ⟨b⟩ is unchanged.
    # u* = ‖τ_w‖ is rotation-invariant — only the τ_zx / τ_zy split changes.
    _rc, _rs = np.cos(alpha), np.sin(alpha)   # rotation cos/sin (functions.rotate_pair)
    AvgPhU,   AvgPhW   = rotate_pair(AvgPhU,   AvgPhW,   _rc, _rs)
    AvgPhU_i, AvgPhW_i = rotate_pair(AvgPhU_i, AvgPhW_i, _rc, _rs)
    AvgPhU_j, AvgPhW_j = rotate_pair(AvgPhU_j, AvgPhW_j, _rc, _rs)
    DispVelU, DispVelW = rotate_pair(DispVelU, DispVelW, _rc, _rs)
    rey_uv,   rey_vw   = rotate_pair(rey_uv,   rey_vw,   _rc, _rs)
    # in-plane stress tensor τ'_ij = R_im R_jn τ_mn  (both indices in the u–w plane).
    # Tuple-assigned so the RHS sees the ORIGINAL uu/uw/ww; trace-preserving.
    _c2, _s2, _cs = _rc*_rc, _rs*_rs, _rc*_rs
    rey_uu, rey_uw, rey_ww = (
        _c2*rey_uu - 2.0*_cs*rey_uw + _s2*rey_ww,
        _cs*(rey_uu - rey_ww) + (_c2 - _s2)*rey_uw,
        _s2*rey_uu + 2.0*_cs*rey_uw + _c2*rey_ww,
    )
    # ══════════════════════════════════════════════════════════════════════════
    # STRESS-TENSOR NOMENCLATURE (triple decomposition, Raupach & Shaw 1982)
    #   u_i = ⟪u_i⟫(y)  +  ũ_i(x,y)  +  u''_i          (mean + dispersive + turbulent)
    #   Total  ⟨u_i u_j⟩ = ⟪u_i⟫⟪u_j⟫(+cross) + ũ_iũ_j + ⟨u''_i u''_j⟩
    #   Reynolds ⟨u'_i u'_j⟩ (deviation from the mean only) = ũ_iũ_j + ⟨u''_i u''_j⟩
    #                                                       = DISPERSIVE + TURBULENT
    # `rey_*` (loaded from uu_d.npy) is the TURBULENT part ⟨u''_i u''_j⟩ ONLY — it is
    # NOT the full Reynolds stress.  Build the dispersive stress ũ_iũ_j (ROTATED frame:
    # from the already-rotated dispersive velocities; the UU_disp..WW_disp loaded from
    # uu_t.npy above are UNROTATED, so overwrite them here) and the true Reynolds
    # stress reyn_* = turbulent + dispersive.
    UU_disp = DispVelU * DispVelU            # ũũ  (streamwise–streamwise)
    UV_disp = DispVelU * DispVelV            # ũṽ  (streamwise–wall-normal) → τ_yx
    UW_disp = DispVelU * DispVelW            # ũw̃  (streamwise–spanwise)
    VV_disp = DispVelV * DispVelV            # ṽṽ  (wall-normal–wall-normal)
    VW_disp = DispVelV * DispVelW            # ṽw̃  (wall-normal–spanwise)  → τ_yz
    WW_disp = DispVelW * DispVelW            # w̃w̃  (spanwise–spanwise)
    reyn_uu = rey_uu + UU_disp               # ⟨u'u'⟩ = turbulent + dispersive
    reyn_uv = rey_uv + UV_disp               # ⟨u'v'⟩
    reyn_uw = rey_uw + UW_disp               # ⟨u'w'⟩
    reyn_vv = rey_vv + VV_disp               # ⟨v'v'⟩
    reyn_vw = rey_vw + VW_disp               # ⟨v'w'⟩
    reyn_ww = rey_ww + WW_disp               # ⟨w'w'⟩
    # ══════════════════════════════════════════════════════════════════════════
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
    du_dy, du_dx, dv_dy, dv_dx, dw_dy, dw_dx = \
        load_or_compute(['du_dy_rot', 'du_dx_rot', 'dv_dy_rot', 'dv_dx_rot', 'dw_dy_rot', 'dw_dx_rot'],
                        recompute_derivatives,
                        lambda: compute_vel_derivs(cd, AvgPhU_j, AvgPhU_i, AvgPhV_j,
                                                   AvgPhV_i, AvgPhW_j, AvgPhW_i,
                                                   mask_intr, DY_METHOD),
                        label='velocity derivatives (rotated frame)')

    # ── Dispersive velocity gradients ─────────────────────────────────────────
    dud_dy, dvd_dy, dwd_dy, dud_dx, dvd_dx, dwd_dx = \
        load_or_compute(['dud_dy_rot', 'dvd_dy_rot', 'dwd_dy_rot', 'dud_dx_rot', 'dvd_dx_rot', 'dwd_dx_rot'],
                        recompute_derivatives,
                        lambda: compute_disp_derivs(cd, DispVelU, DispVelV, DispVelW,
                                                    mask_intr, DY_METHOD),
                        label='dispersive velocity derivatives (rotated frame)')

    # ── Second-order velocity derivatives and Reynolds/pressure gradients ─────
    # d2u_dy2 uses the D2Y_METHOD scheme (config.py; default 'compact').
    d2u_dx2, d2u_dy2, dreyuu_dx, dreyuv_dy, dP_dx, dP_dy = \
        load_or_compute(['d2u_dx2_rot', 'd2u_dy2_rot', 'dreyuu_dx_rot', 'dreyuv_dy_rot', 'dP_dx_rot', 'dP_dy_rot'],
                        recompute_derivatives,
                        lambda: compute_misc_derivs(cd, AvgPhU_i, AvgPhU_j, rey_uu, rey_uv,
                                                    AvgP_i, AvgP_j, mask_intr,
                                                    DY_METHOD, D2Y_METHOD),
                        label='second-order and stress/pressure derivatives')

    # Method 2 — friction velocity from the Ekman momentum-integral balance.
    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║  🔒 LOCKED — STANDARD SHEAR-STRESS BUDGET FORMULA.  DO NOT MODIFY.         ║
    # ║  Steady (∂/∂t=0), intrinsic (fluid-only) averaged, integrated Ekman        ║
    # ║  momentum balance (K&A 2024 eq. 4.2). f=1. The sign combination below is   ║
    # ║  the UNIQUE one that keeps Total = C+V+R height-constant (= surface stress);║
    # ║  verified on smooth+rough .nc (u*/veer match stored values) & fig. 4.      ║
    # ║    τ_zx(y) = C_zx + V_zx + R_zx                                             ║
    # ║       C_zx = ∫₀ʸ(G_z − ⟨w⟩)dy' = −I_corr_yx    (I_corr_yx = ∫(⟨w⟩−G_z))     ║
    # ║       V_zx = (1/Re_Λ) ∂⟨u⟩/∂y                                              ║
    # ║       R_zx = −⟨u'v'⟩ = −(turb_yx + disp_yx)                                ║
    # ║    τ_zy(y) = C_zy + V_zy + R_zy                                             ║
    # ║       C_zy = ∫₀ʸ(⟨u⟩ − G_x)dy' = +I_corr_yz    (I_corr_yz = ∫(⟨u⟩−G_x))     ║
    # ║       V_zy = (1/Re_Λ) ∂⟨w⟩/∂y                                              ║
    # ║       R_zy = −⟨w'v'⟩ = −(turb_yz + disp_yz)                                ║
    # ║  ⚠ Levi-Civita ε_{ik3}: the SPANWISE Coriolis is +I_corr_yz — the OPPOSITE ║
    # ║    sign of the streamwise −I_corr_yx.  (v = wall-normal = engineering idx 1;║
    # ║    ⟨u'v'⟩=rey_uv, ⟨w'v'⟩=rey_vw.)                                          ║
    # ║  u* = (T_zx_plateau² + T_zy_plateau²)^¼  (rotation-invariant = ‖τ_wall‖^½). ║
    # ║  DISPLAY ONLY: the τ_zy PANELS are negated for paper handedness via         ║
    # ║  config.fig4_paper_spanwise_sign (see the Fig-4 assembly block); the        ║
    # ║  physical total_tau_yz / u_star here are NOT negated.                       ║
    # ╚══════════════════════════════════════════════════════════════════════════╝
    corr_yx = (AvgPhW - Gz)*mask0
    I_corr_yx = vIntegral(np.mean(corr_yx, axis=1), ny, y)
    visc_yx = (1/Re_lambda) * (avg_c(eps, du_dy, axis=1))
    # Momentum flux = full Reynolds stress ⟨u'v'⟩ = TURBULENT ⟨u''v''⟩ + DISPERSIVE ũṽ.
    # Kept as two separate contributions (turb_yx + disp_yx) so each is plotted on its
    # own; their sum is the Reynolds flux that enters the balance (was turb_yx alone).
    turb_yx = (avg_c(eps, rey_uv, axis=1))              # turbulent ⟨u''v''⟩(z)
    disp_yx = (avg_c(eps, UV_disp, axis=1))             # dispersive ũṽ(z)
    rey_flux_yx = turb_yx + disp_yx                     # full Reynolds ⟨u'v'⟩(z)
    # Tau_zx(z) = - Coriolis + Viscous - Reynolds(turb + disp)   (Temporal ≈ 0, steady)
    total_tau_yx = - I_corr_yx + visc_yx - turb_yx - disp_yx

    # $f \int_0^z \epsilon_{2 1 3}\left(\langle\bar{u}\rangle_k-g_u\right) \mathrm{d} z + \frac{1}{\operatorname{Re} e_{\Lambda}} \frac{\partial\langle\bar{v}\rangle}{\partial z}-\left\langle\overline{v^{\prime} w^{\prime}}\right\rangle $
    corr_yz = (AvgPhU - Gx)*mask0
    I_corr_yz = vIntegral(np.mean(corr_yz, axis=1), ny, y) # Coriolis is positive
    visc_yz = (1/Re_lambda) * (avg_c(eps, dw_dy, axis=1))
    # Momentum flux = full Reynolds stress ⟨v'w'⟩ = TURBULENT ⟨v''w''⟩ + DISPERSIVE ṽw̃.
    turb_yz = avg_c(eps, rey_vw, axis=1)                # turbulent ⟨v''w''⟩(z)
    disp_yz = avg_c(eps, VW_disp, axis=1)               # dispersive ṽw̃(z)
    rey_flux_yz = turb_yz + disp_yz                     # full Reynolds ⟨v'w'⟩(z)
    # STANDARD shear-stress budget (see CLAUDE.md "Standard shear-stress budget
    # formulation"):  Ty = C_zy + V_zy + R_zy   with
    #   C_zy = +I_corr_yz = ∫(⟨u⟩−Gx),  V_zy = visc_yz,  R_zy = −(turb_yz + disp_yz).
    # Levi-Civita ε_{ik3}: the spanwise Coriolis is +I_corr_yz, OPPOSITE the
    # streamwise −I_corr_yx.  (Earlier code used −I_corr_yz — the old plot_fig4
    # sign, corrected 2026-07-11 after the fig-4 reproduction verified +I_corr_yz
    # is the sign that keeps Total height-constant and recovers the stored u*/veer.)
    total_tau_yz = I_corr_yz + visc_yz - turb_yz - disp_yz

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
    # u_star: this case's representative Method-2 friction velocity, used for inner
    # scaling throughout.  Use the constant-flux-layer PLATEAU of the u_star2(z)
    # profile (NOT the column mean): in a rotating layer the direct stress decays
    # monotonically with height, so the mean is biased low — the plateau is the
    # representative wall value (settled choice; cf. CLAUDE.md "Method 2").  This
    # overrides config.u_star (the prescribed 0.076 is the grid-generation value
    # only).  For a finite-Fr run this is genuinely different from the neutral u*.
    u_star = plateau_value(u_star2, y)

    # ── Friction velocity from the alternative (integrate→cavg) Coriolis term ──
    # Same momentum-balance formula, but using I_corr_*_c (per-column vertical
    # integral, THEN intrinsic fluid-only average) in place of I_corr_* (x-mean
    # THEN integral).  Compared in the Friction-Velocity comparison plot; inner
    # scaling (u_star) above is unchanged and still uses the original u_star2.
    # Full Reynolds flux (turb + disp) and the Fig-4 τ_zy sign, matching total_tau_* above.
    total_tau_yx_c = -I_corr_yx_c + visc_yx - turb_yx - disp_yx
    total_tau_yz_c = I_corr_yz_c + visc_yz - turb_yz - disp_yz
    u_star2_c = ((total_tau_yx_c**2 + total_tau_yz_c**2)**0.5)**0.5
    u_star_c  = np.mean(u_star2_c)

    y_inner =  y*(u_star/nu)
    y_outer = y/u_star

    # ── Refresh all inner/outer-unit scalings to this case's Method-2 u_star ──
    # x_in/y_in (and x_oro_in/y_oro_in, Re_tau, l_visc/wall_units, l_out) were built
    # at module load from config's PRESCRIBED u_star (0.076 — the grid-generation
    # value).  Re-derive them here from the per-case plateau u_star so every pickled
    # inner-unit array is consistent with y_inner/y_outer/u_plus (= y_in now equals
    # y_inner).  config.u_star itself is left untouched (still used by EditGrid).
    l_in       = nu / u_star
    l_visc     = l_in
    wall_units = l_in
    l_out      = u_star
    Re_tau     = (u_star**2) / nu
    x_in       = x / l_in
    y_in       = y / l_in
    x_oro_in   = x_oro / l_in
    y_oro_in   = y_oro / l_in

    # Turbulent Kinetic Energy
    TKE = 0.5*(rey_uu + rey_vv + rey_ww)

    dudt = np.mean(du_dt[:,:,0], axis=1)
    dwdt = np.mean(du_dt[:,:,2], axis=1)

    # Streamwise momentum budget — x-averaged 1D profiles
    # Equation: Temporal + MeanAdv + TurbAdv = Viscous + Coriolis
    mom_temporal  = dudt                                                         # ∂ū/∂t (≈ 0, steady)
    mom_mean_adv  = avg_c(eps, AvgPhU * du_dx + AvgPhV * du_dy, axis=1)         # ū ∂ū/∂x + v̄ ∂ū/∂y
    mom_turb_adv  = avg_c(eps, dreyuu_dx + dreyuv_dy, axis=1)                   # ∂(u''u'')/∂x + ∂(u''v'')/∂y (turbulent)
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
    # SIGN-FLIPPED FOR DISPLAY: w_plus_rot = -⟨W_rot⟩.  The PHYSICAL rotated
    # spanwise mean is NEGATIVE near the wall (Ekman veer left of G with f>0;
    # the smooth reference rW is negative too — same chirality, verified against
    # AvgPhW.npy + Re500 avg_all.nc).  The negation only puts the hodograph /
    # spanwise profile in the positive quadrant; every reference overlay applies
    # the same flip to the smooth/rough curves (-sw, -GblW_s, -arctan(alpha_s)).
    # NB: this flipped sign is what gets PICKLED (w_plus_rot, inst_alpha); the
    # pickled 2-D AvgPhW and the fluxes rey_vw / VW_disp keep the physical sign.
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

    # ── Stratification correction (Obukhov 1971), gated by config.Fr ──────────
    # Neutral runs (Fr = np.inf): φ ≡ 1 and the loop below is the classical OLS
    # of u⁺ vs ln(z⁺ − d⁺) — bit-for-bit unchanged.
    # Stratified runs (finite Fr): the mean-wind gradient obeys
    #     √φ(Ri)·κ·z·dū/dz = u★                          (Obukhov eqs 17 & 22)
    # with the energy-balance universal function
    #     φ(Ri) = (1 − Ri/Ri_cr)^(1/2)                   (Obukhov eq 38),
    # which integrates to u⁺ = (1/κ)·Ξ(z⁺) + B with the stability-modified abscissa
    #     Ξ(z⁺) = ∫ (1 − Ri/Ri_cr)^(−1/4) / (z⁺ − d⁺) dz⁺   ( → ln(z⁺−d⁺) as Ri→0 ).
    # Ri is the gradient Richardson number from the measured profiles,
    #     Ri = (∂⟨b⟩/∂z)/(∂⟨ū⟩/∂z)²   (AvgScal is the buoyancy b; ū = rotated mean).
    # Ri is capped just below Ri_cr (where φ→0, turbulence ceases); unstable Ri<0
    # is kept (φ>1 ⇒ enhanced mixing) so eq 38 covers both stable and unstable.
    _stratified = bool(np.isfinite(Fr))
    if _stratified:
        _b1d  = avg_c(eps, AvgScal, axis=1)              # ⟨b⟩(z), fluid-only
        _dbdz = np.gradient(_b1d, y)[_fit_mask]          # ∂⟨b⟩/∂z on the fit window
        _dudz = np.gradient(u_plus_rot, y)[_fit_mask]    # ∂⟨ū_rot⟩/∂z
        with np.errstate(divide='ignore', invalid='ignore'):
            _Ri_fit = _dbdz / _dudz**2
        _Ri_fit   = np.nan_to_num(_Ri_fit, nan=0.0, posinf=Ri_cr, neginf=0.0)
        _Ri_fit   = np.minimum(_Ri_fit, 0.999 * Ri_cr)   # cap at Ri_cr (φ → 0)
        _phi_corr = (1.0 - _Ri_fit / Ri_cr) ** (-0.25)   # φ^(−1/2)  (eqs 22 & 38)
    else:
        _Ri_fit   = np.zeros_like(_z_fit)
        _phi_corr = np.ones_like(_z_fit)

    # cumtrapz0 (cumulative trapezoid from 0) lives in functions.py — shared by
    # this Ξ-integral fit and the fig-4 budget computation (PLOT 32r).

    # Defaults (fallback if no valid κ found in constrained range)
    kappa_loglaw = loglaw_kappa_default
    d_m_loglaw   = loglaw_d_default
    z0m_loglaw   = loglaw_z0m_default
    _best_r2     = -np.inf

    if _u_fit.size >= 3:
        for _d in np.linspace(0.0, 0.9 * _fit_lo, 1001):
            _zs = _z_fit - _d
            if np.any(_zs <= 0):
                break
            # Stability-modified log abscissa Ξ; ≡ ln(z⁺−d⁺) in the neutral limit.
            if _stratified:
                _x = np.log(_zs[0]) + cumtrapz0(_phi_corr / _zs, _z_fit)
            else:
                _x = np.log(_zs)
            _slope, _intercept, _r, *_ = linregress(_x, _u_fit)
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

    _law = "Obukhov stratified" if _stratified else "neutral"
    print(f"Wall-law fit [{_law}] (z+ ∈ [{_fit_lo:.0f},{_fit_hi:.0f}]):  "
          f"κ_m={kappa_loglaw:.4f}  d_m+={d_m_loglaw:.2f}  "
          f"z_0m+={z0m_loglaw:.5f}  R²={_best_r2:.4f}")
    if _stratified:
        print(f"   stratified (Fr={Fr:.2e}, Ri_cr={Ri_cr:.3f}):  "
              f"⟨Ri⟩_fit={np.mean(_Ri_fit):+.4f}  Ri_max={np.max(_Ri_fit):+.4f}")

    # ── Obukhov (1971) MODIFIED log-law fit (paper-faithful, Sec. 6) ──────────
    # GATED on config.Fr: the modified (stability-corrected) law is fitted ONLY
    # for the stratified runs (finite Fr = 1, 0.1, 0.01, …).  For the NEUTRAL run
    # (Fr = np.inf) it is skipped entirely — only the original neutral log law
    # above runs — and v_star_mod/L1⁺/offset/R²/Ri_cr_implied stay NaN, so the
    # plot overlay, summary rows and pickle all fall back to "skipped".
    #   u⁺(z⁺) = (v*/k) psi(z⁺/L1⁺) + offset          (k = obu_kappa = 0.4)
    # Free parameters: v* (in u_star units — v*≈1 ⇔ profile-implied friction
    # velocity equals Method-2 u_star), L1⁺ (dynamic-turbulence scale, wall
    # units; +stable/−unstable), and an additive offset (roughness/intercept).
    # As stratification → 0, L1⁺ → ∞ and psi → ln(z⁺): the modified law would
    # collapse onto the neutral log law, so the two are directly comparable.
    # Fit window: fixed z⁺ ∈ [70, 150] (the log-law region for this flow), so the
    # modified-law parameters are fitted over a consistent, user-chosen band.
    _mod_lo, _mod_hi = modlaw_zmin, modlaw_zmax
    obu_fit = None
    v_star_mod = L1_plus_mod = offset_mod = r2_mod = float('nan')
    if _stratified:                                     # finite Fr only
        _mod_mask = (y_in >= _mod_lo) & (y_in <= _mod_hi) & np.isfinite(u_h_plus)
        if np.count_nonzero(_mod_mask) >= 4:
            # The surface buoyancy flux sign (B_s, computed below) is not yet
            # available here; seed the stable branch — curve_fit migrates to the
            # unstable branch on its own if the profile is better matched there.
            obu_fit = fit_modified_loglaw(y_in[_mod_mask], u_h_plus[_mod_mask])
            if obu_fit and obu_fit.get('ok'):
                v_star_mod  = obu_fit['v_star']
                L1_plus_mod = obu_fit['L1']
                offset_mod  = obu_fit['offset']
                r2_mod      = obu_fit['r2']
                print(f"Modified log-law (Obukhov 1971) fit [z⁺∈[{_mod_lo:.0f},{_mod_hi:.0f}]]:"
                      f"  v*/u★={v_star_mod:.4f}  L1⁺={L1_plus_mod:+.3e}  "
                      f"offset={offset_mod:.3f}  R²={r2_mod:.4f}")
            else:
                print(f"Modified log-law (Obukhov 1971) fit: FAILED "
                      f"({obu_fit.get('err','<4 pts') if obu_fit else 'no data'})")
        # Table-III self-test (unit-independent) — provenance for the modified law.
        validate_obukhov_tableIII(verbose=True)
    else:
        print("Modified log-law (Obukhov 1971): skipped — neutral run "
              "(Fr=∞); only the classical neutral log law is fitted.")

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

    print("\n══ Method 2  components at z[94] ══════════════════════════════")
    print(f"  Coriolis-yx  −I_corr_yx : {-I_corr_yx[jc]:+.6f}")
    print(f"  Viscous-yx    visc_yx   : {visc_yx[jc]:+.6f}")
    print(f"  Reynolds-yx  −rey_uv    : {-rey_uv_avg[jc]:+.6f}")
    print(f"  total_tau_yx            : {total_tau_yx[jc]:+.6f}")
    print("  ---")
    print(f"  Coriolis-yz  +I_corr_yz : {I_corr_yz[jc]:+.6f}")
    print(f"  Viscous-yz    visc_yz   : {visc_yz[jc]:+.6f}")
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

    print("\n══ Three-method comparison (ref = u_star = plateau of Method 2 profile) ══")
    ref = u_star
    for label, val in [("Method 2 plateau (reference)", u_star),
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
    d = most_d_factor*u_star
    y0 = most_y0_factor*u_star
    u_most = (1/kappa)*np.log(y_inner) + most_B
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

    # ══════════════════════════════════════════════════════════════════════════
    # ░░  RESEARCH DIAGNOSTICS  ░░   (8 prioritised goals — Research.md:536-550)
    # ──────────────────────────────────────────────────────────────────────────
    # Per-run, gracefully gated: every stratification term degrades to
    # neutral / N/A / Obukhov→∞ when the run carries no buoyancy or the reference
    # data is absent.  Cross-case aggregation (Fr on the stability axis,
    # dispersive-share vs Ri_B, Re=500 vs 750) lives in results.py.
    # Engineering arrays: u=streamwise, v=WALL-NORMAL, w=spanwise; the
    # meteorological "vertical" buoyancy flux ⟨w'θ'⟩ is the wall-normal ⟨v'θ'⟩.
    # ══════════════════════════════════════════════════════════════════════════
    _FLUX_EPS = 1e-12
    G_mag = float(G_inf)                       # pickled alias of the geostrophic magnitude G_inf (= √(Gx²+Gz²), rotated → = Gx); computed once above

    # ── Buoyancy field (scalar IS buoyancy b) and its double-average split ─────
    b_xmean  = avg_c(eps, AvgScal, axis=1)                       # ⟨b⟩(z), fluid-only
    # Intrinsic (fluid-only) dispersive buoyancy for the flux decomposition; mirrors
    # the avg_c-based `dispP` used in the form-drag budget (vs the np.mean-based
    # plotting field `DispScal` loaded above, which parallels `DispP`).
    dispScal = (AvgScal - b_xmean[:, None]) * mask0              # dispersive b̃(x,z)
    _finite_bx = b_xmean[np.isfinite(b_xmean)]
    _strat = bool(_finite_bx.size and np.ptp(_finite_bx) > 1e-9) # buoyancy actually varies?

    # ── Buoyancy cross-moments (Route C) from PhAvgAllPlanes.py, if present ────
    try:
        MeanUTheta = np.load('MeanUTheta.npy')
        MeanVTheta = np.load('MeanVTheta.npy')
        MeanWTheta = np.load('MeanWTheta.npy')
        MeanThTh   = np.load('MeanThTh.npy')
        _have_flux = True
    except (FileNotFoundError, OSError):
        MeanUTheta = MeanVTheta = MeanWTheta = MeanThTh = np.zeros_like(AvgScal)
        _have_flux = False
        print("[research] Mean*Theta.npy absent — turbulent buoyancy flux = 0 "
              "(run PhAvgAllPlanes.py with save_avg=1 to generate them).")

    # Frame-consistency of the HORIZONTAL flux components: MeanUTheta/MeanWTheta are
    # the raw products ⟨u·s⟩/⟨w·s⟩ loaded AFTER the frame rotation, so they are still
    # in the unrotated frame while AvgPhU/AvgPhW are already rotated. The products are
    # LINEAR in the velocity and s is a rotation-invariant scalar, so the horizontal
    # buoyancy-flux vector rotates EXACTLY like the velocity — apply the SAME
    # rotate_pair (_rc,_rs) used on AvgPhU/AvgPhW (LOCKED block above) so the turbulent
    # split below is differenced against consistently-rotated means. The wall-normal
    # ⟨v·s⟩ is the rotation axis → invariant, left untouched.
    if _have_flux:
        MeanUTheta, MeanWTheta = rotate_pair(MeanUTheta, MeanWTheta, _rc, _rs)

    # Buoyancy-flux VECTOR components ⟨u_i'θ'⟩ = dispersive (form-induced) ũ_i·b̃
    #   + temporal (Route C) ⟨ū_i·θ̄⟩_t − ū_i·θ̄.  Engineering: u=streamwise,
    #   v=WALL-NORMAL (meteorological vertical), w=spanwise. DispVelU/W are rotated,
    #   DispVelV is the wall-normal rotation axis.
    utheta_disp = DispVelU * dispScal                       # rotated streamwise
    vtheta_disp = DispVelV * dispScal                       # wall-normal (met. vertical)
    wtheta_disp = DispVelW * dispScal                       # rotated spanwise
    if _have_flux:
        utheta_temp = (MeanUTheta - AvgPhU * AvgScal) * mask0
        vtheta_temp = (MeanVTheta - AvgPhV * AvgScal) * mask0
        wtheta_temp = (MeanWTheta - AvgPhW * AvgScal) * mask0
        thetavar    = (MeanThTh  - AvgScal * AvgScal) * mask0   # scalar variance ⟨θ'θ'⟩ (temporal)
    else:                                                   # no cross-moments → temporal flux unknown (0)
        utheta_temp = np.zeros_like(AvgScal)
        vtheta_temp = np.zeros_like(AvgScal)
        wtheta_temp = np.zeros_like(AvgScal)
        thetavar    = np.zeros_like(AvgScal)

    # x-averaged (fluid-only) buoyancy-flux profiles ⟨u_i'b'⟩(z): disp + temporal + total.
    Uflux_disp = avg_c(eps, utheta_disp, axis=1)           # ⟨ũb̃⟩(z) streamwise
    Uflux_temp = avg_c(eps, utheta_temp, axis=1)           # ⟨u'b'⟩(z) temporal
    Uflux      = Uflux_disp + Uflux_temp                   # total streamwise buoyancy flux(z)
    Bflux_disp = avg_c(eps, vtheta_disp, axis=1)           # ⟨ṽb̃⟩(z)
    Bflux_temp = avg_c(eps, vtheta_temp, axis=1)           # ⟨v'θ'⟩(z) temporal (wall-normal)
    Bflux      = Bflux_disp + Bflux_temp                   # total wall-normal buoyancy flux(z)
    Wflux_disp = avg_c(eps, wtheta_disp, axis=1)           # ⟨w̃b̃⟩(z) spanwise
    Wflux_temp = avg_c(eps, wtheta_temp, axis=1)           # ⟨w'b'⟩(z) temporal
    Wflux      = Wflux_disp + Wflux_temp                   # total spanwise buoyancy flux(z)

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL 1 — Translate the control: B_0, Ri_B, Obukhov length, stability axis
    # ──────────────────────────────────────────────────────────────────────────
    _surf_j = np.minimum(eps_hgt, ny - 1)                   # local wall (first-fluid) row per column
    B_0 = float(np.mean(AvgScal[_surf_j, np.arange(nx)]))   # surface buoyancy (x-mean Dirichlet)

    delta_run = float(u_star / f)                           # this run's depth δ = u*/f
    if delta_neutral is not None:
        delta_neu_eff = float(delta_neutral)
    else:
        delta_neu_eff = delta_run
        print("[research] config.delta_neutral is None — using this run's δ = "
              f"{delta_run:.5f} for Ri_B (set it to the matched neutral depth).")
    Ri_B = float(B_0 * delta_neu_eff / G_mag**2)

    # Surface wall-normal buoyancy flux: first level fully fluid for all x (crest+1)
    _j_surf_ref = min(int(hill_hgt) + 1, ny - 1)
    B_s    = float(Bflux[_j_surf_ref])                      # x-mean surface buoyancy flux
    b_star = (-B_s / u_star) if abs(u_star) > _FLUX_EPS else float('nan')

    # obukhov_length / local_obukhov_length / stability_class / bl_scales → functions.py
    L_obukhov_col = obukhov_length(u_star, B_s, kappa, _FLUX_EPS)  # column (domain) Obukhov length
    L_col_plus    = L_obukhov_col * u_star / nu             # in wall units

    # ── Ri_cr implied by the modified-log-law fit (Obukhov eqs 23a/26/27) ─────
    # Obukhov's dynamic-turbulence scale is  L1 = α·Ri_cr·v*³ / (k·|g u|), with
    # β = 1/(α·Ri_cr) and the buoyancy-flux group (g u) → the DNS surface flux
    # B_s.  Inverting for Ri_cr, with α = K_T/K = 1/Pr_t and k = obu_kappa:
    #     Ri_cr = L1_phys · k · |B_s| · Pr_t / v*_phys³.
    # CAREFUL with the two rescalings — they use DIFFERENT friction velocities:
    #   L1_phys  = L1⁺ · l_in ,  and  l_in = ν/u★_config  (config.u_star sets the
    #              grid's viscous length, so z⁺ = y/l_in is scaled by u★_config,
    #              NOT by the measured Method-2 plateau).
    #   v*_phys  = v_star_mod · u_star ,  because the fit was performed on
    #              u_h_plus = u_plus_rot/u_star (the MEASURED plateau).
    # Writing L1_phys as L1⁺·ν/u_star would silently rescale Ri_cr by
    # u★_config/u★_measured (≈1.16 here).  Use l_in explicitly.
    # [FLAG] This uses B_s (= <w'b'>, dispersive + Route-C temporal); when
    # Mean*Theta.npy is absent B_s is dispersive-only, so Ri_cr_implied is a
    # lower bound.  α = 1/Pr_t (config.Pr_t) is a closure choice, not measured.
    Ri_cr_implied = float('nan')
    if (np.isfinite(L1_plus_mod) and np.isfinite(v_star_mod) and abs(v_star_mod) > _FLUX_EPS
            and np.isfinite(B_s) and abs(B_s) > _FLUX_EPS and u_star > 0):
        _L1_phys_ric = L1_plus_mod * l_in            # wall units → physical
        _v_phys_ric  = v_star_mod * u_star           # fit is in measured-u★ units
        Ri_cr_implied = (_L1_phys_ric * obu_kappa * abs(B_s) * Pr_t
                         / _v_phys_ric**3)
        _sign_ok = (L1_plus_mod > 0) == (B_s < 0)   # stable: L1>0 with downward flux
        print(f"Modified log-law → implied Ri_cr={Ri_cr_implied:+.4f} "
              f"(α=1/Pr_t={1.0/Pr_t:.3f}; config Ri_cr={Ri_cr:.3f}; "
              f"branch {'stable' if L1_plus_mod > 0 else 'unstable'}, "
              f"flux-sign {'consistent' if _sign_ok else 'INCONSISTENT'})")

    # Local (per-station) friction velocity from near-wall shear, and local L⁺
    _tau_loc   = nu * np.sqrt(du_dy[_surf_j, np.arange(nx)]**2 + dw_dy[_surf_j, np.arange(nx)]**2)
    u_star_loc = np.sqrt(np.abs(_tau_loc))                  # per-column u*(x)
    _vtheta_tot = vtheta_disp + vtheta_temp

    # Stations: windward = left flank, floor = valley bottom, lee = right flank
    _stn  = {nm: int(fr * nx) for nm, fr in station_fracs.items()}
    L_loc = {nm: (local_obukhov_length(i, eps_hgt, ny, u_star_loc, _vtheta_tot, kappa, _FLUX_EPS)
                  * u_star / nu) for nm, i in _stn.items()}   # L⁺ per station

    stab_class    = stability_class(Ri_B, Ri_B_bins)
    collapse_flag = bool(np.isfinite(L_col_plus) and abs(L_col_plus) < Lplus_collapse)

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL 3 — u*, δ, Ψ, H/δ, H⁺ per run (3 friction-velocity methods, joint)
    # ──────────────────────────────────────────────────────────────────────────
    H_phys = float(y[hill_hgt])                             # valley crest height (physical)
    scales   = {'M2': bl_scales(u_star,  f, L_x, H_phys, nu),
                'M1': bl_scales(u_star1, f, L_x, H_phys, nu),
                'M3': bl_scales(u_star3, f, L_x, H_phys, nu)}
    Psi      = scales['M2']['Psi']                          # headline (Method 2)
    H_delta  = scales['M2']['H_delta']
    H_plus_r = scales['M2']['H_plus']
    Lx_plus  = scales['M2']['Lx_plus']

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL 4 — Double-average + split fluxes (momentum & buoyancy); disp. share
    # ──────────────────────────────────────────────────────────────────────────
    rey_uv_x  = avg_c(eps, rey_uv,  axis=1)                 # turbulent ⟨u''v''⟩(z)  (v = wall-normal)
    UV_disp_x = avg_c(eps, UV_disp, axis=1)                 # dispersive ũṽ(z)
    disp_share_mom  = np.abs(UV_disp_x) / (np.abs(UV_disp_x) + np.abs(rey_uv_x) + _FLUX_EPS)
    disp_share_buoy = np.abs(Bflux_disp) / (np.abs(Bflux_disp) + np.abs(Bflux_temp) + _FLUX_EPS)

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL 5 — Local similarity φ_m, φ_h at windward / floor / lee vs MOST
    # ──────────────────────────────────────────────────────────────────────────
    db_dy = cd.ddy(AvgScal, method=DY_METHOD) * mask_intr   # ∂⟨b⟩/∂z (approximate at the interface)
    phi_m_st = {}; phi_h_st = {}; zeta_st = {}; phi_m_dep = {}; phi_h_dep = {}
    for nm, i in _stn.items():
        js = int(min(eps_hgt[i], ny - 1))
        zc = y[js:] - y[js]                                 # height above the local surface
        Lloc = local_obukhov_length(i, eps_hgt, ny, u_star_loc, _vtheta_tot, kappa, _FLUX_EPS)
        with np.errstate(divide='ignore', invalid='ignore'):
            phim = (kappa * zc / u_star) * du_dy[js:, i]
            zeta = (zc / Lloc) if np.isfinite(Lloc) else np.zeros_like(zc)
            if _strat and np.isfinite(b_star) and abs(b_star) > _FLUX_EPS:
                phih = (kappa * zc / b_star) * db_dy[js:, i]
            else:
                phih = np.full_like(zc, np.nan)
            phim_most = 1.0  + beta_m * zeta
            phih_most = Pr_t + beta_h * zeta
        phi_m_st[nm] = phim; phi_h_st[nm] = phih; zeta_st[nm] = zeta
        _mm = np.isfinite(phim) & np.isfinite(phim_most)
        _mh = np.isfinite(phih) & np.isfinite(phih_most)
        phi_m_dep[nm] = float(np.sqrt(np.mean((phim[_mm] - phim_most[_mm])**2))) if _mm.any() else float('nan')
        phi_h_dep[nm] = float(np.sqrt(np.mean((phih[_mh] - phih_most[_mh])**2))) if _mh.any() else float('nan')

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL 7 — Wave diagnostics: wall-normal wave fluxes + sponge reflection guard
    # ──────────────────────────────────────────────────────────────────────────
    Ly       = float(y[-1])
    sponge_j = int(min(np.searchsorted(y, sponge_frac * Ly), ny - 1))   # sponge bottom index
    bl_top_j = int(min(max(np.searchsorted(y, delta_run), 1), ny - 1))  # BL top ≈ δ
    wave_mom_flux  = UV_disp_x.copy()                       # standing-wave momentum flux(z)
    wave_buoy_flux = Bflux.copy()                           # wall-normal wave buoyancy flux(z)
    # Reflection guard: between BL top and sponge base the wave flux must not grow.
    _w = np.abs(wave_mom_flux)
    if (sponge_j > bl_top_j + 2) and np.isfinite(_w[bl_top_j]) and (_w[bl_top_j] > _FLUX_EPS):
        reflection_ok = bool(_w[sponge_j - 1] <= _w[bl_top_j])
    else:
        reflection_ok = True

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL 2 — Flat-wall stratified reference (hook; data-gated)
    # ──────────────────────────────────────────────────────────────────────────
    strat_ref_available = bool(len(stratified_ref_paths) > 0)
    if not strat_ref_available:
        print("[research] flat-wall stratified reference: data absent "
              "(all .nc are ri00.00 = neutral); neutral baseline overlaid only.")

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL 8 — Reynolds robustness: emit inner & outer measures (Re-agnostic)
    # ──────────────────────────────────────────────────────────────────────────
    re750_note = ("Re_D=750: data absent (neutral run 'running, no statistics')"
                  if int(Re) != 750 else "Re_D=750 active")

    # ──────────────────────────────────────────────────────────────────────────
    # GOAL 6 — Intermittency (Ansorge & Mellado 2016)
    # ──────────────────────────────────────────────────────────────────────────
    # γ(z) ≡ ⟨H(|ω'| − ω₀)⟩  (eq 4.1): horizontal average of the Heaviside
    # indicator on the fluctuation-vorticity magnitude.  Two choices are dictated
    # by the paper (§4.1), NOT a naive enstrophy cut:
    #   • threshold ω₀ = e_ω ≡ ω_rms(δ) — the rms fluctuation-vorticity at the BL
    #     edge (eq 4.2), a fixed PHYSICAL reference, NOT a fraction of the max
    #     (a max-fraction cut collapses to ~0 because the max is set by the near-
    #     wall shear and the spurious ∂u/∂y spike at the IBM valley interface);
    #   • the vorticity is taken from the high-pass (here Reynolds-fluctuation,
    #     the paper's documented "second filter") field, so mean shear / large-
    #     scale background do not mask the small-scale turbulent activity.
    # Primary source: planesK.* (or flow.*.1/2) IN THIS DIRECTORY — real
    # instantaneous frames, time-averaged here exactly as the paper defines γ,
    # with NO spatial averaging in any direction. Only when no such raw frames
    # are available locally does this fall back to a single VERTICAL (x, wall-
    # normal z) γ plane written by MyPyLib/Intermittency.py on the cluster —
    # either a `<prefix>_planesK_k####.npz` (--from-planesK, PREFERRED; carries
    # the instantaneous high-pass |ω'_z| too) or the older
    # `<prefix>_slice_z####.npz` (--slice z, fixed spanwise index) — deliberately
    # NOT the spanwise-mean `<prefix>_xy.npz` nor the HORIZONTAL
    # `<prefix>_planesJ_j####.npz`, which average/reduce away the real patchy
    # turbulent/quiescent structure this diagnostic exists to show.
    gamma_z = None;  gamma_field = None;  omega_rms_z = None
    e_omega = None;  omega0 = None;  omega_inst_raw = None;  omega_inst_hp = None
    if (1 == compute_intermittency):
        import glob as _glob
        import ast as _ast

        # interior-fluid mask: drop solid AND the fluid ring touching solid
        _mb  = mask0.astype(bool)
        _mom = _mb & np.roll(_mb, 1, 1) & np.roll(_mb, -1, 1)   # x-neighbours (periodic)
        _mom[1:, :] &= _mb[:-1, :]; _mom[:-1, :] &= _mb[1:, :]  # y-neighbours
        _mom = _mom.astype(float)

        _NK, _NV, _KP = planesK_n_kplanes, planesK_nvars, planesK_kplane_idx  # planesK layout
        # Frame sources: planesK.* if present, otherwise z-plane 1 of the 3-D
        # velocity component files flow.*.1 (u) / flow.*.2 (v) — fall back to the
        # field files when planesK is unavailable (each snapshot tag = one frame).
        _pk = sorted(_glob.glob(cwd + 'planesK.*'))
        if _pk:
            _frames = [('pk', _p) for _p in _pk];  _src_kind = 'planesK.*'
        else:
            _frames = []
            for _uf in sorted(_glob.glob(cwd + 'flow.*.1')):
                _vf = _uf[:-1] + '2'                       # flow.<tag>.1 → .2
                if os.path.exists(_vf):
                    _frames.append(('flow', _uf, _vf))
            _src_kind = 'flow.*.1/2 (z-plane 1)'

        if _frames:
            def _load_uv(_src):
                """(u, v) as (ny,nx): a planesK frame, or z-plane 1 of flow.*."""
                if _src[0] == 'pk':
                    _pl = read_all_planes(_src[1], nx, ny, _NK, _NV, _KP)
                    return _pl[0], _pl[1]                  # tlab idx0=u, idx1=v
                _hu = read_header(_src[1])[0]; _hv = read_header(_src[2])[0]
                return (readplane(_src[1], nx, ny, 1, _hu),
                        readplane(_src[2], nx, ny, 1, _hv))

            def _read_omega(_src):
                """Return (raw ω_z, high-pass ω'_z) for one frame; each (ny,nx)."""
                _u, _v = _load_uv(_src)                    # idx0=u, idx1=v (wall-normal)
                _raw = cd.ddx(_v) - cd.ddy(_u, method=DY_METHOD)
                # Plain row mean (NOT the fluid-only/intrinsic avg_c-style average):
                # at rows where the fluid width collapses (e.g. near the terrain),
                # dividing by a near-zero fluid-cell count made the row mean noisy/
                # unstable, which showed up as a spurious horizontal band in the
                # high-pass field once ddy differenced across that row.
                _up  = _u - np.mean(_u, axis=1, keepdims=True)   # horizontal-mean fluct.
                _vp  = _v - np.mean(_v, axis=1, keepdims=True)
                _hp  = cd.ddx(_vp) - cd.ddy(_up, method=DY_METHOD)
                return _raw, _hp

            # pass 1 — ω_rms(z) and the threshold ω₀ = e_ω = ω_rms(δ) (eq 4.2)
            _sumsq = np.zeros((ny, nx)); _ng = 0
            for _src in _frames:
                try:
                    _raw, _hp = _read_omega(_src)
                except (ValueError, OSError):
                    continue
                if (_ng == 0):
                    omega_inst_raw = _raw * _mom      # first frame → Fig-2-style field
                    omega_inst_hp  = _hp  * _mom
                _sumsq += _hp**2; _ng += 1
            if (_ng > 0):
                _num = np.sum(_sumsq * _mom, axis=1)
                _den = _ng * np.sum(_mom, axis=1)
                omega_rms_z = np.sqrt(np.divide(_num, _den, out=np.zeros_like(_num),
                                                where=_den > 0))
                e_omega = float(omega_rms_z[bl_top_j])            # rms vorticity at δ
                omega0  = omega_thresh_factor * e_omega           # ω₀ = e_ω
                # pass 2 — γ(x,z) = time-avg H(|ω'| − ω₀);  γ(z) = horizontal avg
                _acc = np.zeros((ny, nx)); _n2 = 0
                for _src in _frames:
                    try:
                        _raw, _hp = _read_omega(_src)
                    except (ValueError, OSError):
                        continue
                    _acc += (np.abs(_hp) > omega0).astype(float); _n2 += 1
                gamma_field = (_acc / _n2) * _mom
                _gden = np.sum(_mom, axis=1)
                gamma_z = np.divide(np.sum(gamma_field, axis=1), _gden,
                                    out=np.zeros_like(_gden), where=_gden > 0)
                print(f"[research] intermittency from {_n2} {_src_kind} frame(s); "
                      f"ω₀ = {omega_thresh_factor:g}·ω_rms(δ) = {omega0:.4g}, "
                      f"max γ = {float(np.nanmax(gamma_z)):.2f}.")
        else:
            # No raw instantaneous data locally — fall back to a single VERTICAL
            # (x, wall-normal z) γ plane written by Intermittency.py on the
            # cluster.  Two producers, both shape (ny, nx):
            #   • `<prefix>_planesK_k####.npz`  (--from-planesK: ω_z' on a fixed
            #     spanwise k-slot) — the purpose-built γ(x,z) plane, PREFERRED,
            #     and it carries the instantaneous high-pass |ω'_z| ('omega_hp')
            #     for the Fig-2 panel below;
            #   • `<prefix>_slice_z####.npz`    (--slice z: a fixed spanwise index
            #     of the full 3-D γ) — the older single-index slice.
            # Deliberately NOT the spanwise-mean *_xy.npz nor the HORIZONTAL
            # *_planesJ_j####.npz ((nz,nx) x–z_span plane, wrong orientation for
            # γ(z)): averaging/reducing over the spanwise extent is exactly the
            # destructive reduction this diagnostic exists to avoid. Multiple
            # matching planes (different spanwise locations) are NOT averaged
            # together either, for the same reason — the first found is used as-is.
            _kplanes = (sorted(_glob.glob(cwd + '*_planesK_k*.npz')) or
                        sorted(_glob.glob(cwd + '*_slice_z*.npz')))
            if not _kplanes:
                print("[research] intermittency: no planesK.*, flow.*.[12], or "
                      "cluster γ plane (*_planesK_k*.npz / *_slice_z*.npz) — γ skipped.")
            else:
                _npz_path = _kplanes[0]
                _d = np.load(_npz_path, allow_pickle=True)
                _gshape = tuple(_d['gamma'].shape) if 'gamma' in _d.files else None
                if _gshape != (ny, nx):
                    print(f"[research] intermittency: {os.path.basename(_npz_path)} "
                          f"gamma {_gshape or 'absent'} ≠ grid ({ny},{nx}) — γ skipped "
                          "(a *_planesJ_j*.npz horizontal plane will trip this).")
                else:
                    try:
                        _meta = _ast.literal_eval(str(_d['meta']))
                    except (ValueError, SyntaxError):
                        _meta = {}
                    gamma_field = np.nan_to_num(_d['gamma'], nan=0.0) * _mom
                    _gden = np.sum(_mom, axis=1)
                    gamma_z = np.divide(np.sum(gamma_field, axis=1), _gden,
                                        out=np.zeros_like(_gden), where=_gden > 0)
                    # instantaneous high-pass |ω'_z| for the Fig-2 panel, if the
                    # plane carried it (planesK writes 'omega_hp'); no RAW field
                    # is stored, so panel (c) then shows the high-pass alone.
                    if ('omega_hp' in _d.files and
                            tuple(_d['omega_hp'].shape) == (ny, nx)):
                        omega_inst_hp = np.nan_to_num(_d['omega_hp'], nan=0.0) * _mom
                    _o = _meta.get('omega0')
                    omega0 = float(_o) if _o is not None else float('nan')
                    _n2 = _meta.get('n_used', _meta.get('n_snapshots', '?'))
                    _src = _meta.get('source', 'slice-z')
                    if len(_kplanes) > 1:
                        print(f"[research] {len(_kplanes)} cluster γ plane(s) found; "
                              f"using {os.path.basename(_npz_path)} only (one "
                              f"spanwise location — not averaged across planes).")
                    print(f"[research] intermittency from {os.path.basename(_npz_path)} "
                          f"(Intermittency.py {_src} plane, cluster, {_n2} snapshot(s) "
                          f"time-averaged); ω₀ = {omega0:.4g}, "
                          f"max γ = {float(np.nanmax(gamma_z)):.2f}.")
    # ══════════════════════════════════════════════════════════════════════════

    # ── Coupling: global surface veer (computed HERE, before the pickle) ─────
    # Method-2 total stress is height-independent (= surface stress), so the veer
    # angle of the total-stress vector is the global surface veer (Research.md
    # candidate finding #3 / §6.14.5).  γ_veer = α_oro/α_smooth is formed in
    # results.py, where the smooth reference is available — the smooth .nc is
    # loaded in PhAvg_rotated.py only AFTER this pickle, so it cannot be used here.
    veer_oro = float(np.abs(np.degrees(np.arctan2(total_tau_yz[-1], total_tau_yx[-1]))))

    # ── Local vertical-wavenumber field m(x,z) — gravity-wave propagation ──────
    # Goal: a full-field map of the wave's vertical wavenumber, so the direction
    # of vertical energy propagation can be read everywhere without hand-picking a
    # velocity value at the wave.  Method: signed Hilbert phase-gradient of the
    # DISPERSIVE velocity.  The dispersive field ṽ = ⟨v⟩(x,z) − ⟨v⟩(z) already
    # removes the horizontal-mean background profile, isolating the topography-
    # locked wave perturbation; the phase-averaged field still carries the mean
    # shear ū(z) whose wall-normal gradient is NOT wave phase and would swamp
    # ∂(phase)/∂z.  Taking the analytic signal along the periodic x fixes the
    # dominant horizontal mode with k>0, so sign(m) encodes the phase-line tilt:
    # for the mountain-wave branch k·m<0 ⇒ upward energy propagation, k·m>0 ⇒
    # downward.  Computed from BOTH DispVelV (vertical; cleanest — mean vertical
    # velocity ≈ 0) and DispVelU (streamwise; carries more mean-shear residue),
    # per the user's request to compare the two.  local_wavenumbers (numpy-only
    # Hilbert phase-gradient) lives in functions.py.
    k_dispV, m_dispV = local_wavenumbers(DispVelV, x, y)    # vertical (meteo w)
    k_dispU, m_dispU = local_wavenumbers(DispVelU, x, y)    # streamwise
    km_dispV = k_dispV * m_dispV                            # <0 ⇒ upward energy prop.

    # Trustworthy window: above the valley crest, below the Rayleigh sponge.
    _wj0, _wj1 = int(hill_hgt), int(sponge_j)
    if _wj1 > _wj0:
        _wm = (mask0[_wj0:_wj1, :] > 0.5)
        _frac_up = (float(np.mean(km_dispV[_wj0:_wj1, :][_wm] < 0))
                    if _wm.any() else float('nan'))
    else:
        _frac_up = float('nan')
    print('[research] vertical-wavenumber field m(x,z) from DispVelV & DispVelU '
          '(signed Hilbert phase-gradient).')
    print('[research]   crest→sponge window z+ = %.1f → %.1f; fraction with k·m<0 '
          '(upward energy propagation) = %.2f' % (y_in[_wj0], y_in[_wj1], _frac_up))
    if not np.isfinite(Fr):
        print('[research]   NOTE: Fr=inf (neutral) — no stratification, so m(x,z) '
              'reflects topographic forcing, NOT a propagating internal gravity '
              'wave. The field is physically a wave wavenumber only for finite Fr.')

    # ──────────────────────────────────────────────────────────────────────────
    # Instantaneous fluctuation planes  inst_u/v/w/scal  (raw-record read → pickle)
    # ──────────────────────────────────────────────────────────────────────────
    # The instantaneous-snapshot figures (results.py P19–P22) need one x–y plane of
    # the turbulent fluctuation u'ᵢ = uᵢ − ⟨uᵢ⟩ₓ.  That is the ONLY quantity derived
    # from a RAW record (flow.<tag>.{1,2,3} / scal.<tag>.1); reading raw records is
    # a stage-a/b job, never stage c.  So it is computed HERE (stage b, which already
    # reads raw planes for the phase-average + intermittency) and pickled, so
    # results.py merely READS inst_* — no flow.*/scal.* touched downstream.
    # Fluctuation = z-plane 1 minus its x-mean, solid zeroed by mask0, stored float32
    # (identical to the former results.py _inst_fluct).  Component→file map matches
    # results.py: inst_u=flow.*.1, inst_v=flow.*.2 (wall-normal), inst_w=flow.*.3
    # (spanwise), inst_scal=scal.*.1.  A truncated/absent download is skipped, and
    # the corresponding key is simply not pickled (results.py then skips that panel).
    import glob as _iglob
    inst_u = inst_v = inst_w = inst_scal = None
    for _icomp, _iname in (('1', 'inst_u'), ('2', 'inst_v'),
                           ('3', 'inst_w'), ('scal', 'inst_scal')):
        _ipat  = (cwd + 'scal.*.1') if _icomp == 'scal' else (cwd + 'flow.*.' + _icomp)
        _ihits = sorted(_iglob.glob(_ipat))
        if not _ihits:
            continue
        _ipath = _ihits[-1]
        try:
            _ihdr, _inx, _iny, *_ = read_header(_ipath)
        except Exception as _ie:
            print('[inst] header read failed for %s: %s' % (os.path.basename(_ipath), _ie))
            continue
        if _ihdr is None or _inx is None or _iny is None:
            continue
        # Only the header + first x–y plane are guaranteed present (downloads are
        # deliberately truncated to ~header+one plane); confirm before reading.
        if os.path.getsize(_ipath) < int(_ihdr) + int(_inx) * int(_iny) * 8:
            print('[inst] %s truncated below one full %dx%d plane — skipped.'
                  % (os.path.basename(_ipath), _inx, _iny))
            continue
        try:
            _ipl = readplane(_ipath, _inx, _iny, 1, _ihdr)
        except Exception as _ie:
            print('[inst] plane read failed for %s: %s' % (os.path.basename(_ipath), _ie))
            continue
        _ifl = (_ipl - _ipl.mean(axis=1, keepdims=True))
        if _ifl.shape == mask0.shape:
            _ifl = _ifl * mask0
        globals()[_iname] = _ifl.astype(np.float32)
        print('[inst] %s ← %s (z-plane 1 fluctuation, pickled)'
              % (_iname, os.path.basename(_ipath)))

    # Bundle every post-processed field listed in IO.var_names into sim1_results.pkl
    # (consumed cross-case by results.py).  IO.write_results_pickle skips names not
    # yet in globals() — a gated diagnostic or a cluster-absent reference — so one
    # missing key cannot drop the whole pickle; results.py treats absent keys as
    # None/NaN.
    IO.write_results_pickle(globals())
        
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
    ustr_M2_plateau_o  = plateau_value(u_star2, y_inner)  # orographic plateau (u_star2 = Method-2 profile)
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
    # NOTE: veer_oro (the pickled global surface veer) is computed earlier, before
    # the pickle dump, from total_tau_*[-1] (== _fp_veer_Ho here).  γ_veer vs the
    # smooth case is formed in results.py from the smooth reference loaded there.

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
        ('u* - M2 (plateau momentum balance)',   u_star,                    '.5f'),
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
        ('Mod. log-law v*/u* (Obukhov 71)',      v_star_mod,                '.4f'),
        ('Mod. log-law L1+ (wall units)',        L1_plus_mod,               '.3e'),
        ('Mod. log-law offset',                  offset_mod,                '.3f'),
        ('Mod. log-law fit R^2',                 r2_mod,                    '.4f'),
        ('Mod. log-law implied Ri_cr',           Ri_cr_implied,             '.4f'),
    ])
    # ──────────────────────────────────────────────────────────────────────────

    # ══════════════════════════════════════════════════════════════════════════
    # ═══  RESEARCH DIAGNOSTICS SUMMARY  → IO.print_research_summary  ════════════
    # (8 goals — Research.md:536-550; the reporting table + its _Lfmt/_trio format
    #  helpers moved to IO.py.  Reads every quantity from globals().)
    IO.print_research_summary(globals())
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
    # ── UNIT CONVENTION (settled with the user) ──────────────────────────────────
    # Every plot in this file is non-dimensional; nothing carries physical units.
    #   • CONTOUR / 2-D maps  → INNER units  (x_in = x/l_in, y_in = y/l_in; z+ axes).
    #   • LINE profiles        → paired: a ZOOMED near-wall view in INNER units and a
    #     ZOOMED-OUT full-depth view in OUTER units.
    # OUTER scaling (Ekman, f = 1 ⇒ δ = u*/f = u_star, δ⁺ = Re_tau = u*²/ν):
    #   wall-normal  z/δ  = y_in / Re_tau           (valley)
    #   velocity     /G   = / G_mag                 (geostrophic magnitude ≈ 1)
    #   stress       /G²  = / G_mag**2
    #   advection    ·δ/G² = · u_star / G_mag**2    (u*³/ν → inner; G²/δ → outer)
    # Reference cases carry their own u* (ustr_s1, ustr_r1) and, with f = 1, their own
    # δ⁺ = u*²/ν, so their outer wall-normal coordinate is y_*_p / Re_tau_*.
    z_out    = y_in / Re_tau                     # z/δ  — outer wall-normal (valley)
    Re_tau_s = ustr_s1**2 / nu                   # smooth reference δ⁺ (= δ_s marker used below)
    z_out_s  = y_s_p / Re_tau_s
    Re_tau_r = ustr_r1**2 / nu_rough             # rough r1 reference δ⁺
    z_out_r  = y_r_p / Re_tau_r
    adv_in   = u_star**3 / nu                    # inner advection scale (u*³/ν)
    adv_out  = G_mag**2 / u_star                 # outer advection scale (G²/δ, δ = u_star)

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
    # [PLOT 06] Phase-averaged mean velocity field (in-plane streamlines +
    #           spanwise-velocity contour; yaw angle; 3-D speed)
    plot_phavg_velocity_3D(x_in, y_in[:limity],
                           AvgPhU[:limity,:], AvgPhV[:limity,:], AvgPhW[:limity,:],
                           eps[:limity,:], 1000,
                           x_oro_in, y_oro_in,
                           cwd + '/fig/' + 'PhAvg_3D_velocity.png',
                           title=r'Phase-averaged mean velocity field  '
                                 r'$\langle\overline{u}_i\rangle(x^+,z^+)$')

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
    # [PLOT 10b] DispScal (dispersive potential temperature / buoyancy)
    plot2D_div(x_in, y_in[:limity], DispScal[:limity,:],'',r'$\widetilde{\theta}(x,z) = \langle\overline{\theta}\rangle(x,z) - \langle\overline{\theta}\rangle(z)$',r'$x^+$',r'$z^+$', cwd + '/fig/' + 'DispScal' + '.png', x_oro_in, y_oro_in, 1000)
    # [PLOT 11] Dispersive velocity field (in-plane dispersive streamlines +
    #           spanwise dispersive-velocity contour; yaw angle; 3-D speed)
    plot_phavg_velocity_3D(x_in, y_in[:limity],
                           DispVelU[:limity,:], DispVelV[:limity,:], DispVelW[:limity,:],
                           eps[:limity,:], 1000,
                           x_oro_in, y_oro_in,
                           cwd + '/fig/' + 'Disp_3D_velocity.png',
                           title=r'Dispersive velocity field  '
                                 r'$\widetilde{u}_i(x^+,z^+)$')
    
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
    # Stress-tensor 2D maps — the FOUR families, 6 independent components each
    # (symmetric tensor).  [PLOTS 20-25 were the single mislabeled "Reynolds" set.]
    #   Total      ⟨u_i u_j⟩       (raw second moment / momentum flux)
    #   Reynolds   ⟨u'_i u'_j⟩     = dispersive + turbulent  (deviation from the mean)
    #   Turbulent  ⟨u''_i u''_j⟩   = rey_*  (deviation from the phase/coherent average)
    #   Dispersive ũ_i ũ_j        = *_disp (coherent/form-induced)
    # Total is the exact identity ⟨u_i u_j⟩ = ⟨u_i⟩⟨u_j⟩ + ⟨u''_i u''_j⟩, reconstructed
    # from the rotated phase-mean product and the turbulent stress.  Meteorological
    # display labels swap v↔w (u=streamwise, w=wall-normal, v=spanwise), matching the
    # rest of this file.  Output: fig/<Family>_R<meteo>.png  (24 PNGs).
    _mean_prod = {'uu': AvgPhU*AvgPhU, 'uv': AvgPhU*AvgPhV, 'uw': AvgPhU*AvgPhW,
                  'vv': AvgPhV*AvgPhV, 'vw': AvgPhV*AvgPhW, 'ww': AvgPhW*AvgPhW}
    _turb_f = {'uu': rey_uu, 'uv': rey_uv, 'uw': rey_uw,
               'vv': rey_vv, 'vw': rey_vw, 'ww': rey_ww}
    _disp_f = {'uu': UU_disp, 'uv': UV_disp, 'uw': UW_disp,
               'vv': VV_disp, 'vw': VW_disp, 'ww': WW_disp}
    _reyn_f = {'uu': reyn_uu, 'uv': reyn_uv, 'uw': reyn_uw,
               'vv': reyn_vv, 'vw': reyn_vw, 'ww': reyn_ww}
    _tot_f  = {_k: _mean_prod[_k] + _turb_f[_k] for _k in _turb_f}
    _meteo  = {'uu': 'uu', 'uv': 'uw', 'uw': 'uv', 'vv': 'ww', 'vw': 'wv', 'ww': 'vv'}
    _stress_families = [
        ('Total',      _tot_f,  r'Total momentum $\langle %s%s\rangle$'),
        ('Reynolds',   _reyn_f, r"Reynolds stress $\langle %s'%s'\rangle$"),
        ('Turbulent',  _turb_f, r"Turbulent stress $\langle %s''%s''\rangle$"),
        ('Dispersive', _disp_f, r'Dispersive stress $\widetilde{%s}\widetilde{%s}$'),
    ]
    for _fname, _fld, _tfmt in _stress_families:
        for _ek in ('uu', 'uv', 'uw', 'vv', 'vw', 'ww'):
            _ml = _meteo[_ek]
            _title = _tfmt % (_ml[0], _ml[1])
            _fn = '%s_R%s.png' % (_fname, _ml)
            plot2D_div(x_in, y_in[:limity], _fld[_ek][:limity, :], '', _title,
                       r'$x^+$', r'$z^+$', cwd + '/fig/' + _fn, x_oro_in, y_oro_in, 1000)
    
    # %%###########################################################################
    # Vorticity
    # plot2D_div(x, y[:limity], omega_x[:limity,:], '', 'Vorticity X', r'$x$',r'$z$', cwd + '/fig/' + 'VorticityX' + '.png', x_oro, y_oro, 50)
    # plot2D_div(x, y[:300], omega_y[:300,:], '', 'Vorticity Z', r'$x$',r'$z$', cwd + '/fig/' + 'VorticityZ' + '.png', x_oro, y_oro, 50)
    # plot2D_div(x, y[:200], omega_z[:200,:], '', 'Vorticity Y', r'$x$',r'$z$', cwd + '/fig/' + 'VorticityY' + '.png', x_oro, y_oro, 50)
    # plot2D_streamlines_vorticityX(x, y[:limity], AvgPhU[:limity,:], AvgPhV[:limity,:],omega_y[:limity,:],'','',r'$x$',r'$z$', cwd + '/fig/' + 'Streamlinezx' + '.png', x_oro, y_oro,1000)
    
    # %%###########################################################################
    # Vorticity contour map
    # [PLOT 26] Dispersion velocity vorticity in XZ plane — INNER units:
    # axes x/l_in, z/l_in (z+); vorticity scaled by the viscous time ν/u*²
    # (same inner scaling as the near-wall vorticity map below).
    plt.figure(figsize=(8,6))
    plt.contourf(x_in, y_in[:limity], disp_vortz[:limity,:]*nu/u_star**2,
                 levels=50, cmap='RdBu_r')
    plt.fill(x_oro_in, y_oro_in, facecolor='black')          # IBM solid
    plt.colorbar(label=r'$\widetilde{\omega}_z\,\nu/u_*^2$')
    plt.xlabel(r'$x^+$')
    plt.ylabel(r'$z^+$')
    plt.title('Dispersion velocity vorticity in XZ plane')
    # plt.savefig(savename, dpi=300)
    plt.show()

    # %%###########################################################################
    # Vertical-wavenumber field m(x,z) — gravity-wave propagation direction
    # Signed local vertical wavenumber from the dispersive velocity (see compute
    # block above the pickle dump).  Plotted in inner units (m·l_in) on the crest→
    # sponge window; sign(m) (with the Hilbert-fixed k>0) gives the phase-line tilt.
    # [PLOT 26b] Wavenumber_m_DispV / _DispU / _compare
    # plot_wavenumber_field lives in PlotField.py (per-run plotting moved out).
    _wave_note = ('' if np.isfinite(Fr)
                  else '  [Fr=inf: topographic forcing, not a propagating GW]')
    _wave_args = (x_in, y_in, x_oro_in, y_oro_in, mask0, l_in, sponge_j, hill_hgt, fig_dir)
    plot_wavenumber_field(m_dispV, 'Wavenumber_m_DispV.png',
                          r'Vertical wavenumber $m(x,z)$ from $\widetilde{W}_y$' + _wave_note,
                          *_wave_args)
    plot_wavenumber_field(m_dispU, 'Wavenumber_m_DispU.png',
                          r'Vertical wavenumber $m(x,z)$ from $\widetilde{U}_y$' + _wave_note,
                          *_wave_args)

    # [PLOT 26c] explicit up/down energy-propagation map  sign(k·m) from DispVelV
    _limc = int(min(sponge_j, ny - 1))
    _kmsign = np.sign(km_dispV[:_limc, :]) * mask0[:_limc, :]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    _cf = ax.contourf(x_in, y_in[:_limc], _kmsign,
                      levels=[-1.5, -0.5, 0.5, 1.5], cmap='coolwarm')
    ax.fill(x_oro_in, y_oro_in, facecolor='black')
    ax.axhline(y_in[int(hill_hgt)], color='g', ls='--', lw=0.8, label='crest $h$')
    ax.axhline(y_in[int(sponge_j)], color='m', ls=':',  lw=1.0, label='sponge')
    _cb = plt.colorbar(_cf, ax=ax, ticks=[-1, 0, 1])
    _cb.ax.set_yticklabels(['up (k·m<0)', '', 'down (k·m>0)'])
    ax.set_xlabel(r'$x^+$'); ax.set_ylabel(r'$z^+$')
    ax.set_title(r'Energy-propagation direction  $\mathrm{sign}(k\cdot m)$ from $\widetilde{W}_y$' + _wave_note)
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Wavenumber_m_compare.png'), dpi=300)
    plt.show()


    # %%###########################################################################
    # Hodograph
    # [PLOT 27] Hodograph
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(u_plus_rot, w_plus_rot, label='valley', color='blue', linestyle='-')
    ref_plot(plot_ref_smooth, su, -sw, label='smooth', color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    # ref_plot(plot_ref_rough, su_r, -sw_r, label='rough r1', color=ROUGH_COLOR, linestyle=ROUGH_LS)
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
    # inst_alpha = w_plus_rot/u_plus_rot is a TANGENT ratio, not an angle: scaling
    # it by 180/π mis-treats a tangent as radians AND diverges to ±thousands of
    # degrees wherever u_plus_rot crosses zero (the strongly-stratified Fr=0.0015
    # run reverses near the surface).  Plot the BOUNDED veer angle arctan2(w,u) in
    # degrees (∈[-180,180]) — consistent with the arctan2 veer values above (7a)
    # and results.py's _veer_deg.  References use arctan (bounded ±90) of their
    # tangent ratios rather than the small-angle *(180/π) approximation.
    alpha_deg = np.degrees(np.arctan2(w_plus_rot, u_plus_rot))   # bounded veer (deg)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(y_inner[1:], alpha_deg[1:], label='valley', color='blue', linestyle='-')
    ref_plot(plot_ref_smooth, y_s_p[1:], -np.degrees(np.arctan(alpha_s[1:])), label='smooth', color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    ref_plot(plot_ref_rough, y_r_p[1:], -np.degrees(np.arctan(alpha_r[1:])), label='rough r1', color=ROUGH_COLOR, linestyle=ROUGH_LS)
    mark_layers(y_inner, alpha_deg, _LYR_ORO, filled=True)
    ref_mark(plot_ref_smooth, mark_layers, y_s_p, -np.degrees(np.arctan(alpha_s)), _LYR_SMO, filled=False)
    mark_h(y_in[h_idx], 'v')
    plt.title('Rotation angle')
    plt.ylabel(r'$\alpha (\degree)$')
    plt.xlabel(r'$z^{+}$')
    plt.xscale("log")
    plt.legend()
    add_marker_legend()
    plt.grid(True)
    plt.show()

    # [PLOT 28b] Rotation angle — ZOOMED-OUT (full depth) in OUTER units (z/δ, linear).
    # Outer-unit counterpart of [PLOT 28]; collapses the Ekman veer over the whole layer.
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(z_out[1:], alpha_deg[1:], label='valley', color='blue', linestyle='-')
    ref_plot(plot_ref_smooth, z_out_s[1:], -np.degrees(np.arctan(alpha_s[1:])), label='smooth', color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    ref_plot(plot_ref_rough, z_out_r[1:], -np.degrees(np.arctan(alpha_r[1:])), label='rough r1', color=ROUGH_COLOR, linestyle=ROUGH_LS)
    mark_h(z_out[h_idx], 'v')
    plt.title('Rotation angle (outer units)')
    plt.ylabel(r'$\alpha (\degree)$')
    plt.xlabel(r'$z/\delta$')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Rotation_angle_outer.png'), dpi=300)
    plt.show()

    # [PLOT 28c] Rotation angle — 2-D CONTOUR alpha(x+, z+) in RADIANS.
    # Single-case counterpart of results.py / results_Re.py "P29b": the wind
    # turning angle of the PHASE-AVERAGED field,
    #     alpha(x,z) = arctan2(<W>, <U>)          [engineering AvgPhW / AvgPhU]
    # i.e. met. arctan(<v>/<u>) with v = spanwise.  arctan2 (not arctan of the
    # ratio) keeps it bounded in [-pi, pi] where <U> reverses inside the valley,
    # the same fix the 1-D [PLOT 28] uses.  Solid cells are NaN'd so they neither
    # set the colour limits nor paint a spurious arctan2(0,0) = 0 in the
    # recirculation region.  RADIANS here (the 1-D plots above stay in degrees).
    _ang_2d = np.arctan2(AvgPhW, AvgPhU)
    _ang_2d = np.where(eps >= 0.5, np.nan, _ang_2d)
    _amax   = float(np.nanmax(np.abs(_ang_2d[:limity, :]))) if np.any(
        np.isfinite(_ang_2d[:limity, :])) else np.pi
    plt.figure(figsize=(8, 6), dpi=300)
    _cfa = plt.contourf(x_in, y_in[:limity], _ang_2d[:limity, :],
                        levels=50, cmap='RdBu_r', vmin=-_amax, vmax=_amax)
    _cla = plt.contour(x_in, y_in[:limity], np.nan_to_num(_ang_2d[:limity, :]),
                       levels=12, colors='k', linewidths=0.4, alpha=0.5)
    plt.clabel(_cla, inline=True, fontsize=6, fmt='%.2f')
    plt.fill(x_oro_in, y_oro_in, facecolor='black')          # IBM solid
    plt.colorbar(_cfa, label=r'$\alpha$ (rad)')
    plt.xlabel(r'$x^+$')
    plt.ylabel(r'$z^+$')
    plt.title(r'Wind turning angle $\alpha=\arctan(\langle v\rangle/\langle u\rangle)$ (rad)')
    plt.savefig(os.path.join(fig_dir, 'P28c_TurningAngle2D.png'), dpi=300)
    plt.show()

    # %%###########################################################################
    # ╔══════════════════════════════════════════════════════════════════════════╗
    # ║  🔒 LOCKED — VALIDATED FIG-4 ASSEMBLY + PLOTS 29–32.  DO NOT MODIFY.       ║
    # ║  Signs = the LOCKED formula banner above (τ_zx: C=−I_corr_yx, R=−(turb+disp);║
    # ║  τ_zy: C=+I_corr_yz, R=−(turb+disp)).  Data = the LOCKED rotation banner    ║
    # ║  (α=config.alpha; velocity rank-1, momentum-flux rank-2 tensor rotation).   ║
    # ║  The τ_zy plot aliases are negated for paper handedness under              ║
    # ║  config.fig4_paper_spanwise_sign (DISPLAY ONLY — physical quantities and    ║
    # ║  u_star untouched).  Verified vs K&A 2024 fig. 4.                           ║
    # ╚══════════════════════════════════════════════════════════════════════════╝
    # ── Fig-4 convention assembly (STANDARD shear-stress budget, shared with plot_fig4_budget) ──
    # Orographic: rotated so g ∥ x = (Gx,Gz)=(1,0).  Smooth: stored g ∥ x.  Rough: the
    # loader reads its geostrophic vector from the profile top (stored at ~18.7° off x),
    # so its τ-split lives in that stored frame — the u* magnitude is frame-invariant, the
    # τ_zx/τ_zy split for the rough overlay is not exactly co-framed with the orographic.
    # Each component's curves (meteorological u=streamwise, v=spanwise, z=wall-normal;
    # Rxy=⟨u'v'⟩, Ryz=⟨w'v'⟩):
    #     C_zx = ∫(g2 − ⟨w⟩) = −I_corr_yx ;  C_zy = ∫(⟨u⟩ − g1) = +I_corr_yz  (Levi-Civita: opposite signs)
    #     V = +ν d⟨vel⟩/dz    R = −⟨flux⟩    T = C + V + R  (≈const = surface-stress comp.)
    # STANDARD (CLAUDE.md "Standard shear-stress budget formulation"), verified against
    # Kostelecky & Ansorge (2024) fig. 4.  u* = (T_zx_plateau² + T_zy_plateau²)^¼ (rotation-invariant).
    # orographic (rotated; 1-D intrinsic profiles already).  The Reynolds flux −⟨flux⟩
    # is split into a TURBULENT curve (Rzx=−turb_yx) and a DISPERSIVE curve
    # (Dzx=−disp_yx); together C + V + R_turb + D_disp = T (Total now includes disp).
    Czx_o = -I_corr_yx; Vzx_o = visc_yx; Rzx_o = -turb_yx; Dzx_o = -disp_yx; Tzx_o = total_tau_yx
    Czy_o = I_corr_yz; Vzy_o = visc_yz; Rzy_o = -turb_yz; Dzy_o = -disp_yz; Tzy_o = total_tau_yz
    # Reynolds contribution (valley/orographic) = turbulent + dispersive = -(turb+disp).
    RYzx_o = Rzx_o + Dzx_o          # = -rey_flux_yx  (the full Reynolds shear term)
    RYzy_o = Rzy_o + Dzy_o          # = -rey_flux_yz
    # smooth reference (loader; collapse the (ny,nt)/(ny,1) arrays to 1-D profiles)
    Czx_s = -I_corr_yx_s; Vzx_s = np.mean(visc_yx_s, axis=1); Rzx_s = -np.mean(Rxy_s, axis=1)
    Tzx_s = Czx_s + Vzx_s + Rzx_s
    Czy_s = I_corr_yz_s; Vzy_s = np.mean(visc_yz_s, axis=1); Rzy_s = -np.mean(Ryz_s, axis=1)
    Tzy_s = Czy_s + Vzy_s + Rzy_s
    # rough r1 reference (loader)
    Czx_r = -I_corr_yx_r; Vzx_r = np.mean(visc_yx_r, axis=1); Rzx_r = -np.mean(Rxy_r, axis=1)
    Tzx_r = Czx_r + Vzx_r + Rzx_r
    Czy_r = I_corr_yz_r; Vzy_r = np.mean(visc_yz_r, axis=1); Rzy_r = -np.mean(Ryz_r, axis=1)
    Tzy_r = Czy_r + Vzy_r + Rzy_r
    # ── Spanwise DISPLAY handedness (config.fig4_paper_spanwise_sign) ──────────
    # Same flip as the PLOT 32r validation figures and fig4_smooth_standalone.py:
    # our tlab f-sign gives the closing τ_zy budget as C_zy<0 / R_zy>0 — the exact
    # MIRROR of K&A.  Negate the SPANWISE plot aliases (τ_zy → −τ_zy) for BOTH the
    # orographic (…_o) and the smooth/rough references (…_s, …_r) so Coriolis reads
    # positive like the paper.  τ_zx is untouched, and this touches ONLY these plot
    # aliases — total_tau_yz / u_star / veer_oro (physical, pickled) are unchanged.
    if fig4_paper_spanwise_sign:
        Czy_o, Vzy_o, Rzy_o, Dzy_o, RYzy_o, Tzy_o = (
            -Czy_o, -Vzy_o, -Rzy_o, -Dzy_o, -RYzy_o, -Tzy_o)
        Czy_s, Vzy_s, Rzy_s, Tzy_s = -Czy_s, -Vzy_s, -Rzy_s, -Tzy_s
        Czy_r, Vzy_r, Rzy_r, Tzy_r = -Czy_r, -Vzy_r, -Rzy_r, -Tzy_r

    # %%###########################################################################
    # Shear Stress XY  (Fig-4 convention: Coriolis +C, Viscous +V, Reynolds −⟨flux⟩, Total C+V+R)
    # [PLOT 29] Shear stress τ_zx — ZOOMED-OUT (full depth) in OUTER units:
    # wall-normal z/δ, stress /G² (each reference in its own δ, G ≈ 1).  The
    # near-wall INNER-unit counterpart is [PLOT 30] below.
    plt.figure(figsize=(10, 6))
    plt.plot(z_out[:], Czx_o[:]/G_mag**2, label='Coriolis', color='blue', linestyle='-')
    plt.plot(z_out[:], Vzx_o[:]/G_mag**2, label='Viscous', color='orange', linestyle='-')
    plt.plot(z_out[:], Rzx_o[:]/G_mag**2, label='Turbulent', color='magenta', linestyle='-')
    plt.plot(z_out[:], Dzx_o[:]/G_mag**2, label='Dispersive', color='cyan', linestyle='-')
    plt.plot(z_out[:], RYzx_o[:]/G_mag**2, label='Reynolds', color='gold', linestyle='-')
    plt.plot(z_out[:], dudt/G_mag**2, label='Temporal', color='saddlebrown', linestyle='-')
    plt.plot(z_out[:], Tzx_o[:]/G_mag**2, label='Total', color='black', linestyle='-')
    # Reynolds (= turbulent + dispersive) in gold: valley (solid, RYzx_o above) +
    # references (dashed/dotted, below).
    ref_plot(plot_ref_smooth, z_out_s, Czx_s, color='blue', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, z_out_s, Vzx_s, color='orange', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, z_out_s, Rzx_s, color='gold', linestyle=SMOOTH_LS)
    # rough r1 (Re=1000) overlay — Method-2 terms, own outer units (z_out_r)
    ref_plot(plot_ref_rough, z_out_r, Czx_r, color='blue', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, z_out_r, Vzx_r, color='orange', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, z_out_r, Rzx_r, color='gold', linestyle=ROUGH_LS)
    mark_layers_multi(z_out, [Czx_o/G_mag**2, Vzx_o/G_mag**2, Rzx_o/G_mag**2,
                              Dzx_o/G_mag**2, dudt/G_mag**2, Tzx_o/G_mag**2], _LYR_ORO, filled=True)
    ref_mark(plot_ref_smooth, mark_layers_multi, z_out_s, [Czx_s, Vzx_s,
                              Rzx_s], _LYR_SMO, filled=False)
    mark_h(z_out[h_idx], 'v')
    plt.title(r'Shear stress $\tau_{zx}$ (outer units)')
    plt.xlabel(r'$z/\delta$')
    plt.ylabel(r'${\langle \bar{\tau} \rangle}_{zx}/G^2$')
    plt.legend(handles=[
        mlines.Line2D([], [], color='blue',       linestyle='-',  label='Coriolis'),
        mlines.Line2D([], [], color='orange',     linestyle='-',  label='Viscous'),
        mlines.Line2D([], [], color='gold',     linestyle='-',  label='Reynolds'),
        mlines.Line2D([], [], color='cyan',       linestyle='-',  label='Dispersive'),
        mlines.Line2D([], [], color='magenta',    linestyle='-',  label='Turbulent'),
        mlines.Line2D([], [], color='saddlebrown',linestyle='-',  label='Temporal'),
        mlines.Line2D([], [], color='black',      linestyle='-',  label='Total'),
    ])
    add_marker_legend(case_lines=True, shade_case=True, smooth_ls=SMOOTH_LS, smooth_color=SMOOTH_COLOR)
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Shear Stress XY.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # Zoomed plot
    # [PLOT 30] Shear stress $\tau_{zx}$
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Valley case (solid lines)
    plt.plot(y_inner[:limity], Czx_o[:limity]/u_star**2, color='blue', linestyle='-', label='Coriolis')
    plt.plot(y_inner[:limity], Vzx_o[:limity]/u_star**2, color='orange', linestyle='-', label='Viscous')
    plt.plot(y_inner[:limity], Rzx_o[:limity]/u_star**2, color='magenta', linestyle='-', label='Turbulent')
    plt.plot(y_inner[:limity], Dzx_o[:limity]/u_star**2, color='cyan', linestyle='-', label='Dispersive')
    plt.plot(y_inner[:limity], RYzx_o[:limity]/u_star**2, color='gold', linestyle='-', label='Reynolds')
    plt.plot(y_inner[:limity], dudt[:limity]/u_star**2, color='saddlebrown', linestyle='-', label='Temporal')
    plt.plot(y_inner[:limity], Tzx_o[:limity]/u_star**2, color='black', linestyle='-', label='Total')
    # Smooth case (dashed); Reynolds (gold) = turbulent + dispersive — valley solid + refs dashed
    ref_plot(plot_ref_smooth, y_s_p, Czx_s/ustr_s1**2, color='blue', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Vzx_s/ustr_s1**2, color='orange', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Rzx_s/ustr_s1**2, color='gold', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, np.zeros(nys), color='saddlebrown', linestyle=SMOOTH_LS)
    # rough r1 — own inner units (ustr_r1)
    ref_plot(plot_ref_rough, y_r_p, Czx_r/ustr_r1**2, color='blue', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Vzx_r/ustr_r1**2, color='orange', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Rzx_r/ustr_r1**2, color='gold', linestyle=ROUGH_LS)

    mark_layers_multi(y_inner, [Czx_o/u_star**2, Vzx_o/u_star**2,
                                Rzx_o/u_star**2, Dzx_o/u_star**2, dudt/u_star**2,
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
        mlines.Line2D([], [], color='orange',      linestyle='-',       label='Viscous'),
        mlines.Line2D([], [], color='gold',      linestyle='-',       label='Reynolds'),
        mlines.Line2D([], [], color='cyan',        linestyle='-',       label='Dispersive'),
        mlines.Line2D([], [], color='magenta',     linestyle='-',       label='Turbulent'),
        mlines.Line2D([], [], color='saddlebrown', linestyle='-',       label='Temporal'),
        mlines.Line2D([], [], color='black',       linestyle='-',       label='Total'),
    ])
    add_marker_legend(case_lines=True, shade_case=True, smooth_ls=SMOOTH_LS, smooth_color=SMOOTH_COLOR)
    plt.grid(True)
    plt.xlim(0, 200)
    plt.ylim(-0.1, 1.0)
    plt.savefig(os.path.join(fig_dir, 'Zoomed Shear Stress XY.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################
    # Shear Stress ZY  (Fig-4 convention: Coriolis +C, Viscous +V, Reynolds −⟨flux⟩, Total C+V+R)
    # [PLOT 31] Shear stress τ_zy — ZOOMED-OUT (full depth) in OUTER units:
    # wall-normal z/δ, stress /G².  Near-wall INNER-unit counterpart is [PLOT 32].
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(z_out[:], Czy_o[:]/G_mag**2, label='Coriolis', color='blue', linestyle='-')
    plt.plot(z_out[:], Vzy_o[:]/G_mag**2, label='Viscous', color='orange', linestyle='-')
    plt.plot(z_out[:], Rzy_o[:]/G_mag**2, label='Turbulent', color='magenta', linestyle='-')
    plt.plot(z_out[:], Dzy_o[:]/G_mag**2, label='Dispersive', color='cyan', linestyle='-')
    plt.plot(z_out[:], RYzy_o[:]/G_mag**2, label='Reynolds', color='gold', linestyle='-')
    plt.plot(z_out[:], dwdt/G_mag**2, label='Temporal', color='saddlebrown', linestyle='-')
    plt.plot(z_out[:], Tzy_o[:]/G_mag**2, label='Total', color='black', linestyle='-')
    # Reynolds (= turbulent + dispersive) in gold: valley (solid, RYzy_o above) + references (dashed).
    ref_plot(plot_ref_smooth, z_out_s, Czy_s, color='blue', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, z_out_s, Vzy_s, color='orange', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, z_out_s, Rzy_s, color='gold', linestyle=SMOOTH_LS)
    # rough r1 (Re=1000) overlay — Method-2 terms, own outer units (z_out_r)
    ref_plot(plot_ref_rough, z_out_r, Czy_r, color='blue', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, z_out_r, Vzy_r, color='orange', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, z_out_r, Rzy_r, color='gold', linestyle=ROUGH_LS)
    mark_layers_multi(z_out, [Czy_o/G_mag**2, Vzy_o/G_mag**2,
                              Rzy_o/G_mag**2, Dzy_o/G_mag**2, dwdt/G_mag**2, Tzy_o/G_mag**2],
                      _LYR_ORO, filled=True)
    ref_mark(plot_ref_smooth, mark_layers_multi, z_out_s, [Czy_s, Vzy_s,
                              Rzy_s], _LYR_SMO, filled=False)
    mark_h(z_out[h_idx], 'v')
    plt.title(r'Shear stress $\tau_{zy}$ (outer units)')
    plt.xlabel(r'$z/\delta$')
    plt.ylabel(r'${\langle \bar{\tau} \rangle}_{zy}/G^2$')
    plt.legend(handles=[
        mlines.Line2D([], [], color='blue',        linestyle='-',       label='Coriolis'),
        mlines.Line2D([], [], color='orange',      linestyle='-',       label='Viscous'),
        mlines.Line2D([], [], color='gold',      linestyle='-',       label='Reynolds'),
        mlines.Line2D([], [], color='cyan',        linestyle='-',       label='Dispersive'),
        mlines.Line2D([], [], color='magenta',     linestyle='-',       label='Turbulent'),
        mlines.Line2D([], [], color='saddlebrown', linestyle='-',       label='Temporal'),
        mlines.Line2D([], [], color='black',       linestyle='-',       label='Total'),
    ])
    add_marker_legend(case_lines=True, shade_case=True, smooth_ls=SMOOTH_LS, smooth_color=SMOOTH_COLOR)
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Shear Stress ZY.png'), dpi=300)
    plt.show()
    
    # %%###########################################################################    
    # Zoomed plot
    # [PLOT 32] Shear stress $\tau_{zy}$
    plt.figure(figsize=(8, 6), dpi=300)

    # Valley case (solid lines)
    plt.plot(y_inner[:limity], Czy_o[:limity]/u_star**2, color='blue', linestyle='-', label='Coriolis')
    plt.plot(y_inner[:limity], Vzy_o[:limity]/u_star**2, color='orange', linestyle='-', label='Viscous')
    plt.plot(y_inner[:limity], Rzy_o[:limity]/u_star**2, color='magenta', linestyle='-', label='Turbulent')
    plt.plot(y_inner[:limity], Dzy_o[:limity]/u_star**2, color='cyan', linestyle='-', label='Dispersive')
    plt.plot(y_inner[:limity], RYzy_o[:limity]/u_star**2, color='gold', linestyle='-', label='Reynolds')
    plt.plot(y_inner[:limity], dwdt[:limity]/u_star**2, color='saddlebrown', linestyle='-', label='Temporal')
    plt.plot(y_inner[:limity], Tzy_o[:limity]/u_star**2, color='black', linestyle='-', label='Total')
    # Smooth case (dashed); Reynolds (gold) = turbulent + dispersive — valley solid + refs dashed
    ref_plot(plot_ref_smooth, y_s_p, Czy_s/ustr_s1**2, color='blue', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Vzy_s/ustr_s1**2, color='orange', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, Rzy_s/ustr_s1**2, color='gold', linestyle=SMOOTH_LS)
    ref_plot(plot_ref_smooth, y_s_p, np.zeros(nys), color='saddlebrown', linestyle=SMOOTH_LS)
    # rough r1 — own inner units (ustr_r1)
    ref_plot(plot_ref_rough, y_r_p, Czy_r/ustr_r1**2, color='blue', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Vzy_r/ustr_r1**2, color='orange', linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, Rzy_r/ustr_r1**2, color='gold', linestyle=ROUGH_LS)

    mark_layers_multi(y_inner, [Czy_o/u_star**2, Vzy_o/u_star**2,
                                Rzy_o/u_star**2, Dzy_o/u_star**2, dwdt/u_star**2,
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
        mlines.Line2D([], [], color='orange',      linestyle='-',       label='Viscous'),
        mlines.Line2D([], [], color='gold',      linestyle='-',       label='Reynolds'),
        mlines.Line2D([], [], color='cyan',        linestyle='-',       label='Dispersive'),
        mlines.Line2D([], [], color='magenta',     linestyle='-',       label='Turbulent'),
        mlines.Line2D([], [], color='saddlebrown', linestyle='-',       label='Temporal'),
        mlines.Line2D([], [], color='black',       linestyle='-',       label='Total'),
    ])
    add_marker_legend(case_lines=True, shade_case=True, smooth_ls=SMOOTH_LS, smooth_color=SMOOTH_COLOR)
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
    # [PLOT 33] Friction Velocity — OUTER units: u*/G vs z/δ (full depth).
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(u_star2[:]/G_mag, z_out[:], label=r'$u_*/G$', color='blue', linestyle='-')
    mark_h(z_out[h_idx], 'h')
    plt.title('Friction Velocity')
    plt.ylabel(r'$z/\delta$')
    plt.xlabel(r'$u_{*}/G$')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Friction velocity.png'), dpi=300)
    plt.show()

    # %%###########################################################################
    # Friction Velocity — comparison of the two Coriolis-integral approaches
    # [PLOT 33b] Friction Velocity (Coriolis integral: mean→integrate vs integrate→cavg)
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(u_star2[:]/G_mag,   z_out[:], label=r'mean$\to$integrate (old)', color='blue', linestyle='-')
    plt.plot(u_star2_c[:]/G_mag, z_out[:], label=r'integrate$\to$cavg (new)', color='red',  linestyle='--')
    mark_h(z_out[h_idx], 'h')
    plt.title('Friction Velocity — Coriolis-integral approaches')
    plt.ylabel(r'$z/\delta$')
    plt.xlabel(r'$u_{*}/G$')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Friction velocity comparison.png'), dpi=300)
    plt.show()

    # %%###########################################################################
    # Friction Velocity — Method-2 for all reference cases (orographic / smooth / rough r1)
    # [PLOT 33c] Friction Velocity (Method-2: orographic vs smooth vs rough)
    # Each case in OUTER units (u*/G vs z/δ); the dotted vertical lines mark the
    # constant-flux plateau values.  Smooth/rough shown per the config switches.
    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(u_star2[:]/G_mag, z_out[:],
             label=f'orographic Re=500 (plateau {ustr_M2_plateau_o:.4f})',
             color='blue', linestyle='-')
    ref_plot(plot_ref_smooth, ustr_M2_s, z_out_s,
             label=f'smooth Re=500 M2 (plateau {ustr_M2_plateau_s:.4f}, stored {ustr_s1:.4f})',
             color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    # Roughness reference not needed here — commented out.
    # ref_plot(plot_ref_rough, ustr_M2_r, z_out_r,
    #          label=f'rough r1 Re=1000 (plateau {ustr_r1:.4f})',
    #          color=ROUGH_COLOR, linestyle=ROUGH_LS)
    plt.axvline(ustr_M2_plateau_o/G_mag, color='blue', linestyle=':', linewidth=1)
    if plot_ref_smooth:
        plt.axvline(ustr_s1, color=SMOOTH_COLOR, linestyle=':', linewidth=1)  # stored smooth u* (G_s ≈ 1)
    # if plot_ref_rough:
    #     plt.axvline(ustr_r1, color=ROUGH_COLOR, linestyle=':', linewidth=1)
    mark_h(z_out[h_idx], 'h')
    plt.title('Friction Velocity — Method 2 (all cases)')
    plt.ylabel(r'$z/\delta$')
    plt.xlabel(r'$u_{*}/G$')
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Friction velocity all-cases.png'), dpi=300)
    plt.show()

    # %%###########################################################################
    # [PLOT 32r] Kostelecky & Ansorge (2024) figure-4 validation of Method 2:
    # integrated momentum budget (C, V, R, Total) for the smooth and rough r1
    # reference cases, in inner (a,b) and outer (c,d) units.  Produced for both
    # cases regardless of the overlay switches (dedicated validation figures).
    # The budget is COMPUTED HERE, in the main script (paper eq. 4.2, directly
    # from the horizontally-averaged profiles) — IO.read_ekman_budget_profiles
    # only loads, PlotField.plot_fig4_budget only draws what it is passed:
    #     ⟨τ⟩_zi(z) = C + V + R            (temporal tendency ≈ 0, steady avg)
    #     C_zx = f∫₀ᶻ(g₂−⟨v⟩)dz' ,  C_zy = f∫₀ᶻ(⟨u⟩−g₁)dz'    (f = 1)
    #       (the spanwise Coriolis integrand carries the OPPOSITE sign to the
    #        streamwise one, ε₂ₖ₃ = −ε₁ₖ₃; with ∫(⟨u⟩−g₁) the total T_zy is
    #        height-constant and u*/veer match the reference file's stored
    #        FrictionVelocity/FrictionAngle)
    #     V = (1/Re_Λ) d⟨u_i⟩/dz ,  R = −⟨u_i'w'⟩
    #     g = (g₁,g₂) read at fig4_top_frac·Ly (free stream, below the sponge);
    #     u* = (T_zx_plat² + T_zy_plat²)^¼ over the fig4_plateau_lo…y_top window.
    for _nc_ref, _nu_ref, _lbl_ref in ((smooth_nc_path, nu,       'smooth_Re500'),
                                       (rough_nc_path,  nu_rough, 'rough_r1_Re1000')):
        y_ref, u_ref, v_ref, Ruw_ref, Rvw_ref = IO.read_ekman_budget_profiles(_nc_ref)
        _top_ref = int(fig4_top_frac * y_ref.size)
        g1_ref, g2_ref = u_ref[_top_ref], v_ref[_top_ref]   # geostrophic vector at BL top
        G_ref = float(np.hypot(g1_ref, g2_ref))
        C_zx_ref = cumtrapz0(g2_ref - v_ref, y_ref)         # Coriolis  f∫(g₂−⟨v⟩)
        V_zx_ref = _nu_ref * np.gradient(u_ref, y_ref)      # viscous
        R_zx_ref = -Ruw_ref                                 # Reynolds −⟨u'w'⟩
        C_zy_ref = cumtrapz0(u_ref - g1_ref, y_ref)         # opposite-sign integrand
        V_zy_ref = _nu_ref * np.gradient(v_ref, y_ref)
        R_zy_ref = -Rvw_ref
        # Display handedness: our tlab f-sign gives C_zy<0/R_zy>0 (mirror of K&A);
        # negate the spanwise budget so the panel matches the paper & the standalone
        # (Coriolis positive).  τ_zx, u*, and closure are untouched (τ_zy → −τ_zy).
        if fig4_paper_spanwise_sign:
            C_zy_ref, V_zy_ref, R_zy_ref = -C_zy_ref, -V_zy_ref, -R_zy_ref
        T_zx_ref = C_zx_ref + V_zx_ref + R_zx_ref
        T_zy_ref = C_zy_ref + V_zy_ref + R_zy_ref
        _plw = (y_ref > fig4_plateau_lo * y_ref[_top_ref]) & (y_ref < y_ref[_top_ref])
        ustar_ref = float((T_zx_ref[_plw].mean()**2 + T_zy_ref[_plw].mean()**2) ** 0.25)
        veer_ref  = float(np.degrees(np.arctan2(g2_ref, g1_ref)))
        plot_fig4_budget(y_ref * ustar_ref / _nu_ref, y_ref / ustar_ref,
                         C_zx_ref, V_zx_ref, R_zx_ref, T_zx_ref,
                         C_zy_ref, V_zy_ref, R_zy_ref, T_zy_ref,
                         ustar_ref, G_ref, veer_ref, _lbl_ref, fig_dir)
        print(f"  [Fig4 budget] {_lbl_ref}: Method-2 u* = {ustar_ref:.4f}  (G={G_ref:.3f}, "
              f"τ_zx plateau={T_zx_ref[_plw].mean():.3e}, τ_zy plateau={T_zy_ref[_plw].mean():.3e})")

    # %%###########################################################################
    # Velocity profile
    # [PLOT 34] Velocity Profile
    plt.figure(figsize=(8,6))
    plt.plot(y_in[(eps_hgt[0]-1):]  -  y_in[(eps_hgt[0]-1)]          ,u_pl_rot2D[(eps_hgt[0]-1):,0]/ustr_s1           , label='Valley top', color='blue', linestyle='-')
    plt.plot(y_in[(eps_hgt[eps_lf]-1):]  -  y_in[eps_hgt[eps_lf]]    ,u_pl_rot2D[(eps_hgt[eps_lf]-1):,eps_lf]/ustr_s1 , label='Left flank', color='saddlebrown', linestyle='-')
    plt.plot(y_in[eps_hgt[512]:]  -  y_in[eps_hgt[512]]              ,u_pl_rot2D[(eps_hgt[512]):,512]/ustr_s1         , label='Valley bottom', color='red', linestyle='-')
    plt.plot(y_in[(eps_hgt[eps_rf]-1):]-y_in[(eps_hgt[eps_rf]-1)]    ,u_pl_rot2D[(eps_hgt[eps_rf]-1):,eps_rf]/ustr_s1 , label='Right flank', color='magenta', linestyle='-')
    
    plt.plot(y_in[(eps_hgt[0]-1):]  -  y_in[(eps_hgt[0]-1)]           ,-w_pl_rot2D[(eps_hgt[0]-1):,0]/ustr_s1           , label='Valley top', color='blue', linestyle='--')
    plt.plot(y_in[(eps_hgt[eps_lf]-1):] - y_in[(eps_hgt[eps_lf]-1)]   ,-w_pl_rot2D[(eps_hgt[eps_lf]-1):,eps_lf]/ustr_s1 , label='Left flank', color='saddlebrown', linestyle='--')
    plt.plot(y_in[eps_hgt[512]:]  -  y_in[eps_hgt[512]]               ,-w_pl_rot2D[eps_hgt[512]:,512]/ustr_s1           , label='Valley bottom', color='red', linestyle='--')
    plt.plot(y_in[(eps_hgt[eps_rf]-1):] - y_in[(eps_hgt[eps_rf] - 1)] ,-w_pl_rot2D[(eps_hgt[eps_rf]-1):,eps_rf]/ustr_s1 , label='Right flank', color='magenta', linestyle='--')
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
    # Each valley curve starts at its OWN local surface (eps_hgt[col]: crest,
    # flanks, valley bottom) and is plotted on a surface-relative z+ axis
    # (y_in[start:] - y_in[shift]).  So the layer markers must be remapped into
    # each curve's shifted axis — the absolute-z+ indices _iv_* would land at the
    # wrong place.  _mark_oro rebuilds z+ relative to that curve's surface and
    # places 'o' (visc, z+~5), '^' (log start, z+~75), 'D' (log top, z+~200)
    # above the local surface, plus 's' (canopy = positive peak of THIS column's
    # dispersive uv below the log start).  start/shift match each plotted line.
    def _mark_oro(start, shift, col, field, sign):
        zsh = y_in[start:] - y_in[shift]
        yc  = sign * field[start:, col] / ustr_s1
        mk  = {'o': _zidx(zsh, 5), '^': _zidx(zsh, 75), 'D': _zidx(zsh, 200)}
        cmax  = _zidx(zsh, 75)
        cprof = UV_disp[start:, col] * mask0[start:, col]
        if cmax >= 1 and cprof.size:
            mk['s'] = int(np.argmax(cprof[:cmax + 1]))
        mark_layers(zsh, yc, mk, filled=True)
    # (start, shift, col) for the u (solid) and -w (dashed) curves, exactly as
    # plotted above so the markers sit ON each curve.
    for s0, sh, col in [(eps_hgt[0]-1,      eps_hgt[0]-1,      0),
                        (eps_hgt[eps_lf]-1, eps_hgt[eps_lf],   eps_lf),
                        (eps_hgt[512],      eps_hgt[512],      512),
                        (eps_hgt[eps_rf]-1, eps_hgt[eps_rf]-1, eps_rf)]:
        _mark_oro(s0, sh, col, u_pl_rot2D,  1.0)
    for s0, sh, col in [(eps_hgt[0]-1,      eps_hgt[0]-1,      0),
                        (eps_hgt[eps_lf]-1, eps_hgt[eps_lf]-1, eps_lf),
                        (eps_hgt[512],      eps_hgt[512],      512),
                        (eps_hgt[eps_rf]-1, eps_hgt[eps_rf]-1, eps_rf)]:
        _mark_oro(s0, sh, col, w_pl_rot2D, -1.0)
    # Smooth curve keeps its absolute-z+ (unshifted) hollow markers.
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

    # [PLOT 34b] Velocity Profile — ZOOMED-OUT (full depth) in OUTER units:
    # global intrinsic mean u/G, v/G vs z/δ (linear).  Outer-unit counterpart of the
    # inner-unit [PLOT 34]; collapses the wake / free-stream region.  References are
    # already stored /G (loader), so they are plotted directly against their own z/δ.
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(z_out, u_plus_rot/G_mag, color='red',  linestyle='-', label='Streamwise (valley)')
    plt.plot(z_out, w_plus_rot/G_mag, color='blue', linestyle='-', label='Spanwise (valley)')
    ref_plot(plot_ref_smooth, z_out_s, GblU_s,  color=SMOOTH_COLOR, linestyle=SMOOTH_LS, label='Streamwise (smooth)')
    ref_plot(plot_ref_smooth, z_out_s, -GblW_s, color=SMOOTH_COLOR, linestyle=SMOOTH_LS, alpha=0.5, label='Spanwise (smooth)')
    ref_plot(plot_ref_rough, z_out_r, GblU_r,   color=ROUGH_COLOR, linestyle=ROUGH_LS, label='Streamwise (rough r1)')
    ref_plot(plot_ref_rough, z_out_r, -GblW_r,  color=ROUGH_COLOR, linestyle=ROUGH_LS, alpha=0.5, label='Spanwise (rough r1)')
    mark_h(z_out[h_idx], 'v')
    plt.title('Velocity Profile (outer units)')
    plt.ylabel(r'$\langle \bar{u}_i \rangle / G$')
    plt.xlabel(r'$z/\delta$')
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'Velocity_Profile_outer.png'), dpi=300)
    plt.show()

    # # %%###########################################################################
    # # zoomed
    # # [PLOT 35] Velocity Profile 
    # plt.figure(figsize=(8,6))
    # plt.plot(y_in[(eps_hgt[0]-1):limity]-y_in[(eps_hgt[0]-1)] ,u_plus[(eps_hgt[0]-1):limity,0], label='top', color='blue', linestyle='-')
    # plt.plot(y_in[(eps_hgt[eps_lf]-1):limity]-y_in[eps_hgt[eps_lf]] ,u_plus[(eps_hgt[eps_lf]-1):limity,eps_lf], label='Flank left', color='saddlebrown', linestyle='-')
    # plt.plot(y_in[eps_hgt[512]:limity]-y_in[eps_hgt[512]]     ,u_plus[(eps_hgt[512]):limity,512], label='Bottom', color='red', linestyle='-')
    # plt.plot(y_in[(eps_hgt[eps_rf]-1):limity]-y_in[(eps_hgt[eps_rf]-1)] ,u_plus[(eps_hgt[eps_rf]-1):limity,eps_rf], label='Flank right', color='magenta', linestyle='-')
    
    # plt.plot(y_in[(eps_hgt[0]-1):limity]-y_in[(eps_hgt[0]-1)],  w_plus[(eps_hgt[0]-1):limity,0], label='top', color='blue', linestyle='--')
    # plt.plot(y_in[(eps_hgt[eps_lf]-1):limity]-y_in[(eps_hgt[eps_lf]-1)],w_plus[(eps_hgt[eps_lf]-1):limity,eps_lf], label='Flank left', color='saddlebrown', linestyle='--')
    # plt.plot(y_in[eps_hgt[512]:limity]-y_in[eps_hgt[512]],w_plus[eps_hgt[512]:limity,512], label='Bottom', color='red', linestyle='--')
    # plt.plot(y_in[(eps_hgt[eps_rf]-1):limity]-y_in[(eps_hgt[eps_rf]-1)],w_plus[(eps_hgt[eps_rf]-1):limity,eps_rf], label='Flank right', color='magenta', linestyle='--')
    
    # # Valley curves are surface-relative (shifted) and there is no smooth curve
    # # here, so only the crest line h is marked (absolute-z+ layer markers omitted).
    # mark_h(y_in[h_idx], 'v')
    # plt.axvline(x=(Re_tau), color='black', linestyle='--', linewidth=1)
    # plt.text((Re_tau), 0.5, r'$\delta$', rotation=90, verticalalignment='center', horizontalalignment='right')
    
    # custom_labels = ['Valley top', 'Left flank', 'Valley bottom', 'Right flank', r'$\langle \bar{u} \rangle$', r'$\langle \bar{v} \rangle$']
    # color_handles = [
    # Line2D([0], [0], color='blue', lw=4, label='Blue'),
    # Line2D([0], [0], color='saddlebrown', lw=4, label='SaddleBrown'),
    # Line2D([0], [0], color='red', lw=4, label='Red'),
    # Line2D([0], [0], color='magenta', lw=4, label='Magenta')]
    # style_handles = [
    # Line2D([0], [0], color='black', linestyle='-', lw=2, label='(-)'),
    # Line2D([0], [0], color='black', linestyle='--', lw=2, label='(--)')]
    # custom_handles = color_handles + style_handles
    # plt.title('Velocity Profile ')
    # plt.ylabel(r'$\langle \bar{u}_i \rangle ^+$')
    # plt.xlabel(r'$z^{+}$')
    # plt.xscale("log")
    # plt.legend(custom_handles, custom_labels, loc='upper left')
    # plt.grid(True)
    # plt.savefig(os.path.join(fig_dir, 'Zoomed_LogLaw.png'), dpi=300)
    # plt.show()
    
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

    # --- Obukhov (1971) MODIFIED log-law overlay for the valley ---------------
    # obu_wind_profile returns u⁺ in u_star units (that is how it was fitted);
    # the plotted valley curve is normalised by 0.0617, so rescale by u_star/0.0617
    # to land on _valley_u.  Drawn only when the modified fit succeeded.
    if np.isfinite(v_star_mod) and np.isfinite(L1_plus_mod):
        _z_mod_plot = np.geomspace(max(_mod_lo, y_in[1]), _mod_hi, 300)
        u_mod_valley_plot = (obu_wind_profile(_z_mod_plot, v_star_mod,
                                              L1_plus_mod, offset_mod)
                             * (u_star / 0.0617))
    else:
        _z_mod_plot = np.array([])
        u_mod_valley_plot = np.array([])

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
# %%


    # ═════════════════════════════════════════════════════════════════════════
    # MODIFIED LOG-LAW TEST BED — manual trial-and-error curve fitting
    # ═════════════════════════════════════════════════════════════════════════
    # Experiment with the Obukhov (1971) modified-log-law parameters by hand for
    # the stratified runs (finite Fr = 1, 0.1, 0.01, …).  Each trial below is
    # drawn next to the measured profile and the automatic curve_fit result, on
    # BOTH a dedicated figure ([PLOT 36t] Modified_loglaw_testbed.png) and the
    # velocity-profile comparison plot ([PLOT 36]).  Every trial is scored with
    # an R² against the measured rotated profile over the trial window, so
    # guesses can be ranked directly against the automatic fit (r2_mod).
    #
    # Model:  u⁺(z⁺) = (v*/κ)·ψ(z⁺/L1⁺) + offset      (obu_wind_profile)
    #
    # HOW TO USE — edit ONLY between the ▼▼▼ / ▲▲▲ markers and re-run this cell:
    #   modlaw_testbed        1 = draw the trial curves, 0 = testbed off
    #   modlaw_test_zlo/zhi   z⁺ window over which trials are DRAWN and SCORED
    #                         (defaults to the auto-fit window [modlaw_zmin,zmax])
    #   modlaw_trials         one dict per guess:
    #     v_star : profile friction velocity in u_star units (auto fit: v_star_mod;
    #              v*≈1 ⇔ profile-implied u★ equals the Method-2 u_star)
    #     L1     : Obukhov dynamic scale L1⁺ in wall units; >0 stable, <0 unstable
    #              (|L1|→∞ recovers the neutral log law; auto fit: L1_plus_mod)
    #     offset : additive intercept / roughness constant (auto fit: offset_mod)
    #     kappa  : OPTIONAL von Kármán constant (default config.obu_kappa = 0.4)

    # ▼▼▼ EDIT HERE — trial parameters for the modified log-law ▼▼▼
    modlaw_testbed  = 1                 # 1 = plot trial curves, 0 = off
    modlaw_test_zlo = 50           # trial window low   (default = fit window)
    modlaw_test_zhi = 120           # trial window high  (default = fit window)
    modlaw_trials = [
        # dict(v_star=0.30, L1=+300.0, offset=19.0),
        dict(v_star=0.50, L1=+40.0, offset=17.0, kappa=0.4),
        # dict(v_star=1.10, L1=-500.0, offset=19.0),     # unstable branch (L1 < 0)
    ]
    # ▲▲▲ EDIT HERE ▲▲▲

    _tb_curves = []       # per trial: (z⁺, u⁺ in u_star units, legend label)
    if modlaw_testbed and _stratified and modlaw_trials:
        _tb_zlo = max(float(modlaw_test_zlo), float(y_in[1]))
        _tb_zhi = float(modlaw_test_zhi)
        _tb_z   = np.geomspace(_tb_zlo, _tb_zhi, 300)
        _tb_msk = (y_in >= _tb_zlo) & (y_in <= _tb_zhi) & np.isfinite(u_h_plus)
        print(f"Modified log-law TEST BED (z+ ∈ [{_tb_zlo:.0f},{_tb_zhi:.0f}]; "
              f"auto fit: v*={v_star_mod:.4f} L1+={L1_plus_mod:+.3e} "
              f"offset={offset_mod:.3f} R²={r2_mod:.4f}):")
        for _it, _t in enumerate(modlaw_trials):
            _kap   = float(_t.get('kappa', obu_kappa))
            _u_mod = obu_wind_profile(_tb_z, _t['v_star'], _t['L1'],
                                      _t['offset'], kappa=_kap)
            # R² of this guess against the measured profile inside the window
            _u_dat = u_h_plus[_tb_msk]
            _u_hat = obu_wind_profile(y_in[_tb_msk], _t['v_star'], _t['L1'],
                                      _t['offset'], kappa=_kap)
            _sst   = float(np.sum((_u_dat - _u_dat.mean())**2))
            _r2t   = (1.0 - float(np.sum((_u_dat - _u_hat)**2))/_sst
                      if _sst > 0 else float('nan'))
            print(f"  trial {_it+1}: v*={_t['v_star']:.3f}  L1+={_t['L1']:+.1f}  "
                  f"offset={_t['offset']:.3f}  κ={_kap:.3f}  →  R²={_r2t:.4f}")
            _tb_curves.append((_tb_z, _u_mod,
                rf"trial {_it+1}: $v_*$={_t['v_star']:.2f}, "
                rf"$L_1^+$={_t['L1']:+.0f}, off={_t['offset']:.2f} "
                rf"($R^2$={_r2t:.3f})"))
    elif modlaw_testbed and modlaw_trials and not _stratified:
        print("Modified log-law TEST BED: skipped — neutral run (Fr=∞).")

    _TB_COLORS = ['magenta', 'teal', 'purple', 'saddlebrown', 'olive', 'deeppink']

    # [PLOT 36t] Dedicated test-bed figure (u_star units — the fit's own frame):
    # measured profile + automatic curve_fit + every manual trial, window shaded.
    if _tb_curves:
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(y_in, u_h_plus, color='red', linestyle='-',
                 label='Measured (rotated streamwise, $u/u_\\star$)')
        if np.isfinite(v_star_mod) and np.isfinite(L1_plus_mod):
            _z_auto = np.geomspace(max(_mod_lo, y_in[1]), _mod_hi, 300)
            plt.plot(_z_auto,
                     obu_wind_profile(_z_auto, v_star_mod, L1_plus_mod, offset_mod),
                     color='darkorange', linestyle='-.', linewidth=2,
                     label=(rf'auto fit: $v_*$={v_star_mod:.2f}, '
                            rf'$L_1^+$={L1_plus_mod:+.0f}, off={offset_mod:.2f} '
                            rf'($R^2$={r2_mod:.3f})'))
        for _ic, (_tz, _tu, _tl) in enumerate(_tb_curves):
            plt.plot(_tz, _tu, color=_TB_COLORS[_ic % len(_TB_COLORS)],
                     linestyle='--', linewidth=1.5, label=_tl)
        plt.axvspan(_tb_zlo, _tb_zhi, color='grey', alpha=0.12,
                    label='trial window (drawn + scored)')
        plt.xscale('log')
        plt.xlabel(r'$z^+$')
        plt.ylabel(r'$\langle \bar{u} \rangle / u_\star$')
        plt.legend(fontsize=7)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title('Modified log-law (Obukhov 1971) — manual test bed')
        plt.savefig(os.path.join(fig_dir, 'Modified_loglaw_testbed.png'), dpi=300)
        plt.show()
# %%


    # Rough Re=1000 STABLE LADDER (ri00.00 → ri18.78) — LOG-LAW overlay only, drawn
    # in each case's OWN inner units (z⁺ = y·u*/ν_rough, u⁺ = ⟨ū⟩/u*).  Gated on the
    # config flag `plot_ref_rough_ladder`; the loader reads only mean rU + the stored
    # FrictionVelocity per file (memory-light) and returns [] when the data directory
    # is absent (e.g. this code-prep repo).  Colour = Ri gradient (viridis).
    _rough_ladder = (load_rough_ladder_loglaw(rough_ladder_dir, nu_rough,
                                              rough_ladder_pattern,
                                              u_star_default=rough_ladder_ustar)
                     if plot_ref_rough_ladder else [])
    _ladder_colors = (plt.cm.viridis(np.linspace(0.12, 0.92, len(_rough_ladder)))
                      if _rough_ladder else [])
    # When the ladder is drawn, the log-law figure must also carry the smooth
    # reference (the orographic/valley case is always drawn), so force the smooth
    # curves on for PLOT 36 even if plot_ref_smooth is off.  Smooth data is loaded
    # unconditionally (load_smooth_case above), so this cannot reference undefined
    # variables.  Only PLOT 36 uses this gate; the other figures keep plot_ref_smooth.
    _ll_show_smooth = bool(plot_ref_smooth) or bool(_rough_ladder)

    # [PLOT 36] Velocity Profile with and without Orography
    plt.figure(figsize=(8, 6), dpi=300)
    # Valley case (solid lines)
    plt.plot(y_in, _valley_u,          color='red',  linestyle='-')
    plt.plot(y_in, w_plus_rot/0.0617,  color='blue', linestyle='-')
    # Smooth case (dashed grey) — forced on when the ladder is drawn (_ll_show_smooth)
    ref_plot(_ll_show_smooth, y_s_p, _smooth_u,                   color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    ref_plot(_ll_show_smooth, y_s_p, -np.mean(W_s_p, axis=1),     color=SMOOTH_COLOR, linestyle=SMOOTH_LS, alpha=0.4)
    # rough r1 (Re=1000) — own inner units (y_r_p), magnitude profiles
    ref_plot(plot_ref_rough, y_r_p, np.mean(U_r_p, axis=1),       color=ROUGH_COLOR, linestyle=ROUGH_LS)
    ref_plot(plot_ref_rough, y_r_p, -np.mean(W_r_p, axis=1),      color=ROUGH_COLOR, linestyle=ROUGH_LS, alpha=0.4)
    # Rough Re=1000 STABLE LADDER — streamwise log-law u⁺(z⁺) per case, own inner
    # units (colour = Ri gradient).  Only drawn when plot_ref_rough_ladder is on.
    for _rc, _rcol in zip(_rough_ladder, _ladder_colors):
        plt.plot(_rc['z_plus'], _rc['u_plus'], color=_rcol,
                 linestyle='-', linewidth=0.8, alpha=0.75, zorder=2)
    # Valley log-law — extended 5 u+ units below fit-range start
    plt.plot(_z_vll_plot, u_loglaw_valley_plot, color='red', linestyle='dotted', linewidth=2)
    # Valley MODIFIED log-law (Obukhov 1971) — stability-corrected profile
    if u_mod_valley_plot.size > 0:
        plt.plot(_z_mod_plot, u_mod_valley_plot, color='darkorange',
                 linestyle='-.', linewidth=2)
    # TEST BED trial curves (manual parameters — see the EDIT HERE block above);
    # model curves are in u_star units, rescale by u_star/0.0617 like the auto fit.
    for _ic, (_tz, _tu, _tl) in enumerate(_tb_curves):
        plt.plot(_tz, _tu * (u_star / 0.0617),
                 color=_TB_COLORS[_ic % len(_TB_COLORS)],
                 linestyle='--', linewidth=1.5)
    # Smooth log-law — extended 5 u+ units below fit-range start
    if _z_sml_plot.size > 0:
        ref_plot(_ll_show_smooth, _z_sml_plot, u_loglaw_smooth_plot, color=SMOOTH_COLOR, linestyle='dotted', linewidth=2)
    # Canopy fit (best of exponential vs power law), z+ ∈ [0, 20]
    plt.plot(_z_can_v, u_canopy_v, color='green', linestyle='--')
    plt.axvline(x=(Re_tau), color='black', linestyle='-', linewidth=1)
    plt.text((Re_tau), 0.5, r'$\delta_{o}$', rotation=90, verticalalignment='center', horizontalalignment='right')
    # Smooth-case BL depth in ITS OWN wall units: δ_s⁺ = u★_s²/ν (u★_s = ustr_s1,
    # the SAME u★ that scales y_s_p), so it lands at the true edge of the smooth
    # curve.  Gated/coloured with the smooth reference (grey dotted) to distinguish
    # it from the solid orographic δ_o and the dashed crest line 'h'.
    if _ll_show_smooth:
        plt.axvline(x=ustr_s1**2/nu, color=SMOOTH_COLOR, linestyle=':', linewidth=1)
        plt.text(ustr_s1**2/nu, 0.5, r'$\delta_{s}$', rotation=90,
                 verticalalignment='center', horizontalalignment='right', color=SMOOTH_COLOR)
    mark_layers_multi(y_in, [_valley_u, w_plus_rot/0.0617], _LYR_ORO, filled=True)
    ref_mark(_ll_show_smooth, mark_layers_multi, y_s_p, [_smooth_u, -np.mean(W_s_p, axis=1)], _LYR_SMO, filled=False)
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
    ] + ([
        Line2D([0], [0], color='darkorange', linestyle='-.',      linewidth=2,
               label=(rf'Obukhov (1971) mod. log-law '
                      rf'($v_*/u_\star$={v_star_mod:.2f}, $L_1^+$={L1_plus_mod:+.0f})'))
    ] if (u_mod_valley_plot.size > 0) else []) + [
        Line2D([0], [0], color=_TB_COLORS[_ic % len(_TB_COLORS)], linestyle='--',
               linewidth=1.5, label=_tl)
        for _ic, (_tz, _tu, _tl) in enumerate(_tb_curves)
    ] + [
        Line2D([0], [0], color='green',      linestyle='--',      label=canopy_legend),
    ]
    if _rough_ladder:
        legend_elements.append(
            Line2D([0], [0], color=plt.cm.viridis(0.5), linestyle='-', linewidth=1.2,
                   label=r'rough Re1000 stable ladder (own $u_\star$)'))
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

    # [PLOT 38b] TKE profile — ZOOMED-OUT (full depth) in OUTER units: TKE/G² vs z/δ.
    # Outer-unit counterpart of the inner-unit [PLOT 38] (each case /its own G², G ≈ 1).
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(z_out, avg_c(eps, TKE, axis=1)/G_mag**2, label='valley', color='blue', linestyle='-')
    ref_plot(plot_ref_smooth, z_out_s, np.mean(TKE_s, axis=1)/G_mag**2, label='smooth', color=SMOOTH_COLOR, linestyle=SMOOTH_LS)
    ref_plot(plot_ref_rough, z_out_r, np.mean(TKE_r, axis=1)/G_mag**2, label='rough r1', color=ROUGH_COLOR, linestyle=ROUGH_LS)
    mark_h(z_out[h_idx], 'v')
    plt.title('TKE profile (outer units)')
    plt.xlabel(r'$z/\delta$')
    plt.ylabel(r'$TKE/G^2$')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'TKE_profile_outer.png'), dpi=300)
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
    # [PLOT 40] Advection — ZOOMED-OUT (full depth) in OUTER units:
    # advection /(G²/δ) vs z/δ.  Near-wall INNER-unit counterpart is [PLOT 41].
    plt.figure(figsize=(6, 5))
    plt.plot(conv_top[:450]/adv_out,    z_out[:450], label='Valley top', color="yellow")
    plt.plot(conv_lf[:450]/adv_out,     z_out[:450], label='Left flank', color="red")
    plt.plot(conv_bottom[:450]/adv_out, z_out[:450], label='Valley bottom', color="black")
    plt.plot(conv_rf[:450]/adv_out,     z_out[:450], label='Right flank', color="blue")
    for _cv in (conv_top, conv_lf, conv_bottom, conv_rf):    # oro layers on every flank curve
        mark_layers(_cv/adv_out, z_out, _LYR_ORO, filled=True)
    mark_h(z_out[h_idx], 'h')
    plt.xlabel(r'$u_{j}\,\partial u_i/\partial x_j \;/\;(G^2/\delta)$')
    plt.ylabel(r'$z/\delta$')
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
    # [PLOT 41] Advection — ZOOMED near-wall in INNER units:
    # advection /(u*³/ν) vs z⁺.
    plt.figure(figsize=(6, 5))
    plt.plot(conv_top[:200]/adv_in,    y_in[:200], label='Valley top', color="yellow")
    plt.plot(conv_lf[:200]/adv_in,     y_in[:200], label='Left flank', color="red")
    plt.plot(conv_bottom[:200]/adv_in, y_in[:200], label='Valley bottom', color="black")
    plt.plot(conv_rf[:200]/adv_in,     y_in[:200], label='Right flank', color="blue")
    for _cv in (conv_top, conv_lf, conv_bottom, conv_rf):    # oro layers on every flank curve
        mark_layers(_cv/adv_in, y_in, _LYR_ORO, filled=True)
    mark_h(y_in[h_idx], 'h')
    plt.xlabel(r'$u_{j}\,\partial u_i/\partial x_j \;/\;(u_*^3/\nu)$')
    plt.ylabel(r'$z^{+}$')
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
    # Plot: ADVERSE PRESSURE GRADIENT near the IBM body — zoomed to z+ <= 200.
    # [PLOT 52] Mean-pressure gradients $\partial\langle\bar p\rangle/\partial x$,
    #           $\partial\langle\bar p\rangle/\partial z$ near IBM (z+ <= 200)
    #
    # The adverse pressure gradient IS the streamwise pressure gradient dP/dx:
    #   dP/dx > 0  -> adverse   (pressure rising downstream, flow decelerating)
    #   dP/dx < 0  -> favorable (pressure falling downstream, flow accelerating)
    # It is datum-independent and identical to d(DispP)/dx (the x-mean removed by
    # the dispersive split depends only on z), so this single field already gives
    # the "relative pressure" view the dispersive plot (PLOT 10) shows -- PLUS the
    # adverse/favorable sign, which a pressure *field* cannot convey directly.
    # Companion panel: the wall-normal gradient dP/dz+ (= engineering dP/dy). The
    # spanwise gradient dP/dy (met.) is identically 0 in this spanwise+phase-
    # averaged 2-D field, so it is not plotted. Both normalised to wall units
    # (nu/u*^3); diverging colormap; each panel on its own symmetric scale.
    # Consistent IBM-solid overlay for every research plot below.  Shade EXACTLY
    # the eps==1 region (this run's own eps) — NOT the analytic cosine polygon
    # x_oro/y_oro, which can miss the true staircase boundary by a cell.  One
    # colour (_IBM_COLOR) is used everywhere so the solid reads the same in all
    # figures.  _mask_solid blanks the solid to NaN so masked-to-0 derivatives
    # do not (a) set the colour scale or (b) draw a spurious 0-isoline hugging
    # the body.
    _IBM_COLOR = 'black'

    def _ibm_overlay(_ax, _xa, _ya, _eps_c):
        if _eps_c is not None and np.nanmax(_eps_c) >= 0.5:
            _ax.contourf(_xa, _ya, _eps_c, levels=[0.5, 1.5],
                         colors=[_IBM_COLOR], zorder=5)

    def _mask_solid(_fld, _eps_c):
        return np.where(_eps_c > 0.5, np.nan, _fld)

    def _robust_vmax(_fld, _pct=98.0):
        _v = np.nanpercentile(np.abs(_fld), _pct)
        return _v if np.isfinite(_v) and _v > 0 else 1.0

    _zP_lim = 200
    _jP_lim = max(int(np.argmin(np.abs(y_inner - _zP_lim))), 2)
    _sc_gradP = nu / u_star**3
    _eps_zoom = eps[:_jP_lim, :]                          # this run's eps (solid=1)

    # Blank the solid so the interior masked-to-0 cells neither wash out the
    # colour scale nor render as a false zero band.
    _apg  = _mask_solid(dP_dx[:_jP_lim, :] * _sc_gradP, _eps_zoom)   # streamwise = APG
    _wng  = _mask_solid(dP_dy[:_jP_lim, :] * _sc_gradP, _eps_zoom)   # wall-normal (dP/dz+)
    # Robust symmetric scale (98th pct of |·|) so a few interface spikes do not
    # flatten the field — the earlier nanmax scaling washed the structure out.
    _vmax_apg = _robust_vmax(_apg)
    _vmax_wng = _robust_vmax(_wng)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
    for _ax, _fld, _vm, _ttl, _cl in (
            (axes[0], _apg, _vmax_apg,
             r'Adverse pressure gradient $\partial\langle\bar p\rangle/\partial x$'
             r'  (red: adverse, blue: favorable)',
             r'$(\partial\langle\bar p\rangle/\partial x)\,\nu/u_*^3$'),
            (axes[1], _wng, _vmax_wng,
             r'Wall-normal pressure gradient $\partial\langle\bar p\rangle/\partial z$',
             r'$(\partial\langle\bar p\rangle/\partial z)\,\nu/u_*^3$')):
        _cf = _ax.contourf(x_in, y_in[:_jP_lim], _fld,
                           levels=np.linspace(-_vm, _vm, 101),
                           cmap='RdBu_r', extend='both')
        plt.colorbar(_cf, ax=_ax, label=_cl)
        _ax.contour(x_in, y_in[:_jP_lim], _fld, levels=[0.0],
                    colors='k', linewidths=0.3, linestyles=':')   # ∂p/∂· = 0 isoline
        _ibm_overlay(_ax, x_in, y_in[:_jP_lim], _eps_zoom)
        for _xi, _ls in ((eps_lf, '--'), (eps_bottom, '-'), (eps_rf, '-.')):
            _ax.axvline(x=x_in[_xi], color='0.4', lw=0.6, ls=_ls, alpha=0.4)
        _ax.set_xlabel(r'$x^+$')
        _ax.set_ylabel(r'$z^+$')
        _ax.set_ylim(0, _zP_lim)
        _ax.set_title(_ttl, fontsize=9)
    plt.tight_layout()
    plt.savefig(cwd + '/fig/gradP_APG_near_IBM.png', dpi=300)
    plt.show()

    # %%###########################################################################
    # FLOW SEPARATION — wall-normal gradient of the tangential velocity.
    # Separation occurs where the near-wall streamwise shear vanishes and reverses:
    #   ∂⟨u⟩/∂z = 0  (z = wall-normal; met. label, engineering ∂u/∂y).  The point
    # where the WALL value (first fluid cell) goes +→- (downstream) is the
    # separation point; -→+ is reattachment; the interval in between (wall shear
    # < 0) is the reversed-flow / recirculation region.  tau_wx = ν(∂u/∂z)|_wall/u*²
    # and tau_wz = ν(∂v/∂z)|_wall/u*² (spanwise; eng. ∂w/∂y) were built above.
    # This is the kinematic counterpart of the adverse-pressure-gradient plot:
    # the APG (dP/dx > 0) is what drives the near-wall flow to separate here.
    #
    # Separation / reattachment x⁺ from sign changes of the wall streamwise shear.
    _txw = tau_wx
    _sign_chg = np.where(_txw[:-1] * _txw[1:] < 0)[0]      # i where sign flips i→i+1

    def _zero_x(_i):                                        # linear zero-crossing x⁺
        _d = _txw[_i + 1] - _txw[_i]
        _f = 0.0 if _d == 0 else -_txw[_i] / _d
        return x_in[_i] + _f * (x_in[_i + 1] - x_in[_i])

    _sep_x = [_zero_x(_i) for _i in _sign_chg if _txw[_i] > 0.0]   # +→- separation
    _rea_x = [_zero_x(_i) for _i in _sign_chg if _txw[_i] < 0.0]   # -→+ reattachment
    print(f"  Flow separation: {len(_sep_x)} separation, {len(_rea_x)} reattachment "
          f"point(s) on the IBM surface")
    if _sep_x:
        print(f"    separation   x⁺: {', '.join('%.0f' % _v for _v in _sep_x)}")
    if _rea_x:
        print(f"    reattachment x⁺: {', '.join('%.0f' % _v for _v in _rea_x)}")

    # ── [PLOT 53] 2-D wall-normal velocity gradients near IBM (z+ <= 200) ──────
    # Left: ∂⟨u⟩/∂z (streamwise shear; blue = reversed = separated).  Right:
    # ∂⟨v⟩/∂z (spanwise shear; eng. ∂w/∂y).  Bold black line = the ∂/∂z = 0
    # isoline; where it meets the wall is the separation/reattachment point.
    _zS_lim  = 200
    _jS_lim  = max(int(np.argmin(np.abs(y_inner - _zS_lim))), 2)
    _sc_shear = nu / u_star**2                              # ∂u/∂z → ∂u⁺/∂z⁺
    _eps_zoomS = eps[:_jS_lim, :]                           # this run's eps (solid=1)

    # Blank the solid to NaN: du_dy/dw_dy are masked to 0 inside the body, so
    # WITHOUT this the ∂/∂z = 0 isoline would trace the solid boundary (the
    # "reversal line overlapping the IBM").  Now the 0-isoline appears only where
    # the shear truly reverses in the fluid.
    _dudz = _mask_solid(du_dy[:_jS_lim, :] * _sc_shear, _eps_zoomS)   # ∂⟨u⟩/∂z (met.)
    _dvdz = _mask_solid(dw_dy[:_jS_lim, :] * _sc_shear, _eps_zoomS)   # ∂⟨v⟩/∂z (eng. ∂w/∂y)
    # Robust symmetric scale (92nd pct of |·|) so reversed pockets are not washed
    # out by the large attached near-wall shear.
    _vmax_u = _robust_vmax(_dudz, 92.0)
    _vmax_v = _robust_vmax(_dvdz, 92.0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
    for _ax, _fld, _vm, _ttl, _cl in (
            (axes[0], _dudz, _vmax_u,
             r'Streamwise shear $\partial\langle u\rangle/\partial z$'
             r'  (blue: reversed $\to$ separated)',
             r'$(\partial\langle u\rangle/\partial z)\,\nu/u_*^2$'),
            (axes[1], _dvdz, _vmax_v,
             r'Spanwise shear $\partial\langle v\rangle/\partial z$',
             r'$(\partial\langle v\rangle/\partial z)\,\nu/u_*^2$')):
        _cf = _ax.contourf(x_in, y_in[:_jS_lim], _fld,
                           levels=np.linspace(-_vm, _vm, 101),
                           cmap='RdBu_r', extend='both')
        plt.colorbar(_cf, ax=_ax, label=_cl)
        _ax.contour(x_in, y_in[:_jS_lim], _fld, levels=[0.0],
                    colors='k', linewidths=0.3, linestyles=':')   # ∂/∂z = 0 isoline
        _ibm_overlay(_ax, x_in, y_in[:_jS_lim], _eps_zoomS)
        for _xv in _sep_x:
            _ax.axvline(x=_xv, color='lime',    lw=1.2, ls='--', zorder=6)
        for _xv in _rea_x:
            _ax.axvline(x=_xv, color='magenta', lw=1.2, ls=':',  zorder=6)
        _ax.set_xlabel(r'$x^+$')
        _ax.set_ylabel(r'$z^+$')
        _ax.set_ylim(0, _zS_lim)
        _ax.set_title(_ttl, fontsize=9)
    axes[0].plot([], [], color='lime',    ls='--', label='separation')
    axes[0].plot([], [], color='magenta', ls=':',  label='reattachment')
    axes[0].legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    plt.savefig(cwd + '/fig/separation_shear_near_IBM.png', dpi=300)
    plt.show()

    # ── [PLOT 54] Surface skin friction & separation points along the IBM ─────
    # The definitive separation-point identifier: wall value of the tangential
    # shear vs x⁺.  Zero-crossing of the streamwise component = separation (+→-)
    # or reattachment (-→+); shaded band = reversed-flow (recirculation) region.
    fig, ax = plt.subplots(figsize=(11, 4), dpi=300)
    ax.plot(x_in, tau_wx, color='blue', lw=1.5,
            label=r'$\partial\langle u\rangle/\partial z|_{\rm wall}\;\nu/u_*^2$ (streamwise)')
    ax.plot(x_in, tau_wz, color='red',  lw=1.2,
            label=r'$\partial\langle v\rangle/\partial z|_{\rm wall}\;\nu/u_*^2$ (spanwise)')
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.fill_between(x_in, tau_wx, 0, where=(tau_wx < 0),
                    color='grey', alpha=0.25, label='reversed flow (separated)')
    for _xv in _sep_x:
        ax.axvline(x=_xv, color='lime',    lw=1.2, ls='--')
    for _xv in _rea_x:
        ax.axvline(x=_xv, color='magenta', lw=1.2, ls=':')
    ax.axvline(x=x_in[eps_bottom], color='k', lw=0.6, ls='-', alpha=0.3, label='valley bottom')
    ax.set_xlabel(r'$x^+$')
    ax.set_ylabel(r'wall shear $/u_*^2$')
    ax.set_title(r'Surface skin friction & separation points along IBM')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(cwd + '/fig/separation_skin_friction.png', dpi=300)
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

    # ══════════════════════════════════════════════════════════════════════════
    # ░░  RESEARCH DIAGNOSTICS PLOTS  ░░   (8 goals — Research.md:536-550)
    # All saved to fig/; gated terms degrade gracefully in the neutral run.
    # ══════════════════════════════════════════════════════════════════════════
    _zr   = y_in[:limity]
    _cols = {'windward': 'b', 'floor': 'g', 'lee': 'r'}

    # ── [R1] Turbulent vs dispersive flux split — momentum & buoyancy (Goal 4) ─
    figR, (axRm, axRb) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    axRm.plot(rey_uv_x[:limity],  _zr, 'b-',  lw=1.5, label=r"turbulent $\langle u''v''\rangle$")
    axRm.plot(UV_disp_x[:limity], _zr, 'r--', lw=1.5, label=r'dispersive $\tilde u\tilde v$')
    axRm.axhline(y_in[hill_hgt], color='k', ls=':', lw=0.8, label='crest $h$')
    axRm.set_xlabel('wall-normal momentum flux'); axRm.set_ylabel(r'$z^+$')
    axRm.set_title('Momentum flux split'); axRm.legend(fontsize=8); axRm.grid(True, ls='--', lw=0.5)
    axRb.plot(Bflux_temp[:limity], _zr, 'b-',  lw=1.5, label=r"turbulent (Route C)")
    axRb.plot(Bflux_disp[:limity], _zr, 'r--', lw=1.5, label=r'dispersive $\tilde w\tilde\theta$')
    axRb.plot(Bflux[:limity],      _zr, 'k:',  lw=1.5, label='total')
    axRb.axhline(y_in[hill_hgt], color='k', ls=':', lw=0.8)
    axRb.set_xlabel(r"wall-normal buoyancy flux $\langle w'\theta'\rangle$"); axRb.set_ylabel(r'$z^+$')
    axRb.set_title('Buoyancy flux split' + ('' if _strat else '  (neutral $\\approx$ 0)'))
    axRb.legend(fontsize=8); axRb.grid(True, ls='--', lw=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'Research_flux_split.png'), dpi=300); plt.show()

    # ── [R1b] Buoyancy-flux VECTOR components ⟨u'b'⟩,⟨v'b'⟩,⟨w'b'⟩ (Goal 4) ──────
    # The three individual components of the phase-averaged buoyancy flux from
    # avg_flux* (streamwise u·s, wall-normal/vertical v·s, spanwise w·s), each split
    # into dispersive (ũ_i b̃) + temporal (Route C). Rotated frame: u,w are in the
    # geostrophic-aligned frame (raw products rotated with rotate_pair above).
    figR, _axF = plt.subplots(1, 3, figsize=(15, 6), dpi=300, sharey=True)
    for _ax, (_dsp, _tmp, _tot, _ttl) in zip(
            _axF,
            ((Uflux_disp, Uflux_temp, Uflux, r"streamwise $\langle u'b'\rangle$"),
             (Bflux_disp, Bflux_temp, Bflux, r"wall-normal $\langle v'b'\rangle$"),
             (Wflux_disp, Wflux_temp, Wflux, r"spanwise $\langle w'b'\rangle$"))):
        _ax.plot(_tmp[:limity], _zr, 'b-',  lw=1.5, label='turbulent (Route C)')
        _ax.plot(_dsp[:limity], _zr, 'r--', lw=1.5, label='dispersive')
        _ax.plot(_tot[:limity], _zr, 'k:',  lw=1.5, label='total')
        _ax.axhline(y_in[hill_hgt], color='k', ls=':', lw=0.8)
        _ax.axvline(0.0, color='0.6', lw=0.6)
        _ax.set_xlabel(_ttl); _ax.grid(True, ls='--', lw=0.5)
    _axF[0].set_ylabel(r'$z^+$'); _axF[0].legend(fontsize=8)
    figR.suptitle('Buoyancy-flux vector components (avg_flux)'
                  + ('' if _strat else '  (neutral $\\approx$ 0)'))
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'Research_flux_components.png'), dpi=300); plt.show()

    # ── [R2] Dispersive flux share — momentum & buoyancy (Goal 4) ──────────────
    figR, axR = plt.subplots(figsize=(7, 6), dpi=300)
    axR.plot(disp_share_mom[:limity],  _zr, 'b-',  lw=1.5, label='momentum')
    axR.plot(disp_share_buoy[:limity], _zr, 'r--', lw=1.5, label='buoyancy')
    axR.axhline(y_in[hill_hgt], color='k', ls=':', lw=0.8, label='crest $h$')
    axR.set_xlim(0, 1); axR.set_xlabel('dispersive share  |disp| / (|disp|+|turb|)')
    axR.set_ylabel(r'$z^+$'); axR.set_title('Dispersive flux share')
    axR.legend(fontsize=8); axR.grid(True, ls='--', lw=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'Research_dispersive_share.png'), dpi=300); plt.show()

    # ── [R3] Local similarity φ_m, φ_h vs z+ at 3 stations vs MOST (Goal 5) ─────
    figR, (axPm, axPh) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    for nm, i in _stn.items():
        js = int(min(eps_hgt[i], ny - 1))
        zcp = (y[js:js + len(phi_m_st[nm])] - y[js]) * (u_star / nu)
        axPm.plot(phi_m_st[nm], zcp, _cols[nm] + '-', lw=1.3, label=nm)
        if _strat:
            axPh.plot(phi_h_st[nm], zcp, _cols[nm] + '-', lw=1.3, label=nm)
    axPm.axvline(1.0, color='k', ls=':', lw=0.8, label='MOST neutral ($\\phi_m{=}1$)')
    axPm.set_xlabel(r'$\phi_m$'); axPm.set_ylabel(r'$z^+$ (from local surface)')
    axPm.set_title('φ_m by station'); axPm.legend(fontsize=8); axPm.grid(True, ls='--', lw=0.5)
    if _strat:
        axPh.axvline(Pr_t, color='k', ls=':', lw=0.8, label=r'MOST neutral ($\phi_h{=}Pr_t$)')
        axPh.legend(fontsize=8)
    else:
        axPh.text(0.5, 0.5, 'neutral run:\nφ_h undefined', ha='center', va='center',
                  transform=axPh.transAxes, fontsize=11)
    axPh.set_xlabel(r'$\phi_h$'); axPh.set_ylabel(r'$z^+$ (from local surface)')
    axPh.set_title('φ_h by station'); axPh.grid(True, ls='--', lw=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'Research_similarity_phi.png'), dpi=300); plt.show()

    # ── [R4] Wave fluxes + sponge reflection guard (Goal 7) ────────────────────
    _top = int(min(sponge_j + 10, ny))
    figR, axR = plt.subplots(figsize=(7, 7), dpi=300)
    axR.plot(wave_mom_flux[:_top], y_in[:_top], 'b-', lw=1.5, label=r'wave momentum flux $\tilde u\tilde v$')
    if _strat:
        axR.plot(wave_buoy_flux[:_top], y_in[:_top], 'r--', lw=1.5, label='wave buoyancy flux')
    axR.axhline(y_in[bl_top_j], color='g', ls='-.', lw=0.9, label=r'BL top $z^+{=}%.0f$' % y_in[bl_top_j])
    axR.axhline(y_in[sponge_j], color='m', ls=':',  lw=1.1, label=r'sponge $z^+{=}%.0f$' % y_in[sponge_j])
    axR.set_xlabel('wall-normal wave flux'); axR.set_ylabel(r'$z^+$')
    axR.set_title('Wave fluxes + sponge guard — reflection OK: %s'
                  % ('yes' if reflection_ok else 'NO'))
    axR.legend(fontsize=8); axR.grid(True, ls='--', lw=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'Research_wave_flux.png'), dpi=300); plt.show()

    # ── [R5] Stability axis — this run's Ri_B vs Ansorge bins (Goal 1) ─────────
    figR, axR = plt.subplots(figsize=(8, 3.2), dpi=300)
    _hi = max(Ri_B_bins[1] * 2.0, abs(Ri_B) * 1.3, Ri_B_bins[1] + 0.05)
    axR.axvspan(0,             Ri_B_bins[0], color='green',  alpha=0.12, label='weak')
    axR.axvspan(Ri_B_bins[0],  Ri_B_bins[1], color='orange', alpha=0.12, label='intermediate')
    axR.axvspan(Ri_B_bins[1],  _hi,          color='red',    alpha=0.12, label='strong')
    axR.axvline(Ri_B, color='k', lw=2.0, label=f'this run  $Ri_B$={Ri_B:.2e}')
    axR.set_xlim(0, _hi); axR.set_yticks([]); axR.set_xlabel(r'$Ri_B = B_0\,\delta_{neu}/G^2$')
    axR.set_title('Stability axis — class: %s' % stab_class)
    axR.legend(fontsize=8, loc='upper right', ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'Research_stability_axis.png'), dpi=300); plt.show()

    # ── [R6] Intermittency (Goal 6) — Ansorge & Mellado (2016) ─────────────────
    # (a) γ(z) profile (eq 4.1) — now spans 0→1 with the physical ω₀=e_ω cut;
    # (b) 2-D local intermittency γ(x,z); (c) instantaneous |ω| field (their fig 2).
    if gamma_z is not None:
        _ztop = y_in[limity - 1]

        # (a) γ(z) profile.
        figR, axR = plt.subplots(figsize=(6, 7), dpi=300)
        axR.plot(gamma_z[:limity], _zr, 'k-', lw=1.5)
        axR.axhline(y_in[hill_hgt], color='b', ls=':', lw=0.8, label='crest $h$')
        axR.axhline(y_in[bl_top_j], color='g', ls='--', lw=0.8,
                    label=r'$\delta$ (BL edge, $\omega_0=e_\omega$)')
        axR.set_xlim(0, 1.02); axR.set_xlabel(r'intermittency $\gamma$'); axR.set_ylabel(r'$z^+$')
        axR.set_title('Intermittency γ(z)'); axR.legend(fontsize=8); axR.grid(True, ls='--', lw=0.5)
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'Research_intermittency_gamma.png'), dpi=300); plt.show()

        # (b) 2-D local intermittency γ(x,z) = ⟨H(|ω'|−ω₀)⟩_t ; cyan line = γ=0.5
        #     turbulent/non-turbulent interface (γ→1 turbulent core, →0 quiescent).
        #     With few independent time frames (e.g. a single planesK/flow snapshot,
        #     N=1) the time-average alone is exactly binary — H() only takes 0/1, so
        #     contourf has no cells to color in between and the colorbar's middle
        #     range goes unused. A short spatial box-filter (periodic in x, masked
        #     in y so solid/near-wall zeros don't bleed into fluid cells) turns the
        #     local turbulent AREA fraction into a graded field spanning the full
        #     colorbar — display only; the raw gamma_field/gamma_z used for the
        #     pickle and the Goal 6 table are untouched. The γ=0.5 border is drawn
        #     from the RAW (unsmoothed) field, i.e. the actual instantaneous
        #     turbulent/non-turbulent interface, not the smoothed display field.
        if gamma_field is not None:
            _knl = 9   # smoothing footprint in grid cells (~a few Δx/Δz); display only
            _num_s = uniform_filter(gamma_field * _mom, size=(_knl, _knl),
                                     mode=('nearest', 'wrap'))
            _den_s = uniform_filter(_mom, size=(_knl, _knl), mode=('nearest', 'wrap'))
            gamma_field_disp = np.divide(_num_s, _den_s, out=np.zeros_like(_num_s),
                                          where=_den_s > 0)

            figG, axG = plt.subplots(figsize=(9, 4.5), dpi=300)
            _cf = axG.contourf(x_in, y_in[:limity], gamma_field_disp[:limity, :],
                               levels=np.linspace(0.0, 1.0, 21), cmap='hot_r')
            axG.contour(x_in, y_in[:limity], gamma_field[:limity, :],
                        levels=[0.5], colors='cyan', linewidths=1.4)
            plt.colorbar(_cf, ax=axG, label=r'intermittency $\gamma$ (box-smoothed, $N=$'
                                            f'{_n2} frame(s))')
            axG.fill(x_oro_in, y_oro_in, color='grey', zorder=3)
            _turb_proxy = mlines.Line2D([], [], color='cyan', lw=1.4,
                                         label=r'turbulent region border ($\gamma=0.5$, raw)')
            axG.legend(handles=[_turb_proxy], loc='upper right', fontsize=8, framealpha=0.85)
            axG.set_xlabel(r'$x^+$'); axG.set_ylabel(r'$z^+$'); axG.set_ylim(0, _ztop)
            axG.set_title(r'Local intermittency $\gamma(x,z)$ — cyan: $\gamma=0.5$')
            plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'Research_intermittency_gamma2D.png'), dpi=300); plt.show()

        # (c) instantaneous spanwise-vorticity magnitude (paper fig 2): raw | high-
        #     pass.  The high-pass panel strips mean shear/background so turbulent
        #     patches stand out; the raw panel is background-dominated (their point).
        if omega_inst_hp is not None:
            _sc = nu / u_star**2
            # raw |ω_z| may be absent — a cluster γ plane (*_planesK_k*.npz)
            # stores only the high-pass field — so plot whichever panels exist.
            _panels = [(_f, _t) for _f, _t in
                       ((omega_inst_raw, r'raw $|\omega_z|$'),
                        (omega_inst_hp,  r"high-pass $|\omega'_z|$")) if _f is not None]
            figO, axO = plt.subplots(1, len(_panels), figsize=(7 * len(_panels), 4.5),
                                     dpi=300, sharey=True, squeeze=False)
            axO = axO[0]
            for _ax, (_fld, _ttl) in zip(axO, _panels):
                _m = np.abs(_fld[:limity, :]) * _sc
                _vmax = float(np.nanpercentile(_m[_m > 0], 99)) if np.any(_m > 0) else 1.0
                _cf = _ax.contourf(x_in, y_in[:limity], _m,
                                   levels=np.linspace(0.0, _vmax, 40), cmap='magma', extend='max')
                plt.colorbar(_cf, ax=_ax, label=r'$|\omega|\,\nu/u_*^2$')
                _ax.fill(x_oro_in, y_oro_in, color='grey', zorder=3)
                _ax.axhline(y_in[bl_top_j], color='cyan', ls='--', lw=0.8)
                _ax.set_xlabel(r'$x^+$'); _ax.set_title(_ttl); _ax.set_ylim(0, _ztop)
            axO[0].set_ylabel(r'$z^+$')
            figO.suptitle('Instantaneous spanwise vorticity magnitude '
                          '(cf. Ansorge & Mellado 2016, fig. 2)')
            plt.tight_layout(); plt.savefig(os.path.join(fig_dir, 'Research_intermittency_omega_field.png'), dpi=300); plt.show()

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
    # planesK layout + animation window/render settings (all from config §13)
    N_KPLANES  = planesK_n_kplanes    # k-planes saved per file  (kplanes%n in TLAB)
    # File variable order in tlab: u(streamwise)=0, v(wall-normal)=1, w(spanwise)=2, s1=3, p=4
    NVARS      = planesK_nvars
    KPLANE_IDX = planesK_kplane_idx   # which k-plane to show (0-based)
    NY_ANIM    = anim_ny              # wall-normal points to include
    FIRST_ITER = anim_first_iter
    LAST_ITER  = anim_last_iter
    STEP       = anim_iter_step
    FPS        = anim_fps
    OUTPUT_MP4 = cwd + 'planesK_animation.mp4'
    # Frames where max|u| or max|v| (meteo: streamwise idx 0, spanwise idx 2) exceeds
    # this threshold are considered unphysical and are discarded before rendering.
    VEL_MAX_THRESHOLD = planesK_vel_max
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

# %%
###############################################################################
######## Streamwise energy spectra — Kolmogorov -5/3 inertial-range check ######
###############################################################################
# Purpose: test whether a -5/3 inertial subrange is present in this simulation.
#
# Why planesK.* (instantaneous), NOT the phase-averaged fields: a -5/3 range is a
# property of the *resolved turbulent fluctuations*.  The phase-averaged AvgPh /
# DispVel fields are coherent (time-mean) quantities and carry no inertial-range
# cascade, so spectra of those (spectra.py) cannot show -5/3.  The planesK.* files
# store instantaneous z-planes of (u, v, w, theta, p) — exactly the snapshots
# needed.  We take the 1-D streamwise (periodic x) FFT of the velocity
# fluctuations u' = u - <u>_x at each wall-normal height, average |F|^2 over all
# available frames, and convert to a spectral density E(kx) normalised so that
# integral E(kx) dkx = variance.  The longitudinal spectrum E_uu(kx) ~ kx^{-5/3}
# in the inertial subrange (Kolmogorov); the transverse spectra E_vv, E_ww share
# the same slope.  A compensated plot kx^{5/3} E(kx) is flat where -5/3 holds.
#
# Coordinate / index convention (tlab order, see the animation block above):
#   idx 0 = u  streamwise         (meteo u)
#   idx 1 = v  wall-normal        (meteo w)   -> "vertical"
#   idx 2 = w  spanwise           (meteo v)
# The 1-D spectrum is always taken along x (streamwise), at fixed height z.
#
# Standalone: reads only the grid + planesK.* + (u_star, nu, l_in); does not need
# the cal_Avg averages, so it runs with cal_Avg=0.  u_star/nu/l_in are the
# Method-2 values when the cal_Avg block ran, otherwise the config.py scalars.
if plot_spectra == 1:
    import glob as _glob

    _SP_NK, _SP_NV, _SP_KP = planesK_n_kplanes, planesK_nvars, planesK_kplane_idx  # planesK layout
    _SP_VEL_MAX = planesK_vel_max       # discard frames with |u|,|v| above this (unphysical)
    _SP_COMPS = [(0, r"$E_{uu}$ (streamwise $u'$)",  'tab:blue'),
                 (1, r"$E_{ww}$ (wall-normal $w'$)", 'tab:green'),
                 (2, r"$E_{vv}$ (spanwise $v'$)",    'tab:red')]
    # Wall-normal heights (z+) at which to draw 1-D spectra — picked in the
    # log/inertial region, above the valley crest and below the Rayleigh sponge.
    _SP_Z_TARGETS = spectra_z_targets

    # Frame sources, in order of preference.  Each descriptor yields one
    # instantaneous (ny, nx) plane per velocity component (tlab idx 0=u streamwise,
    # 1=v wall-normal, 2=w spanwise):
    #   ('planesK', path)          — one saved k-plane file (full time series)
    #   ('flow',    {vi: path})    — first z-plane of the raw flow field component
    # A single instantaneous plane is enough for a -5/3 spectrum (one realisation,
    # just noisier), so when NO planesK.* time series is present we fall back to the
    # raw flow field, which is always in the case folder.  That flow field is a
    # BROKEN file — only the first few z-planes were copied — so ONLY the first
    # plane (pl_id=1) is read.  (The intermittency γ(z) above is a statistical
    # average over many frames and is NOT given this single-plane fallback.)
    _sp_files = sorted(_glob.glob(cwd + 'planesK.*'))
    # keep only real data files (drop e.g. planesK_animation.* and .npy/.gif)
    _sp_files = [f for f in _sp_files
                 if os.path.isfile(f) and os.path.basename(f).split('.')[-1].isdigit()]

    _frames = [('planesK', f) for f in _sp_files]
    if not _frames:
        # Fallback: first k-plane of the raw flow field (tlab velocity triplet).
        # flow.old.1/2/3 = u / v(wall-normal) / w(spanwise); also try flow.1/2/3.
        for _stem in ('flow.old.', 'flow.'):
            _flow = {vi: cwd + _stem + str(vi + 1) for vi in (0, 1, 2)}
            _flow = {vi: p for vi, p in _flow.items() if os.path.isfile(p)}
            if 0 in _flow:                      # need at least u for E_uu
                _frames = [('flow', _flow)]
                print('[spectra] no planesK.* time series — using the first k-plane '
                      f'(plane 1 only; broken flow file) of {sorted(_flow.values())}.')
                break

    if not _frames:
        print('[spectra] no planesK.* and no flow field found — -5/3 spectra skipped.')
    else:
        # uniform streamwise grid spacing (== the global `dx`; recomputed locally
        # as a float for the FFT frequency axis)
        dx_grid = float(x[1] - x[0])
        L_x     = nx * dx_grid
        kx      = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx_grid)   # rad / length, (nk,)
        dk      = 2.0 * np.pi / L_x

        def _streamwise_psd(fluc_2d):
            """1-D streamwise spectral density E(kx) per row; integral E dkx = variance.

            fluc_2d : (ny, nx) fluctuation field, periodic in x (x-mean already removed).
            Returns E with shape (ny, nk) on the wavenumbers `kx` above.
            """
            F = np.fft.rfft(fluc_2d, axis=1)            # (ny, nk)
            P = (np.abs(F) ** 2) / (nx ** 2)            # variance per mode (two-sided)
            P[:, 1:] *= 2.0                             # fold negative wavenumbers (skip DC)
            if nx % 2 == 0:                             # Nyquist mode is not doubled
                P[:, -1] /= 2.0
            return P / dk                               # -> spectral density

        # Accumulate E(kx, z) per component, averaged over valid frames.  A frame
        # may carry only a subset of components (a broken flow file may have only u),
        # so accumulate per component and remember which ones appeared.
        nk      = kx.size
        _E_acc  = {vi: np.zeros((ny, nk)) for vi, _, _ in _SP_COMPS}
        _E_cnt  = {vi: 0 for vi, _, _ in _SP_COMPS}
        _nf     = 0
        for _kind, _src in _frames:
            if _kind == 'planesK':
                try:
                    _pl = read_all_planes(_src, nx, ny, _SP_NK, _SP_NV, _SP_KP)
                except (ValueError, OSError):
                    continue
                _comp = {vi: _pl[vi] for vi, _, _ in _SP_COMPS}
                _tag  = os.path.basename(_src)
            else:   # 'flow' — first z-plane (pl_id=1) of each available component
                _comp = {}
                for vi, _p in _src.items():
                    try:
                        _hdr = read_header(_p)[0]
                        _comp[vi] = readplane(_p, nx, ny, 1, _hdr)
                    except (ValueError, OSError):
                        continue
                _tag = 'flow field (plane 1)'
            if 0 not in _comp:
                continue
            # velocity sanity check (streamwise idx0; spanwise idx2 if present)
            _bad = float(np.max(np.abs(_comp[0]))) > _SP_VEL_MAX
            if 2 in _comp:
                _bad = _bad or float(np.max(np.abs(_comp[2]))) > _SP_VEL_MAX
            if _bad:
                print(f'[spectra] discarding {_tag}: velocity over threshold.')
                continue
            for vi in _comp:
                _f = _comp[vi]
                _fluc = _f - _f.mean(axis=1, keepdims=True)   # remove x-mean per row
                _E_acc[vi] += _streamwise_psd(_fluc)
                _E_cnt[vi] += 1
            _nf += 1

        if _nf == 0 or _E_cnt[0] == 0:
            print('[spectra] all frames rejected (no usable u plane) — -5/3 spectra skipped.')
        else:
            for vi in _E_acc:                     # per-component frame average
                if _E_cnt[vi] > 0:
                    _E_acc[vi] /= _E_cnt[vi]
            _avail = [vi for vi in _E_acc if _E_cnt[vi] > 0]
            print(f'[spectra] streamwise spectra from {_nf} frame(s); '
                  f'components available: {sorted(_avail)}.')

            # Inner (wall) units: kx+ = kx*l_in,  E+ = E/(u*^2 * l_in)  (dimensionless)
            kx_plus = kx * l_in
            E_plus  = {vi: _E_acc[vi] / (u_star ** 2 * l_in) for vi in _E_acc}

            # ---- Export for the cross-case comparison (stage c, spectra.py) -----
            # This block runs LONG after IO.write_results_pickle() (line ~1846), so
            # these arrays cannot ride in sim1_results.pkl.  Write a small .npz
            # instead -- a sanctioned pipeline intermediate (cf. Intermittency.py),
            # so spectra.py never has to touch a raw record itself.
            _spec_npz = os.path.join(cwd, 'spectra_turb.npz')
            _spec_out = {
                'kx_plus': kx_plus,          # (nk,)  inner-scaled wavenumber
                'y_in':    np.asarray(y_in), # (ny,)  z+ of each row
                'u_star':  float(u_star),
                'l_in':    float(l_in),
                'n_frames': int(_nf),
                'source':  ('flow-field plane 1' if _frames[0][0] == 'flow'
                            else 'planesK'),
            }
            for _vi, _nm in ((0, 'E_uu'), (1, 'E_vv'), (2, 'E_ww')):
                if _E_cnt.get(_vi, 0) > 0:   # only components that actually appeared
                    _spec_out[_nm] = E_plus[_vi]          # (ny, nk), density in + units
                    _spec_out[_nm + '_nframes'] = int(_E_cnt[_vi])
            np.savez_compressed(_spec_npz, **_spec_out)
            print(f'[spectra] saved {_spec_npz} '
                  f'({", ".join(k for k in ("E_uu", "E_vv", "E_ww") if k in _spec_out)}; '
                  f'{_nf} frame(s)) for cross-case spectra.py.')

            # Choose heights above the crest, below the sponge, nearest the targets.
            _j_lo = int(hill_hgt) + 2
            _j_hi = int(0.5 * ny)                 # stay well below the Rayleigh sponge
            _sel  = []
            for _zt in _SP_Z_TARGETS:
                _j = int(np.argmin(np.abs(y_in - _zt)))
                if _j_lo <= _j <= _j_hi and _j not in [j for _, j in _sel]:
                    _sel.append((_zt, _j))
            if not _sel:                          # fallback: a few rows in the lower half
                _sel = [(float(y_in[_j]), _j)
                        for _j in np.linspace(_j_lo, _j_hi, 4).astype(int)]

            # Positive wavenumbers only (skip DC at index 0) for the log-log plots.
            _ks = slice(1, None)
            kxp = kx_plus[_ks]

            # ---- Figure 1: E_uu(kx+) at several heights + -5/3 guide & compensated ----
            figS, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))
            _cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(_sel)))
            for (_zt, _j), _col in zip(_sel, _cmap):
                _E = E_plus[0][_j, _ks]           # streamwise (u') spectrum at this height
                axA.loglog(kxp, _E, color=_col, lw=1.4,
                           label=fr'$z^+ \approx {y_in[_j]:.0f}$')
                axB.loglog(kxp, kxp ** (5.0 / 3.0) * _E, color=_col, lw=1.4,
                           label=fr'$z^+ \approx {y_in[_j]:.0f}$')

            # -5/3 reference line anchored to the lowest selected height in the
            # low-wavenumber (inertial) part of the resolved range.
            _Eref = E_plus[0][_sel[0][1], _ks]
            _good = np.isfinite(_Eref) & (_Eref > 0)
            if _good.any():
                _i0 = np.where(_good)[0][max(1, len(kxp) // 12)]
                _k_line = kxp[_i0:]
                _C53 = _Eref[_i0] * kxp[_i0] ** (5.0 / 3.0)
                axA.loglog(_k_line, _C53 * _k_line ** (-5.0 / 3.0),
                           'k--', lw=1.6, label=r'$k_x^{-5/3}$ (Kolmogorov)')
                axB.axhline(_C53, color='k', ls='--', lw=1.2,
                            label=r'$-5/3$ plateau')

            axA.set_xlabel(r'$k_x^+ = k_x\,\nu/u_*$')
            axA.set_ylabel(r'$E_{uu}^+ = E_{uu}/(u_*^2\,\nu/u_*)$')
            axA.set_title('Streamwise spectrum  $E_{uu}(k_x^+)$')
            axA.grid(True, which='both', ls=':', alpha=0.4)
            axA.legend(fontsize=8)

            axB.set_xlabel(r'$k_x^+ = k_x\,\nu/u_*$')
            axB.set_ylabel(r'$(k_x^+)^{5/3}\,E_{uu}^+$')
            axB.set_title('Compensated  $k_x^{5/3}E_{uu}$  (flat $\\Rightarrow$ -5/3)')
            axB.set_xscale('log')
            axB.grid(True, which='both', ls=':', alpha=0.4)
            axB.legend(fontsize=8)

            _src_lbl = ('flow-field plane 1' if _frames[0][0] == 'flow'
                        else f'{_nf} planesK frame(s)')
            figS.suptitle('Streamwise energy spectra — inertial-range (-5/3) check '
                          f'({_src_lbl})', fontsize=12)
            figS.tight_layout()
            _outS = os.path.join(fig_dir, 'Spectra_Euu_kx.png')
            figS.savefig(_outS, dpi=300)
            plt.show()
            print(f'[spectra] saved {_outS}')

            # ---- Figure 2: three velocity components at one representative height ----
            _zt2, _j2 = _sel[len(_sel) // 2]      # middle of the selected heights
            figC, axC = plt.subplots(figsize=(7, 5.5))
            for vi, _lbl, _col in _SP_COMPS:
                if vi not in _avail:              # broken flow file may lack v/w
                    continue
                axC.loglog(kxp, E_plus[vi][_j2, _ks], color=_col, lw=1.4, label=_lbl)
            _Ec = E_plus[0][_j2, _ks]
            _gc = np.isfinite(_Ec) & (_Ec > 0)
            if _gc.any():
                _i0 = np.where(_gc)[0][max(1, len(kxp) // 12)]
                _k_line = kxp[_i0:]
                _C = _Ec[_i0] * kxp[_i0] ** (5.0 / 3.0)
                axC.loglog(_k_line, _C * _k_line ** (-5.0 / 3.0),
                           'k--', lw=1.6, label=r'$k_x^{-5/3}$')
            axC.set_xlabel(r'$k_x^+ = k_x\,\nu/u_*$')
            axC.set_ylabel(r'$E^+ = E/(u_*^2\,\nu/u_*)$')
            axC.set_title(fr'Velocity-component spectra at $z^+\approx{y_in[_j2]:.0f}$')
            axC.grid(True, which='both', ls=':', alpha=0.4)
            axC.legend(fontsize=8)
            figC.tight_layout()
            _outC = os.path.join(fig_dir, 'Spectra_components.png')
            figC.savefig(_outC, dpi=300)
            plt.show()
            print(f'[spectra] saved {_outC}')

    # %%###########################################################################
    #  PLAIN-LANGUAGE SUMMARY → IO.print_run_summary (console reporting).
    #  The ~250-line report + its format helpers moved to IO.py for readability;
    #  it reads every quantity from globals().  Kept at this location (inside the
    #  plot_spectra block) so the run behaviour is unchanged.
    ###############################################################################
    IO.print_run_summary(globals())

    # %%
