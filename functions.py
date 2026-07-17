#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 16:42:14 2026

@author: shreyad95
"""
import os
import struct
import pickle
import numpy as np
import netCDF4 as nc
from scipy.integrate import simpson
from scipy.integrate import trapezoid
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d


###############################################################################
############################## Function defintion #############################

class _Tee(object):
    """Duplicate stdout writes to the terminal AND a log file (sim_stats.log).

    Used so the post-processing statistics are both printed and saved.
    File-write failures are swallowed so a read-only dir can't break the run.
    """
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh
    def write(self, data):
        self._stream.write(data)
        try:
            self._fh.write(data)
        except Exception:
            pass
    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._fh.flush()
        except Exception:
            pass


def start_stats_log(path='sim_stats.log'):
    """Tee everything printed from now on to `path` as well as the terminal.

    `path` is (re)created fresh on each run (mode 'w'). Call this once, at the
    point in PhAvg(_rotated).py from which the run statistics should start being
    recorded (the 'Computing ghost-cell interpolated fields (PCHIP) ...' line).
    Returns the open file handle. Safe to call when stdout is already a _Tee
    (re-wraps the current stream); no-op on failure to open the file.
    """
    import sys
    try:
        fh = open(path, 'w')
    except Exception:
        return None
    sys.stdout = _Tee(sys.stdout, fh)
    return fh


def load_or_compute(names, recompute, compute_fn, save=True, verbose=True,
                    label=None):
    """
    Load a group of cached .npy arrays, or compute them fresh.

    Parameters
    ----------
    names      : list[str]   — base names; files are '<name>.npy'.
    recompute  : bool        — if True, always recompute (ignore caches).
    compute_fn : callable    — returns a tuple/list of arrays in the same
                               order as `names` (used only when computing).
    save       : bool        — save freshly computed arrays to '<name>.npy'.
    verbose    : bool        — print a load/compute message.
    label      : str | None  — human label for the message (defaults to names).

    Returns
    -------
    tuple of arrays, in the order of `names`, so callers can keep flat
    assignments, e.g.  a, b, c = load_or_compute([...], flag, lambda: (...)).
    """
    tag = label or ', '.join(names)
    if (not recompute) and all(os.path.exists(n + '.npy') for n in names):
        if verbose:
            print(f'Loading cached {tag} ...')
        return tuple(np.load(n + '.npy') for n in names)

    if verbose:
        print(f'Computing {tag} ...')
    vals = tuple(compute_fn())
    if len(vals) != len(names):
        raise ValueError(
            f'load_or_compute: compute_fn returned {len(vals)} arrays '
            f'but {len(names)} names were given.')
    if save:
        for n, v in zip(names, vals):
            np.save(n + '.npy', v)
        if verbose:
            print(f'{tag} saved.')
    return vals


def plateau_value(prof, y=None, skip_frac=0.05, win_frac=0.10):
    """Representative 'stable' value of a 1-D profile = mean over its flattest window.

    Used to read the constant-flux-layer plateau of a Method-2 friction-velocity
    profile u*(z): skip the near-wall / roughness region (first `skip_frac` of the
    points), then slide a window of width `win_frac·N` and return the mean of the
    window with the smallest coefficient of variation (std/|mean|).

    `y` is accepted for API symmetry / future spacing-aware weighting; the current
    implementation works on index windows (robust enough for a scalar estimate).
    """
    prof = np.asarray(prof, dtype=np.float64)
    n = prof.size
    if n == 0:
        return float('nan')
    lo = max(2, int(skip_frac * n))
    w  = max(3, int(win_frac * n))
    seg = prof[lo:]
    if seg.size < w:
        return float(np.nanmean(seg)) if seg.size else float('nan')
    best_cv, best_val = np.inf, float(np.nanmean(seg))
    for s in range(0, seg.size - w + 1):
        wv = seg[s:s + w]
        m = np.nanmean(wv)
        if not np.isfinite(m) or abs(m) < 1e-12:
            continue
        cv = np.nanstd(wv) / abs(m)
        if cv < best_cv:
            best_cv, best_val = cv, m
    return float(best_val)


def load_ekman_nc_case(nc_path, x_grid, nu, Re_lambda):
    """
    Load and post-process ONE horizontally-averaged Ekman case (tlab avg_all.nc)
    and return every derived quantity under GENERIC keys.  Single source of truth
    for any external reference case (smooth flat wall, rough r1, …); the file's
    variable names must follow the tlab convention (y, fU/fV/fW, rU/rV/rW,
    Rxx/Rxy/Ryy/Ryz/Rzz).

    Friction velocity:
      * if the file stores `FrictionVelocity` (smooth case) it is read into
        `ustr_stored` and used for the inner scaling (legacy behaviour);
      * the Ekman momentum-integral ("Method 2") friction-velocity profile
        `ustr_M2 = (tau_yx**2 + tau_yz**2)**0.25` is ALWAYS computed, with its
        constant-flux plateau scalar `ustr_M2_plateau`;
      * when no stored value exists (rough case) the inner scaling falls back to
        `ustr_M2_plateau` (`ustr_scale`).

    Uses this module's own `diffu_dy` (7-point Fornberg, order 1), `vIntegral`
    and `plateau_value`.  The geostrophic unit vector (G_x, G_z) is built from the
    SCALAR surface friction angle (stored FrictionAngle [deg] for smooth; near-wall
    stress direction for rough), so cor_yx = -(GblW + G_z), cor_yz = (GblU - G_x).

    Parameters mirror load_smooth_case (nc_path, x_grid, nu, Re_lambda).

    Returns dict with generic keys: sy, nys, U, V, W, su, sw, alpha, ustr_stored,
        alpha_str, y, y_p, rU, rV, rW, G_x, G_z, G, U_p, W_p, GblU, GblW, Rxx,
        Rxy, Rxz, Ryy, Ryz, Rzz, TKE, case_v, cor_yx, I_corr_yx, du_dy, visc_yx,
        tau_yx, cor_yz, I_corr_yz, dw_dy, visc_yz, tau_yz, AVG_TKE_V, x,
        AVG_TKE_V_i, ustr_M2, ustr_M2_plateau, ustr_scale.
    """
    s1 = nc.Dataset(nc_path, 'r')
    sy = (s1.variables['y'][:])
    nys = np.size(sy)
    U = (s1.variables['fU'][:]).T
    V = (s1.variables['fV'][:]).T
    W = (s1.variables['fW'][:]).T
    su = np.mean(U, axis=1)
    sw = np.mean(W, axis=1)
    alpha = (sw / su)                          # turning-angle profile
    # Stored friction velocity / angle exist only for the smooth file
    ustr_stored = (float(np.mean(s1.variables['FrictionVelocity'][:]))
                   if 'FrictionVelocity' in s1.variables else None)
    alpha_str = (float(np.mean(s1.variables['FrictionAngle'][:]))
                 if 'FrictionAngle' in s1.variables else None)
    y = s1.variables['y'][:]
    rU = (s1.variables['rU'][:]).T
    rV = (s1.variables['rV'][:]).T
    rW = (s1.variables['rW'][:]).T
    # Mean pressure profile (present in these avg files; used only as a smooth /
    # reference 2-D panel in results.py).  Guarded: a file without rP returns None.
    rP = (s1.variables['rP'][:]).T if 'rP' in s1.variables else None
    # Mean scalar ⟨s⟩ — the direct solution of the Boussinesq scalar transport eq
    # (tlab `rs`, avg_scal_xz.f90 `rS`); the smooth-case analog of AvgScal.  NB: the
    # buoyancy ⟨b⟩ (tlab `rB`) is a DERIVED quantity (Gravity_Buoyancy ÷ froude);
    # this returns the raw scalar.  ≡0 in the neutral (ri00.00) reference; guarded.
    rs = (s1.variables['rs'][:]).T if 'rs' in s1.variables else None
    # ── Geostrophic magnitude for inner scaling (|G|=1) ──────────────────────
    # G_x/G_z/G below are the SCALAR magnitude used only to normalise U_p/W_p; the
    # geostrophic VECTOR that enters the Coriolis integral is read from the profile
    # top in the Method-2 block (do not confuse the two).  _fric_deg kept for info.
    if alpha_str is not None:
        _fric_deg = alpha_str                                          # stored FrictionAngle [deg]
    else:
        _fric_deg = np.degrees(np.arctan2(sw[4] - sw[0], su[4] - su[0]))  # rough: from near-wall stress dir
    # G_x = np.cos(_fric_deg * np.pi / 180.0)
    # G_z = -np.sin(_fric_deg * np.pi / 180.0)
    G_x = 1
    G_z = 0
    G = np.sqrt(G_x**2 + G_z**2)
    GblU = np.mean(rU, axis=1)
    GblW = np.mean(rW, axis=1)
    Rxx = (s1.variables['Rxx'][:]).T
    Rxy = (s1.variables['Rxy'][:]).T
    # Rxz = ⟨u'w'⟩ (streamwise-spanwise); guarded — an avg file may omit it.
    Rxz = (s1.variables['Rxz'][:]).T if 'Rxz' in s1.variables else None
    Ryy = (s1.variables['Ryy'][:]).T
    Ryz = (s1.variables['Ryz'][:]).T
    Rzz = (s1.variables['Rzz'][:]).T
    TKE = 0.5 * (Rxx + Ryy + Rzz)              # TKE = 0.5*(⟨u'u'⟩+⟨v'v'⟩+⟨w'w'⟩)
    case_v = np.zeros((nys, 1)).astype(int)
    case_v[:, :] = 4; case_v[0, 0] = 1; case_v[1, 0] = 2; case_v[2, 0] = 3
    case_v[-3, 0] = 5; case_v[-2, 0] = 6; case_v[-1, 0] = 7
    # ── Method 2: vertically integrated Ekman momentum balance ───────────────
    # STANDARD shear-stress budget (see CLAUDE.md "Standard shear-stress budget
    # formulation").  The geostrophic vector g=(g1,g2) is read from the mean wind
    # at the BL TOP (0.8·domain height, below the sponge) — NOT hardcoded (1,0):
    # a stored file may be in a rotated frame (e.g. the rough r1 case is stored
    # with the geostrophic at ~18.7° off x → forcing (1,0) gave a spurious
    # u*=0.125; reading g from the profile recovers 0.058 and closes the budget).
    _gtop = int(0.8 * y.size)
    g1, g2 = float(GblU[_gtop]), float(GblW[_gtop])        # (streamwise, spanwise) geostrophic
    #   C_zx = ∫(g2 − ⟨w⟩) = −I_corr_yx ;  C_zy = ∫(⟨u⟩ − g1) = +I_corr_yz
    #   R = −⟨u_i'v'⟩ ;  Total = C + V + R  (height-constant surface stress)
    cor_yx = (GblW - g2)                                   # I_corr_yx = ∫(⟨w⟩ − g2)
    I_corr_yx = vIntegral(cor_yx, y.size, y)
    du_dy = diffu_dy((np.reshape(GblU, (y.size, 1))), y.size, 1, case_v, y, 1)
    visc_yx = (1 / Re_lambda) * du_dy
    tau_yx = -I_corr_yx + np.mean(visc_yx, axis=1) - np.mean(Rxy, axis=1)
    cor_yz = (GblU - g1)                                   # I_corr_yz = ∫(⟨u⟩ − g1)
    I_corr_yz = vIntegral(cor_yz, y.size, y)
    dw_dy = diffu_dy((np.reshape(GblW, (y.size, 1))), y.size, 1, case_v, y, 1)
    visc_yz = (1 / Re_lambda) * dw_dy
    tau_yz = I_corr_yz + np.mean(visc_yz, axis=1) - np.mean(Ryz, axis=1)
    # Method-2 friction-velocity profile and its constant-flux plateau scalar
    ustr_M2 = (tau_yx**2 + tau_yz**2) ** 0.25
    ustr_M2_plateau = plateau_value(ustr_M2, y)
    # Inner-scaling u*: stored value if available, else the Method-2 plateau
    ustr_scale = ustr_stored if ustr_stored is not None else ustr_M2_plateau
    y_p = (y * ustr_scale) / nu
    U_p = (U / ustr_scale) / G        # G is now a scalar (|g|=1) → broadcasts directly
    W_p = (W / ustr_scale) / G
    AVG_TKE_V = np.mean(TKE, axis=0)
    x = np.linspace(0, 1, AVG_TKE_V.size)   # normalized index (= 250 for smooth, 382 for rough)
    AVG_TKE_V_i = np.interp(x_grid, x, AVG_TKE_V)
    s1.close()

    return {
        'sy': sy, 'nys': nys, 'U': U, 'V': V, 'W': W,
        'su': su, 'sw': sw, 'alpha': alpha, 'ustr_stored': ustr_stored,
        'alpha_str': alpha_str, 'y': y, 'y_p': y_p,
        'rU': rU, 'rV': rV, 'rW': rW, 'rP': rP, 'rs': rs,
        'G_x': G_x, 'G_z': G_z, 'G': G,
        'U_p': U_p, 'W_p': W_p, 'GblU': GblU, 'GblW': GblW,
        'Rxx': Rxx, 'Rxy': Rxy, 'Rxz': Rxz, 'Ryy': Ryy, 'Ryz': Ryz, 'Rzz': Rzz,
        'TKE': TKE, 'case_v': case_v,
        'cor_yx': cor_yx, 'I_corr_yx': I_corr_yx, 'du_dy': du_dy,
        'visc_yx': visc_yx, 'tau_yx': tau_yx,
        'cor_yz': cor_yz, 'I_corr_yz': I_corr_yz, 'dw_dy': dw_dy,
        'visc_yz': visc_yz, 'tau_yz': tau_yz,
        'AVG_TKE_V': AVG_TKE_V, 'x': x, 'AVG_TKE_V_i': AVG_TKE_V_i,
        'ustr_M2': ustr_M2, 'ustr_M2_plateau': ustr_M2_plateau,
        'ustr_scale': ustr_scale,
    }


def load_smooth_case(nc_path, x_grid, nu, Re_lambda):
    """
    Smooth-wall reference (flat, neutral, Re=500) — thin wrapper around
    load_ekman_nc_case that remaps the generic keys to the historical `_s` names
    used by PhAvg.py and results.py (single source of truth — unchanged public
    contract).  `ustr_s1` stays the STORED FrictionVelocity; new keys
    `ustr_M2_s` / `ustr_M2_plateau_s` expose the Method-2 estimate for comparison.
    """
    d = load_ekman_nc_case(nc_path, x_grid, nu, Re_lambda)
    return {
        'sy': d['sy'], 'nys': d['nys'], 'U_s': d['U'], 'V_s': d['V'], 'W_s': d['W'],
        'su': d['su'], 'sw': d['sw'], 'alpha_s': d['alpha'], 'ustr_s1': d['ustr_stored'],
        'alpha_str_s': d['alpha_str'], 'y_s': d['y'], 'y_s_p': d['y_p'],
        'rU_s': d['rU'], 'rV_s': d['rV'], 'rW_s': d['rW'], 'rP_s': d['rP'],
        'rs_s': d['rs'],
        'G_x_s': d['G_x'], 'G_z_s': d['G_z'], 'G_s': d['G'],
        'U_s_p': d['U_p'], 'W_s_p': d['W_p'], 'GblU_s': d['GblU'], 'GblW_s': d['GblW'],
        'Rxx_s': d['Rxx'], 'Rxy_s': d['Rxy'], 'Rxz_s': d['Rxz'],
        'Ryy_s': d['Ryy'], 'Ryz_s': d['Ryz'],
        'Rzz_s': d['Rzz'], 'TKE_s': d['TKE'], 'case_v_s': d['case_v'],
        'cor_yx_s': d['cor_yx'], 'I_corr_yx_s': d['I_corr_yx'], 'du_dy_s': d['du_dy'],
        'visc_yx_s': d['visc_yx'], 'tau_yx_s': d['tau_yx'],
        'cor_yz_s': d['cor_yz'], 'I_corr_yz_s': d['I_corr_yz'], 'dw_dy_s': d['dw_dy'],
        'visc_yz_s': d['visc_yz'], 'tau_yz_s': d['tau_yz'],
        'AVG_TKE_V_s': d['AVG_TKE_V'], 'x_s': d['x'], 'AVG_TKE_V_s_i': d['AVG_TKE_V_i'],
        # new: Method-2 friction velocity for the smooth case (vs stored ustr_s1)
        'ustr_M2_s': d['ustr_M2'], 'ustr_M2_plateau_s': d['ustr_M2_plateau'],
    }


def load_loglaw_nc(nc_path, nu, u_star_default=0.0618):
    """MINIMAL log-law loader for a tlab avg_all.nc reference case.

    Deliberately loads ONLY what the log-law needs — the mean streamwise velocity
    profile (plus the tiny 1-D `y` grid needed for the inner-unit abscissa).  This
    is NOT `load_ekman_nc_case`: the full budget / Reynolds stresses are never read.

    Friction velocity: the log-law only needs a NORMALISING scale.  These rough
    stable files store no `FrictionVelocity`, so u* falls back to `u_star_default`
    (0.0618) — the stored value is used only if the file happens to carry one.

    Memory-light per the intended use over a whole ladder of large files: `rU` is
    reduced to a time-mean profile (`np.mean(rU, axis=1)`) and then freed.

    Returns
    -------
    dict {'z_plus', 'u_plus', 'u_star', 'y'}  — z⁺ = y·u*/ν, u⁺ = ⟨ū⟩/u* (inner
        units) — or **None** only if `rU` / `y` cannot be read.
    """
    ds = nc.Dataset(nc_path, 'r')
    if 'FrictionVelocity' in ds.variables:                 # use it only if present
        u_star = float(np.mean(ds.variables['FrictionVelocity'][:]))
    else:
        u_star = float(u_star_default)                     # log-law needs only a scale
    y = np.asarray(ds.variables['y'][:], dtype=float)
    rU = np.asarray(ds.variables['rU'][:], dtype=float).T   # → (ny, nt) like load_ekman_nc_case
    ds.close()
    U_mean = np.mean(rU, axis=1) if rU.ndim == 2 else rU   # time-mean profile ⟨ū⟩(y)
    del rU                                                  # release memory
    if not (np.isfinite(u_star) and u_star > 0):
        return None
    return {'z_plus': y * u_star / nu, 'u_plus': U_mean / u_star,
            'u_star': u_star, 'y': y}


def load_rough_ladder_loglaw(ladder_dir, nu, pattern='ri*_avg.nc',
                             u_star_default=0.0618):
    """Log-law overlay data for a whole rough Re=1000 stability ladder.

    Globs `pattern` inside `ladder_dir` and, for EACH matching avg_all.nc, calls
    `load_loglaw_nc` (mean `rU` only; u* = `u_star_default` since these files store
    no FrictionVelocity — see there).  The bulk Richardson number and the sim tag
    are parsed from the file name (`ri<NN.NN>_..._<tag>_avg.nc`) for labelling /
    ordering.

    Returns a list of per-case dicts sorted by increasing Ri:
        {'ri', 'tag', 'label', 'path', 'z_plus', 'u_plus', 'u_star'}
    An empty list is returned if the directory is absent (e.g. running in the
    code-prep repo rather than on the machine that holds the data).
    """
    import glob as _glob
    import re as _re
    out = []
    if not os.path.isdir(ladder_dir):
        print('[rough-ladder] directory not found -> NO ladder curves plotted: %s'
              % ladder_dir)
        print('               (run on the machine that holds the data, or copy the '
              'ri*_avg.nc files there.)')
        return out
    _files = sorted(_glob.glob(os.path.join(ladder_dir, pattern)))
    if not _files:
        print('[rough-ladder] directory found but NO files match %r in %s -> '
              'NO ladder curves plotted.' % (pattern, ladder_dir))
        return out
    _skipped = 0
    for p in _files:
        d = load_loglaw_nc(p, nu, u_star_default=u_star_default)
        _base = os.path.basename(p)
        if d is None:
            print('  [rough-ladder] skipped (rU/y unreadable): %s' % _base)
            _skipped += 1
            continue
        _mri  = _re.search(r'ri(\d+\.\d+)', _base)
        _mtag = _re.search(r'_([a-z]\d*[a-z]?)_avg\.nc$', _base)
        d['ri']    = float(_mri.group(1)) if _mri else np.nan
        d['tag']   = _mtag.group(1) if _mtag else ''
        d['path']  = p
        d['label'] = ('rough Ri=%.2f' % d['ri']) if np.isfinite(d['ri']) else _base
        if d['tag']:
            d['label'] += ' (%s)' % d['tag']
        out.append(d)
    print('[rough-ladder] %d/%d case(s) loaded from %s (u*=%.4f)%s'
          % (len(out), len(_files), ladder_dir, u_star_default,
             (' (%d skipped: rU/y unreadable)' % _skipped) if _skipped else ''))
    out.sort(key=lambda e: (np.inf if not np.isfinite(e['ri']) else e['ri'], e['tag']))
    return out


bias_patterns = {
    "1": [0,1,2,3,4,5,6],
    "2": [-1,0,1,2,3,4,5],
    "3": [-2,-1,0,1,2,3,4],
    "4":  [-3,-2,-1,0,1,2,3],
    "5": [-4,-3,-2,-1,0,1,2],
    "6": [-5,-4,-3,-2,-1,0,1],
    "7": [-6,-5,-4,-3,-2,-1,0]
}

def fornberg_weights(x_stencil, x0, m):
    n = len(x_stencil)
    c = np.zeros((n, m+1))
    c1 = 1.0
    c4 = x_stencil[0] - x0
    c[0, 0] = 1.0

    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x_stencil[i] - x0

        for j in range(i):
            c3 = x_stencil[i] - x_stencil[j]
            c2 *= c3

            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k*c[i-1, k-1] - c5*c[i-1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i-1, 0] / c2

            for k in range(mn, 0, -1):
                c[j, k] = (c4*c[j, k] - k*c[j, k-1]) / c3
            c[j, 0] = c4*c[j, 0] / c3

        c1 = c2

    return c[:, m]

def read_fortran_record(f_h, dtype):
    dum1 = np.fromfile(f_h, dtype, count=1)[0]
    return dum1

def read_header(FilePath):
    # Define sizes based on Fortran implementation
    int_dtype = np.dtype('<i4')  # 4-byte integer, little-endian
    float_dtype = np.dtype('<f8')  # 8-byte float, little-endian
    sizeofint = 4
    sizeofreal = 8
    
    try:
        with open(FilePath, 'rb') as f:
            # Read the offset first
            offset = read_fortran_record(f, int_dtype)

            if offset <= sizeofint:
                raise ValueError("Offset value is too small, it nust be greater than the size of an integer.")

            # Read the grid dimensions and nt
            nx = read_fortran_record(f, np.dtype('<i4'))
            ny = read_fortran_record(f, np.dtype('<i4'))
            nz = read_fortran_record(f, np.dtype('<i4'))
            nt = read_fortran_record(f, np.dtype('<i4'))
            # Calculate the size of params
            remaining_header_size = offset - 5 * sizeofint
            params_size = int(remaining_header_size/sizeofreal)

            # Read params if there are any
            params = []
            if params_size > 0:
                for i in range (params_size):
                    params_record = read_fortran_record(f, np.dtype('<f8'))  # 'f8' for double precision float
                    params.append(params_record)

            return offset, nx, ny, nz, nt, params

    except Exception as e:
        # Print the error message and return a default value
        # print(f'Error reading header: {e}')
        return None, None, None, None, None, None
    
def read_grid(path):
    #---------------------------------------------------------------------------#
    # Read grid
    #---------------------------------------------------------------------------#

    # open grid file
    seek = 0
    f = open(path+'grid','rb')
    f.seek(seek,0)

    # header - number of nodes
    print("--------------------------------------------------")       
    h = np.fromfile(f, '<i4', 1)
    print('iheader length = ', h)
    nmax = np.fromfile(f, '<i4', 3)
    h = np.fromfile(f, '<i4', 1)
    print('check iheader  = ', h)

    # header - grid scales
    print("--------------------------------------------------")       
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    scales = np.fromfile(f, '<f8', 3)
    print('scales         = ', scales)
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)

    # x - nodes
    print("--------------------------------------------------")  
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    x = np.fromfile(f, '<f8', nmax[0])
    print('x-nodes       =  ', x[:5])
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)

    # y - nodes
    print("--------------------------------------------------")  
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    y = np.fromfile(f, '<f8', nmax[1])
    print('y-nodes       =  ', y[:5])
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)

    # z - nodes
    print("--------------------------------------------------")  
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    z = np.fromfile(f, '<f8', nmax[2])
    print('z-nodes       =  ', z[:5])
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h) 
    print("--------------------------------------------------")  

    # close grid file
    f.close()
    return x, y, z

def epsfield():
    #-----------------------------------------------------------------------------#
    # data specification of eps field
    #-----------------------------------------------------------------------------#
    # path to data
    current_path = os.getcwd() + '/'
    path         = current_path
    fname        ='eps0.1'

    # data types (little endian)
    type_i1 = np.dtype('<i1'); type_i4 = np.dtype('<i4'); type_f8 = np.dtype('<f8')
    sizeofdata_int1 = 1; sizeofdata_int4 = 4; sizeofdata_float = 8

    # header
    head_params = 5 
    head_size   = head_params * sizeofdata_int4

    #-----------------------------------------------------------------------------#
    # read
    #-----------------------------------------------------------------------------#
    # header
    f = open(path + fname,'rb')
    f.seek(0,0)
    header = np.fromfile(f, type_i4, head_params)
    f.close()
    print('Header size           :', header[0])
    print('Grid   size (nx*ny*nz):', header[1]*8,'x',header[2],'x',header[3])

    # data size (attention: h[1] = grid.nx*8!)
    bsize = np.prod(header[1:3])
    rsize = bsize * 8

    # read eps field as int1
    f = open(path + fname,'rb')
    f.seek(header[0],0)
    data = np.fromfile(f, np.dtype('<i1'), bsize)
    f.close()

    #-----------------------------------------------------------------------------#
    # convert to bitwise 
    #-----------------------------------------------------------------------------#

    eps = np.zeros(rsize)
    eps = int2bit_2(eps,data) # eps = int2bit_2(eps,data) # faster
    eps = eps.reshape((header[1]*8,header[2]),order='F') # (attention: h[1] = grid.nx*8!)
    return eps.T #eps[:,:,1].T

def int2bit_2(out,data): # option 2 (bit faster then option 1)
    bsize = data.size
    for i in range(bsize):
        ip = i * 8
        by   = struct.pack('b',data[i])
        by2b = ''.join(format(ord(by), '08b') for byte in by)
        j = 0
        for k in range(-1,-9,-1):
            out[j+ip] = int(str(by2b)[k])
            j += 1
    return out

def _eps_log(msg, _logpath='eps.log'):
    """Append msg to eps.log (NOT printed to the console).

    Used by epsVolume to record the surface-cell classification (the
    'Case undefined' diagnostics) so they can be inspected later if there
    are any issues. Writing failures are non-fatal. eps.log is created in
    the current working directory (the case dir from which PhAvg.py /
    PhAvg_rotated.py are run) and is rewritten on each run.
    """
    try:
        with open(_logpath, 'a') as _f:
            _f.write(str(msg) + '\n')
    except Exception:
        pass

def epsVolume(eps,ny,nx, hill_hgt):
    eps_vol = np.zeros((ny,nx))

    # start a fresh eps.log for this run (truncate any previous one)
    try:
        open('eps.log', 'w').close()
    except Exception:
        pass

    for j in range (hill_hgt+1):
        for i in range (nx):
            if i == 1023:
                _eps_log('i: %d' % i)

            # Top
            if j == 0:
                # Top left cornor
                if i == 0:
                    if (eps[j,i] + eps[j+1,i+1] + eps[j+1,i] + eps[j,i+1] == 4):
                        eps_vol[j,i] = 1
                    else:
                        _eps_log('i: %d j: %d Case undefined' % (i, j))
                        
                # Top right cornor
                elif i == nx-1:
                    if (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 4):
                        eps_vol[j,i] = 1
                    else:
                        _eps_log('i: %d j: %d Case undefined' % (i, j))
                        
                # Top edge
                if i != 0 and i != nx-1:
                    if (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] + eps[j+1,i+1] + eps[j,i+1] == 6):
                            eps_vol[j,i] = 1
                            
                    elif (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] + eps[j+1,i+1] + eps[j,i+1] == 5):
                            eps_vol[j,i] = 0.75
                            
                    elif (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] + eps[j+1,i+1] + eps[j,i+1] == 4):
                            eps_vol[j,i] = 0.5
                    
                    elif (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] + eps[j+1,i+1] + eps[j,i+1] == 2):
                            eps_vol[j,i] = 0.25
                        
                    elif (eps[j,i] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] + eps[j+1,i+1] + eps[j,i+1] == 3):
                        if (eps[j+1,i] == 0) and ((eps[j,i+1] == 0) or (eps[j,i-1] == 0)):
                            eps_vol[j,i] = 0.25
                        else:
                            eps_vol[j,i] = 0.5
                    else:
                        _eps_log('i: %d j: %d Case undefined' % (i, j))
                
            # Generalized area
            elif i != 0 and j != 0 and i != nx-1:
                if (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 9):
                    eps_vol[j,i] = 1
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 8):
                    eps_vol[j,i] = 0.75
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 7):
                    eps_vol[j,i] = 0.5
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 6):
                    eps_vol[j,i] = 0.5
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 5):
                    eps_vol[j,i] = 0.25
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] + eps[j+1,i-1] + eps[j,i-1] + eps[j-1,i-1] == 4):
                    eps_vol[j,i] = 0.25
                else:
                    _eps_log('i: %d j: %d Case undefined' % (i, j))
                    
            # Left edge
            elif i == 0 and j != 0:
                if (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] == 6):
                    eps_vol[j,i] = 1
                
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] == 5):
                    eps_vol[j,i] = 0.5
                    
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i+1] + eps[j,i+1] + eps[j+1,i+1] + eps[j+1,i] == 4):
                    eps_vol[j,i] = 0.5
                    
                else:
                    _eps_log('i: %d j: %d Case undefined' % (i, j))
                    
            # Right edge
            elif i == nx-1 and j != 0:
                if (eps[j,i] + eps[j-1,i] + eps[j-1,i-1] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 6):
                    eps_vol[j,i] = 1
                
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i-1] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 5):
                    eps_vol[j,i] = 0.5
                    
                elif (eps[j,i] + eps[j-1,i] + eps[j-1,i-1] + eps[j,i-1] + eps[j+1,i-1] + eps[j+1,i] == 4):
                    eps_vol[j,i] = 0.5
                else:
                    _eps_log('i: %d j: %d Case undefined' % (i, j))
                    
            else:
                _eps_log('i: %d j: %d Case undefined' % (i, j))
    return eps_vol

def writefield(path, Nx, Ny, Nz, field):
    output_FilePath = path
    data_block = np.zeros((Ny,Nx))
    ofile=open(output_FilePath,'ab')
    ofile.seek(52)
    for iz in range(Nz):
        data_block[:,:]=field[:,:,iz]
        ofile.write(data_block)
        
def readfield(path, Nx, Ny, Nz, hdr):
    field = np.zeros((Ny,Nx,Nz))
    input_FilePath = path 
    data_block = np.zeros((Ny,Nx))
    ifile=open(input_FilePath,'rb')
    ifile.seek(hdr)
    for iz in range(Nz):
        data_block[:,:]=np.fromfile(ifile, dtype=np.float64, count=Nx*Ny).reshape([Ny,Nx])
        field[:,:,iz] = data_block
    ifile.close()
    return field

def readplane(path, Nx, Ny, pl_id, hdr):
    plane = np.zeros((Ny,Nx))
    input_FilePath = path 
    ifile=open(input_FilePath,'rb')
    ifile.seek(hdr + Nx*Ny*(pl_id-1)*8)
    plane[:,:]=np.fromfile(ifile, dtype=np.float64, count=Nx*Ny).reshape([Ny,Nx])
    ifile.close()
    return plane

def diff_cases(eps,nx,ny):
    case_h = np.zeros((ny,nx)).astype(int)
    case_v = np.zeros((ny,nx)).astype(int)
    # For horizontal cases
    # Case1 : Forward derivatives. current point is solid The fluid is on the next 6 points
    # Case2 : Forward2 there is 1 fluid point behind and 5 fluid points ahead
    # Case3 : Forward3 there are 2 fluid points behind and and 4 fluid points ahead
    # Case4 : Central there are 3 fluid points behind and 3 fluid points ahead
    # Case5 : Backward there are 4 fluid points behind and 2 ahead and after which there is solid
    # Case6 : Backward there are 5 fluid points behind and 1 ahead and after which there is solid
    # Case7 : Backward there are 6 fluid points behind and 0 ahead and after which there is solid
    for j in range(ny):
        for i in range(nx):
            #### Solid #####
            if eps[j,i] == 1:
                if  i == 0:
                    case_h[j,i] = 4
                elif i == 1:
                    if eps[j,i+1] == 0:
                        case_h[j,i] = 4
                    else:
                        case_h[j,i] = 4
                elif i == 2:
                    if eps[j,i+1] == 0:
                        case_h[j,i] = 4
                    else:
                        case_h[j,i] = 4
                elif ((i > 2) and (i < nx-3)):
                    lhs = sum(eps[j,i-3:i])
                    rhs = sum(eps[j,i+1:i+4])
                    if (lhs == rhs):
                        case_h[j,i] = 4
                    elif (lhs == 3) and (rhs == 0):
                        case_h[j,i] = 1
                    elif (lhs == 3) and (rhs == 1):
                        case_h[j,i] = 6
                    elif (lhs == 3) and (rhs == 2):
                        case_h[j,i] = 5 
                    elif (lhs == 0) and (rhs == 3):
                        case_h[j,i] = 7
                    elif (lhs == 1) and (rhs == 3):
                        case_h[j,i] = 2
                    elif (lhs == 2) and (rhs == 3):
                        case_h[j,i] = 3
                    
                elif (i == nx-3):
                    if eps[j,i+1] == 0:
                        case_h[j,i] = 4
                    else:
                        case_h[j,i] = 4
                elif (i == nx-2):
                    case_h[j,i] = 4
                elif (i == nx-1):
                    case_h[j,i] = 4
                else:
                    case_h[j,i] = 4
            #### FLuid ####
            elif eps[j,i] == 0:
                if  i == 0:
                    case_h[j,i] = 4
                elif i == 1:
                    case_h[j,i] = 4
                
                elif i == 2:
                    if eps[j,i-1] == 1:
                        case_h[j,i] = 4
                    else:
                        case_h[j,i] = 4
                    
                elif ((i > 2) and (i < nx-3)):
                    lhs = sum(eps[j,i-3:i])
                    rhs = sum(eps[j,i+1:i+4])
                    if (lhs == rhs):
                        case_h[j,i] = 4
                    elif (lhs == 3 and lhs != rhs):
                        case_h[j,i] = 2
                    elif (lhs == 2 and lhs != rhs):
                        case_h[j,i] = 3
                    elif (lhs == 1 and lhs != rhs):
                        case_h[j,i] = 4
                    elif (rhs == 3 and lhs != rhs):
                        case_h[j,i] = 6
                    elif (rhs == 2 and lhs != rhs):
                        case_h[j,i] = 5
                    elif ((lhs == 1) or (rhs == 1)):
                        case_h[j,i] = 4
                elif (i == nx-3):
                    if (eps[j,i+1] == 1):
                        case_h[j,i] = 4
                    else:
                        case_h[j,i] = 4
                elif (i == nx-2):
                    case_h[j,i] = 4
                elif (i == nx-1):
                    case_h[j,i] = 4
                else:
                    case_h[j,i] = 4
            # =============================================================== #    
            #### Verical case
            # =============================================================== #   
            if (j == 0):
                case_v[j,i] = 1
            elif (j == 1):
                if (eps[j,i] == 1 and eps[j+1,i] == 0):
                    case_v[j,i] = 1
                else:
                    case_v[j,i] = 2
            elif (j == 2):
                if (eps[j,i] == 1 and eps[j+1,i] == 0):
                    case_v[j,i] = 1
                elif ((eps[j,i] == 1) and (eps[j-1,i] == 1) and (eps[j-2,i] == 1)):
                    case_v[j,i] = 3
                elif ((eps[j,i] == 0) and (eps[j-1,i] == 0) and (eps[j-2,i] == 0)):
                    case_v[j,i] = 3
                else:
                    case_v[j,i] = 2
            elif (j == 3):
                bottom = sum(eps[j-3:j, i])
                top = sum(eps[j+1:j+4, i])
                if (eps[j,i] == 1 and eps[j+1,i] == 0):
                    case_v[j,i] = 1
                elif (bottom == top):
                    case_v[j,i] = 4
                elif (bottom == 2):
                    case_v[j,i] = 3
                else:
                    case_v[j,i] = 4
            elif (j < ny-3):
                bottom = sum(eps[j-3:j, i])
                top = sum(eps[j+1:j+4, i])
                # Inside Solids 
                if (eps[j,i] == 1): 
                    if ((top == 0) and (bottom > 0)):
                        case_v[j,i] = 1
                        
                    elif ((top == 1) and (bottom > 1)):
                        if (j > 4):
                            case_v[j,i] = 6
                        else:
                            case_v[j,i] = 5
                    
                    elif ((top == 2) and (bottom > 1)):
                        if (j > 3):
                            case_v[j,i] = 5
                        else:
                            case_v[j,i] = 4
                    # Equal number of points on both sides
                    elif (top == bottom ):
                        case_v[j,i] = 4
                    
                    elif ((top > 1) and (bottom == 0)):
                        case_v[j,i] = 4
                    
                    elif ((top > 1) and (bottom == 1)):
                        if (j+5 < nx):
                            case_v[j,i] = 2
                        else:
                            case_v[j,i] = 4
                    
                    elif ((top > 1) and (bottom == 2)):
                        if (j+4 < nx):
                            case_v[j,i] = 3
                        else:
                            case_v[j,i] = 4
                # Inside fluid            
                elif (eps[j,i] == 0):
                    if ((bottom == 1) or (bottom == 0)):
                        case_v[j,i] = 4
                        
                    elif (bottom == 2):
                        case_v[j,i] = 3
                    
                    elif (bottom == 3):
                        case_v[j,i] = 2
                    
                    elif (bottom == top):
                        case_v[j,i] = 4
                    
                    else:
                        case_v[j,i] = 0
            # Top boundary
            elif (j == ny-3):
                if (eps[j,i] == 1 and eps[j-1,i] == 0):
                    case_v[j,i] = 7
                else:
                    case_v[j,i] = 5
            elif (j == ny-2):
                if (eps[j,i] == 1 and eps[j-1,i] == 0):
                    case_v[j,i] = 7
                else:
                    case_v[j,i] = 6
            elif (j == ny-1):
                case_v[j,i] = 7
            else:
                case_v[j,i] = 0          
    # for i in range (nx):
    #     for j in range (ny):
    #         if j == 0:
    #             case_v[j,i] = 1
    #         if j == 1:
    #             case_v[j,i] = 2
    #         if j == 2:
    #             case_v[j,i] = 3
    #         if j >= 3 and j <= ny-4:
    #             case_v[j,i] = 4
    #         if j == ny-3:
    #             case_v[j,i] = 5
    #         if j == ny-2:
    #             case_v[j,i] = 6
    #         if j == ny-1:
    #             case_v[j,i] = 7
    # case_h[:,:] = 4
    return case_v, case_h

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
    fld_x[eps_sum, np.arange(Nx)] = 0; fld_y[eps_sum, np.arange(Nx)] = 0
    return fld_x, fld_y
                
def get_weights(x, i, derivative_order, bias_type):
    offsets = bias_patterns[bias_type]
    idx = [i + o for o in offsets]
    x_stencil = x[idx]
    return fornberg_weights(x_stencil, x[i], derivative_order), idx

def diffu_dy(field, ny, nx, case_v, y, order):    # ny is number of points in vertical
    du = np.zeros((ny,nx))
    for i in range (nx):
        for j in range (ny):
            if (case_v[j,i] != 0):
                w, jdx = get_weights(y, j, order, str(case_v[j,i]))
                du[j,i] = np.dot(w, field[jdx, i])
            else:
                w = np.zeros((7))
                du[j,i] = 0
    return du

def diffu_dx(field, ny, nx, case_h, x, order):    # ny is number of points in vertical
    du = np.zeros((ny,nx))
    for j in range (ny):
        for i in range (nx):
            if (case_h[j,i] != 0):
                w, idx = get_weights(x, i, order, str(case_h[j,i]))
                idx = np.linspace(i-3,i+3,7).astype(int)
                idx = (idx%nx).astype(int)
                du[j,i] = np.dot(w, field[j, idx])
            else:
                du[j,i] = 0
    return du

def local_wavenumbers(field, xc, yc):
    """Signed local horizontal (k) and vertical (m) wavenumbers of a real 2-D
    field via the analytic-signal (Hilbert) phase-gradient method.

    Used for the gravity-wave vertical-wavenumber diagnostic m(x,z): the sign of
    m (with the Hilbert-fixed dominant k>0) encodes the phase-line tilt, and
    k*m<0 ⇒ upward energy propagation for the mountain-wave branch.

    Parameters
    ----------
    field : (ny, nx) real array, periodic & uniform in x (axis 1), non-uniform
            yc allowed (axis 0).
    xc, yc : 1-D coordinate arrays for x (axis 1) and y (axis 0).

    Returns
    -------
    k, m : (ny, nx) arrays — local horizontal and vertical wavenumbers, via the
           unwrap-free identity  wavenumber = Im((d/ds A)/A), with the analytic
           signal A = field + i*Hilbert_x[field] built from numpy FFT only
           (x is periodic, so the FFT-based Hilbert is exact).
    """
    ny_, nx_ = field.shape
    dxc = float(xc[1] - xc[0])
    F = np.fft.fft(field, axis=1)
    h = np.zeros(nx_)                       # one-sided (analytic) multiplier
    if nx_ % 2 == 0:
        h[0] = 1.0; h[nx_ // 2] = 1.0; h[1:nx_ // 2] = 2.0
    else:
        h[0] = 1.0; h[1:(nx_ + 1) // 2] = 2.0
    kx_ = 2.0 * np.pi * np.fft.fftfreq(nx_, d=dxc)
    A   = np.fft.ifft(F * h[np.newaxis, :], axis=1)             # analytic signal
    A_x = np.fft.ifft(1j * kx_[np.newaxis, :] * F * h[np.newaxis, :], axis=1)
    A_y = np.gradient(A, yc, axis=0)        # complex-safe, non-uniform yc
    _den = A + 1e-30
    return np.imag(A_x / _den), np.imag(A_y / _den)

def vIntegral(varaible, ny, y): # ny is number of points in vertical
    I = np.zeros((ny))
    for j in range (1,ny):
        if j == 1:
            I[j] = trapezoid(varaible[:j],x=y[:j])
        elif j > 1:
            I[j] = simpson(varaible[:j],x=y[:j])
    return I

def vIntegral2(varaible, ny, y):
    I = np.zeros((ny))
    for j in range (1,ny):
        I[j] = I[j-1] + 0.5*(varaible[j] + varaible[j-1])*(y[j]-y[j-1])
    return I

def vIntegral_2d(field, ny, y):
    """Column-wise cumulative vertical integral of a 2-D field (ny, nx).

    For every x-column i, returns the running integral from the wall
        I[j, i] = ∫_0^{y[j]} field[:, i] dy,
    using the trapezoidal rule (identical scheme to vIntegral2, applied per
    column and vectorised over columns).

    Intended for use over orography: pass a field that is ALREADY zeroed inside
    the solid (e.g. corr*mask0), so solid cells add no contribution and the
    integration naturally skips the immersed body.  This differs from
    vIntegral(np.mean(field, axis=1)) — which x-averages (extrinsically, over all
    nx columns) BEFORE integrating.  Here each column is integrated first, so the
    result can be combined with the intrinsic (fluid-only) average avg_c AFTER
    integration:  avg_c(eps, vIntegral_2d(field, ny, y), axis=1).

    Returns
    -------
    I : ndarray, shape (ny, nx) — cumulative integral per column (I[0, :] = 0).
    """
    field = np.asarray(field, dtype=np.float64)
    y     = np.asarray(y, dtype=np.float64)
    I     = np.zeros_like(field)
    dy    = np.diff(y)                                   # (ny-1,)
    incr  = 0.5 * (field[1:, :] + field[:-1, :]) * dy[:, None]
    I[1:, :] = np.cumsum(incr, axis=0)
    return I

def createIntegrate(surf_hor, n, i_id, variable, x, side):
    if side == 'LHS':
        indj = np.where(surf_hor[0,:int(n)] == i_id)[0]
    else:
        indj = np.where(surf_hor[0,int(n):] == i_id)[0]
    min_ind = indj.min()
    max_ind = indj.max()+1
    sigma = variable[0,min_ind:max_ind]
    coords = x[min_ind:max_ind]
    I = simpson(sigma, x=coords)
    return I

def compute_r_squared(d, z, u):
    # Apply the same filtering to both z and u
    valid_indices = z > d
    z_d = z[valid_indices] - d
    ln_z_d = np.log(z_d)
    u_valid = u[valid_indices]
    slope, intercept, r_value, _, _ = linregress(ln_z_d, u_valid)
    return r_value**2, slope, intercept

def calculate_roughness_function(y_smooth, u_smooth, y_rough, u_rough, utau_smooth, utau_rough, nu, kappa=0.41):
    """
    Calculates the roughness function Delta U+ by comparing smooth and rough log-laws.
    """
    # 1. Convert to wall units (+)
    y_plus_s = y_smooth * utau_smooth / nu
    u_plus_s = u_smooth / utau_smooth
    
    y_plus_r = y_rough * utau_rough / nu
    u_plus_r = u_rough / utau_rough
    
    # 2. Define the Log-Law Region (Standard range is 30 < y+ < 0.2*delta)
    # We'll filter for y+ between 50 and 500 for a stable fit
    mask_s = (y_plus_s > 50) & (y_plus_s < 500)
    mask_r = (y_plus_r > 50) & (y_plus_r < 500)
    
    # 3. Fit: U+ = (1/kappa) * ln(y+) + B
    # Smooth Wall Fit
    slope_s, intercept_s, _, _, _ = linregress(np.log(y_plus_s[mask_s]), u_plus_s[mask_s])
    B_smooth = intercept_s
    
    # Rough Wall Fit
    slope_r, intercept_r, _, _, _ = linregress(np.log(y_plus_r[mask_r]), u_plus_r[mask_r])
    B_rough = intercept_r
    
    # 4. Calculate Delta U+ (The downward shift)
    delta_u_plus = B_smooth - B_rough
    
    return delta_u_plus, B_smooth, B_rough

def avg_c(eps, field, axis):
    n = np.shape(field)[axis]
    eps_y = np.sum((1-eps), axis=axis)
    eps_y = np.where(eps_y < n, eps_y+1, eps_y)
    fld = np.sum(field, axis=axis)
    mean_c = fld/eps_y
    return mean_c

def valley_profile(field_2d, loc, y_in, flk_hgt, lf_ind, rf_ind):
    '''Extract vertical profile at a valley location.
    loc: 'top', 'lf', 'bottom', 'rf'
    Returns (profile, y_coord) pair.
    '''
    if loc == 'top':
        return np.mean(field_2d[94:, 0:5], axis=1), y_in[94:]
    elif loc == 'lf':
        return np.mean(field_2d[flk_hgt:, lf_ind], axis=1), y_in[flk_hgt:]
    elif loc == 'bottom':
        return np.mean(field_2d[:, 507:517], axis=1), y_in[:]
    elif loc == 'rf':
        return np.mean(field_2d[flk_hgt:, rf_ind], axis=1), y_in[flk_hgt:]


def update_frame(frame):
    # Read the corresponding plane file
    filename = f'Plane{frame}.txt'  # Assuming your filenames are Plane0.txt, Plane1.txt, ...
    field = read_field(filename)    # Read the field for this frame
    im.set_data(field)              # Update the data for the image
    return [im]

def plot_frame(ax, x, y, field_2D):
    Y, X = np.meshgrid(y, x)
    if np.max(field_2D) > abs(np.min(field_2D)):
        ll = -np.max(field_2D)
        ul = np.max(field_2D)
    else:
        ll = np.min(field_2D)
        ul = -np.min(field_2D)
    
    contourf = ax.contourf(X, Y, field_2D.T, cmap='seismic', levels=100, vmin=ll, vmax=ul)
    return contourf

def save_sim(sim_id, data_dict):
    filename = f"results_{sim_id}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Saved: {filename}")


def read_plane(filepath, imax, jmax, n_kplanes, nvars, var_idx, kplane_idx):
    """Return a 2-D (imax × jmax) array for one variable from a raw float32 planesK file."""
    expected_bytes = imax * jmax * n_kplanes * nvars * 4
    actual_bytes = os.path.getsize(filepath)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{filepath}: expected {expected_bytes} B "
            f"({imax}×{jmax}×{n_kplanes}×{nvars}×4), got {actual_bytes}. "
            f"Check N_KPLANES / NVARS."
        )
    data = np.fromfile(filepath, dtype=np.float32)
    block_size = imax * jmax * n_kplanes
    var_block  = data[var_idx * block_size : (var_idx + 1) * block_size]
    plane_3d   = var_block.reshape((imax, jmax, n_kplanes), order='F')
    return plane_3d[:, :, kplane_idx]   # shape (imax, jmax)


def read_all_planes(filepath, imax, jmax, n_kplanes, nvars, kplane_idx):
    """Read all nvars variables from one planesK file; return list of (ny, nx) arrays."""
    expected_bytes = imax * jmax * n_kplanes * nvars * 4
    actual_bytes   = os.path.getsize(filepath)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{filepath}: expected {expected_bytes} B "
            f"({imax}×{jmax}×{n_kplanes}×{nvars}×4), got {actual_bytes}."
        )
    data       = np.fromfile(filepath, dtype=np.float32)
    block_size = imax * jmax * n_kplanes
    planes = []
    for vi in range(nvars):
        var_block = data[vi * block_size : (vi + 1) * block_size]
        plane_3d  = var_block.reshape((imax, jmax, n_kplanes), order='F')
        planes.append(plane_3d[:, :, kplane_idx].T)   # (nx,ny) → (ny,nx)
    return planes   # list of nvars arrays each (ny, nx)


# def update_frame(frame):
#     # path =
#     pl_id = frame_ids[frame]  # Get the specific plane ID from the list
#     field = read_plane(path, Nx, Ny, pl_id)  # Read the field for this plane
#     im.set_data(field)  # Update the data for the image
#     return [im]


def power_law_model(z, a, n):
    """Near-wall power law: u⁺ = a · z⁺^n."""
    return a * z ** n


def fmt_val(v, fmt='.5f', na_str='n/a'):
    """Format a scalar for the parameter table; return na_str if None or NaN."""
    if v is None:
        return na_str
    if isinstance(v, float) and np.isnan(v):
        return na_str
    return format(v, fmt)


def print_summary_table(title, rows, width=74, label_w=46):
    """
    Print a consolidated, aligned key-value table to the console.

    Parameters
    ----------
    title : str
    rows  : list — each item is either
              ('section', text)            -> a sub-header / divider line, or
              (label, value)               -> formatted with default '.5f', or
              (label, value, fmt)          -> formatted with `fmt`, or
              (label, value, fmt, unit)    -> value then a trailing unit string.
            `value` may be None/NaN (shown as 'n/a').

    Uses fmt_val so NaN/None are handled gracefully.  Pure output — no return.
    """
    bar = '=' * width
    print('\n' + bar)
    print('  ' + title)
    print(bar)
    for row in rows:
        if row[0] == 'section':
            print('-' * width)
            if len(row) > 1 and row[1]:
                print('  ' + str(row[1]))
                print('-' * width)
            continue
        label = row[0]
        value = row[1]
        fmt   = row[2] if len(row) > 2 else '.5f'
        unit  = row[3] if len(row) > 3 else ''
        cell  = fmt_val(value, fmt) + (f' {unit}' if unit else '')
        print(f'  {label:<{label_w}s}{cell:>{width - label_w - 2}s}')
    print(bar)


###############################################################################
# Plot annotation helpers (valley-crest height + boundary-layer sublayers)
#
# Boundary-layer sublayer marker convention (inner units z+ = y·u*/ν).
# Discrete markers — same symbol = same level in EVERY plot:
#   'o'  viscous sublayer top          z+ ≈ 5      (both cases)
#   's'  canopy top / roughness start  (orographic only) = peak of dispersive
#                                       stress uv_t (x-averaged), where it stops
#                                       rising and starts to dissipate
#   '^'  log-layer start               smooth z+ ≈ 30 ;  orographic z+ ≈ 75
#   'D'  log-layer top                 smooth z+ ≈ 100;  orographic z+ ≈ 200
#   'X'  valley crest h (index 94)     used only where z+ is NOT an axis
#                                       (e.g. hodograph); elsewhere h is a line
# Fill encodes the wall type:
#   filled marker  = orographic (valley) curve
#   hollow marker  = smooth-wall curve
# Layer structure:
#   Smooth     : viscous(≤5) | buffer(5–30)         | log(30–100)
#   Orographic : viscous(≤5) | canopy(5–canopy_top) | roughness(canopy_top–75)
#                | log(75–200)
###############################################################################

# symbol -> human-readable layer-boundary name (for legends / reference)
LAYER_MARKER_NAMES = {
    'o': r'viscous top ($z^+\!\approx\!5$)',
    's': r'canopy/roughness (oro)',
    '^': r'log start ($z^+\!\approx\!30/75$)',
    'D': r'log top ($z^+\!\approx\!100/200$)',
    'X': r'valley crest $h$',
}


# ── Reference-overlay helpers (gated by the config master switches) ──────────
# Plot / mark a reference-case curve only when its switch (plot_ref_smooth /
# plot_ref_rough) is True, so the same plot code serves publication (smooth-only)
# and testing (smooth + rough) without duplicating every figure.
def ref_plot(flag, *args, **kwargs):
    if flag:
        import matplotlib.pyplot as plt
        plt.plot(*args, **kwargs)


def ref_mark(flag, fn, *args, **kwargs):
    if flag:
        fn(*args, **kwargs)


def cumtrapz0(fvals, xvals):
    """Cumulative trapezoidal integral of fvals over xvals, starting at 0.

    Vectorised (O(n)) counterpart of the loop-based vIntegral2.  Used by the
    Obukhov Ξ-integral wall-law fit and by the fig-4 momentum-budget
    computation (PLOT 32r) in PhAvg[_rotated].py.

    NOTE: the old mixed-role plot_fig4_budget (nc read + eq. 4.2 budget + u* +
    panel drawing in one \"plotting\" function) was split per module role:
    loading → IO.read_ekman_budget_profiles, computation → in-line in
    PhAvg[_rotated].py PLOT 32r, drawing → PlotField.plot_fig4_budget."""
    return np.concatenate(([0.0],
                           np.cumsum(0.5 * (fvals[1:] + fvals[:-1])
                                     * np.diff(xvals))))


def mark_h(pos, orient='v', label=True, lblpos=0.04, ha='right', ax=None,
           color='black', linestyle='--', linewidth=0.8):
    """Mark a reference height with a dashed line labelled 'h'.

    pos    : coordinate value of the line — e.g. y_in[hill_hgt], the valley
             crest height (z+ = y_in[94]) in inner units.
    orient : 'v' draws a vertical line (use when z+ is on the x-axis);
             'h' draws a horizontal line (use when z+ is on the y-axis).
    The 'h' label is placed in axes-fraction coordinates along the line so it
    renders correctly on linear or log axes and after axis re-scaling.
    """
    import matplotlib.pyplot as plt
    a = ax if ax is not None else plt.gca()
    if orient == 'v':
        a.axvline(x=pos, color=color, linestyle=linestyle, linewidth=linewidth)
        if label:
            a.text(pos, lblpos, r'$h$', rotation=90, va='bottom', ha=ha,
                   fontsize=9, transform=a.get_xaxis_transform())
    else:
        a.axhline(y=pos, color=color, linestyle=linestyle, linewidth=linewidth)
        if label:
            a.text(lblpos, pos, r'$h$', va='bottom', ha='left',
                   fontsize=9, transform=a.get_yaxis_transform())


def mark_layers(xdata, ydata, idx_dict, ax=None, filled=True, color='black',
                size=5, zorder=6, edgewidth=0.8):
    """Place discrete boundary-layer markers on a curve.

    xdata, ydata : the x- and y-data arrays of the curve to mark on (e.g. z+ and
                   u+, or the (u, w) components of a hodograph).  The markers are
                   drawn at (xdata[idx], ydata[idx]) so they sit ON the curve.
    idx_dict     : mapping {marker-symbol: index}, e.g.
                   {'o': i_visc, 's': i_canopy, '^': i_logstart, 'D': i_logtop}.
    filled       : True  -> filled marker  (orographic / valley curve);
                   False -> hollow marker  (smooth-wall curve).
    """
    import matplotlib.pyplot as plt
    a = ax if ax is not None else plt.gca()
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    n = min(xdata.size, ydata.size)
    for sym, idx in idx_dict.items():
        if idx is None or idx < 0 or idx >= n:
            continue
        a.plot(xdata[idx], ydata[idx], linestyle='none', marker=sym,
               markersize=size,
               markerfacecolor=(color if filled else 'none'),
               markeredgecolor=color, markeredgewidth=edgewidth, zorder=zorder)


def mark_layers_multi(xdata, ydata_list, idx_dict, **kwargs):
    """Apply mark_layers to several curves that share the same x-data.

    Use this so EVERY curve of a case carries its layer markers, e.g. all the
    valley term-curves of a momentum budget (shared x = y_inner) or all the
    smooth term-curves (shared x = y_s_p).
    """
    for ydata in ydata_list:
        mark_layers(xdata, ydata, idx_dict, **kwargs)


def layer_legend_handles(symbols=('o', 's', '^', 'D'), oro=True, smooth=True,
                         markersize=5):
    """Return Line2D legend handles describing the layer-marker symbols.

    markersize keeps the legend glyphs small (the on-curve markers drawn by
    mark_layers are deliberately larger).  Pass the result to ax.legend(), or
    use add_marker_legend() to drop a fine-print legend at the bottom.
    """
    from matplotlib.lines import Line2D
    h = []
    for sym in symbols:
        h.append(Line2D([0], [0], ls='none', marker=sym, color='black',
                        markerfacecolor='black', markeredgecolor='black',
                        markersize=markersize,
                        label=LAYER_MARKER_NAMES.get(sym, sym)))
    if oro:
        h.append(Line2D([0], [0], ls='none', marker='o', color='black',
                        markerfacecolor='black', markeredgecolor='black',
                        markersize=markersize, label='valley (filled)'))
    if smooth:
        h.append(Line2D([0], [0], ls='none', marker='o', color='black',
                        markerfacecolor='none', markeredgecolor='black',
                        markersize=markersize, label='smooth (hollow)'))
    return h


def add_marker_legend(ax=None, oro=True, smooth=True, fontsize=6, markersize=5,
                      case_lines=False, shade_case=False,
                      smooth_ls='--', smooth_color='grey'):
    """Add a separate, fine-print legend explaining the layer markers BELOW the
    x-axis label (outside the data area), without disturbing the plot's existing
    (line) legend.

    The on-curve markers are large for visibility; this legend uses small glyphs
    (markersize) and small text (fontsize) so it reads as a discreet footnote.
    It is anchored just under the x-axis label (loc='upper center',
    bbox_to_anchor=(0.5, -0.18)) so it no longer overlaps the curves.  The bottom
    figure margin is widened (subplots_adjust) so the footnote clears the x-axis
    label and is captured by a plain savefig; tight_layout also reserves room for
    it because the legend is an in-layout child of the axes (the matplotlib
    default).  Call it AFTER the main legend has been created on the same axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    a = ax if ax is not None else plt.gca()
    main = a.get_legend()                 # main (line) legend already on the axes

    # CASE / style keys: filled ● = valley, hollow ○ = smooth, plus (optional,
    # case_lines) the line-style keys Valley (solid) and Smooth (dashed) that
    # otherwise clutter the main line legend.
    case_h = []
    if oro:
        case_h.append(Line2D([0], [0], ls='none', marker='o', color='black',
                             markerfacecolor='black', markeredgecolor='black',
                             markersize=markersize, label='valley (filled)'))
    if smooth:
        case_h.append(Line2D([0], [0], ls='none', marker='o', color='black',
                             markerfacecolor='none', markeredgecolor='black',
                             markersize=markersize, label='smooth (hollow)'))
    if case_lines:
        case_h.append(Line2D([0], [0], color='black', ls='-', label='Valley'))
        case_h.append(Line2D([0], [0], color=smooth_color, ls=smooth_ls, label='Smooth'))

    if not shade_case:
        # ── Original single-legend footnote (default behaviour, unchanged) ──
        handles = layer_legend_handles(oro=oro, smooth=smooth, markersize=markersize)
        if case_lines:
            handles = handles + case_h[-2:]        # append only the two line keys
        a.figure.subplots_adjust(bottom=0.24)
        leg = a.legend(handles=handles, loc='upper center',
                       bbox_to_anchor=(0.5, -0.18), ncol=min(4, len(handles)),
                       fontsize=fontsize, handletextpad=0.3, columnspacing=0.9,
                       frameon=True, framealpha=0.7, facecolor='white',
                       edgecolor='none')
        if main is not None:
            a.add_artist(main)            # restore the main legend a.legend() displaced
        return leg

    # ── Split footnote: layer-SHAPE markers (unshaded, top) and CASE/style
    # markers (SHADED grey box, bottom) so the two marker kinds read as distinct
    # groups.  The shaded box groups valley/smooth fill + the Valley/Smooth lines.
    shape_h = layer_legend_handles(oro=False, smooth=False, markersize=markersize)
    a.figure.subplots_adjust(bottom=0.30)
    leg_shape = a.legend(handles=shape_h, loc='upper center',
                         bbox_to_anchor=(0.5, -0.15), ncol=min(4, len(shape_h)),
                         fontsize=fontsize, handletextpad=0.3, columnspacing=0.9,
                         frameon=True, framealpha=0.7, facecolor='white',
                         edgecolor='none')
    a.add_artist(leg_shape)
    leg_case = a.legend(handles=case_h, loc='upper center',
                        bbox_to_anchor=(0.5, -0.29), ncol=min(4, len(case_h)),
                        fontsize=fontsize, handletextpad=0.3, columnspacing=0.9,
                        frameon=True, framealpha=0.9, facecolor='0.85',
                        edgecolor='0.4')
    if main is not None:
        a.add_artist(main)                # restore the main line legend
    return leg_case


def streamfunction_2d(U, V, x, y, mask=None):
    """2-D streamfunction psi(x, z) of a spanwise-mean (x, wall-normal) flow.

    For an incompressible 2-D projection with u = +d(psi)/dz and v = -d(psi)/dx,
    psi at fixed x is the wall-normal integral of the streamwise velocity:

        psi[j, i] = integral_0^{y[j]} U[:, i] dz' .

    This is the wall-anchored cumulative integral (psi = 0 at the wall), computed
    column-wise with vIntegral_2d.  V is accepted for API symmetry / future use
    (the continuity-consistent reconstruction) and to document the convention; the
    leading-order streamfunction needs only U.  If `mask` (1 in fluid, 0 in solid)
    is given, U is zeroed in the solid before integrating so the immersed body
    adds no circulation.

    Parameters
    ----------
    U, V : ndarray (ny, nx) — streamwise and wall-normal mean velocity.
    x, y : 1-D coordinates (physical or inner units — psi inherits those units).
    mask : optional (ny, nx) fluid mask (e.g. 1 - eps); applied to U if shapes match.

    Returns
    -------
    psi : ndarray (ny, nx) — streamfunction, psi[0, :] = 0.
    """
    U = np.asarray(U, dtype=np.float64)
    if mask is not None and np.shape(mask) == np.shape(U):
        U = U * mask
    ny = U.shape[0]
    return vIntegral_2d(U, ny, np.asarray(y, dtype=np.float64))


def terrain_follow_remap(field, y, y_surf_x, zeta=None, nzeta=None):
    """Remap a 2-D field (ny, nx) from floor-referenced height y to a
    terrain-following height zeta = (y - local surface elevation) per column.

    Each x-column i is shifted down by its local surface height y_surf_x[i] and
    re-sampled (np.interp) onto a COMMON zeta axis, so a constant-zeta row sits at
    a constant distance ABOVE the local surface rather than a constant distance
    from the domain floor.  This removes the leading-order kinematic crest/valley
    artefact from horizontally-averaged or phase-averaged maps (Research.md Ch. 6,
    Figs. 6.4-6.6 / 6.17 terrain-following re-evaluation).

    Points below the local surface (zeta < 0) and above the original top map to
    NaN (left/right fill), so they are excluded from any subsequent average/plot.

    Parameters
    ----------
    field    : ndarray (ny, nx).
    y        : 1-D wall-normal coordinate (ny,), monotonically increasing.
    y_surf_x : 1-D local surface elevation per column (nx,), same units as y.
    zeta     : optional common terrain-following axis; if None it is built from
               0 to (y[-1] - min(y_surf_x)) with nzeta points.
    nzeta    : number of points for the auto zeta axis (default ny).

    Returns
    -------
    out  : ndarray (len(zeta), nx) — field on the terrain-following axis (NaN fill).
    zeta : 1-D terrain-following axis used.
    """
    field    = np.asarray(field, dtype=np.float64)
    y        = np.asarray(y, dtype=np.float64)
    y_surf_x = np.asarray(y_surf_x, dtype=np.float64)
    ny, nx   = field.shape
    if zeta is None:
        _zmax = float(y[-1] - np.nanmin(y_surf_x))
        zeta  = np.linspace(0.0, max(_zmax, 0.0), int(nzeta or ny))
    out = np.full((zeta.size, nx), np.nan, dtype=np.float64)
    for i in range(nx):
        _zcol = y - y_surf_x[i]                       # height above local surface
        out[:, i] = np.interp(zeta, _zcol, field[:, i], left=np.nan, right=np.nan)
    return out, zeta


###############################################################################
#  Obukhov (1971) stability-corrected surface-layer wind profile
#  "Turbulence in an Atmosphere with a Non-Uniform Temperature",
#   Bound.-Layer Meteorol. 2, 7-29.
# -----------------------------------------------------------------------------
#  Paper-faithful modified wall-law option (Obukhov Section-6 parametric surface-
#  layer profile), used ALONGSIDE the neutral log law so the DNS mean wind can be
#  fitted with the modified law and compared with the neutral log law.  Moved here
#  from PhAvg_rotated.py (it is a self-contained set of helpers).
#
#  All quantities are NON-DIMENSIONAL (the DNS has g = 1, f = 1).
#  Parametric substitution (eq 39, unified stable + unstable branches):
#      xi  = z / L1 = 1/u' - u'^3            (u' = auxiliary parameter)
#      eta = Ri/Ri_cr = 1 - u'^4
#      phi(Ri) = sqrt(1 - eta) = u'^2       (eq 38)
#    u' in (0,1]   -> xi >= 0  STABLE   (Ri > 0, phi < 1, mixing suppressed)
#    u' in [1,inf) -> xi <= 0  UNSTABLE (Ri > 0, phi > 1, mixing enhanced)
#  Wind gradient (eq 22):  sqrt(phi)*k*z*dv/dz = v*  ->  dv/dz = v*/(k z u'),
#    v(z) = (v*/k) * psi(xi),   psi(xi) = int dxi / (xi u').
###############################################################################
# obu_kappa (the paper's fixed von Kármán constant, 0.4) lives in config so every
# constant stays in one place; imported here as the default for the fit helpers.
from config import obu_kappa

# --- monotone lookup table over u' spanning xi in ~[-100, +1e4] --------------
_OBU_U_LO, _OBU_U_HI = 1.0e-4, 4.5
_OBU_U = np.concatenate([
    np.linspace(_OBU_U_HI, 1.0, 4000, endpoint=False),   # unstable side (u'>1)
    np.linspace(1.0, _OBU_U_LO, 12000)])                 # stable side  (u'<=1)
_OBU_XI = 1.0 / _OBU_U - _OBU_U**3                        # ascending along grid

# psi_hat(xi) = psi(xi) - ln|xi|  is finite through xi -> 0.
#   d psi_hat / du' = (1 + 3u'^4)(u' - 1) / (u'^2 (1 - u'^4)) ; limit -1/2 at u'=1
with np.errstate(divide='ignore', invalid='ignore'):
    _OBU_DPSIH = (1.0 + 3.0*_OBU_U**4)*(_OBU_U - 1.0) / (_OBU_U**2 * (1.0 - _OBU_U**4))
_OBU_DPSIH[~np.isfinite(_OBU_DPSIH)] = -0.5
_OBU_I0 = int(np.argmin(np.abs(_OBU_U - 1.0)))
_OBU_PSIH = np.zeros_like(_OBU_U)
_OBU_PSIH[1:] = np.cumsum(0.5*(_OBU_DPSIH[1:] + _OBU_DPSIH[:-1])*np.diff(_OBU_U))
_OBU_PSIH -= _OBU_PSIH[_OBU_I0]                           # psi_hat(xi=0) = 0


def obu_up_of_xi(xi):
    """Auxiliary parameter u'(xi), stable & unstable, via monotone interp."""
    return np.interp(np.asarray(xi, float), _OBU_XI, _OBU_U)


def obu_eta_of_xi(xi):
    """eta(xi) = Ri/Ri_cr = 1 - u'^4  (Table III column 'eta')."""
    return 1.0 - obu_up_of_xi(xi)**4


def obu_psi(xi):
    """Wind function psi(xi) = (k/v*) v = ln|xi| + psi_hat(xi)  (Table III 'psi').

    Additive constant is arbitrary (it is absorbed by the fit's offset / z0);
    as xi -> 0 psi -> ln|xi|, so the modified law reduces to the neutral log law
    in the unstratified limit (L1 -> inf)."""
    xi = np.asarray(xi, float)
    psih = np.interp(xi, _OBU_XI, _OBU_PSIH)
    with np.errstate(divide='ignore'):
        return np.log(np.abs(xi)) + psih


def obu_wind_profile(z, v_star, L1, offset, kappa=obu_kappa):
    """Modified log-law wind  v(z) = (v*/k) psi(z/L1) + offset.
    L1 > 0 stable, L1 < 0 unstable (sign of xi follows sign of L1)."""
    return (v_star/kappa)*obu_psi(np.asarray(z, float)/L1) + offset


def obu_K_unstable(z, u_flux, kappa=obu_kappa):
    """Eq (40) unstable-branch asymptote  K(z) = k^(4/3) (g u)^(1/3) z^(4/3);
    the dimensional group  g*u  -> non-dimensional buoyancy-flux magnitude."""
    return kappa**(4.0/3.0)*np.abs(u_flux)**(1.0/3.0)*np.asarray(z, float)**(4.0/3.0)


def fit_modified_loglaw(z, u, kappa=obu_kappa, L1_0=None):
    """Nonlinear least-squares fit (scipy curve_fit) of the modified log law
        u(z) = (v*/k) psi(z/L1) + offset
    for (v_star, L1, offset).  Returns a dict (or None if too few points).
    Sign of L1_0 seeds the stable(+)/unstable(-) branch."""
    z = np.asarray(z, float); u = np.asarray(u, float)
    good = np.isfinite(z) & np.isfinite(u) & (z > 0)
    z, u = z[good], u[good]
    if z.size < 4:
        return None
    if L1_0 is None:
        L1_0 = 3.0*float(z.max())                        # weak-stratification seed
    _d = obu_psi(z[-1]/L1_0) - obu_psi(z[0]/L1_0)
    v0   = kappa*(u[-1] - u[0]) / (_d if abs(_d) > 1e-6 else 1e-6)
    off0 = u[0] - (v0/kappa)*obu_psi(z[0]/L1_0)

    def _model(zz, vs, L1, off):
        return (vs/kappa)*obu_psi(np.asarray(zz, float)/L1) + off
    try:
        popt, pcov = curve_fit(_model, z, u, p0=[abs(v0), L1_0, off0], maxfev=20000)
    except Exception as _e:
        return {'ok': False, 'err': str(_e)}
    resid  = u - _model(z, *popt)
    ss_res = float(np.sum(resid**2)); ss_tot = float(np.sum((u - u.mean())**2))
    r2 = (1.0 - ss_res/ss_tot) if ss_tot > 0 else float('nan')
    return {'ok': True, 'v_star': float(popt[0]), 'L1': float(popt[1]),
            'offset': float(popt[2]), 'r2': float(r2),
            'perr': np.sqrt(np.diag(pcov)).tolist()}


# Obukhov (1971) Table III — xi, eta(=Ri/Ri_cr), psi.  The printed xi=1.5 row
# (psi = 5.230) breaks the monotone psi(xi) and is a transcription typo; dropped.
OBU_TBL3 = np.array([
 (0.05,0.055,1.600),(0.10,0.102,2.370),(0.15,0.144,2.742),(0.20,0.189,3.065),
 (0.25,0.231,3.320),(0.30,0.278,3.500),(0.35,0.320,3.662),(0.40,0.359,3.803),
 (0.45,0.398,3.928),(0.50,0.435,4.045),(0.55,0.470,4.157),(0.60,0.502,4.258),
 (0.65,0.533,4.360),(0.70,0.565,4.450),(0.75,0.597,4.560),(0.80,0.626,4.608),
 (0.85,0.650,4.695),(0.90,0.677,4.769),(0.95,0.700,4.839),(1.00,0.723,4.908),
 (1.1,0.76,5.03),(1.2,0.80,5.16),(1.3,0.84,5.29),(1.4,0.86,5.41),(1.6,0.90,5.63),
 (1.7,0.92,5.74),(1.8,0.93,5.85),(1.9,0.94,5.95),(2.0,0.95,6.06),(2.1,0.96,6.16),
 (2.2,0.96,6.27),(2.3,0.97,6.37),(2.4,0.97,6.47),(2.5,0.98,6.57),(2.6,0.98,6.68),
 (2.7,0.98,6.78),(2.8,0.98,6.88),(2.9,0.99,6.99),(3.0,0.99,7.09),(3.5,0.99,7.60),
 (4.0,1.00,8.10),(4.5,1.00,8.60),(5.0,1.00,9.10),(5.5,1.00,9.60),(6.0,1.00,10.10)])


def validate_obukhov_tableIII(verbose=True, tol_eta=0.02, tol_psi=0.06):
    """Reproduce Obukhov (1971) Table III (eta and psi vs xi) from the solver.
    Unit-independent (all quantities dimensionless in the paper).  psi carries an
    arbitrary additive constant, so it is compared after a single best-fit shift
    C (= Obukhov's integration constant, ~4.6)."""
    xi, eta_t, psi_t = OBU_TBL3[:, 0], OBU_TBL3[:, 1], OBU_TBL3[:, 2]
    eta_c = obu_eta_of_xi(xi)
    psi_c = obu_psi(xi)
    C     = float(np.mean(psi_t - psi_c))
    e_eta = np.abs(eta_c - eta_t)
    e_psi = np.abs(psi_c + C - psi_t)
    ok = (e_eta.max() < tol_eta) and (float(np.sqrt(np.mean(e_psi**2))) < tol_psi)
    if verbose:
        print(f"[Obukhov Table III] eta(xi): max|err|={e_eta.max():.4f} "
              f"RMS={np.sqrt(np.mean(e_eta**2)):.4f}")
        print(f"[Obukhov Table III] psi(xi): shift C={C:.3f} max|err|={e_psi.max():.4f} "
              f"RMS={np.sqrt(np.mean(e_psi**2)):.4f}")
        print(f"[Obukhov Table III] {'PASS' if ok else 'FAIL'}  "
              f"(k={obu_kappa}, stable+unstable parametric solver)")
    return ok


###############################################################################
#  PhAvg_rotated helpers — interpolation / rotation / derivative bundles
#  Explicit-argument versions of the former nested closures in PhAvg_rotated.py;
#  the call sites wrap them in a zero-arg lambda for load_or_compute caching, so
#  the numerics are byte-for-byte identical to the in-line closures they replace.
###############################################################################
def rotate_pair(a, b, cos_a, sin_a):
    """Proper 2-D vector rotation by angle α (cos_a=cosα, sin_a=sinα):
        a' = a·cosα − b·sinα ,  b' = a·sinα + b·cosα ."""
    return a * cos_a - b * sin_a, a * sin_a + b * cos_a


def compute_ghost_interp(x, y, nx, ny, eps, fields, ghost_depth, n_anchor, smooth_width):
    """Ghost-cell fill each field in `fields`; return the flattened (Fi, Fj) pairs
    in order  [f0_i, f0_j, f1_i, f1_j, …]  (the order the caller unpacks)."""
    out = []
    for f in fields:
        fi, fj = interpolate_component(x, y, nx, ny, eps, f, ghost_depth=ghost_depth,
                                       n_anchor=n_anchor, smooth_width=smooth_width)
        out.extend((fi, fj))
    return tuple(out)


def compute_vel_derivs(cd, U_j, U_i, V_j, V_i, W_j, W_i, mask_intr, dy_method):
    """(∂u/∂y, ∂u/∂x, ∂v/∂y, ∂v/∂x, ∂w/∂y, ∂w/∂x) — y via dy_method, masked."""
    return (cd.ddy(U_j, method=dy_method) * mask_intr,
            cd.ddx(U_i) * mask_intr,
            cd.ddy(V_j, method=dy_method) * mask_intr,
            cd.ddx(V_i) * mask_intr,
            cd.ddy(W_j, method=dy_method) * mask_intr,
            cd.ddx(W_i) * mask_intr)


def compute_disp_derivs(cd, DU, DV, DW, mask_intr, dy_method):
    """Dispersive-velocity gradients (dud_dy, dvd_dy, dwd_dy, dud_dx, dvd_dx, dwd_dx)."""
    return (cd.ddy(DU, method=dy_method) * mask_intr,
            cd.ddy(DV, method=dy_method) * mask_intr,
            cd.ddy(DW, method=dy_method) * mask_intr,
            cd.ddx(DU) * mask_intr,
            cd.ddx(DV) * mask_intr,
            cd.ddx(DW) * mask_intr)


def compute_misc_derivs(cd, U_i, U_j, rey_uu, rey_uv, P_i, P_j, mask_intr, dy_method, d2y_method):
    """(∂²ū/∂x², ∂²ū/∂y², ∂rey_uu/∂x, ∂rey_uv/∂y, ∂P/∂x, ∂P/∂y) — masked."""
    return (cd.d2dx2(U_i) * mask_intr,
            cd.d2dy2(U_j, method=d2y_method) * mask_intr,
            cd.ddx(rey_uu) * mask_intr,
            cd.ddy(rey_uv, method=dy_method) * mask_intr,
            cd.ddx(P_i) * mask_intr,
            cd.ddy(P_j, method=dy_method) * mask_intr)


###############################################################################
#  PhAvg_rotated research-diagnostic scale helpers (stratification / Goal 1,3)
###############################################################################
def obukhov_length(us, bs, kappa, flux_eps=1e-12):
    """Obukhov length  L = -u*^3 / (kappa * B_s);  +inf in the neutral limit."""
    if (not np.isfinite(bs)) or abs(bs) < flux_eps or us <= 0:
        return float('inf')
    return -us**3 / (kappa * bs)


def local_obukhov_length(i, eps_hgt, ny, u_star_loc, vtheta_tot, kappa, flux_eps=1e-12):
    """Physical local Obukhov length at x-column i (evaluated at the local BL top)."""
    js = int(min(eps_hgt[i], ny - 1))
    return obukhov_length(float(u_star_loc[i]), float(vtheta_tot[js, i]), kappa, flux_eps)


def stability_class(ri, ri_b_bins):
    """Stability-class label from a bulk/gradient Richardson number `ri`."""
    if (not np.isfinite(ri)) or abs(ri) < 1e-6:
        return 'neutral'
    if ri < ri_b_bins[0]:
        return 'weakly stable'
    if ri < ri_b_bins[1]:
        return 'intermediately stable'
    return 'strongly stable'


def bl_scales(us, f, L_x, H_phys, nu):
    """Boundary-layer scale set for friction velocity `us`:
    depth δ=us/f, Ψ=Lx/(2δ), H/δ, H⁺=H·us/ν, Lx⁺=Lx·us/ν."""
    d = us / f
    return {'u_star': float(us), 'delta': float(d),
            'Psi':     float(L_x / (2.0 * d)),
            'H_delta': float(H_phys / d),
            'H_plus':  float(H_phys * us / nu),
            'Lx_plus': float(L_x * us / nu)}
