#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buckingham_pi.py  --  Dimensional-mapping tool for the tlab Ekman-over-valley DNS.

Every quantity in the simulation is non-dimensional.  This standalone tool turns
the fixed Buckingham-Pi groups of the run into a real-world mapping: pick any TWO
anchor quantities from {G, latitude(->f), L0, nu}, and every other physical scale
and derived boundary-layer quantity follows from the matched Pi groups.

    Fixed Pi groups (from tlab.ini / config.py, see DimensionalAnalysis.md):
        Re = U0 L0 / nu            = 125000   (= 1/2 Re_D^2, Re_D = 500)
        Ro = U0 / (L0 f)           = 1        -> pins  L0 = G / f
        Fr = U0^2 / (b0 L0)        = inf,1,0.1,0.01   -> N = f / sqrt(Fr)
        Sc = nu / kappa            = 1
        alpha (rotation angle)     = -0.430511 rad

Because Ro = 1 removes one length degree of freedom and Re fixes the viscosity,
the dimensional family {G, f, L0, nu} has exactly TWO free degrees of freedom:
fix any two, the other two (and all scales below) are determined.

Each slider is shaded with the range a real *terrestrial* atmosphere can occupy
(green = Earth, red = off-Earth) and a live ON EARTH / OFF EARTH verdict is shown
-- SOFT guides only, the sliders are never clamped.  Off-Earth just means the
mapping is a rotating-tank lab or another planet, not Earth's air.  The verdict
also reports the lowest Froude number Earth's stratification allows at the current
latitude, Fr_min = (f/N_max)^2  (see DimensionalAnalysis.md sec. 8).

Usage
-----
    python3 buckingham_pi.py                 # interactive sliders (needs a GUI backend)
    python3 buckingham_pi.py --static        # headless: print table + save fig PNG
    python3 buckingham_pi.py --G 10 --lat 45 --Fr 0.01 --static
    python3 buckingham_pi.py --anchors G,nu --G 10 --nu 1.5e-5 --static
    python3 buckingham_pi.py --data-dir /path/to/case    # where the .npy live

Reads only config.py + this run's local .npy (y, AvgPhU, AvgScal, eps_save) --
no cluster data required.  The case directory is auto-located (cwd, then the
script/symlink dir, then the repo root) or given explicitly with --data-dir;
when none is found the profile panels show a notice instead of staying blank.
If no GUI window can be opened the tool falls back to --static automatically
(probed in a subprocess: Qt aborts the whole process when no display is
usable, which a Python except cannot catch).  All def's are grouped at the
top (repo convention).
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from functions import avg_c

# ============================================================================
# Environmental / physical constants  (edit these for a different fluid/planet)
# ============================================================================
OMEGA   = 7.2921159e-5    # Earth angular velocity              [rad/s]
G_GRAV  = 9.81            # gravitational acceleration          [m/s^2]
T0_REF  = 298.0           # reference temperature (tlab TREF)   [K]
RHO0    = 1.225           # reference air density               [kg/m^3]
NU_AIR  = 1.5e-5          # molecular kinematic visc. of air    [m^2/s]  (reference)

# ============================================================================
# Fixed Pi groups of the simulation  (source of truth = config.py / tlab.ini)
# ============================================================================
RE_PI   = float(config.Re_lambda)   # U0 L0 / nu  = 125000   (tlab "Reynolds")
RO_PI   = 1.0                        # U0 / (L0 f) = 1        (tlab "Rossby")
SC_PI   = 1.0                        # nu / kappa  = 1        (tlab "Schmidt")
ALPHA   = float(config.alpha)        # rotation angle                     [rad]
USTAR_C = float(config.u_star)       # friction velocity in code units (u*/G)
NU_CODE = float(config.nu)           # = 1 / RE_PI  (code-unit viscosity)

FR_LADDER = [np.inf, 1.0, 0.1, 0.01]      # the four simulated Froude cases

# Domain extents in code units (tlab.ini [IniGridO*] scales_1); fallback to arrays.
LX_CODE_DEFAULT = 0.2652427184466019
LY_CODE_DEFAULT = 0.3505250551104695
LZ_CODE_DEFAULT = 0.2652427184466019

ANCHOR_KEYS  = ('G', 'f', 'L0', 'nu')
ANCHOR_PAIRS = [('G', 'f'), ('G', 'L0'), ('G', 'nu'),
                ('f', 'L0'), ('f', 'nu'), ('L0', 'nu')]

# ============================================================================
# Earth-atmosphere feasibility envelope  (SOFT visual guides, NOT clamps)
# ============================================================================
# For every dimensional quantity the sliders expose, the range that a real
# *terrestrial* atmospheric Ekman layer can actually occupy.  These are drawn on
# the interactive window (green = Earth, red = off-Earth) and drive a live
# "on-Earth / off-Earth" verdict — but they never limit the sliders: sliding
# outside just means the mapped flow is no longer Earth's atmosphere.  It may
# still be a perfectly valid flow in a rotating-tank lab or on another planet
# (a faster-spinning body raises the f ceiling; a different fluid moves nu).
#
#   lo/hi      : the "practical Earth" band, shaded green on each slider.
#   hard_lo/hi : the absolute terrestrial limit; crossing it flags OFF-EARTH.
# Physical basis of each ceiling is given inline.
EARTH = {
    # geostrophic wind: green = typical winds (light breeze .. jet-stream core);
    # hard_lo down to near-calm 0.05 m/s (a real, if very light, terrestrial wind).
    'G':  dict(lo=1.0, hi=50.0, hard_lo=0.05, hard_hi=100.0, unit='m/s'),
    # Coriolis f = 2 Omega sin(lat): latitude in [0,90] => f in [0, 2 Omega].
    # A larger f can only come from a faster-rotating planet.
    'f':  dict(hard_hi=2.0 * OMEGA, unit='1/s'),
    # reference length L0 = G/f (Rossby radius).  With Ro=1 the turbulent BL
    # depth is delta = u*/f = USTAR_C * L0, so L0 = delta / USTAR_C.  Earth: the
    # turbulent Ekman layer runs from a ~30 m thin stable layer up to the
    # troposphere top ~15 km (absolute ceiling ~20 km); below ~1 m it is a lab.
    'L0': dict(lo=30.0 / USTAR_C, hi=15.0e3 / USTAR_C,
               hard_lo=1.0 / USTAR_C, hard_hi=20.0e3 / USTAR_C, unit='m'),
    # kinematic viscosity: a *real fluid's molecular* value.  Water 1e-6 ..
    # sea-level air 1.5e-5 .. thin upper-atmosphere air ~1e-3.  Anything above
    # ~1e-3 is eddy-viscosity magnitude, i.e. a Reynolds-reduced model, not air.
    'nu': dict(lo=1.0e-6, hi=1.0e-3, hard_lo=1.0e-7, hard_hi=1.0e-3, unit='m^2/s'),
    # buoyancy frequency N = f/sqrt(Fr).  Neutral .. strongest *sustained* Earth
    # inversion ~0.05 rad/s; the sharpest thin inversions reach ~0.1 (extreme).
    'N':  dict(lo=0.0, hi=0.05, hard_hi=0.1, unit='rad/s'),
}


# ---------------------------------------------------------------------------
# f  <->  latitude
# ---------------------------------------------------------------------------
def f_from_lat(lat_deg):
    """Coriolis parameter f = 2 Omega sin(latitude)."""
    return 2.0 * OMEGA * np.sin(np.deg2rad(lat_deg))


def lat_from_f(f):
    """Inverse of f_from_lat, clipped to a valid arcsin argument."""
    s = np.clip(f / (2.0 * OMEGA), -1.0, 1.0)
    return np.rad2deg(np.arcsin(s))


# ---------------------------------------------------------------------------
# The 2-DOF closure: given exactly two of {G, f, L0, nu}, solve the other two.
# Uses the fixed Pi groups:  L0 = G/f (Ro=1)  and  nu = G*L0/Re (Re fixed).
# ---------------------------------------------------------------------------
def solve_gflnu(known):
    """known: dict with EXACTLY two of {'G','f','L0','nu'} -> full dict of all four.

    Closed forms for the six anchor pairs (Ro=1, Re=RE_PI):
        L0 = G/f ,  nu = G*L0/Re = G^2/(f*Re)
    """
    keys = tuple(sorted(known.keys()))
    Re = RE_PI

    if keys == ('L0', 'f'):
        f, L0 = known['f'], known['L0']
        G  = f * L0
        nu = G * L0 / Re
    elif keys == ('G', 'f'):
        G, f = known['G'], known['f']
        L0 = G / f
        nu = G * L0 / Re
    elif keys == ('G', 'L0'):
        G, L0 = known['G'], known['L0']
        f  = G / L0
        nu = G * L0 / Re
    elif keys == ('G', 'nu'):
        G, nu = known['G'], known['nu']
        L0 = Re * nu / G
        f  = G / L0
    elif keys == ('f', 'nu'):
        f, nu = known['f'], known['nu']
        L0 = np.sqrt(Re * nu / f)        # nu = f*L0^2/Re
        G  = f * L0
    elif keys == ('L0', 'nu'):
        L0, nu = known['L0'], known['nu']
        G  = Re * nu / L0
        f  = G / L0
    else:
        raise ValueError(f"need exactly two distinct anchors from "
                         f"{ANCHOR_KEYS}, got {list(known.keys())}")

    return {'G': float(G), 'f': float(f), 'L0': float(L0), 'nu': float(nu)}


# ---------------------------------------------------------------------------
# Full scale + derived-quantity mapping for a given (G,f,L0,nu) and Froude number
# ---------------------------------------------------------------------------
def compute_mapping(gflnu, Fr):
    """Return an ordered list of (label, value, unit) tuples: conversion factors
    (units of one code unit) followed by derived physical quantities.

    Everything is a plain float in SI.  Fr = np.inf -> neutral (no buoyancy)."""
    G, f, L0, nu = gflnu['G'], gflnu['f'], gflnu['L0'], gflnu['nu']
    stratified = np.isfinite(Fr) and Fr > 0.0

    # --- unit scales (value of "1 code unit" in SI) ------------------------
    U_scale    = G                       # velocity   [m/s]
    L_scale    = L0                      # length     [m]
    T_scale    = 1.0 / f                 # time       [s]   (= L0/U0, Ro=1)
    vort_scale = f                       # vorticity  [1/s]
    tau_scale  = RHO0 * G**2             # stress/pressure [Pa]
    b_scale    = G * f / Fr if stratified else 0.0   # buoyancy [m/s^2] = U0^2/(Fr L0)
    dTheta     = b_scale * T0_REF / G_GRAV           # temperature [K] (b = g alpha_T dT)

    # --- stratification ----------------------------------------------------
    N          = f / np.sqrt(Fr) if stratified else 0.0     # buoyancy freq [rad/s]
    N_period   = 2.0 * np.pi / N if N > 0 else np.inf        # [s]

    # --- boundary-layer quantities ----------------------------------------
    D_ekman    = np.sqrt(2.0 * nu / f)   # laminar Ekman depth  sqrt(2 nu/f) [m]
    u_star     = USTAR_C * G             # friction velocity                 [m/s]
    delta_BL   = u_star / f              # turbulent BL depth  u*/f          [m]
    wall_unit  = nu / u_star             # viscous length l+ = nu/u*         [m]
    Re_tau     = u_star * delta_BL / nu  # friction Reynolds (dimensionless invariant)
    Ri_B       = (N * delta_BL / G)**2 if stratified else 0.0   # bulk Richardson

    # --- domain / time -----------------------------------------------------
    T_inertial = 2.0 * np.pi / f         # inertial period 2 pi / f          [s]

    rows = [
        ('--- anchors (G, f, L0, nu) ---', None, ''),
        ('Geostrophic wind  G',            G,          'm/s'),
        ('Coriolis f (latitude %.1f deg)' % lat_from_f(f), f, '1/s'),
        ('Reference length  L0 = G/f',     L0,         'm'),
        ('Kinematic viscosity  nu',        nu,         'm^2/s'),
        ('--- unit conversion factors (1 code unit =) ---', None, ''),
        ('velocity  U0',                   U_scale,    'm/s'),
        ('length    L0',                   L_scale,    'm'),
        ('time      T0 = 1/f',             T_scale,    's'),
        ('vorticity 1/T0',                 vort_scale, '1/s'),
        ('stress    rho0 U0^2',            tau_scale,  'Pa'),
        ('buoyancy  b0 = U0^2/(Fr L0)',    b_scale,    'm/s^2'),
        ('temperature  dTheta = b0 T0/g',  dTheta,     'K'),
        ('--- derived boundary-layer quantities ---', None, ''),
        ('friction velocity  u* = %.4f G' % USTAR_C, u_star, 'm/s'),
        ('Ekman depth  D = sqrt(2 nu/f)',  D_ekman,    'm'),
        ('turbulent BL depth  delta = u*/f', delta_BL, 'm'),
        ('viscous wall unit  l+ = nu/u*',  wall_unit,  'm'),
        ('friction Reynolds  Re_tau',      Re_tau,     '-'),
        ('inertial period  2 pi/f',        T_inertial, 's'),
        ('--- stratification (Fr = %s) ---' % _fmt_fr(Fr), None, ''),
        ('buoyancy frequency  N = f/sqrt(Fr)', N,      'rad/s'),
        ('buoyancy period  2 pi/N',        N_period,   's'),
        ('N / f',                          (N / f) if stratified else 0.0, '-'),
        ('bulk Richardson  Ri_B',          Ri_B,       '-'),
    ]
    return rows


# ---------------------------------------------------------------------------
# Load this run's actual profiles (code units) via intrinsic x-average
# ---------------------------------------------------------------------------
_REQUIRED_NPY = ('y.npy', 'eps_save.npy')


def _find_data_dir(explicit=None):
    """Locate the case directory holding this run's .npy files.

    Search order: --data-dir (if given), cwd, the script/symlink dir (repo
    convention: a symlinked script lives IN its case dir), then the repo root
    above the real MyPyLib location.  Returns (dir_or_None, searched_dirs).
    Fixes the launch-dir dependence that left the profile panels blank when
    the tool was started from anywhere but a case directory."""
    here = os.path.dirname(os.path.abspath(__file__))            # symlink dir
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    cands = ([explicit] if explicit else []) + [os.getcwd(), here, root]
    searched = []
    for d in cands:
        d = os.path.abspath(d)
        if d in searched:
            continue
        searched.append(d)
        if all(os.path.exists(os.path.join(d, f)) for f in _REQUIRED_NPY):
            return d, searched
    return None, searched


def load_profiles(base=None):
    """x-average AvgPhU / AvgScal to 1-D profiles vs y (code units).

    base=None auto-locates the case dir (see _find_data_dir).  Returns dict
    {y, U, b, dir, searched}; entries stay None for absent files.  Solid
    cells are zeroed with mask0 = 1-eps before avg_c (fluid-only average),
    exactly as flow_params.py does."""
    base, searched = _find_data_dir(base)
    out = {'y': None, 'U': None, 'b': None, 'dir': base, 'searched': searched}
    if base is None:
        print('  [load_profiles] no %s found in any of: %s'
              % (' + '.join(_REQUIRED_NPY), ', '.join(searched)))
        print('  [load_profiles] profile panels will show a notice; '
              'run from a case dir or pass --data-dir')
        return out
    print('  [load_profiles] reading profiles from %s' % base)
    try:
        eps = np.load(os.path.join(base, 'eps_save.npy'))
        mask0 = 1.0 - eps
        out['y'] = np.load(os.path.join(base, 'y.npy'))
        upath = os.path.join(base, 'AvgPhU.npy')
        if os.path.exists(upath):
            out['U'] = avg_c(eps, np.load(upath) * mask0, axis=1)
        bpath = os.path.join(base, 'AvgScal.npy')
        if os.path.exists(bpath):
            out['b'] = avg_c(eps, np.load(bpath) * mask0, axis=1)
    except Exception as exc:                       # pragma: no cover
        print('  [load_profiles] could not read local profiles: %s' % exc)
    return out


def _no_profile_msg(prof):
    """Explanatory text for an empty profile panel (never leave it blank)."""
    if prof.get('dir') is None:
        return ('no local .npy profiles found\nsearched:\n%s\n'
                'run from a case dir or use --data-dir'
                % '\n'.join(prof.get('searched', [])))
    return 'AvgPhU.npy / AvgScal.npy absent in\n%s' % prof['dir']


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_fr(Fr):
    return 'inf (neutral)' if not np.isfinite(Fr) else ('%g' % Fr)


def _fmt_val(v, unit):
    """Human-friendly SI value with an auto-scaled companion for length/time."""
    if v is None:
        return ''
    if not np.isfinite(v):
        return '   inf'
    s = '%12.4g %s' % (v, unit)
    if unit == 'm' and abs(v) >= 1000:
        s += '   (%.3g km)' % (v / 1000.0)
    elif unit == 'm' and 0 < abs(v) < 0.01:
        s += '   (%.3g mm)' % (v * 1000.0)
    elif unit == 's' and abs(v) >= 3600:
        s += '   (%.3g h)' % (v / 3600.0)
    elif unit == 's' and 60 <= abs(v) < 3600:
        s += '   (%.3g min)' % (v / 60.0)
    return s


def format_table(rows):
    """Render the mapping rows as a printable text block."""
    lines = []
    for label, val, unit in rows:
        if val is None:                            # section header
            lines.append('')
            lines.append(label)
        else:
            lines.append('  %-34s %s' % (label, _fmt_val(val, unit)))
    return '\n'.join(lines)


def print_pi_summary():
    """Print the fixed Pi groups (constant across every mapping)."""
    print('=' * 72)
    print(' Fixed Buckingham-Pi groups of the DNS (matched exactly)')
    print('=' * 72)
    print('  Re = U0 L0 / nu      = %g   (= 1/2 Re_D^2, Re_D = %d)'
          % (RE_PI, int(round(np.sqrt(2 * RE_PI)))))
    print('  Ro = U0 / (L0 f)     = %g   -> L0 = G/f (Rossby radius)' % RO_PI)
    print('  Sc = nu / kappa      = %g' % SC_PI)
    print('  alpha (rot. angle)   = %.6f rad  (%.2f deg)'
          % (ALPHA, np.rad2deg(ALPHA)))
    print('  u*/G (code units)    = %.4f' % USTAR_C)
    print('  Re_tau = u*^2/(f nu) = %.1f  (dimensionless invariant)'
          % (USTAR_C**2 / (RO_PI * NU_CODE)))


# ---------------------------------------------------------------------------
# Earth-atmosphere feasibility: is THIS mapping a real terrestrial flow?
# ---------------------------------------------------------------------------
def check_earth_feasibility(gflnu, Fr):
    """Test a solved (G,f,L0,nu) + Froude number against the EARTH envelope.

    Returns a dict:
        ok         : bool  -- all hard terrestrial limits satisfied
        fails      : list  -- human-readable reasons it is off-Earth (empty if ok)
        N          : float -- buoyancy frequency f/sqrt(Fr) [rad/s] (0 if neutral)
        fr_floor   : float -- lowest Fr Earth's stratification allows at THIS f,
                              using the strong-inversion N=0.05  (= (f/0.05)^2)
        fr_floor_ext : float -- same with the extreme N=0.1 ceiling
    Nothing here clamps anything; it only reports."""
    G, f, L0, nu = gflnu['G'], gflnu['f'], gflnu['L0'], gflnu['nu']
    strat = np.isfinite(Fr) and Fr > 0.0
    N = f / np.sqrt(Fr) if strat else 0.0
    fails = []
    if not (EARTH['G']['hard_lo'] <= G <= EARTH['G']['hard_hi']):
        fails.append('G=%.3g m/s off Earth' % G)
    if f > EARTH['f']['hard_hi'] * (1.0 + 1e-9):
        fails.append('lat>90 (f=%.3g/s: rotating tank)' % f)
    if not (EARTH['L0']['hard_lo'] <= L0 <= EARTH['L0']['hard_hi']):
        fails.append('BL depth %.2g m off Earth' % (USTAR_C * L0))
    if not (EARTH['nu']['hard_lo'] <= nu <= EARTH['nu']['hard_hi']):
        fails.append('nu=%.2g m2/s (eddy-visc, not a real fluid)' % nu)
    if N > EARTH['N']['hard_hi']:
        fails.append('N=%.3g rad/s > Earth extreme 0.1' % N)
    return dict(ok=(not fails), fails=fails, N=N, strat=strat,
                fr_floor=(f / EARTH['N']['hi']) ** 2,
                fr_floor_ext=(f / EARTH['N']['hard_hi']) ** 2)


def earth_status_lines(gflnu, Fr):
    """Two-line verdict string + a light-green/light-red colour for a status box."""
    v = check_earth_feasibility(gflnu, Fr)
    if v['ok']:
        head = 'ON EARTH  ✓   this mapping is a real terrestrial atmospheric flow'
        colour = '#d8f5d8'
    else:
        head = 'OFF EARTH ✗   ' + ' ; '.join(v['fails'])
        colour = '#f7d6d6'
    if v['strat']:
        second = ('N=%.2e rad/s (Earth max ~0.05, extreme 0.1) | '
                  'lowest Earth Fr here: %.1e .. %.1e (extreme)'
                  % (v['N'], v['fr_floor'], v['fr_floor_ext']))
    else:
        second = 'neutral (Fr = inf): no stratification, Froude limit N/A'
    return head + '\n' + second, colour


def _earth_shade(ax, lo, hi):
    """Paint an axes track red (off-Earth) then green over its Earth band [lo,hi].

    Coordinates are the slider's own (linear for G/lat, log10 for L0/nu).  Drawn
    behind the slider's value bar (low zorder); xlim is preserved."""
    x0, x1 = ax.get_xlim()
    ax.axvspan(x0, x1, facecolor='#e7b3b3', alpha=0.35, lw=0, zorder=0.0)
    g0, g1 = max(lo, x0), min(hi, x1)
    if g1 > g0:
        ax.axvspan(g0, g1, facecolor='#77cc77', alpha=0.45, lw=0, zorder=0.1)
    ax.set_xlim(x0, x1)


def draw_earth_bands(sliders):
    """Shade the Earth-feasible band on each of the four sliders (green/red) and
    mark the reference molecular viscosities of air and water on the nu slider."""
    _earth_shade(sliders['G'].ax,  EARTH['G']['lo'],  EARTH['G']['hi'])
    _earth_shade(sliders['f'].ax,  0.0,               90.0)          # all Earth
    _earth_shade(sliders['L0'].ax, np.log10(EARTH['L0']['lo']),
                                   np.log10(EARTH['L0']['hi']))
    _earth_shade(sliders['nu'].ax, np.log10(EARTH['nu']['lo']),
                                   np.log10(EARTH['nu']['hi']))
    nu_ax = sliders['nu'].ax
    nu_ax.axvline(np.log10(NU_AIR), color='0.25', lw=1.0, ls=':', zorder=0.2)
    nu_ax.axvline(np.log10(1.0e-6), color='0.25', lw=0.8, ls=':', zorder=0.2)


# ---------------------------------------------------------------------------
# Static / headless run: print the table and save a dashboard PNG
# ---------------------------------------------------------------------------
def run_static(anchors, values, Fr, savepath='fig/buckingham_dashboard.png',
               data_dir=None):
    known = {k: values[k] for k in anchors}
    gflnu = solve_gflnu(known)
    rows = compute_mapping(gflnu, Fr)

    print_pi_summary()
    print('\n' + '=' * 72)
    print(' Real-world mapping   anchors = %s,  Fr = %s' % (anchors, _fmt_fr(Fr)))
    print('=' * 72)
    print(format_table(rows))

    verdict, _ = earth_status_lines(gflnu, Fr)
    print('\n' + '-' * 72)
    print(' EARTH FEASIBILITY')
    print('-' * 72)
    for ln in verdict.split('\n'):
        print('  ' + ln)

    prof = load_profiles(data_dir)
    _save_dashboard(rows, gflnu, Fr, prof, savepath)
    return gflnu, rows


def _dimensional_profiles(prof, gflnu, Fr):
    """Convert code-unit profiles to SI: z[m], U[m/s], dTheta[K]."""
    if prof is None or prof['y'] is None:
        return None
    G, L0, f = gflnu['G'], gflnu['L0'], gflnu['f']
    stratified = np.isfinite(Fr) and Fr > 0.0
    b_scale = G * f / Fr if stratified else 0.0
    dTheta_scale = b_scale * T0_REF / G_GRAV
    z = prof['y'] * L0
    U = prof['U'] * G if prof['U'] is not None else None
    # buoyancy scalar (units of b0) -> temperature perturbation about its top value
    if prof['b'] is not None and dTheta_scale != 0.0:
        b = prof['b'] - prof['b'][-1]
        Th = b * dTheta_scale
    else:
        Th = None
    return {'z': z, 'U': U, 'Th': Th}


def _save_dashboard(rows, gflnu, Fr, prof, savepath):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 7))
    ax_txt = fig.add_axes([0.02, 0.02, 0.46, 0.96]); ax_txt.axis('off')
    ax_txt.text(0.0, 1.0, format_table(rows),
                family='monospace', fontsize=8, va='top', ha='left')

    dp = _dimensional_profiles(prof, gflnu, Fr)
    axU = fig.add_axes([0.55, 0.58, 0.42, 0.37])
    axT = fig.add_axes([0.55, 0.10, 0.42, 0.37])
    if dp is not None and dp['U'] is not None:
        axU.plot(dp['U'], dp['z'], 'b-')
    else:
        axU.text(.5, .5, _no_profile_msg(prof), ha='center', va='center',
                 transform=axU.transAxes, fontsize=8, color='0.35')
    axU.set_xlabel('mean wind U [m/s]'); axU.set_ylabel('z [m]')
    axU.set_title('mean wind  U(z)   (Fr = %s)' % _fmt_fr(Fr))
    axU.grid(alpha=.3)
    if dp is not None and dp['Th'] is not None:
        axT.plot(dp['Th'], dp['z'], 'r-')
        axT.set_title('buoyancy scalar as temperature  dTheta(z)')
    else:
        axT.text(.5, .5, 'no buoyancy (Fr = inf)\nor scalar file absent'
                 if dp is not None else _no_profile_msg(prof),
                 ha='center', va='center', transform=axT.transAxes,
                 fontsize=8, color='0.35')
        axT.set_title('temperature perturbation')
    axT.set_xlabel('dTheta [K]'); axT.set_ylabel('z [m]'); axT.grid(alpha=.3)

    os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True)
    fig.savefig(savepath, dpi=120)
    plt.close(fig)
    print('\n  dashboard saved -> %s' % savepath)


# ---------------------------------------------------------------------------
# Interactive run: 4 sliders (G, latitude, log10 L0, log10 nu) + anchor radios
# ---------------------------------------------------------------------------
def run_interactive(values, Fr0, anchors=('G', 'f'), data_dir=None):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons, Button

    prof = load_profiles(data_dir)
    state = {'anchors': tuple(anchors), 'Fr': Fr0, 'updating': False}

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('tlab Ekman DNS  ->  real-world dimensional mapping '
                 '(pick 2 anchors, slide, read the rest)', fontsize=11)

    ax_txt = fig.add_axes([0.02, 0.30, 0.44, 0.60]); ax_txt.axis('off')
    txt = ax_txt.text(0.0, 1.0, '', family='monospace', fontsize=8,
                      va='top', ha='left')

    # live Earth-feasibility verdict (updated in refresh), in the top strip
    ax_status = fig.add_axes([0.02, 0.905, 0.52, 0.052]); ax_status.axis('off')
    status_txt = ax_status.text(
        0.0, 0.5, '', family='monospace', fontsize=7.5, va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.6'))

    # profile panels: right column, x >= 0.56.  ALL controls stay left of
    # x = 0.55 -- the old layout drew the Fr radio box and the reset button
    # on top of the temperature axes.
    axU = fig.add_axes([0.585, 0.57, 0.385, 0.35])
    axT = fig.add_axes([0.585, 0.09, 0.385, 0.35])
    lineU, = axU.plot([], [], 'b-'); axU.set_xlabel('U [m/s]'); axU.set_ylabel('z [m]')
    axU.set_title('mean wind U(z)'); axU.grid(alpha=.3)
    lineT, = axT.plot([], [], 'r-'); axT.set_xlabel('dTheta [K]'); axT.set_ylabel('z [m]')
    axT.set_title('buoyancy scalar as temperature'); axT.grid(alpha=.3)
    # a panel is never silently blank: these carry the reason when empty
    msgU = axU.text(.5, .5, '', ha='center', va='center',
                    transform=axU.transAxes, fontsize=8, color='0.35')
    msgT = axT.text(.5, .5, '', ha='center', va='center',
                    transform=axT.transAxes, fontsize=8, color='0.35')

    # sliders: raw slider value == physical for G/lat, log10 for L0/nu; the
    # value text right of each slider always shows the PHYSICAL number
    sax_G   = fig.add_axes([0.075, 0.225, 0.25, 0.03])
    sax_lat = fig.add_axes([0.075, 0.175, 0.25, 0.03])
    sax_L0  = fig.add_axes([0.075, 0.125, 0.25, 0.03])
    sax_nu  = fig.add_axes([0.075, 0.075, 0.25, 0.03])
    s_G   = Slider(sax_G,   'G [m/s]',   0.5, 40.0, valinit=values['G'])
    s_lat = Slider(sax_lat, 'lat [deg]', 5.0, 89.0,
                   valinit=lat_from_f(values['f']))
    s_L0  = Slider(sax_L0,  'L0 [m]',    -2.0, 5.5,
                   valinit=np.log10(values['L0']))
    s_nu  = Slider(sax_nu,  'nu [m2/s]', -6.0, 2.0,
                   valinit=np.log10(values['nu']))
    sliders = {'G': s_G, 'f': s_lat, 'L0': s_L0, 'nu': s_nu}

    # shade each slider's Earth-feasible band (green) vs off-Earth (red); soft
    # guides only — the sliders themselves are never limited.
    draw_earth_bands(sliders)
    fig.text(0.02, 0.018,
             'L0/nu sliders move in log10 steps (value shown is physical).  '
             'shading: green = Earth atmosphere, red = off-Earth '
             '(rotating-tank lab / another planet).  nu dots: air, water.',
             fontsize=7.5, color='0.35')

    def slider_phys(key):
        if key == 'G':
            return s_G.val
        if key == 'f':
            return f_from_lat(s_lat.val)
        if key == 'L0':
            return 10.0**s_L0.val
        if key == 'nu':
            return 10.0**s_nu.val

    def set_slider_phys(key, val):
        """Push a derived value onto its (display-only) slider, clamped to the
        track so the bar never draws outside its axes; the value text shows
        the exact physical number regardless (set in refresh)."""
        s = sliders[key]
        raw = {'G': lambda v: v, 'f': lat_from_f,
               'L0': np.log10, 'nu': np.log10}[key](val)
        s.set_val(float(np.clip(raw, s.valmin, s.valmax)))

    # anchor-pair + Fr selectors + reset, grouped in the bottom-centre strip
    # (left of the profile panels).  Initial selection honours the CLI.
    rax_pair = fig.add_axes([0.40, 0.045, 0.065, 0.21])
    rax_pair.set_title('anchors', fontsize=8)
    rb_pair = RadioButtons(rax_pair, ['%s,%s' % p for p in ANCHOR_PAIRS],
                           active=ANCHOR_PAIRS.index(state['anchors']))
    fr_labels = ['inf' if not np.isfinite(v) else '%g' % v for v in FR_LADDER]
    if np.isfinite(Fr0) and not any(lbl != 'inf' and np.isclose(float(lbl), Fr0)
                                    for lbl in fr_labels):
        fr_labels.append('%g' % Fr0)               # off-ladder --Fr still shown
    fr_active = next(i for i, lbl in enumerate(fr_labels)
                     if (lbl == 'inf') == (not np.isfinite(Fr0))
                     and (lbl == 'inf' or np.isclose(float(lbl), Fr0)))
    rax_fr = fig.add_axes([0.475, 0.04, 0.05, 0.16])
    rax_fr.set_title('Fr', fontsize=8)
    rb_fr = RadioButtons(rax_fr, fr_labels, active=fr_active)
    bax = fig.add_axes([0.475, 0.225, 0.05, 0.035]); btn = Button(bax, 'reset')

    def refresh(_=None):
        if state['updating']:
            return
        state['updating'] = True
        try:
            a = state['anchors']
            known = {k: slider_phys(k) for k in a}
            gflnu = solve_gflnu(known)
            for k in ANCHOR_KEYS:               # push derived values onto the other sliders
                if k not in a:
                    set_slider_phys(k, gflnu[k])
            for k in ANCHOR_KEYS:               # grey the derived sliders, colour the active ones
                sliders[k].poly.set_color('tab:blue' if k in a else '0.7')
            # value texts: exact physical numbers (not raw log10 positions)
            s_G.valtext.set_text('%.3g' % gflnu['G'])
            s_lat.valtext.set_text('%.1f' % lat_from_f(gflnu['f']))
            s_L0.valtext.set_text('%.3g' % gflnu['L0'])
            s_nu.valtext.set_text('%.3g' % gflnu['nu'])
            rows = compute_mapping(gflnu, state['Fr'])
            txt.set_text(format_table(rows))
            verdict, colour = earth_status_lines(gflnu, state['Fr'])
            status_txt.set_text(verdict)
            status_txt.get_bbox_patch().set_facecolor(colour)
            dp = _dimensional_profiles(prof, gflnu, state['Fr'])
            if dp is not None and dp['U'] is not None:
                msgU.set_text('')
                lineU.set_data(dp['U'], dp['z']); axU.relim(); axU.autoscale_view()
            else:
                lineU.set_data([], [])
                msgU.set_text(_no_profile_msg(prof))
            if dp is not None and dp['Th'] is not None:
                msgT.set_text('')
                lineT.set_data(dp['Th'], dp['z']); axT.relim(); axT.autoscale_view()
            else:
                lineT.set_data([], [])
                msgT.set_text('no buoyancy (Fr = inf)\nor AvgScal.npy absent'
                              if dp is not None else _no_profile_msg(prof))
            fig.canvas.draw_idle()
        finally:
            state['updating'] = False

    def on_pair(lbl):
        state['anchors'] = tuple(lbl.split(',')); refresh()

    def on_fr(lbl):
        state['Fr'] = np.inf if lbl == 'inf' else float(lbl); refresh()

    def on_reset(_):
        state['updating'] = True
        s_G.reset(); s_lat.reset(); s_L0.reset(); s_nu.reset()
        state['updating'] = False
        refresh()

    for s in sliders.values():
        s.on_changed(refresh)
    rb_pair.on_clicked(on_pair)
    rb_fr.on_clicked(on_fr)
    btn.on_clicked(on_reset)

    refresh()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
NON_INTERACTIVE_BACKENDS = ('agg', 'pdf', 'ps', 'svg', 'template')

# run in a throwaway child interpreter: opens (and closes) one figure window.
# Exit 1 = headless backend; SIGABRT = Qt found no usable platform plugin.
_GUI_PROBE = ("import sys\n"
              "import matplotlib.pyplot as plt\n"
              "if plt.get_backend().lower() in %r:\n"
              "    sys.exit(1)\n"
              "plt.figure(); plt.close('all')\n" % (NON_INTERACTIVE_BACKENDS,))


def _gui_preflight():
    """Can a GUI window actually be created?  Qt aborts the *process* when no
    platform plugin initialises (SIGABRT is not a catchable Python exception),
    so probe in a subprocess and read its exit status instead."""
    import subprocess
    try:
        return subprocess.run(
            [sys.executable, '-c', _GUI_PROBE], timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    except Exception:
        return False


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--static', action='store_true',
                   help='headless: print table + save dashboard PNG (no GUI)')
    p.add_argument('--anchors', default=None,
                   help='comma pair from {G,f,L0,nu}, e.g. "G,f" or "G,nu"')
    p.add_argument('--G',   type=float, default=10.0, help='geostrophic wind [m/s]')
    p.add_argument('--lat', type=float, default=45.0, help='latitude [deg] -> f')
    p.add_argument('--f',   type=float, default=None, help='Coriolis f [1/s] (overrides --lat)')
    p.add_argument('--L0',  type=float, default=None, help='reference length [m]')
    p.add_argument('--nu',  type=float, default=None, help='kinematic viscosity [m^2/s]')
    p.add_argument('--Fr',  default=str(config.Fr),
                   help='Froude number: inf | 1 | 0.1 | 0.01 (default from config.py)')
    p.add_argument('--data-dir', default=None,
                   help='case directory holding y.npy/eps_save.npy/AvgPh*.npy '
                        '(default: auto-search cwd, script dir, repo root)')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    Fr = np.inf if str(args.Fr).lower() in ('inf', 'np.inf', 'infty') else float(args.Fr)

    f_val = args.f if args.f is not None else f_from_lat(args.lat)
    cli_vals = {'G': args.G, 'f': f_val}
    if args.L0 is not None:
        cli_vals['L0'] = args.L0
    if args.nu is not None:
        cli_vals['nu'] = args.nu

    # choose the anchor pair, canonicalised to its ANCHOR_PAIRS ordering (so
    # "nu,G" works and the GUI radio can preselect it)
    if args.anchors:
        pick = set(a.strip() for a in args.anchors.split(','))
        anchors = next((p for p in ANCHOR_PAIRS if set(p) == pick), None)
        if anchors is None:
            sys.exit('--anchors must be two distinct keys of %s' % (ANCHOR_KEYS,))
    else:
        anchors = ('G', 'f')

    # complete `values`: fill L0/nu from the (G,f) closure so an --anchors
    # pair naming them never KeyErrors when --L0/--nu were not given, let any
    # CLI-given numbers override, then re-solve from the CHOSEN anchors so
    # all four are mutually consistent with that pair.
    values = {**solve_gflnu({'G': cli_vals['G'], 'f': cli_vals['f']}), **cli_vals}
    values = solve_gflnu({k: values[k] for k in anchors})

    if args.static:
        run_static(anchors, values, Fr, data_dir=args.data_dir)
        return

    # interactive; fall back to static when no GUI can open.  The backend-name
    # check catches headless matplotlib; the subprocess preflight catches the
    # harsher case where a GUI backend is selected but Qt cannot initialise
    # any platform plugin and would SIGABRT the whole process (uncatchable).
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if backend in NON_INTERACTIVE_BACKENDS:
        print('  [interactive unavailable: non-interactive backend %r]'
              '  -> falling back to --static\n' % backend)
        run_static(anchors, values, Fr, data_dir=args.data_dir)
        return
    if not _gui_preflight():
        print('  [interactive unavailable: backend %r cannot open a window '
              'here (no usable display / Qt platform plugin)]'
              '  -> falling back to --static\n' % backend)
        run_static(anchors, values, Fr, data_dir=args.data_dir)
        return
    try:
        run_interactive(values, Fr, anchors=anchors, data_dir=args.data_dir)
    except Exception as exc:
        print('  [interactive failed: %s]  -> falling back to --static\n' % exc)
        run_static(anchors, values, Fr, data_dir=args.data_dir)


if __name__ == '__main__':
    main()
