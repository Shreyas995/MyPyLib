#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STANDALONE: Method-2 friction velocity + log-law comparison across Reynolds number.

Single responsibility (does NOT touch PhAvg_rotated.py or any pickle):
  1. For every reference case that lives in a tlab avg_all.nc, compute the
     vertically-integrated ("Method 2") friction velocity u* — the SAME momentum-
     integral balance used by functions.load_ekman_nc_case (Kostelecky & Ansorge
     2024, eq. 4.2), which is the only correct u* in a rotating Ekman layer.
  2. Overlay every case's mean log-law profile in its OWN TRUE inner units
     (z+ = y*u*/nu, u+ = <u>/u*), grouped by Reynolds number, so Re=500 and
     Re=1000 can be compared on the universal wall scaling.  A larger Re shows a
     LONGER log layer (higher Re_tau = u*^2/nu); the log region should overlap.
  3. Print a table (u*_M2, Re_tau, veer) and plot u*/Re_tau vs the ladder Ri.

Cases (paths + constants come from config.py — single source of truth):
  * Re=500  smooth  reference     — config.smooth_nc_path      (nu,       Re_lambda)
  * Re=1000 rough r1 reference    — config.rough_nc_path       (nu_rough, Re_lambda_rough)
  * Re=1000 stable ladder (16)    — config.rough_ladder_dir/glob (nu_rough, Re_lambda_rough)
  * (optional) Re=500 orographic  — a PhAvg_rotated pickle, see PICKLE_CASES below

Run:   python3 compare_Re_loglaw.py
Out:   fig_compareRe/loglaw_Re500_vs_Re1000.png
       fig_compareRe/ustar_Re_tau_vs_Ri.png
       + a console summary table.
Reads ONLY .nc (and, if configured, one .pkl); nothing is written to the raw data.
"""
import os
import sys
import glob
import re
import pickle

# Make this file's own directory importable so `config`/`functions` resolve whether
# it is run from MyPyLib or from a case dir where setup.sh symlinked it.
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import config as cfg
from functions import load_ekman_nc_case

# ── Optional Re=500 orographic (valley) case(s) from a PhAvg_rotated pickle ────
# The valley case is not an avg_all.nc, so its Method-2 u* is the one already
# computed by PhAvg_rotated (u_star2, crest value).  Leave empty to skip; add
# tuples (label, pickle_path, nu) to include it in the comparison.
#   e.g. ('Re500 orographic', '/home/.../Ekman18/sim1_results.pkl', cfg.nu)
PICKLE_CASES = [
    # ('Re500 orographic', 'sim1_results.pkl', cfg.nu),
]

_XDUMMY = np.linspace(0.0, 1.0, 16)     # load_ekman_nc_case x_grid arg (unused here)
_FIGDIR = 'fig_compareRe'


def _parse_ri_tag(basename):
    """(Ri, tag) parsed from an ri<NN.NN>_..._<tag>_avg.nc file name."""
    _mri  = re.search(r'ri(\d+\.\d+)', basename)
    _mtag = re.search(r'_([a-z]\d*[a-z]?)_avg\.nc$', basename)
    return ((float(_mri.group(1)) if _mri else np.nan),
            (_mtag.group(1) if _mtag else ''))


def _nc_case(nc_path, nu, Re_lambda, label, Re, ri=np.nan):
    """Load ONE avg_all.nc, compute Method-2 u*, return the log-law overlay dict.
    Returns None (with a printed reason) if the file is unreadable / lacks the
    tlab velocity+stress variables load_ekman_nc_case needs."""
    try:
        d = load_ekman_nc_case(nc_path, _XDUMMY, nu, Re_lambda)
    except Exception as e:
        print('  [compare-Re] skipped %s (%s)' % (os.path.basename(nc_path), e))
        return None
    ustar = float(d['ustr_M2_plateau'])
    if not (np.isfinite(ustar) and ustar > 0):
        print('  [compare-Re] skipped %s (bad Method-2 u*=%r)'
              % (os.path.basename(nc_path), ustar))
        return None
    y, U = d['y'], d['GblU']
    # Surface veer (deg) between the geostrophic and the near-wall stress direction.
    veer = float(np.degrees(np.arctan2(d['GblW'][2] - d['GblW'][0],
                                       d['GblU'][2] - d['GblU'][0])))
    return dict(label=label, Re=Re, ri=ri, nu=nu, ustar=ustar,
                Re_tau=ustar**2 / nu, veer=veer,
                zp=y * ustar / nu, up=U / ustar)


def _pickle_case(label, pkl_path, nu, Re=500):
    """Optional valley case from a PhAvg_rotated pickle (uses its own u_star2)."""
    if not os.path.exists(pkl_path):
        print('  [compare-Re] pickle not found, skipped: %s' % pkl_path)
        return None
    with open(pkl_path, 'rb') as f:
        p = pickle.load(f)
    try:
        y   = np.asarray(p['y'], dtype=float)
        upr = np.asarray(p['u_plus_rot'], dtype=float)
        us2 = np.asarray(p['u_star2'], dtype=float)
    except Exception as e:
        print('  [compare-Re] pickle missing keys (%s), skipped: %s' % (e, pkl_path))
        return None
    ustar = float(np.nanmax(us2))            # crest Method-2 u* for the valley
    return dict(label=label, Re=Re, ri=np.nan, nu=nu, ustar=ustar,
                Re_tau=ustar**2 / nu, veer=np.nan,
                zp=y * ustar / nu, up=upr / ustar)


def build_cases():
    """Assemble every available case (Re=500 + Re=1000), each with Method-2 u*."""
    cases = []
    # Re=500 smooth reference
    c = _nc_case(cfg.smooth_nc_path, cfg.nu, cfg.Re_lambda,
                 'Re500 smooth', 500)
    if c: cases.append(c)
    # Re=1000 rough r1 reference
    c = _nc_case(cfg.rough_nc_path, cfg.nu_rough, cfg.Re_lambda_rough,
                 'Re1000 rough r1', 1000)
    if c: cases.append(c)
    # Re=1000 stable ladder
    lad_glob = os.path.join(cfg.rough_ladder_dir, cfg.rough_ladder_pattern)
    lad_files = sorted(glob.glob(lad_glob))
    if not lad_files:
        print('  [compare-Re] no ladder files match %s' % lad_glob)
    for p in lad_files:
        ri, tag = _parse_ri_tag(os.path.basename(p))
        lbl = ('Re1000 Ri=%.2f' % ri if np.isfinite(ri) else os.path.basename(p))
        if tag:
            lbl += ' (%s)' % tag
        c = _nc_case(p, cfg.nu_rough, cfg.Re_lambda_rough, lbl, 1000, ri=ri)
        if c: cases.append(c)
    # Optional pickle (valley) cases
    for lbl, pkl, nu in PICKLE_CASES:
        c = _pickle_case(lbl, pkl, nu)
        if c: cases.append(c)
    return cases


def print_table(cases):
    print('=' * 74)
    print('METHOD-2 FRICTION VELOCITY — all cases (u* from the momentum-integral '
          'balance)')
    print('-' * 74)
    print('  %-26s %5s %10s %9s %8s' % ('case', 'Re', 'u*_M2', 'Re_tau', 'veer'))
    for c in cases:
        _veer = '   n/a' if not np.isfinite(c['veer']) else '%6.2f' % c['veer']
        print('  %-26s %5d %10.5f %9.1f %8s'
              % (c['label'], c['Re'], c['ustar'], c['Re_tau'], _veer))
    print('=' * 74)


def plot_loglaw(cases):
    """Log-law overlay in each case's OWN true inner units, grouped by Re."""
    os.makedirs(_FIGDIR, exist_ok=True)
    re500  = [c for c in cases if c['Re'] == 500]
    re1000 = [c for c in cases if c['Re'] == 1000]
    # Ladder colour = Ri gradient; non-ladder Re1000 (rough r1) drawn black-dashed.
    ladder = [c for c in re1000 if np.isfinite(c['ri'])]
    ri_vals = [c['ri'] for c in ladder]
    norm = (Normalize(vmin=min(ri_vals), vmax=max(ri_vals))
            if ri_vals else Normalize(0, 1))
    cmap = plt.cm.viridis

    plt.figure(figsize=(8, 6), dpi=300)
    # Reference neutral log law  u+ = (1/0.41) ln z+ + 5.0
    _zr = np.geomspace(1.0, 3000.0, 200)
    plt.plot(_zr, (1.0 / 0.41) * np.log(_zr) + 5.0, color='0.4',
             linestyle=':', linewidth=1.2, zorder=1, label=r'log law ($\kappa$=0.41)')
    # Re=500 group — solid, cool colours
    _c500 = ['tab:blue', 'tab:cyan', 'navy']
    for i, c in enumerate(re500):
        plt.plot(c['zp'], c['up'], color=_c500[i % len(_c500)], linestyle='-',
                 linewidth=1.6, label=r'%s ($u_\star$=%.4f, $Re_\tau$=%.0f)'
                 % (c['label'], c['ustar'], c['Re_tau']))
    # Re=1000 rough r1 — black dashed
    for c in re1000:
        if np.isfinite(c['ri']):
            continue
        plt.plot(c['zp'], c['up'], color='k', linestyle='--', linewidth=1.6,
                 label=r'%s ($u_\star$=%.4f, $Re_\tau$=%.0f)'
                 % (c['label'], c['ustar'], c['Re_tau']))
    # Re=1000 ladder — viridis by Ri
    for c in ladder:
        plt.plot(c['zp'], c['up'], color=cmap(norm(c['ri'])), linestyle='-',
                 linewidth=0.9, alpha=0.85, zorder=2)
    plt.xscale('log')
    plt.xlabel(r'$z^+ = z\,u_\star/\nu$  (each case in its OWN wall units)')
    plt.ylabel(r'$\langle\bar u\rangle^+ = \langle\bar u\rangle/u_\star$')
    plt.title('Log-law comparison — Re=500 vs Re=1000 (Method-2 $u_\\star$)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    handles = plt.gca().get_legend_handles_labels()[0]
    if ladder:
        handles.append(Line2D([0], [0], color=cmap(0.5), linestyle='-',
                              linewidth=1.2, label='Re1000 stable ladder'))
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
        cb.set_label(r'ladder $Ri$')
    plt.legend(handles=handles, fontsize=7)
    _out = os.path.join(_FIGDIR, 'loglaw_Re500_vs_Re1000.png')
    plt.savefig(_out, dpi=300)
    print('  wrote %s' % _out)
    plt.show()


def plot_ustar_vs_ri(cases):
    """u* and Re_tau vs Ri across the ladder, with Re=500/Re=1000 reference lines."""
    ladder = sorted([c for c in cases if np.isfinite(c['ri'])], key=lambda c: c['ri'])
    if not ladder:
        return
    ri = [c['ri'] for c in ladder]
    us = [c['ustar'] for c in ladder]
    rt = [c['Re_tau'] for c in ladder]
    smooth = next((c for c in cases if c['label'] == 'Re500 smooth'), None)
    roughr1 = next((c for c in cases if c['label'] == 'Re1000 rough r1'), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    ax1.plot(ri, us, 'o-', color='tab:red')
    ax1.set_xlabel(r'$Ri$ (ladder)'); ax1.set_ylabel(r'$u_\star$ (Method 2)')
    ax1.set_title(r'Friction velocity vs stability')
    if smooth:  ax1.axhline(smooth['ustar'],  color='tab:blue', ls='--', lw=1,
                            label='Re500 smooth')
    if roughr1: ax1.axhline(roughr1['ustar'], color='k', ls=':', lw=1,
                            label='Re1000 rough r1')
    ax1.legend(fontsize=8); ax1.grid(True, ls='--', lw=0.4)

    ax2.plot(ri, rt, 's-', color='tab:purple')
    ax2.set_xlabel(r'$Ri$ (ladder)'); ax2.set_ylabel(r'$Re_\tau=u_\star^2/\nu$')
    ax2.set_title(r'Friction Reynolds number vs stability')
    if smooth:  ax2.axhline(smooth['Re_tau'],  color='tab:blue', ls='--', lw=1)
    if roughr1: ax2.axhline(roughr1['Re_tau'], color='k', ls=':', lw=1)
    ax2.grid(True, ls='--', lw=0.4)

    fig.tight_layout()
    _out = os.path.join(_FIGDIR, 'ustar_Re_tau_vs_Ri.png')
    fig.savefig(_out, dpi=300)
    print('  wrote %s' % _out)
    plt.show()


def main():
    cases = build_cases()
    if not cases:
        print('[compare-Re] no cases loaded — check config paths / data location.')
        return
    print_table(cases)
    plot_loglaw(cases)
    plot_ustar_vs_ri(cases)


if __name__ == '__main__':
    main()
