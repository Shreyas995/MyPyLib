#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_eps_geometry.py — is the SAME valley encoded in two (or more) eps fields?

Motivation
----------
The IBM indicator `eps0.1` is generated once and then carried to every new grid
with `MyPyLib/transfer_eps.py` (nearest-neighbour, so it stays strictly {0,1}).
If that transfer is faithful, the valley must be the SAME OBJECT IN PHYSICAL
SPACE on every grid: same crest height, same floor, same depth, same width, same
solid cross-section — regardless of nx/ny/nz.  This script checks exactly that,
and separates the two things that are easy to conflate:

  (a) IS THE GEOMETRY THE SAME?      -> compare in PHYSICAL units (this script).
  (b) DOES IT LOOK THE SAME ON A PLOT? -> depends on the z-axis NORMALIZATION,
      which is a plotting choice, not a property of eps.  A valley of fixed
      physical height h has

          h+     = h * u*/nu        (inner / wall units)  -- Re-DEPENDENT
          h/delta= h * f/u*         (outer units, delta=u*/f) -- weakly Re-dep.
          h/h    = 1                (topography units)     -- Re-INDEPENDENT

      so the SAME valley legitimately draws at different heights on a z+ axis.
      `--norm` prints that table so an apparent "the valley grew" can be traced
      to the axis rather than to the geometry.

eps file format (tlab bit-packed, little endian, no Fortran record markers) —
identical to the one documented in verify_eps.py / transfer_eps.py:

    offset 0   int32   head_size (= 20 = 5*int32)
    offset 4   int32   nx8 = nx/8      <-- packed, 8 cells per byte
    offset 8   int32   ny
    offset 12  int32   nz
    offset 16  int32   (unused)
    offset 20  int8 *  nx8*ny*nz       bits LSB-first, x fastest then y
    eps == 1 inside the solid.  The geometry is spanwise (z) invariant, so only
    plane k=0 is read here (verify_eps.py is the tool that proves invariance).

Usage
-----
    python3 compare_eps_geometry.py 1024x832x1024 1152x816x1152_Re1000
    python3 compare_eps_geometry.py DIR_A DIR_B --plot fig/eps_compare.png
    python3 compare_eps_geometry.py A/eps0.1 B/eps0.1 --grid-a A/grid --grid-b B/grid
    python3 compare_eps_geometry.py A B --norm 500 750      # + wall-unit table

A "target" is either an eps file or a directory.  For a directory the script
takes the first `eps*.1` in it and auto-detects a `grid*` file whose dimensions
match; a directory that holds a grid but no eps (e.g. 1024x832x1024) falls back
to an eps of matching (nx,ny,nz) found anywhere under the repo root.

Everything is opened READ-ONLY; the only file written is the optional PNG.
Exit status 0 when every target matches the first one to within one grid cell.
"""

import os
import sys
import glob
import argparse

import numpy as np

HEAD_PARAMS = 5                      # int32 words in the eps header
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# readers (read-only)
# --------------------------------------------------------------------------- #
def read_grid(path):
    """tlab grid (Fortran sequential unformatted, 5 records) -> nmax,scales,x,y,z."""
    with open(path, 'rb') as f:
        def rec(dtype, count):
            np.fromfile(f, '<i4', 1)                 # leading record marker
            val = np.fromfile(f, dtype, count)
            np.fromfile(f, '<i4', 1)                 # trailing record marker
            return val
        nmax   = rec('<i4', 3)
        scales = rec('<f8', 3)
        x = rec('<f8', int(nmax[0]))
        y = rec('<f8', int(nmax[1]))
        z = rec('<f8', int(nmax[2]))
    return nmax, scales, x, y, z


def read_eps_header(path):
    """(head_size, nx, ny, nz); raises when the file size contradicts the header."""
    h = np.fromfile(path, '<i4', HEAD_PARAMS)
    if h.size != HEAD_PARAMS:
        raise ValueError('%s: too short for a %d-word header' % (path, HEAD_PARAMS))
    head, nx8, ny, nz = int(h[0]), int(h[1]), int(h[2]), int(h[3])
    if min(head, nx8, ny, nz) <= 0:
        raise ValueError('%s: implausible header %s' % (path, h.tolist()))
    expect, actual = head + nx8 * ny * nz, os.path.getsize(path)
    if actual != expect:
        raise ValueError('%s: size mismatch — header implies %d bytes '
                         '(%d + %d*%d*%d), file has %d'
                         % (path, expect, head, nx8, ny, nz, actual))
    return head, nx8 * 8, ny, nz


def read_eps_plane(path):
    """Plane k=0 of the bit-packed eps -> eps[y, x] uint8 in {0,1}."""
    head, nx, ny, nz = read_eps_header(path)
    with open(path, 'rb') as f:
        f.seek(head, 0)
        data = np.fromfile(f, np.uint8, (nx // 8) * ny)
    return np.unpackbits(data, bitorder='little').reshape((ny, nx)), (nx, ny, nz)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def _grid_candidates(directory):
    for p in sorted(glob.glob(os.path.join(directory, 'grid*'))):
        if os.path.isfile(p) and not p.endswith(('.sts', '.png', '.py')):
            yield p


def find_grid(directory, nx, ny, nz, wider=True):
    """A grid file whose nmax == (nx,ny,nz): first beside the eps, then repo-wide."""
    places = [directory] + ([_REPO] + sorted(
        d for d in glob.glob(os.path.join(_REPO, '*')) if os.path.isdir(d))
        if wider else [])
    for d in places:
        for cand in _grid_candidates(d):
            try:
                nmax, scales, x, y, z = read_grid(cand)
            except Exception:
                continue
            if (int(nmax[0]), int(nmax[1]), int(nmax[2])) == (nx, ny, nz):
                return cand, scales, x, y, z
    return None, None, None, None, None


def find_eps_for_grid(nx, ny, nz):
    """Any eps in the repo whose header matches (nx,ny,nz) — used when a case
    directory carries a grid but no eps of its own."""
    pats = [os.path.join(_REPO, 'eps*'), os.path.join(_REPO, '*', 'eps*')]
    for p in sorted({q for pat in pats for q in glob.glob(pat)}):
        if not os.path.isfile(p) or p.endswith(('.png', '.py', '.npy')):
            continue
        try:
            _h, _nx, _ny, _nz = read_eps_header(p)
        except Exception:
            continue
        if (_nx, _ny, _nz) == (nx, ny, nz):
            return p
    return None


def resolve_target(t, grid_override=None):
    """Turn a CLI target (eps file or directory) into (eps_path, grid_path)."""
    if os.path.isfile(t):
        return t, grid_override
    if not os.path.isdir(t):
        alt = os.path.join(_REPO, t)                 # allow a bare case name
        if os.path.isdir(alt) or os.path.isfile(alt):
            return resolve_target(alt, grid_override)
        raise SystemExit('compare_eps_geometry.py: %s does not exist' % t)

    found = sorted(glob.glob(os.path.join(t, 'eps*.1')))
    if found:
        return found[0], grid_override
    # directory holds a grid but no eps -> look for an eps of the same shape
    for g in _grid_candidates(t):
        try:
            nmax, _s, _x, _y, _z = read_grid(g)
        except Exception:
            continue
        ep = find_eps_for_grid(int(nmax[0]), int(nmax[1]), int(nmax[2]))
        if ep:
            return ep, (grid_override or g)
    raise SystemExit('compare_eps_geometry.py: no eps*.1 (and no eps matching a '
                     'grid) found in %s' % t)


# --------------------------------------------------------------------------- #
# geometry extraction
# --------------------------------------------------------------------------- #
def describe(eps_path, grid_path=None, label=None):
    """Physical description of the valley encoded in one eps field."""
    eps, (nx, ny, nz) = read_eps_plane(eps_path)
    directory = os.path.dirname(os.path.abspath(eps_path))
    if grid_path is None:
        grid_path, scales, x, y, z = find_grid(directory, nx, ny, nz)
    else:
        nmax, scales, x, y, z = read_grid(grid_path)
        if (int(nmax[0]), int(nmax[1]), int(nmax[2])) != (nx, ny, nz):
            raise SystemExit('%s is %dx%dx%d but eps %s is %dx%dx%d'
                             % (grid_path, nmax[0], nmax[1], nmax[2],
                                eps_path, nx, ny, nz))
    if grid_path is None:
        raise SystemExit('no grid of shape %dx%dx%d found for %s'
                         % (nx, ny, nz, eps_path))

    # topmost solid row per column (-1 where the column is pure fluid)
    solid_col = eps.max(axis=0) > 0
    j_top = np.where(solid_col, (ny - 1) - np.argmax(eps[::-1, :], axis=0), -1)
    j_crest = int(j_top.max())
    h_surf = np.where(solid_col, y[np.clip(j_top, 0, None)], np.nan)   # h(x)

    dy = np.gradient(y)
    area = float((eps * dy[:, None]).sum() * (x[1] - x[0]))            # ∫∫ eps dx dy

    h_crest = float(y[j_crest])
    h_floor = float(np.nanmin(h_surf))
    below   = np.nan_to_num(h_surf, nan=0.0) < 0.999 * h_crest         # carved part
    i_lo, i_hi = (int(np.argmax(below)), int(len(below) - 1 - np.argmax(below[::-1]))
                  ) if below.any() else (0, 0)

    return dict(
        label=label or os.path.relpath(eps_path, _REPO),
        eps_path=eps_path, grid_path=grid_path,
        nx=nx, ny=ny, nz=nz, Lx=float(scales[0]), Ly=float(scales[1]),
        Lz=float(scales[2]), dx=float(x[1] - x[0]), dy_wall=float(y[1] - y[0]),
        x=x, y=y, eps=eps, h_surf=h_surf, solid_col=solid_col,
        j_crest=j_crest, h_crest=h_crest, h_floor=h_floor,
        depth=h_crest - h_floor, area=area,
        width=float(x[i_hi] - x[i_lo]), solid_frac=float(eps.mean()),
        n_solid_cols=int(solid_col.sum()))


def surface_diff(ref, other):
    """Compare h(x) after mapping both onto the reference x/Lx (grids differ in nx).

    The comparison is in PHYSICAL y.  Because the transfer is nearest-neighbour,
    a column's surface can land one node higher or lower on the finer grid, so
    the meaningful tolerance is ONE wall-normal cell of the coarser of the two
    grids — not zero."""
    xr = ref['x'] / ref['Lx']
    hr = np.nan_to_num(ref['h_surf'], nan=0.0)
    xo = other['x'] / other['Lx']
    ho = np.nan_to_num(other['h_surf'], nan=0.0)
    d = np.interp(xr, xo, ho) - hr
    tol = max(ref['dy_wall'], other['dy_wall'])
    return dict(max_abs=float(np.abs(d).max()), rms=float(np.sqrt((d ** 2).mean())),
                mean=float(d.mean()), tol=float(tol),
                within=bool(np.abs(d).max() <= tol + 1e-15), d=d, x=xr)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _nu(Re_D):
    """tlab convention: nu = 1/Re_lambda, Re_lambda = 0.5*Re_D^2."""
    return 1.0 / (0.5 * float(Re_D) ** 2)


def print_report(cases, norm_Re=None, ustar=None):
    print('=' * 100)
    print('EPS GEOMETRY — PHYSICAL DESCRIPTION (the valley as an object, '
          'independent of any plot axis)')
    print('=' * 100)
    print('%-26s %5s %5s %5s %11s %11s %13s %13s %13s'
          % ('case', 'nx', 'ny', 'nz', 'Lx', 'Ly', 'h_crest', 'h_floor', 'depth'))
    for c in cases:
        print('%-26s %5d %5d %5d %11.8f %11.8f %13.10g %13.10g %13.10g'
              % (c['label'], c['nx'], c['ny'], c['nz'], c['Lx'], c['Ly'],
                 c['h_crest'], c['h_floor'], c['depth']))
    print()
    print('%-26s %13s %13s %13s %11s %11s'
          % ('case', 'width', 'area', 'solid_frac', 'dx', 'dy_wall'))
    for c in cases:
        print('%-26s %13.10g %13.10g %13.8f %11.4e %11.4e'
              % (c['label'], c['width'], c['area'], c['solid_frac'],
                 c['dx'], c['dy_wall']))

    ref = cases[0]
    ok = True
    if len(cases) > 1:
        print()
        print('-' * 100)
        print('PHYSICAL SURFACE h(x) vs REFERENCE  %s' % ref['label'])
        print('  tolerance = one wall-normal cell of the coarser grid '
              '(nearest-neighbour transfer can shift a column by one node)')
        print('%-26s %12s %12s %12s %12s %8s'
              % ('case', 'max|dh|', 'rms dh', 'tol (1 cell)', 'd h_crest', 'verdict'))
        for c in cases[1:]:
            s = surface_diff(ref, c)
            ok &= s['within']
            print('%-26s %12.5g %12.5g %12.5g %12.3g %8s'
                  % (c['label'], s['max_abs'], s['rms'], s['tol'],
                     c['h_crest'] - ref['h_crest'],
                     'MATCH' if s['within'] else 'DIFFER'))
        print()
        print('  VERDICT: %s' % ('the same valley on every grid (all differences '
                                 'below one grid cell)' if ok else
                                 'at least one field encodes a DIFFERENT valley'))

    if norm_Re:
        print()
        print('=' * 100)
        print('WHY THE SAME VALLEY CAN LOOK DIFFERENT — z-AXIS NORMALIZATION')
        print('=' * 100)
        print('  h is FIXED in physical units, so its plotted height is set '
              'entirely by the axis:')
        print('    z+      = z*u*/nu        inner  (wall units)   -> h+ scales '
              'like u*/nu  ==> STRONGLY Re-dependent')
        print('    z-      = z*f/u*         outer  (delta=u*/f)   -> h/delta '
              'scales like 1/u* ==> weakly Re-dependent')
        print('    zeta    = z/h            topography units      -> always 1 '
              '==> Re-INDEPENDENT (valleys coincide)')
        print()
        print('%-10s %12s %12s %12s %12s %12s %12s'
              % ('Re_D', 'nu', 'u*', 'l_in=nu/u*', 'h+ = h/l_in', 'h/delta',
                 'z/h'))
        h = ref['h_crest']
        for Re in norm_Re:
            us = (ustar or {}).get(float(Re), (ustar or {}).get(int(Re)))
            if us is None:
                print('%-10s %12.4e %12s %12s %12s %12s %12s'
                      % (Re, _nu(Re), '(u* unknown)', '-', '-', '-', '1.000'))
                continue
            li = _nu(Re) / us
            print('%-10s %12.4e %12.5f %12.4e %12.2f %12.4f %12.3f'
                  % (Re, _nu(Re), us, li, h / li, h * 1.0 / us, 1.0))
        print()
        print('  => a valley that "grew" between two Reynolds numbers on a z+ '
              'axis has NOT grown;')
        print('     h+ = h*u*/nu simply rises with Re because nu falls.  Use '
              'z/h (or physical z)')
        print('     for a panel-to-panel comparison in which the topography '
              'must coincide.')
    return ok


def make_plot(cases, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ref = cases[0]
    ncase = len(cases)
    fig, axes = plt.subplots(2, max(ncase, 2), figsize=(5.2 * max(ncase, 2), 7.6),
                             squeeze=False)

    # row 0 — each eps plane on its own physical axes, common z-extent
    ztop = 2.2 * max(c['h_crest'] for c in cases)
    for i, c in enumerate(cases):
        ax = axes[0][i]
        ax.pcolormesh(c['x'], c['y'], c['eps'], cmap='Greys',
                      shading='auto', vmin=0, vmax=1)
        ax.set_ylim(0, ztop)
        ax.set_xlim(c['x'][0], c['x'][-1])
        ax.set_title('%s\n%dx%dx%d   h=%.6g' % (c['label'], c['nx'], c['ny'],
                                                c['nz'], c['h_crest']),
                     fontsize=9)
        ax.set_xlabel('x (physical)')
        if i == 0:
            ax.set_ylabel('z (physical)')
    for j in range(ncase, axes.shape[1]):
        axes[0][j].axis('off')

    # row 1a — overlaid surfaces in physical units
    ax = axes[1][0]
    for c in cases:
        ax.plot(c['x'] / c['Lx'], c['h_surf'], lw=1.2, alpha=0.85,
                label='%s (ny=%d)' % (c['label'], c['ny']))
    ax.set_xlabel('x / L_x')
    ax.set_ylabel('surface height h(x)  [physical]')
    ax.set_title('IBM surface, PHYSICAL units — must coincide', fontsize=9)
    ax.grid(True, ls='--', lw=0.4)
    ax.legend(fontsize=7)

    # row 1b — difference vs reference, in units of one wall-normal cell
    ax = axes[1][1] if axes.shape[1] > 1 else axes[1][0]
    if ncase > 1:
        for c in cases[1:]:
            s = surface_diff(ref, c)
            ax.plot(s['x'], s['d'] / s['tol'], lw=1.0,
                    label='%s  (max %.2f cell)'
                          % (c['label'], s['max_abs'] / s['tol']))
        ax.axhline(1.0, color='k', ls='--', lw=0.7)
        ax.axhline(-1.0, color='k', ls='--', lw=0.7)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xlabel('x / L_x')
        ax.set_ylabel('(h - h_ref) / one grid cell')
        ax.set_title('surface difference vs %s\n(|.|<1 cell = nearest-neighbour '
                     'rounding only)' % ref['label'], fontsize=9)
        ax.grid(True, ls='--', lw=0.4)
        ax.legend(fontsize=7)
    for j in range(2, axes.shape[1]):
        axes[1][j].axis('off')

    fig.suptitle('eps geometry comparison — physical space', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    odir = os.path.dirname(os.path.abspath(out_png))
    if odir and not os.path.isdir(odir):
        os.makedirs(odir)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print('  wrote %s' % out_png)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description='Compare the valley encoded in two or more eps fields.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='targets are eps files or case directories; the FIRST is the '
               'reference everything else is compared against.')
    p.add_argument('targets', nargs='+', help='eps file(s) and/or case directories')
    p.add_argument('--grid-a', default=None, help='explicit grid for target 1')
    p.add_argument('--grid-b', default=None, help='explicit grid for target 2')
    p.add_argument('--plot', nargs='?', const='eps_compare.png', default=None,
                   metavar='PNG', help='write a comparison figure')
    p.add_argument('--norm', nargs='*', type=float, default=None, metavar='Re_D',
                   help='also print the z-axis normalization table for these Re_D')
    p.add_argument('--ustar', nargs='*', default=None, metavar='Re=u*',
                   help='friction velocity per Re for --norm, e.g. 500=0.0663 '
                        '750=0.067 (default: the CLAUDE.md physics table)')
    a = p.parse_args(argv)

    ustar = {500.0: 0.077, 750.0: 0.06732, 1000.0: 0.06372}   # CLAUDE.md table
    for kv in (a.ustar or []):
        k, v = kv.split('=')
        ustar[float(k)] = float(v)

    overrides = [a.grid_a, a.grid_b] + [None] * len(a.targets)
    cases = []
    for i, t in enumerate(a.targets):
        ep, gp = resolve_target(t, overrides[i])
        cases.append(describe(ep, gp, label=os.path.basename(os.path.normpath(t))))
        print('  %-26s eps=%s' % (cases[-1]['label'],
                                  os.path.relpath(ep, _REPO)))
        print('  %-26s grid=%s' % ('', os.path.relpath(cases[-1]['grid_path'], _REPO)))

    ok = print_report(cases, norm_Re=a.norm, ustar=ustar)
    if a.plot:
        make_plot(cases, a.plot)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
