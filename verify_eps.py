#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_eps.py -- check and verify a bit-packed IBM eps field.

Goal (two jobs, in this order):

  1. PLOT the FIRST spanwise plane (k = 0) of the eps field in the XOY plane
     (x horizontal, y wall-normal) so the sinusoidal valley can be eyeballed.
  2. VERIFY that EVERY remaining plane k = 1 .. nz-1 is an EXACT match of
     plane 0.  The valley geometry is spanwise (z) invariant by construction,
     so any differing plane is a bug in the eps generation / transfer.

The comparison is done on the RAW PACKED BYTES: each plane occupies exactly
nx8*ny bytes and byte equality is bit equality, so this is an exact test and
needs no unpacking on the fast path.  Bits are only unpacked for plane 0 (to
plot) and for planes that actually differ (to localise the difference).

eps file format (tlab bit-packed, little endian, no Fortran record markers):

    offset 0   int32   head_size (= 20 = 5 * int32)
    offset 4   int32   nx8  = nx / 8   <-- ATTENTION: packed, 8 cells per byte
    offset 8   int32   ny
    offset 12  int32   nz
    offset 16  int32   (unused / 0)
    offset 20  int8 *  nx8 * ny * nz   packed bits, LSB-first within each byte

    Layout: nz contiguous z-planes of nx8*ny bytes; within a plane x is the
    fastest index, y the slowest  ->  eps[y, x] after unpacking.
    eps == 1 inside the solid, 0 in the fluid (same convention as eps_save.npy).

Usage
-----
    python3 verify_eps.py                          # eps0.1 + grid in cwd
    python3 verify_eps.py 1152x816x1152_Re1000     # a case directory
    python3 verify_eps.py path/to/eps0.1 --grid path/to/grid_1152x816x1152
    python3 verify_eps.py DIR1 DIR2 --outdir fig   # several fields at once
    python3 verify_eps.py --stride 16              # quick partial scan
    python3 verify_eps.py --no-plot                # verification only

Exit status: 0 if every checked plane matches plane 0, 1 otherwise.
Reads the eps/grid files READ-ONLY; writes only the PNG.
"""

import os
import sys
import glob
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# tunables
# ---------------------------------------------------------------------------
BLOCK_PLANES = 64        # planes read per I/O block (memory ~ BLOCK*nx8*ny bytes)
MAX_REPORT   = 10        # detailed report for at most this many bad planes
HEAD_PARAMS  = 5         # int32 words in the eps header


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def read_eps_header(path):
    """Return (head_size, nx8, ny, nz) and validate the file size."""
    h = np.fromfile(path, '<i4', HEAD_PARAMS)
    if h.size != HEAD_PARAMS:
        raise ValueError('%s: too short to hold a %d-word header'
                         % (path, HEAD_PARAMS))
    head_size, nx8, ny, nz = (int(h[0]), int(h[1]), int(h[2]), int(h[3]))
    if min(nx8, ny, nz) <= 0 or head_size <= 0:
        raise ValueError('%s: implausible header %s' % (path, h.tolist()))

    expected = head_size + nx8 * ny * nz
    actual   = os.path.getsize(path)
    if actual != expected:
        raise ValueError('%s: size mismatch -- header implies %d bytes '
                         '(%d + %d*%d*%d), file has %d'
                         % (path, expected, head_size, nx8, ny, nz, actual))
    return head_size, nx8, ny, nz


def read_planes_raw(fh, head_size, plane_bytes, k0, nplanes):
    """Read nplanes packed planes starting at k0 -> uint8[nplanes, plane_bytes]."""
    fh.seek(head_size + k0 * plane_bytes, 0)
    buf = np.fromfile(fh, np.uint8, plane_bytes * nplanes)
    if buf.size != plane_bytes * nplanes:
        raise IOError('short read at plane %d (got %d of %d bytes)'
                      % (k0, buf.size, plane_bytes * nplanes))
    return buf.reshape((nplanes, plane_bytes))


def unpack_plane(raw, nx, ny):
    """Packed bytes of one plane -> eps[y, x] as uint8 {0,1} (LSB-first)."""
    return np.unpackbits(raw, bitorder='little').reshape((ny, nx))


def read_grid(path):
    """tlab grid (Fortran sequential unformatted, 5 records) -> nmax, x, y, z."""
    with open(path, 'rb') as f:
        def rec(dtype, count):
            np.fromfile(f, '<i4', 1)                 # leading record marker
            val = np.fromfile(f, dtype, count)
            np.fromfile(f, '<i4', 1)                 # trailing record marker
            return val
        nmax = rec('<i4', 3)
        rec('<f8', 3)                                # Lx, Ly, Lz (unused here)
        x = rec('<f8', int(nmax[0]))
        y = rec('<f8', int(nmax[1]))
        z = rec('<f8', int(nmax[2]))
    return nmax, x, y, z


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def find_grid(directory, nx, ny, nz):
    """Pick a grid file in `directory` whose dimensions match (nx, ny, nz)."""
    for cand in sorted(glob.glob(os.path.join(directory, 'grid*'))):
        if not os.path.isfile(cand) or cand.endswith(('.sts', '.png', '.py')):
            continue
        try:
            nmax, x, y, z = read_grid(cand)
        except Exception:
            continue
        if (int(nmax[0]), int(nmax[1]), int(nmax[2])) == (nx, ny, nz):
            return cand, x, y, z
    return None, None, None, None


def resolve_targets(args_targets):
    """Turn CLI targets (files and/or directories) into a list of eps paths."""
    targets = args_targets or ['.']
    eps_paths = []
    for t in targets:
        if os.path.isdir(t):
            found = sorted(glob.glob(os.path.join(t, 'eps*.1')))
            if not found:
                print('  ! no eps*.1 found in %s -- skipped' % t)
            eps_paths.extend(found)
        elif os.path.isfile(t):
            eps_paths.append(t)
        else:
            print('  ! %s does not exist -- skipped' % t)
    return eps_paths


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_first_plane(eps0, x, y, meta, bad, diff_count, out_png):
    """Plot plane 0 (+ a disagreement map when planes differ)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ny, nx = eps0.shape
    xs = x if x is not None else np.arange(nx)
    ys = y if y is not None else np.arange(ny)
    xlab = 'x' if x is not None else 'x index'
    ylab = 'y (wall-normal)' if y is not None else 'y index'

    # zoom the wall-normal axis to a few times the solid height
    solid_rows = np.where(eps0.max(axis=1) > 0)[0]
    if solid_rows.size:
        j_top = int(solid_rows.max())
        ytop  = ys[min(ny - 1, max(int(2.5 * j_top), j_top + 5))]
    else:
        ytop = ys[-1]

    npanel = 1 if not bad else 2
    fig, axes = plt.subplots(1, npanel, figsize=(9.6 if npanel == 1 else 15.2,
                                                 4.8), squeeze=False)
    ax = axes[0][0]
    ax.pcolormesh(xs, ys, eps0, cmap='Greys', shading='auto', vmin=0, vmax=1)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_ylim(ys[0], ytop)
    ax.set_title('plane k = 0   (solid fraction %.6f)' % eps0.mean())

    if bad:
        ax2 = axes[0][1]
        m = ax2.pcolormesh(xs, ys, diff_count, cmap='inferno', shading='auto')
        fig.colorbar(m, ax=ax2, label='# planes differing from k = 0')
        ax2.set_xlabel(xlab)
        ax2.set_ylim(ys[0], ytop)
        ax2.set_title('DISAGREEMENT MAP  (%d bad plane%s)'
                      % (len(bad), '' if len(bad) == 1 else 's'))

    fig.suptitle('%s   eps XOY plane   nx=%d ny=%d nz=%d\n%s'
                 % (meta['name'], meta['nx'], meta['ny'], meta['nz'],
                    'ALL %d PLANES IDENTICAL TO k = 0' % meta['nz'] if not bad
                    else '%d PLANE(S) DIFFER FROM k = 0' % len(bad)),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# main check for one eps file
# ---------------------------------------------------------------------------
def check_eps(eps_path, grid_path=None, stride=1, do_plot=True, outdir=None,
              max_report=MAX_REPORT):
    """Plot plane 0 and verify every other plane equals it.  Returns n_bad."""
    directory = os.path.dirname(os.path.abspath(eps_path))
    name      = os.path.basename(eps_path)

    head_size, nx8, ny, nz = read_eps_header(eps_path)
    nx = nx8 * 8
    plane_bytes = nx8 * ny

    print('=' * 78)
    print('eps file : %s' % os.path.abspath(eps_path))
    print('header   : head_size=%d  nx=%d (nx8=%d)  ny=%d  nz=%d'
          % (head_size, nx, nx8, ny, nz))
    print('size     : %d bytes = %d + %d planes x %d bytes  [OK]'
          % (os.path.getsize(eps_path), head_size, nz, plane_bytes))

    # ---- grid (optional, for physical axes) -------------------------------
    x = y = z = None
    if grid_path is None:
        grid_path, x, y, z = find_grid(directory, nx, ny, nz)
    elif os.path.isfile(grid_path):
        nmax, x, y, z = read_grid(grid_path)
        if (int(nmax[0]), int(nmax[1]), int(nmax[2])) != (nx, ny, nz):
            print('  ! grid %s is %dx%dx%d, eps is %dx%dx%d -- using indices'
                  % (grid_path, nmax[0], nmax[1], nmax[2], nx, ny, nz))
            grid_path = x = y = z = None
    print('grid     : %s' % (grid_path if grid_path else
                             '(none found -- plotting on index axes)'))

    # ---- plane 0 ----------------------------------------------------------
    with open(eps_path, 'rb') as fh:
        ref_raw = read_planes_raw(fh, head_size, plane_bytes, 0, 1)[0]
        eps0    = unpack_plane(ref_raw, nx, ny)

        n_solid     = int(eps0.sum())
        solid_rows  = np.where(eps0.max(axis=1) > 0)[0]
        j_top       = int(solid_rows.max()) if solid_rows.size else -1
        print('plane 0  : solid cells %d / %d  (fraction %.6f)'
              % (n_solid, eps0.size, n_solid / eps0.size))
        if j_top >= 0:
            hgt = ('%.6f' % y[j_top]) if y is not None else 'n/a'
            print('           topmost solid row j=%d  (y=%s), '
                  'solid columns %d / %d'
                  % (j_top, hgt, int((eps0.max(axis=0) > 0).sum()), nx))
        else:
            print('           ** plane 0 contains NO solid cells **')

        # ---- compare every remaining plane against plane 0 ----------------
        ks = np.arange(1, nz, stride, dtype=int)
        print('checking : %d of %d remaining planes (stride %d) against k=0 ...'
              % (ks.size, nz - 1, stride))

        bad = []
        if stride == 1:
            # contiguous blocks -- one seek per block
            for k0 in range(1, nz, BLOCK_PLANES):
                npl = min(BLOCK_PLANES, nz - k0)
                blk = read_planes_raw(fh, head_size, plane_bytes, k0, npl)
                neq = np.where((blk != ref_raw).any(axis=1))[0]
                bad.extend((k0 + int(i)) for i in neq)
        else:
            for k in ks:
                raw = read_planes_raw(fh, head_size, plane_bytes, int(k), 1)[0]
                if not np.array_equal(raw, ref_raw):
                    bad.append(int(k))

        # ---- localise the differences -------------------------------------
        diff_count = np.zeros((ny, nx), dtype=np.int32)
        if bad:
            print('')
            print('  *** %d plane(s) DIFFER from plane 0 ***' % len(bad))
            print('  first bad k = %d, last bad k = %d' % (bad[0], bad[-1]))
            for n, k in enumerate(bad):
                raw = read_planes_raw(fh, head_size, plane_bytes, k, 1)[0]
                d   = unpack_plane(raw, nx, ny) ^ eps0
                diff_count += d
                if n < max_report:
                    jj, ii = np.nonzero(d)
                    print('    k=%-6d %8d differing cells   '
                          'x-idx [%d, %d]  y-idx [%d, %d]  '
                          'solid frac %.6f (ref %.6f)'
                          % (k, jj.size, ii.min(), ii.max(),
                             jj.min(), jj.max(),
                             unpack_plane(raw, nx, ny).mean(), eps0.mean()))
            if len(bad) > max_report:
                print('    ... %d further bad planes not listed individually'
                      % (len(bad) - max_report))
        else:
            print('')
            print('  OK: all %d checked planes are an EXACT match of plane 0 '
                  '(spanwise invariant).' % ks.size)

    # ---- plot -------------------------------------------------------------
    if do_plot:
        odir = outdir or directory
        if not os.path.isdir(odir):
            os.makedirs(odir)
        tag  = os.path.basename(directory) or 'cwd'
        out  = os.path.join(odir, 'eps_verify_%s_%s.png'
                            % (tag, name.replace('.', '_')))
        meta = {'name': '%s/%s' % (tag, name), 'nx': nx, 'ny': ny, 'nz': nz}
        plot_first_plane(eps0, x, y, meta, bad, diff_count, out)
        print('  wrote %s' % out)

    return len(bad)


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description='Plot eps plane 0 and verify every other plane matches it.')
    p.add_argument('targets', nargs='*',
                   help='eps file(s) and/or directories containing eps*.1 '
                        '(default: current directory)')
    p.add_argument('--grid', default=None,
                   help='explicit grid file (default: auto-detect a matching '
                        'grid* next to the eps file)')
    p.add_argument('--stride', type=int, default=1,
                   help='check every Nth plane instead of all (default 1 = all)')
    p.add_argument('--no-plot', action='store_true',
                   help='verification only, do not write the PNG')
    p.add_argument('--outdir', default=None,
                   help='where to write the PNG (default: next to the eps file)')
    p.add_argument('--max-report', type=int, default=MAX_REPORT,
                   help='detail at most this many differing planes (default %d)'
                        % MAX_REPORT)
    a = p.parse_args(argv)

    if a.stride < 1:
        p.error('--stride must be >= 1')

    eps_paths = resolve_targets(a.targets)
    if not eps_paths:
        print('nothing to check.')
        return 1
    if a.grid is not None and len(eps_paths) > 1:
        p.error('--grid can only be used with a single eps file')

    total_bad = 0
    failed    = []
    for ep in eps_paths:
        try:
            nbad = check_eps(ep, grid_path=a.grid, stride=a.stride,
                             do_plot=not a.no_plot, outdir=a.outdir,
                             max_report=a.max_report)
        except Exception as exc:
            print('=' * 78)
            print('eps file : %s' % ep)
            print('  ERROR: %s' % exc)
            failed.append(ep)
            total_bad += 1
            continue
        total_bad += nbad

    print('=' * 78)
    print('SUMMARY: %d eps field(s) checked, %d unreadable, %d differing plane(s) '
          'in total  ->  %s'
          % (len(eps_paths), len(failed), total_bad - len(failed),
             'PASS' if total_bad == 0 else 'FAIL'))
    return 0 if total_bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
