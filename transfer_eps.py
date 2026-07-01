#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transfer_eps.py

Transfer a bit-packed IBM eps field from one grid to another.

The eps field (fluid=0 / solid=1 indicator for the sinusoidal valley) is stored
in the tlab-style bit-packed binary format documented in
MyPyLib/functions.py::epsfield (lines 427-484):

    header : 5 x int32 (little endian) = [head_size, nx//8, ny, nz, 0]
             head_size = 20 bytes (= 5*int32); data starts at byte head_size.
    data   : nz planes, each plane is (nx//8)*ny bytes.
             Within a plane the 8 x-values of a byte are packed LSB-first
             (bit -1 -> smallest x); byte order is x-block fastest, then y.
             The geometry is spanwise (z) invariant, so every z-plane is
             identical.

The remap is done on the 2-D (z-invariant) plane: the source eps(y,x) is
resampled onto the destination (y,x) node grid with nearest-neighbour
interpolation, which keeps the field strictly {0,1} (no fractional cells at the
IBM interface). The result is replicated over all destination z-planes and
re-encoded in the same bit-packed format.

Usage
-----
    python3 transfer_eps.py \
        --eps      eps0.1 \
        --grid-src 1024x832x1024/grid_1024x832x1024 \
        --grid-dst 1728x784x1728/grid_1728x784x1728 \
        --out      eps0.1_1728x784x1728

Run with --info to only print a summary (no output written).
"""

import os
import argparse
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HEAD_PARAMS = 5          # number of int32 in the header
HEAD_SIZE   = HEAD_PARAMS * 4


# --------------------------------------------------------------------------- #
# Grid I/O (Fortran sequential unformatted, i4 record markers)
# --------------------------------------------------------------------------- #
def read_grid(path):
    """Return (nmax, scales, x, y, z) from a tlab grid file."""
    with open(path, 'rb') as f:
        np.fromfile(f, '<i4', 1)                       # record marker
        nmax = np.fromfile(f, '<i4', 3)
        np.fromfile(f, '<i4', 1)

        np.fromfile(f, '<i4', 1)
        scales = np.fromfile(f, '<f8', 3)
        np.fromfile(f, '<i4', 1)

        np.fromfile(f, '<i4', 1)
        x = np.fromfile(f, '<f8', nmax[0])
        np.fromfile(f, '<i4', 1)

        np.fromfile(f, '<i4', 1)
        y = np.fromfile(f, '<f8', nmax[1])
        np.fromfile(f, '<i4', 1)

        np.fromfile(f, '<i4', 1)
        z = np.fromfile(f, '<f8', nmax[2])
        np.fromfile(f, '<i4', 1)
    return nmax, scales, x, y, z


# --------------------------------------------------------------------------- #
# eps I/O (bit-packed, LSB-first along x, z-invariant)
# --------------------------------------------------------------------------- #
def read_eps_plane(path):
    """Read one z-plane of the bit-packed eps field -> eps[y, x] in {0,1}.

    Mirrors functions.epsfield: only the first plane is read because the
    geometry is spanwise invariant.
    """
    header = np.fromfile(path, '<i4', HEAD_PARAMS)
    head_size, nx8, ny, nz = int(header[0]), int(header[1]), int(header[2]), int(header[3])
    nx = nx8 * 8
    plane_bytes = nx8 * ny
    with open(path, 'rb') as f:
        f.seek(head_size, 0)
        data = np.fromfile(f, '<i1', plane_bytes)
    # unpack LSB-first (matches int2bit_2: bit -1 -> out[0]); bit index = x + j*nx
    bits = np.unpackbits(data.view(np.uint8), bitorder='little')
    eps = bits.reshape((ny, nx)).astype(np.int8)       # eps[j=y, i=x]
    return eps, header


def write_eps(path, eps2d, nz):
    """Write eps2d[y, x] (in {0,1}) as a bit-packed 3-D field of nz z-planes."""
    ny, nx = eps2d.shape
    if nx % 8 != 0:
        raise ValueError(f"nx ({nx}) must be a multiple of 8 for bit packing")
    nx8 = nx // 8

    # pack the single plane: flatten as x + j*nx (C-order over [j,i]), LSB-first
    plane_bits = eps2d.astype(np.uint8).reshape(-1)
    plane_bytes = np.packbits(plane_bits, bitorder='little').view('<i1')
    assert plane_bytes.size == nx8 * ny, (plane_bytes.size, nx8 * ny)

    header = np.array([HEAD_SIZE, nx8, ny, nz, 0], dtype='<i4')
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        pb = plane_bytes.tobytes()
        for _ in range(nz):                            # replicate over z (invariant)
            f.write(pb)
    return header


# --------------------------------------------------------------------------- #
# Remap
# --------------------------------------------------------------------------- #
def remap_eps(eps_src, x_src, y_src, x_dst, y_dst):
    """Nearest-neighbour resample eps_src[y,x] onto the (y_dst, x_dst) grid.

    Nearest-neighbour keeps the field strictly {0,1}. Points of the destination
    grid outside the source extent take the value of the nearest source node
    (fill_value=None -> extrapolate by nearest).
    """
    interp = RegularGridInterpolator(
        (y_src, x_src), eps_src.astype(np.float64),
        method='nearest', bounds_error=False, fill_value=None)
    Yd, Xd = np.meshgrid(y_dst, x_dst, indexing='ij')  # (ny_dst, nx_dst)
    pts = np.column_stack([Yd.ravel(), Xd.ravel()])
    eps_dst = interp(pts).reshape(Yd.shape)
    return np.rint(eps_dst).astype(np.int8)


# --------------------------------------------------------------------------- #
def surface_profile(eps2d):
    """Topmost solid j per column (-1 if none) -- for reporting only."""
    ny, nx = eps2d.shape
    surf = np.full(nx, -1, dtype=int)
    for i in range(nx):
        solid = np.where(eps2d[:, i] == 1)[0]
        if solid.size:
            surf[i] = solid.max()
    return surf


def main():
    ap = argparse.ArgumentParser(description="Transfer a bit-packed eps field between grids.")
    ap.add_argument('--eps',      default='eps0.1',
                    help='input bit-packed eps field (source grid)')
    ap.add_argument('--grid-src', default='1024x832x1024/grid_1024x832x1024',
                    help='source grid file')
    ap.add_argument('--grid-dst', default='1728x784x1728/grid_1728x784x1728',
                    help='destination grid file')
    ap.add_argument('--out',      default='eps0.1_1728x784x1728',
                    help='output bit-packed eps field (destination grid)')
    ap.add_argument('--info', action='store_true',
                    help='only print a summary; do not write the output file')
    args = ap.parse_args()

    # --- source ---
    nS, sclS, xS, yS, zS = read_grid(args.grid_src)
    eps_src, hdr = read_eps_plane(args.eps)
    nyS, nxS = eps_src.shape
    print("-" * 60)
    print(f"source grid : {args.grid_src}")
    print(f"  nmax          = {tuple(int(v) for v in nS)}")
    print(f"  eps header    = {tuple(int(v) for v in hdr)}")
    print(f"  eps plane     = (ny={nyS}, nx={nxS}), solid fraction = {eps_src.mean():.6f}")
    if (nxS, nyS) != (int(nS[0]), int(nS[1])):
        raise ValueError(f"eps plane {(nxS, nyS)} != source grid (nx,ny) "
                         f"{(int(nS[0]), int(nS[1]))}")

    # --- destination ---
    nD, sclD, xD, yD, zD = read_grid(args.grid_dst)
    nxD, nyD, nzD = int(nD[0]), int(nD[1]), int(nD[2])
    print("-" * 60)
    print(f"dest grid   : {args.grid_dst}")
    print(f"  nmax          = {(nxD, nyD, nzD)}")
    print(f"  x extent src/dst = [{xS[0]:.6g},{xS[-1]:.6g}] / [{xD[0]:.6g},{xD[-1]:.6g}]")
    print(f"  y extent src/dst = [{yS[0]:.6g},{yS[-1]:.6g}] / [{yD[0]:.6g},{yD[-1]:.6g}]")

    # --- remap ---
    eps_dst = remap_eps(eps_src, xS, yS, xD, yD)
    print("-" * 60)
    print(f"remapped eps  = (ny={eps_dst.shape[0]}, nx={eps_dst.shape[1]}), "
          f"solid fraction = {eps_dst.mean():.6f}")
    print(f"  values present = {np.unique(eps_dst)}")
    srfS, srfD = surface_profile(eps_src), surface_profile(eps_dst)
    print(f"  source surface j range = [{srfS.min()}, {srfS.max()}] of ny={nyS}")
    print(f"  dest   surface j range = [{srfD.min()}, {srfD.max()}] of ny={nyD}")

    if args.info:
        print("-" * 60)
        print("--info set: no output written.")
        return

    out_hdr = write_eps(args.out, eps_dst, nzD)
    size = os.path.getsize(args.out)
    print("-" * 60)
    print(f"wrote {args.out}")
    print(f"  header        = {tuple(int(v) for v in out_hdr)}")
    print(f"  file size     = {size} bytes "
          f"(= {HEAD_SIZE} + {nxD // 8}*{nyD}*{nzD})")

    # --- round-trip verification of the written file ---
    chk, chk_hdr = read_eps_plane(args.out)
    ok = np.array_equal(chk, eps_dst)
    print(f"  round-trip plane read-back matches: {ok}")
    if not ok:
        raise SystemExit("ERROR: written eps does not decode back to the remapped field")


if __name__ == '__main__':
    main()
