#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BuoyancyFlux3D.py — build tlab `avg_flux` phase-average files from scarce 3-D snapshots.

This is the Python counterpart of the DNS routine
`tlab/src/statistics/avg_phase.f90` (AvgPhaseFlux / AvgPhaseCalcFlux).  On the
cluster the solver accumulates, EVERY iteration, the spanwise(z)-averaged raw
velocity-scalar products and writes

    avg_flux<start>_<end>.{1,2,3}   with   .1 = ⟨u·s⟩_z , .2 = ⟨v·s⟩_z , .3 = ⟨w·s⟩_z

(s = the first scalar = buoyancy b; v = wall-normal, so .2 is the vertical
buoyancy flux ⟨v·s⟩ = meteorological ⟨w·θ⟩).  Each file holds `avg_planes+1`
planes of imax·jmax float64: planes 1..avg_planes are the per-iteration spanwise
means, the final plane is the phase (time) average.  The stored quantity is the
RAW product, spanwise-averaged over ALL k (÷ nz) with no fluctuation subtraction
and no IBM mask — exactly what avg_phase.f90 writes; the mean subtraction that
turns it into a turbulent flux happens downstream in PhAvgAllPlanes.py /
PhAvg_rotated.py.

The Fortran path reads every iteration in-core, so it cannot be used for the
FULL 3-D field dumps (`flow.<it>.{1,2,3}`, `scal.<it>.1`), which are scarce
(≈1 per 500 iterations).  This script reproduces the same output FORMAT from
those scarce snapshots: one space-average plane per 3-D snapshot found, plus a
final phase-average plane.  With a single snapshot the file has exactly 2
identical planes (space + phase) — nothing to time-average over.  The result is
byte-format-compatible with, and consumed by, `MyPyLib/PhAvgAllPlanes.py`.

Coordinates (tlab engineering): axis0 = y wall-normal, axis1 = x streamwise
(periodic), axis2 = z spanwise (periodic).  Velocity on disk: 1 = u streamwise,
2 = v WALL-NORMAL (vertical), 3 = w spanwise.  Scalar/buoyancy: scal.<it>.1.

CLUSTER-SAFE: numpy + stdlib only.  Every field file is opened strictly
read-only ('rb') — the raw flow.*/scal.* records are never modified; only new
avg_flux* output files are created.

Usage
-----
    # in the case directory (needs `grid` and one or more flow/scal snapshots):
    python3 BuoyancyFlux3D.py --workdir /path/to/case --range 501_1000
    #   -> avg_flux501_1000.1 / .2 / .3   (readable by PhAvgAllPlanes.py)
    # <range> defaults to  <firsttag>_<lasttag>  of the discovered snapshots;
    # set it to MATCH the case's existing avg_flow<range> token so
    # PhAvgAllPlanes.py pairs the flux with the flow/stress/scal of the same set.
"""

import os
import re
import sys
import glob
import argparse
import numpy as np

I32 = np.dtype('<i4')
F64 = np.dtype('<f8')

# A snapshot is dropped from the average if ANY component is non-finite (NaN/Inf)
# or holds |value| above this cap (finite but physically impossible — a diverged
# / corrupt restart dump), so its garbage never poisons the spanwise mean.
MAX_ABS_DEFAULT = 1.0e4

_FLOW_ITER_RE = re.compile(r'^flow\.(\d+)\.1$')   # ONLY flow.<iteration>.1


# ─────────────────────────────────────────────────────────────────────────────
# Readers (self-contained; numpy only — mirror Intermittency.py / functions.py)
# ─────────────────────────────────────────────────────────────────────────────
def read_grid(grid_path):
    """tlab grid: Fortran sequential unformatted, 5 records. Returns x, y, z."""
    with open(grid_path, 'rb') as f:
        np.fromfile(f, I32, 1)                         # open record 1
        nmax = np.fromfile(f, I32, 3)                  # nx, ny, nz
        np.fromfile(f, I32, 2)                         # close 1 + open 2
        np.fromfile(f, F64, 3)                         # Lx, Ly, Lz (scales)
        np.fromfile(f, I32, 2)                         # close 2 + open 3
        x = np.fromfile(f, F64, int(nmax[0])); np.fromfile(f, I32, 2)
        y = np.fromfile(f, F64, int(nmax[1])); np.fromfile(f, I32, 2)
        z = np.fromfile(f, F64, int(nmax[2]))
    return x, y, z


def read_full_header(path):
    """Full tlab stream-binary header. Returns (offset, nx, ny, nz, nt, params),
    params being the (offset-20)//8 float64 trailing values (rtime, visc, …)."""
    with open(path, 'rb') as f:
        offset = int(np.fromfile(f, I32, 1)[0])
        dims = np.fromfile(f, I32, 3)
        nt = np.fromfile(f, I32, 1)
        if dims.size < 3 or nt.size < 1:
            raise ValueError("truncated header")
        nparams = (offset - 5 * 4) // 8
        params = np.fromfile(f, F64, nparams) if nparams > 0 else np.empty(0, F64)
    return offset, int(dims[0]), int(dims[1]), int(dims[2]), int(nt[0]), params


def valid_field(path, nx, ny, nz):
    """(ok, reason). A real 3-D field: header dims == grid AND file size EXACTLY
    offset + nx·ny·nz·8 bytes.  Rejects boundary-condition planes (flow.bcs.*),
    truncated/partial writes, and anything else the glob caught whose seek offset
    would otherwise be garbage."""
    try:
        offset, hnx, hny, hnz, _, _ = read_full_header(path)
    except Exception as e:                                 # noqa: BLE001
        return False, f"unreadable header ({e})"
    if offset <= 0:
        return False, f"bad header offset {offset}"
    if (hnx, hny, hnz) != (nx, ny, nz):
        return False, f"header dims {hnx}x{hny}x{hnz} != grid {nx}x{ny}x{nz}"
    try:
        actual = os.path.getsize(path)
    except OSError as e:
        return False, str(e)
    expect = offset + nx * ny * nz * 8
    if actual != expect:
        return False, f"size {actual}B != expected {expect}B (truncated/not a 3-D field)"
    return True, ""


def read_full_field(path, nx, ny, nz, dtype_out=np.float32):
    """Read a component's full 3-D field (ny, nx, nz), one z-plane at a time.
    Opens strictly read-only — never writes back to the raw record."""
    offset = read_full_header(path)[0]
    field = np.empty((ny, nx, nz), dtype=dtype_out)
    with open(path, 'rb') as f:
        f.seek(offset)
        for k in range(nz):
            field[:, :, k] = np.fromfile(f, F64, nx * ny).reshape(ny, nx)
    return field


def field_flaws(fld, max_abs):
    """(n_nan, n_inf, n_big): counts of NaN, of Inf, and of finite |value|>max_abs
    in a 3-D field.  Any nonzero count means the field is corrupt — a single NaN/Inf
    anywhere poisons the spanwise mean, which (like avg_phase.f90) is taken over ALL
    k, so the whole snapshot is skipped.  Mirrors the plane-corruption rule in
    PhAvgAllPlanes.py (NaN/Inf → skip; |val|>MAX_ABS → skip)."""
    nan = int(np.isnan(fld).sum())
    inf = int(np.isinf(fld).sum())
    big = int((np.isfinite(fld) & (np.abs(fld) > max_abs)).sum())
    return nan, inf, big


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot discovery — (u, v, w, scalar, tag) quintuples, one per usable snapshot
# ─────────────────────────────────────────────────────────────────────────────
def snapshot_sets(workdir, nx, ny, nz):
    """List of dicts {'tag','u','v','w','s'} sorted by iteration.

    Anchored on flow.<iteration>.1 (the same numeric-<tag> glob Intermittency.py
    uses so flow.bcs.* is skipped).  All of flow.<tag>.1/2/3 and scal.<tag>.1 must
    exist and pass valid_field(); a snapshot missing/failing any is skipped."""
    sets = []
    for uf in sorted(glob.glob(os.path.join(workdir, 'flow.*.1'))):
        base = os.path.basename(uf)
        m = _FLOW_ITER_RE.match(base)
        if not m:                                          # e.g. flow.bcs.jmax.*.1
            print(f"  [skip] {base} — not a flow.<iteration>.1 field file.")
            continue
        tag = m.group(1)
        vf, wf = uf[:-1] + '2', uf[:-1] + '3'
        sf = os.path.join(os.path.dirname(uf), f'scal.{tag}.1')
        bad = False
        for label, p in (('u', uf), ('v', vf), ('w', wf), ('s', sf)):
            if not os.path.exists(p):
                print(f"  [skip] snapshot {tag} — missing {os.path.basename(p)}.")
                bad = True; break
            ok, why = valid_field(p, nx, ny, nz)
            if not ok:
                print(f"  [skip] {os.path.basename(p)} — {why}.")
                bad = True; break
        if bad:
            continue
        sets.append(dict(tag=tag, u=uf, v=vf, w=wf, s=sf))
    # deterministic, sorted by numeric iteration
    return sorted(sets, key=lambda r: int(r['tag']))


# ─────────────────────────────────────────────────────────────────────────────
# tlab writer — header (copied from the source flow header) + planes
# ─────────────────────────────────────────────────────────────────────────────
def write_avgflux_file(path, offset, nx, ny, nplanes, nt, params, planes):
    """Write a tlab stream-binary avg_flux component file: int32 offset,nx,ny,nz,nt
    then float64 params then `nplanes` planes of (ny,nx) float64 (i-fast/j-slow).
    nz-header field = nplanes (plane count), matching avg_phase.f90.  `offset`,
    `nt` and `params` are copied verbatim from the source flow header (data
    provenance), so the ONLY field that differs from that flow field's header is
    nz (set to the plane count, as the avg format requires)."""
    assert offset == 20 + params.size * 8, \
        f"header offset {offset} inconsistent with {params.size} params"
    assert len(planes) == nplanes
    with open(path, 'wb') as f:
        np.array([offset, nx, ny, nplanes, nt], dtype=I32).tofile(f)
        params.astype(F64).tofile(f)
        for pl in planes:
            # C-order ravel of (ny,nx) → plane[j*nx + i] (i fast, j slow), the tlab layout
            np.ascontiguousarray(pl, dtype=F64).tofile(f)


# ─────────────────────────────────────────────────────────────────────────────
def compute(args):
    workdir = args.workdir
    x, y, z = read_grid(os.path.join(workdir, 'grid'))
    nx, ny, nz = x.size, y.size, z.size
    print(f"Grid: nx={nx}, ny={ny}, nz={nz}")

    sets = snapshot_sets(workdir, nx, ny, nz)
    if not sets:
        sys.exit("ERROR: no usable snapshot (flow.<it>.1/2/3 + scal.<it>.1) found.")
    print(f"Found {len(sets)} candidate snapshot(s): {', '.join(r['tag'] for r in sets)}")

    # Space-average planes per component (1=u·s, 2=v·s, 3=w·s), one per snapshot.
    planes = {1: [], 2: [], 3: []}
    used_tags = []
    used_flow1 = []                                         # flow.<it>.1 of each snapshot used

    for s, rec in enumerate(sets):
        tag = rec['tag']
        print(f"  [{s + 1}/{len(sets)}] snapshot {tag} …", flush=True)
        try:
            u = read_full_field(rec['u'], nx, ny, nz)
            v = read_full_field(rec['v'], nx, ny, nz)
            w = read_full_field(rec['w'], nx, ny, nz)
            sc = read_full_field(rec['s'], nx, ny, nz)
        except (OSError, ValueError) as e:
            print(f"      [skip] unreadable field: {e}")
            continue

        # NaN/Inf guard — drop the WHOLE snapshot if ANY component holds a NaN,
        # an Inf, or a finite |value|>max_abs. One non-finite cell poisons the
        # spanwise mean (taken over all k), and skipping the whole snapshot keeps
        # u,v,w,s from the same instant time-consistent for the products.
        skip = False
        for nm, fld in (('u', u), ('v', v), ('w', w), ('s', sc)):
            nan, inf, big = field_flaws(fld, args.max_abs)
            if nan or inf or big:
                print(f"      [skip] snapshot {tag} — field {nm} has "
                      f"{nan} NaN, {inf} Inf, {big} |val|>{args.max_abs:.0e}.")
                skip = True
                break
        if skip:
            continue

        # Space average = spanwise (z, axis 2) mean over ALL k, matching the
        # Fortran Σ_k /g(3)%size (no fluctuation, no IBM mask). → (ny, nx).
        planes[1].append((u * sc).mean(axis=2))
        planes[2].append((v * sc).mean(axis=2))
        planes[3].append((w * sc).mean(axis=2))
        used_tags.append(tag)
        used_flow1.append(rec['u'])
        del u, v, w, sc

    n_used = len(used_tags)
    if n_used == 0:
        sys.exit("ERROR: every snapshot was corrupt/unreadable — nothing to write.")
    print(f"Averaged over {n_used} clean snapshot(s): {', '.join(used_tags)}")

    # File range token — it MUST match the case's avg_flow<range> token, else
    # PhAvgAllPlanes.py (which discovers tokens from avg_flow*.1 and then looks for
    # avg_flux<token>.{1,2,3}) will not pair the flux.  So when --range is omitted,
    # auto-adopt the directory's avg_flow token; fall back to the snapshot iteration
    # tags only when no avg_flow* is present.
    if args.range:
        rng = args.range
    else:
        flow_toks = sorted({re.fullmatch(r'avg_flow(\d+_\d+)\.1', f).group(1)
                            for f in os.listdir(workdir)
                            if re.fullmatch(r'avg_flow\d+_\d+\.1', f)})
        if len(flow_toks) == 1:
            rng = flow_toks[0]
            print(f"  [range] adopting avg_flow token {rng} → pairs with PhAvgAllPlanes.py.")
        elif len(flow_toks) > 1:
            sys.exit(f"ERROR: several avg_flow tokens present {flow_toks}; pass --range "
                     f"START_END to say which window this flux belongs to.")
        else:
            rng = f"{used_tags[0]}_{used_tags[-1]}"
            print(f"  [range] no avg_flow*.1 found; using snapshot tags → {rng}. Set "
                  f"--range to match the case's avg_flow token if you add one later.")
    if not re.fullmatch(r'\d+_\d+', rng):
        sys.exit(f"ERROR: --range '{rng}' must be START_END (e.g. 501_1000).")

    # Header (offset, nt, params incl. rtime) copied VERBATIM from the source flow
    # field — it is the PROVENANCE of the data: it records exactly which
    # flow.<it>.{1,2,3} snapshot produced this avg_flux, so a mislabelled file can
    # still be traced back to its source iteration/time.  Only the nz-field is
    # changed to the plane count (as the tlab avg format requires).  With several
    # snapshots the newest (last) one is embedded, matching the Fortran nt=itime
    # (the window end).
    src_flow = used_flow1[-1]
    offset, _hnx, _hny, _hnz, nt_src, params = read_full_header(src_flow)
    nplanes = n_used + 1                                    # n_used space planes + 1 phase plane

    for comp in (1, 2, 3):
        space_planes = planes[comp]
        phase_plane = np.mean(space_planes, axis=0)         # phase (time) average
        all_planes = space_planes + [phase_plane]           # final plane = phase average
        out_path = os.path.join(workdir, f"{args.out_prefix}{rng}.{comp}")
        write_avgflux_file(out_path, offset, nx, ny, nplanes, nt_src, params, all_planes)
        print(f"  wrote {os.path.basename(out_path)}: {nplanes} planes "
              f"({n_used} space + 1 phase), {nx}x{ny} each.")
    print(f"  header copied from {os.path.basename(src_flow)} "
          f"(nt={nt_src}, {params.size} params incl. rtime) — data provenance preserved.")

    # Console summary of the wall-normal (vertical) flux ⟨v·s⟩_z, its phase plane.
    vphase = np.mean(planes[2], axis=0)
    print("---------------------------------------------------------------")
    print("avg_flux written. Wall-normal (vertical) buoyancy flux ⟨v·s⟩_z (phase avg):")
    print(f"   range token        : {rng}   ({nplanes} planes/component)")
    print(f"   ⟨v·s⟩_z  mean       : {vphase.mean():+.6e}")
    print(f"   ⟨v·s⟩_z  min / max  : {vphase.min():+.6e} / {vphase.max():+.6e}")
    print("   (turbulent flux = ⟨v·s⟩ − ⟨v⟩⟨s⟩ is formed downstream by "
          "PhAvgAllPlanes.py / PhAvg_rotated.py.)")
    print("---------------------------------------------------------------")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workdir', default='.', help="case directory (holds grid, flow.*, scal.*).")
    ap.add_argument('--range', default=None,
                    help="file range token START_END (e.g. 501_1000). Default: "
                         "<firsttag>_<lasttag> of the discovered snapshots. Set it to "
                         "match the case's avg_flow<range> token.")
    ap.add_argument('--out-prefix', default='avg_flux',
                    help="output filename prefix (default 'avg_flux' → avg_flux<range>.{1,2,3}).")
    ap.add_argument('--max-abs', type=float, default=MAX_ABS_DEFAULT,
                    help="drop a snapshot if any component exceeds this magnitude (default 1e4).")
    args = ap.parse_args()
    compute(args)


if __name__ == '__main__':
    main()
