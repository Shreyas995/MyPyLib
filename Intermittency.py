#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intermittency.py — 3-D external/global intermittency from a tlab velocity field.

Method (Ansorge & Mellado 2016, J. Fluid Mech. 805, 611-635):

    γ(x) = ⟨ H(|ω'| − ω₀) ⟩ ,   ω₀ = e_ω ≡ ω_rms(δ)              (eqs 4.1, 4.2)

    H         Heaviside step,
    |ω'|      magnitude of the vorticity of the HIGH-PASS velocity field,
    ω₀        threshold = the rms fluctuation-vorticity at the BL edge δ — a
              fixed PHYSICAL reference (NOT a fraction of the max: a max-fraction
              cut collapses because the max is set by the near-wall shear / IBM
              interface spike, exactly the failure the paper warns about, §4.1).

The "high-pass" here is the spanwise (z) fluctuation  u' = u − ⟨u⟩_z, i.e. the
turbulent part of the triple decomposition (mean + dispersive removed).  With
the full 3-D field we get the TRUE vorticity magnitude
    |ω'| = sqrt(ω'x² + ω'y² + ω'z²),   ω'x = ∂w'/∂y − ∂v'/∂z, …
(a single plane only gives ω'z).

═════════════════════════════════════════════════════════════════════════════
CLUSTER-SAFE.  The COMPUTE path uses ONLY numpy + the standard library — no
scipy, no matplotlib.  matplotlib is imported lazily, only inside --plot mode,
so this runs where the cluster python lacks it.

WORKFLOW — never download the whole field, only the chosen 2-D plane:
    # 1. on the cluster: compute γ(x,y,z), save it, write ONE requested plane:
    python3 Intermittency.py --workdir /path/to/case --save-full --slice z --index 0
    # 2. copy back the small  *_slice_*.npz  (a few MB), then LOCALLY:
    python3 Intermittency.py --plot intermittency_slice_z0000.npz
    # extra planes later, no recompute (slice the saved 3-D γ):
    python3 Intermittency.py --workdir . --from-full intermittency_gamma3d.npy \\
            --slice y --index 40

Coordinates (tlab engineering): axis0 = y wall-normal, axis1 = x streamwise
(periodic), axis2 = z spanwise (periodic).  Velocity components on disk:
1 = u streamwise, 2 = v wall-normal, 3 = w spanwise  (flow.<tag>.1/2/3).

MEMORY: the compute path holds a few field-sized float32 arrays at once
(≈ 6–8 × nx·ny·nz · 4 B).  Fine up to ~1e9 points on a big-memory node; for
larger grids run on fewer snapshots or a high-memory queue.
"""

import os
import sys
import glob
import argparse
import numpy as np

I32 = np.dtype('<i4')
F64 = np.dtype('<f8')


# ─────────────────────────────────────────────────────────────────────────────
# Readers (self-contained; numpy only — mirror functions.py / prepare_fields.py)
# ─────────────────────────────────────────────────────────────────────────────
def read_grid(grid_path):
    """tlab grid: Fortran sequential unformatted, 5 records. Returns x, y, z."""
    with open(grid_path, 'rb') as f:
        np.fromfile(f, I32, 1)                        # open record 1
        nmax = np.fromfile(f, I32, 3)                 # nx, ny, nz
        np.fromfile(f, I32, 2)                        # close 1 + open 2
        np.fromfile(f, F64, 3)                        # Lx, Ly, Lz (scales)
        np.fromfile(f, I32, 2)                        # close 2 + open 3
        x = np.fromfile(f, F64, int(nmax[0])); np.fromfile(f, I32, 2)
        y = np.fromfile(f, F64, int(nmax[1])); np.fromfile(f, I32, 2)
        z = np.fromfile(f, F64, int(nmax[2]))
    return x, y, z


def read_header(path):
    """tlab stream-binary header. Returns byte offset to the field data."""
    with open(path, 'rb') as f:
        return int(np.fromfile(f, I32, 1)[0])


def read_full_field(path, nx, ny, nz, dtype_out=np.float32):
    """Read a component's full 3-D field (ny, nx, nz), one z-plane at a time."""
    hdr = read_header(path)
    field = np.empty((ny, nx, nz), dtype=dtype_out)
    with open(path, 'rb') as f:
        f.seek(hdr)
        for k in range(nz):
            field[:, :, k] = np.fromfile(f, F64, nx * ny).reshape(ny, nx)
    return field


def load_eps(eps_path, ny, nx):
    """Optional 2-D IBM indicator eps_save.npy (ny, nx); 1 = solid, 0 = fluid."""
    if not eps_path or not os.path.exists(eps_path):
        return None
    eps = np.load(eps_path)
    if eps.shape != (ny, nx):
        print(f"  [warn] eps shape {eps.shape} != ({ny},{nx}) — ignoring eps.")
        return None
    return eps.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Vorticity of the high-pass (spanwise-fluctuation) field
# ─────────────────────────────────────────────────────────────────────────────
def interior_fluid_mask(mask0):
    """Erode the fluid mask by one cell so the IBM interface ring is dropped
    (its ∂u/∂y spike must set neither the threshold nor the field)."""
    m = mask0.astype(bool)
    er = m & np.roll(m, 1, 1) & np.roll(m, -1, 1)        # x-neighbours (periodic)
    er[1:, :] &= m[:-1, :]; er[:-1, :] &= m[1:, :]       # y-neighbours
    return er.astype(np.float64)


def omega_highpass(u, v, w, x, y, z):
    """|ω'| of the spanwise-fluctuation field.  u, v, w are (ny,nx,nz) and are
    OVERWRITTEN in place with their fluctuations to save memory."""
    dx = float(x[1] - x[0]); dz = float(z[1] - z[0])     # x, z uniform + periodic

    def ddx(f):                                          # ∂/∂x  (axis 1, periodic)
        return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) * (0.5 / dx)

    def ddz(f):                                          # ∂/∂z  (axis 2, periodic)
        return (np.roll(f, -1, 2) - np.roll(f, 1, 2)) * (0.5 / dz)

    def ddy(f):                                          # ∂/∂y  (axis 0, non-uniform)
        return np.gradient(f, y, axis=0)

    u -= u.mean(axis=2, keepdims=True)                   # high-pass = remove ⟨·⟩_z
    v -= v.mean(axis=2, keepdims=True)
    w -= w.mean(axis=2, keepdims=True)

    osq = ddy(w) - ddz(v); osq *= osq                    # ω'x = ∂w'/∂y − ∂v'/∂z
    tmp = ddz(u) - ddx(w); osq += tmp * tmp              # ω'y = ∂u'/∂z − ∂w'/∂x
    tmp = ddx(v) - ddy(u); osq += tmp * tmp              # ω'z = ∂v'/∂x − ∂u'/∂y
    return np.sqrt(osq, out=osq)


# ─────────────────────────────────────────────────────────────────────────────
# Compute γ(x,y,z)  (average of the indicator over the given snapshots)
# ─────────────────────────────────────────────────────────────────────────────
def component_triplets(args, workdir):
    """List of (u,v,w) file triplets; each triplet = one snapshot to average."""
    if args.u:
        if not (args.v and args.w):
            sys.exit("ERROR: --u requires --v and --w (all three components).")
        return [(args.u, args.v, args.w)]
    trip = []
    for uf in sorted(glob.glob(os.path.join(workdir, 'flow.*.1'))):
        vf, wf = uf[:-1] + '2', uf[:-1] + '3'            # flow.<tag>.1 → .2/.3
        if os.path.exists(vf) and os.path.exists(wf):
            trip.append((uf, vf, wf))
    return trip


def compute_gamma3d(args, workdir, x, y, z, mask_er):
    nx, ny, nz = x.size, y.size, z.size
    trips = component_triplets(args, workdir)
    if not trips:
        sys.exit("ERROR: no flow.*.1/2/3 velocity triplet found — nothing to do.")
    print(f"  snapshots to average: {len(trips)}")

    j_delta = (int(np.searchsorted(y, args.delta)) if args.delta is not None
               else None)
    gamma3d = np.zeros((ny, nx, nz), dtype=np.float32)
    omega0 = None
    for s, (uf, vf, wf) in enumerate(trips):
        print(f"  [{s + 1}/{len(trips)}] {os.path.basename(uf)} …", flush=True)
        u = read_full_field(uf, nx, ny, nz)
        v = read_full_field(vf, nx, ny, nz)
        w = read_full_field(wf, nx, ny, nz)

        if s == 0 and j_delta is None:                   # δ₉₅ from the mean wind
            Umag = np.sqrt(u.mean(axis=(1, 2)) ** 2 + w.mean(axis=(1, 2)) ** 2)
            j_delta = int(np.argmax(Umag >= 0.95 * Umag.max()))
            print(f"      δ (95% wind) at j={j_delta}, z={float(y[j_delta]):.4g}")

        omega = omega_highpass(u, v, w, x, y, z)         # u,v,w → fluctuations
        del u, v, w

        if s == 0:                                       # ω₀ = e_ω = ω_rms(δ)
            row = omega[j_delta, :, :]                   # (nx, nz) at the BL edge
            if mask_er is not None:
                wgt = mask_er[j_delta, :][:, None]
                denom = max(float(np.sum(wgt)) * nz, 1.0)
                e_omega = float(np.sqrt(np.sum((row ** 2) * wgt) / denom))
            else:
                e_omega = float(np.sqrt(np.mean(row ** 2)))
            omega0 = args.factor * e_omega
            print(f"      e_ω = ω_rms(δ) = {e_omega:.4g};  "
                  f"ω₀ = {args.factor:g}·e_ω = {omega0:.4g}")

        gamma3d += (omega > omega0)                      # accumulate H(|ω'|−ω₀)
        del omega

    gamma3d /= len(trips)                                # → intermittency factor
    return gamma3d, omega0, j_delta


# ─────────────────────────────────────────────────────────────────────────────
# Slice extraction + masking + write
# ─────────────────────────────────────────────────────────────────────────────
def extract_slice(field3d, direction, index, x, y, z):
    """Return (plane2d, axis_h, axis_v, h_label, v_label, resolved_index).
    plane2d has shape (len(axis_v), len(axis_h)) ready for contourf(h, v, plane)."""
    ny, nx, nz = field3d.shape
    d = direction.lower()
    if d == 'z':                                         # fixed spanwise → (x, z_wn)
        idx = index % nz
        return field3d[:, :, idx], x, y, 'x', 'z (wall-normal)', idx
    if d == 'y':                                         # fixed height → (x, z_span)
        idx = index % ny
        return field3d[idx, :, :].T, x, z, 'x', 'z (spanwise)', idx
    if d == 'x':                                         # fixed streamwise → (z_span, z_wn)
        idx = index % nx
        return field3d[:, idx, :], z, y, 'z (spanwise)', 'z (wall-normal)', idx
    raise ValueError("slice direction must be x, y or z")


def mask_plane(plane, direction, idx, mask_er):
    """Blank solid/interface cells (nan) on an extracted plane."""
    if mask_er is None:
        return plane
    if direction == 'z':                                 # plane (ny, nx)
        return np.where(mask_er > 0, plane, np.nan)
    if direction == 'y':                                 # plane (nz, nx); mask over x
        return np.where(mask_er[idx, :][None, :] > 0, plane, np.nan)
    return np.where(mask_er[:, idx][:, None] > 0, plane, np.nan)   # x: (ny, nz)


def write_plane_npz(out_path, plane, axis_h, axis_v, h_label, v_label, meta):
    np.savez_compressed(
        out_path, gamma=plane.astype(np.float32),
        axis_h=np.asarray(axis_h, np.float64), axis_v=np.asarray(axis_v, np.float64),
        h_label=np.array(h_label), v_label=np.array(v_label), meta=np.array(str(meta)))
    print(f"  wrote {out_path}  ({plane.shape[1]}×{plane.shape[0]})")


# ─────────────────────────────────────────────────────────────────────────────
# Plot (LOCAL only — matplotlib imported here, never on the cluster path)
# ─────────────────────────────────────────────────────────────────────────────
def plot_npz(npz_path, out_png=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = np.load(npz_path, allow_pickle=True)
    field, ax_h, ax_v = d['gamma'], d['axis_h'], d['axis_v']
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    cf = ax.contourf(ax_h, ax_v, field, levels=np.linspace(0, 1, 21), cmap='hot_r')
    ax.contour(ax_h, ax_v, np.nan_to_num(field), levels=[0.5],
               colors='cyan', linewidths=1.0)
    plt.colorbar(cf, ax=ax, label=r'intermittency $\gamma$')
    ax.set_xlabel(str(d['h_label'])); ax.set_ylabel(str(d['v_label']))
    ax.set_title(f"Intermittency  [{str(d['meta'])}]", fontsize=8)
    out_png = out_png or os.path.splitext(npz_path)[0] + '.png'
    plt.tight_layout(); plt.savefig(out_png); plt.close(fig)
    print(f"  wrote {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workdir', default='.', help='case dir (grid + flow.*)')
    ap.add_argument('--grid', default=None, help='grid file [workdir/grid]')
    ap.add_argument('--eps', default=None,
                    help='eps_save.npy (ny,nx) [workdir/eps_save.npy]')
    ap.add_argument('--u', default=None, help='explicit u file (else glob flow.*.1)')
    ap.add_argument('--v', default=None, help='explicit v file')
    ap.add_argument('--w', default=None, help='explicit w file')
    ap.add_argument('--factor', type=float, default=1.0,
                    help='ω₀ = factor·e_ω  (Ansorge sweeps 1/8…3; default 1)')
    ap.add_argument('--delta', type=float, default=None,
                    help='BL-edge height (y units) for e_ω; default = auto δ₉₅')
    ap.add_argument('--slice', default=None, choices=['x', 'y', 'z'],
                    help='write this plane for download')
    ap.add_argument('--index', type=int, default=0,
                    help='plane index (0-based; negative wraps from the end)')
    ap.add_argument('--save-full', action='store_true',
                    help='also save the whole 3-D γ (big; stays on cluster)')
    ap.add_argument('--from-full', default=None,
                    help='slice a previously saved 3-D γ .npy (no recompute)')
    ap.add_argument('--out-prefix', default='intermittency')
    ap.add_argument('--plot', default=None, help='LOCAL: plot a saved *.npz')
    args = ap.parse_args()

    # ---- LOCAL plot mode ----------------------------------------------------
    if args.plot:
        plot_npz(args.plot)
        return

    workdir = args.workdir
    x, y, z = read_grid(args.grid or os.path.join(workdir, 'grid'))
    nx, ny, nz = x.size, y.size, z.size
    print(f"  grid: nx={nx} ny={ny} nz={nz}")
    eps = load_eps(args.eps or os.path.join(workdir, 'eps_save.npy'), ny, nx)
    mask_er = interior_fluid_mask(1.0 - eps) if eps is not None else None
    if eps is not None:
        print("  eps loaded → IBM interface ring eroded from stats/output.")

    # ---- γ(x,y,z): recompute, or slice a previously saved one ---------------
    if args.from_full:
        gamma3d = np.load(args.from_full)
        print(f"  loaded γ3d {gamma3d.shape} from {args.from_full}")
        meta = {'source': os.path.basename(args.from_full)}
    else:
        gamma3d, omega0, j_delta = compute_gamma3d(args, workdir, x, y, z, mask_er)
        meta = {'omega0': omega0, 'factor': args.factor, 'j_delta': j_delta}
        if args.save_full:
            fp = os.path.join(workdir, f'{args.out_prefix}_gamma3d.npy')
            np.save(fp, gamma3d)
            print(f"  wrote {fp}  ({gamma3d.nbytes / 1e9:.2f} GB, stays on cluster)")

    # ---- always: spanwise-averaged γ(x,z) — the small valley intermittency --
    g_xy = np.nanmean(gamma3d, axis=2)
    if mask_er is not None:
        g_xy = np.where(mask_er > 0, g_xy, np.nan)
    fp = os.path.join(workdir, f'{args.out_prefix}_gamma_xy.npz')
    write_plane_npz(fp, g_xy, x, y, 'x', 'z (wall-normal)',
                    {**meta, 'reduction': 'spanwise-mean'})

    # ---- requested plane ----------------------------------------------------
    if args.slice:
        plane, ax_h, ax_v, hl, vl, idx = extract_slice(
            gamma3d, args.slice, args.index, x, y, z)
        plane = mask_plane(plane, args.slice, idx, mask_er)
        out = os.path.join(workdir,
                           f'{args.out_prefix}_slice_{args.slice}{idx:04d}.npz')
        write_plane_npz(out, plane, ax_h, ax_v, hl, vl,
                        {**meta, 'slice': f'{args.slice}={idx}'})


if __name__ == '__main__':
    main()
