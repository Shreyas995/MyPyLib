#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intermittency.py — 3-D external/global intermittency from a tlab velocity
(+ optional scalar buoyancy) field.

Velocity (mechanical) intermittency — Ansorge & Mellado (2016), JFM 805:

    γ(x)  = ⟨ H(|ω'| − ω₀) ⟩ ,   ω₀ = e_ω ≡ ω_rms(δ)              (eqs 4.1, 4.2)

Scalar (buoyancy) intermittency — same recipe, substituting the buoyancy
fluctuation for the vorticity fluctuation.  Own physical threshold (NOT a
reuse of ω₀ — different field, different units/dynamics):

    γ_b(x) = ⟨ H(|b'| − b₀) ⟩ ,   b₀ = e_b ≡ b_rms(δ)

Conditional buoyancy statistics — independent of γ_b; uses the *velocity*-
defined instantaneous state H(|ω'|−ω₀) at each snapshot to split the raw
scalar into "inside the mechanically turbulent region" vs "outside it":

    ⟨b⟩_turb(x), ⟨b⟩_quiet(x), ⟨b'²⟩_turb(x), ⟨b'²⟩_quiet(x)

    H         Heaviside step,
    |ω'|      magnitude of the vorticity of the HIGH-PASS velocity field,
    |b'|      magnitude of the HIGH-PASS (spanwise-fluctuation) scalar field,
    ω₀, b₀    thresholds = the rms fluctuation at the BL edge δ — fixed
              PHYSICAL references (NOT a fraction of the max: a max-fraction
              cut collapses because the max is set by near-wall shear / the
              IBM interface spike, exactly the failure the paper warns
              about, §4.1).

The "high-pass" here is the spanwise (z) fluctuation  q' = q − ⟨q⟩_z, i.e.
the turbulent part of the triple decomposition (mean + dispersive removed) —
the same convention PhAvg_rotated.py uses for both velocity and the scalar
buoyancy field (AvgScal IS b; see CLAUDE.md).  With the full 3-D field we get
the TRUE vorticity magnitude
    |ω'| = sqrt(ω'x² + ω'y² + ω'z²),   ω'x = ∂w'/∂y − ∂v'/∂z, …
(a single plane only gives ω'z).

═════════════════════════════════════════════════════════════════════════════
CLUSTER-SAFE.  The COMPUTE path uses ONLY numpy + the standard library — no
scipy, no matplotlib.  matplotlib is imported lazily, only inside --plot
mode, so this runs where the cluster python lacks it.  All field files are
opened strictly read-only ('rb'); nothing in this script ever writes back to
flow.*/scal.* — only new *.npy/*.npz output files are created.

WORKFLOW — never download the whole field, only the chosen 2-D plane(s). The
spanwise-mean <prefix>_xy.npz is ALWAYS written (quick-look / cross-case use,
e.g. results.py's Ri_B collapse view) but it is a time+space AVERAGE: spanwise-
averaging washes out the real patchy turbulent/quiescent structure. For a
single run's spatial detail (what PhAvg_rotated.py's local plots want), request
RAW K/I/J-index planes instead -- no averaging in that direction, so the
plane's other two dimensions keep their real structure:
    # 1. on the cluster: compute γ (+ γ_b + conditional stats, automatically,
    #    if scal.<tag>.1 is present alongside each flow.<tag>.1/2/3), save,
    #    and write RAW planes at the given K (spanwise/z), I (streamwise/x),
    #    J (wall-normal/y) indices -- any/all of the three, comma-separated:
    python3 Intermittency.py --workdir /path/to/case --save-full \\
            --planesK 0,10,20 --planesI 100,500 --planesJ 5,40
    # 2. copy back the small  *_xy.npz / *_slice_*.npz  (a few MB), then LOCALLY:
    python3 Intermittency.py --plot intermittency_slice_z0010.npz
    # extra planes later, no recompute (slices the saved 3-D velocity γ only):
    python3 Intermittency.py --workdir . --from-full intermittency_gamma3d.npy \\
            --planesJ 40
    # skip the scalar path even if scal.* is present (velocity γ only, faster):
    python3 Intermittency.py --workdir . --skip-scalar

── γ FROM THE tlab PLANE FILES (planesK.*/planesJ.*) — the PhAvg_rotated.py port.
A separate, cheaper source than the scarce 3-D flow.* triplets above: the solver
saves planesK.<it> (fixed-z, x–y) and planesJ.<it> (fixed-y, x–z) at MANY
iterations → a long ensemble for eq. 4.1's time-average.  Each slice can resolve
only its ONE in-plane vorticity component, so this is a 2-D-per-plane γ, exactly
like PhAvg_rotated.py's block (NOT the full |ω| of the 3-D path):
    K-plane (x,y): ω_z' = ∂v'/∂x − ∂u'/∂y, high-pass = remove ⟨·⟩_x  (sets ω₀ at δ)
    J-plane (x,z): ω_y' = ∂u'/∂z − ∂w'/∂x, high-pass = remove ⟨·⟩_z  (REUSES K's ω₀)
No destructive field averaging: the only averages are forming the fluctuation and
the eq. 4.1 time-average of the pointwise Heaviside indicator (the definition of γ).
    # K-plane γ(x,y) at k-slot 0 and J-plane γ(x,z) at j-slots 0,2 in one call;
    # --pj-ynodes gives each J-slot's wall-normal grid index for terrain masking
    # (omit if the plane is above the valley crest → all-fluid):
    python3 Intermittency.py --workdir /path/to/case \\
            --from-planesK 0 --from-planesJ 0,2 --pj-ynodes 120,300
    # writes  <prefix>_planesK_kNNNN.npz  and  <prefix>_planesJ_jNNNN.npz
    # layout (nvars, n_kplanes/n_jplanes) is inferred from the file size assuming
    # --pk-nvars/--pj-nvars (default 5 = u,v,w,s1,p); override if PLANES_LOG added
    # log(enstrophy)/log(grad s) columns.  --omega0 lets J run without a K in the
    # same call.  Plot locally the same way: --plot <prefix>_planesK_k0000.npz

Coordinates (tlab engineering): axis0 = y wall-normal, axis1 = x streamwise
(periodic), axis2 = z spanwise (periodic).  Velocity components on disk:
1 = u streamwise, 2 = v wall-normal, 3 = w spanwise  (flow.<tag>.1/2/3).
Scalar (buoyancy) component on disk: scal.<tag>.1, matched to flow.<tag>.*
by <tag> (same tag, missing scal.* for a given tag just drops that snapshot
from the scalar-side averages — the velocity γ is unaffected).

MEMORY: the compute path holds a few field-sized arrays at once (≈6-8x
nx·ny·nz·4B for velocity alone; roughly double that while the scalar path is
also active, since it carries six more float64 accumulators of the same
shape across the whole snapshot loop). Fine up to ~1e9 points on a
big-memory node; for larger grids run on fewer snapshots, --skip-scalar, or
a high-memory queue.
"""

import os
import re
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


def read_field_header(path):
    """Full tlab stream-binary header: (offset, nx, ny, nz, nt)."""
    with open(path, 'rb') as f:
        offset = int(np.fromfile(f, I32, 1)[0])
        dims = np.fromfile(f, I32, 3)
        nt = np.fromfile(f, I32, 1)
    if dims.size < 3 or nt.size < 1:
        raise ValueError("truncated header")
    return offset, int(dims[0]), int(dims[1]), int(dims[2]), int(nt[0])


def valid_field(path, nx, ny, nz):
    """(ok, reason). A real 3-D field file: header dims == grid AND file size
    is EXACTLY offset + nx·ny·nz·8 bytes.  Rejects boundary-condition planes
    (flow.bcs.*), truncated/partial writes, and anything else the glob caught
    whose seek offset would otherwise be garbage."""
    try:
        offset, hnx, hny, hnz, _ = read_field_header(path)
    except Exception as e:                                # noqa: BLE001 (report, don't raise)
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
    Opens the file strictly read-only — never writes back to the raw record."""
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


def _rms_at_row(absfield_row, mask_er, j, nz):
    """rms of |field| at wall-normal index j, masked to interior fluid if
    mask_er is given. absfield_row is (nx, nz)."""
    if mask_er is not None:
        wgt = mask_er[j, :][:, None]
        denom = max(float(np.sum(wgt)) * nz, 1.0)
        return float(np.sqrt(np.sum((absfield_row ** 2) * wgt) / denom))
    return float(np.sqrt(np.mean(absfield_row ** 2)))


def _safe_divide(num, den):
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


# ─────────────────────────────────────────────────────────────────────────────
# Compute γ(x,y,z) [+ γ_b, conditional buoyancy stats] over the given snapshots
# ─────────────────────────────────────────────────────────────────────────────
_FLOW_ITER_RE = re.compile(r'^flow\.(\d+)\.1$')          # ONLY flow.<iteration>.1


def component_quads(args, workdir, nx, ny, nz):
    """List of (u, v, w, scalar_or_None) file quads; each = one snapshot.

    The glob 'flow.*.1' also catches non-field files (e.g. the boundary-
    condition planes flow.bcs.jmax.*.1, whose header offset is garbage and
    crashes the seek).  We therefore accept a velocity file ONLY when its name
    is exactly flow.<digits>.1 AND all three components pass valid_field()
    (header dims == grid, size exactly consistent).  scalar_or_None is
    scal.<tag>.1 matched to flow.<tag>.1 by the SAME numeric <tag>; a snapshot
    missing/failing its scalar file still contributes to gamma (velocity), just
    not to gamma_b / the conditional buoyancy statistics."""
    if args.u:
        if not (args.v and args.w):
            sys.exit("ERROR: --u requires --v and --w (all three components).")
        return [(args.u, args.v, args.w, args.scalar)]
    quads = []
    for uf in sorted(glob.glob(os.path.join(workdir, 'flow.*.1'))):
        base = os.path.basename(uf)
        m = _FLOW_ITER_RE.match(base)
        if not m:                                        # e.g. flow.bcs.jmax.*.1
            print(f"  [skip] {base} — not a flow.<iteration>.1 field file.")
            continue
        vf, wf = uf[:-1] + '2', uf[:-1] + '3'
        bad = False
        for comp in (uf, vf, wf):
            if not os.path.exists(comp):
                print(f"  [skip] {base} — missing {os.path.basename(comp)}.")
                bad = True; break
            ok, why = valid_field(comp, nx, ny, nz)
            if not ok:
                print(f"  [skip] {os.path.basename(comp)} — {why}.")
                bad = True; break
        if bad:
            continue
        tag = m.group(1)                                 # numeric iteration
        sf = os.path.join(os.path.dirname(uf), f'scal.{tag}.1')
        if os.path.exists(sf):
            ok, why = valid_field(sf, nx, ny, nz)
            if not ok:
                print(f"  [skip scalar] scal.{tag}.1 — {why}.")
                sf = None
        else:
            sf = None
        quads.append((uf, vf, wf, sf))
    return quads


def compute_all(args, workdir, x, y, z, mask_er):
    """Returns (fields: dict[name]->3-D array, meta: dict, j_delta).

    A snapshot is DROPPED from every average (not silently counted as fully
    turbulent) if it is unreadable or its |ω'| / b' comes out non-finite — the
    signature of a corrupt or diverged restart/checkpoint dump (e.g. a file
    written at an irregular iteration right before a CFL blow-up: its garbage
    velocities overflow float32, become inf, and inf > ω₀ would otherwise be
    counted as turbulent).  The BL-edge index δ and the thresholds ω₀/b₀ are
    fixed from the FIRST CLEAN snapshot, never a corrupt one."""
    nx, ny, nz = x.size, y.size, z.size
    quads = component_quads(args, workdir, nx, ny, nz)
    if not quads:
        sys.exit("ERROR: no valid flow.<iteration>.1/2/3 velocity triplet found — nothing to do.")
    n_scalar = sum(1 for q in quads if q[3] is not None)
    do_scalar = (not args.skip_scalar) and n_scalar > 0
    print(f"  snapshots to average: {len(quads)}  (with matching scal.*: "
          f"{n_scalar}{'' if do_scalar else ' — scalar path skipped'})")

    j_delta = (int(np.searchsorted(y, args.delta)) if args.delta is not None
               else None)

    gamma3d = np.zeros((ny, nx, nz), dtype=np.float32)
    omega0 = None
    n_used = 0

    if do_scalar:
        gamma3d_b    = np.zeros((ny, nx, nz), dtype=np.float32)
        omega0_b     = None
        n_used_b     = 0
        sum_b_turb   = np.zeros((ny, nx, nz), dtype=np.float64)
        sum_b2_turb  = np.zeros((ny, nx, nz), dtype=np.float64)
        cnt_turb     = np.zeros((ny, nx, nz), dtype=np.float64)
        sum_b_quiet  = np.zeros((ny, nx, nz), dtype=np.float64)
        sum_b2_quiet = np.zeros((ny, nx, nz), dtype=np.float64)
        cnt_quiet    = np.zeros((ny, nx, nz), dtype=np.float64)

    for s, (uf, vf, wf, sf) in enumerate(quads):
        print(f"  [{s + 1}/{len(quads)}] {os.path.basename(uf)} …", flush=True)
        try:
            u = read_full_field(uf, nx, ny, nz)
            v = read_full_field(vf, nx, ny, nz)
            w = read_full_field(wf, nx, ny, nz)
        except (OSError, ValueError) as e:
            print(f"      [skip] unreadable velocity field: {e}")
            continue

        j_delta_cand = j_delta
        if j_delta is None:                               # δ₉₅ from the mean wind
            Umag = np.sqrt(u.mean(axis=(1, 2)) ** 2 + w.mean(axis=(1, 2)) ** 2)
            j_delta_cand = int(np.argmax(Umag >= 0.95 * Umag.max()))

        omega = omega_highpass(u, v, w, x, y, z)          # u,v,w → fluctuations
        del u, v, w

        if not np.isfinite(omega).all():                  # corrupt / diverged snapshot
            nbad = int((~np.isfinite(omega)).sum())
            print(f"      [skip] non-finite |ω'| at {nbad} point(s) — corrupt/diverged field.")
            del omega
            continue

        if j_delta is None:                               # commit δ from this clean snapshot
            j_delta = j_delta_cand
            print(f"      δ (95% wind) at j={j_delta}, z={float(y[j_delta]):.4g}")

        if omega0 is None:                                # ω₀ = e_ω = ω_rms(δ)
            e_omega = _rms_at_row(omega[j_delta, :, :], mask_er, j_delta, nz)
            omega0 = args.factor * e_omega
            print(f"      e_ω = ω_rms(δ) = {e_omega:.4g};  "
                  f"ω₀ = {args.factor:g}·e_ω = {omega0:.4g}")

        H_inst = omega > omega0                           # this snapshot's turbulent mask
        gamma3d += H_inst
        n_used += 1
        del omega

        if do_scalar and sf is not None:
            try:
                b = read_full_field(sf, nx, ny, nz)
            except (OSError, ValueError) as e:
                print(f"      [skip scalar] unreadable scalar field: {e}")
                del H_inst
                continue
            bp = b - b.mean(axis=2, keepdims=True)         # high-pass, same z-mean convention

            if not np.isfinite(bp).all():
                print("      [skip scalar] non-finite b' — corrupt/diverged scalar field.")
                del b, bp, H_inst
                continue

            if omega0_b is None:                           # b₀ = e_b = b_rms(δ)
                e_b = _rms_at_row(np.abs(bp[j_delta, :, :]), mask_er, j_delta, nz)
                omega0_b = args.factor_b * e_b
                print(f"      e_b = b_rms(δ) = {e_b:.4g};  "
                      f"b₀ = {args.factor_b:g}·e_b = {omega0_b:.4g}")

            gamma3d_b += np.abs(bp) > omega0_b

            sum_b_turb   += np.where(H_inst, b, 0.0)
            sum_b2_turb  += np.where(H_inst, bp * bp, 0.0)
            cnt_turb     += H_inst
            sum_b_quiet  += np.where(~H_inst, b, 0.0)
            sum_b2_quiet += np.where(~H_inst, bp * bp, 0.0)
            cnt_quiet    += ~H_inst
            n_used_b     += 1
            del b, bp

        del H_inst

    if n_used == 0:
        sys.exit("ERROR: every snapshot was skipped (unreadable or non-finite) — no γ produced.")
    if n_used < len(quads):
        print(f"  [note] used {n_used}/{len(quads)} snapshots for γ "
              f"({len(quads) - n_used} skipped as unreadable/non-finite).")

    gamma3d /= n_used
    fields = {'gamma': gamma3d}
    meta = {'omega0': omega0, 'factor': args.factor, 'j_delta': j_delta,
            'n_snapshots': len(quads), 'n_used': n_used}

    if do_scalar and n_used_b > 0:
        gamma3d_b /= n_used_b
        fields['gamma_b']      = gamma3d_b
        fields['mean_b_turb']  = _safe_divide(sum_b_turb,  cnt_turb ).astype(np.float32)
        fields['mean_b_quiet'] = _safe_divide(sum_b_quiet, cnt_quiet).astype(np.float32)
        fields['var_b_turb']   = _safe_divide(sum_b2_turb, cnt_turb ).astype(np.float32)
        fields['var_b_quiet']  = _safe_divide(sum_b2_quiet,cnt_quiet).astype(np.float32)
        meta.update({'omega0_b': omega0_b, 'factor_b': args.factor_b,
                     'n_scalar_snapshots': n_scalar, 'n_used_b': n_used_b})
    elif do_scalar:
        print("  [note] scalar path enabled but every scalar snapshot was skipped.")

    return fields, meta, j_delta


# ─────────────────────────────────────────────────────────────────────────────
# Slice extraction + masking + write (multi-field: gamma, gamma_b, bcond stats)
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


def spanwise_mean_fields(fields3d, mask_er):
    """Spanwise-average (axis=2) every field in the dict; nan solid cells."""
    planes = {}
    for name, f3d in fields3d.items():
        p = np.nanmean(f3d, axis=2)
        if mask_er is not None:
            p = np.where(mask_er > 0, p, np.nan)
        planes[name] = p
    return planes


def extract_slice_fields(fields3d, direction, index, x, y, z, mask_er):
    """Slice every field in `fields3d` at the same (direction, index); all
    fields share the grid so axis/label/idx are identical across them."""
    planes = {}
    axis_h = axis_v = h_label = v_label = idx = None
    for name, f3d in fields3d.items():
        plane, axis_h, axis_v, h_label, v_label, idx = extract_slice(
            f3d, direction, index, x, y, z)
        planes[name] = mask_plane(plane, direction, idx, mask_er)
    return planes, axis_h, axis_v, h_label, v_label, idx


def write_plane_npz(out_path, fields, axis_h, axis_v, h_label, v_label, meta):
    """fields: dict[name] -> 2-D array. Each becomes its own array in the npz."""
    payload = {name: f.astype(np.float32) for name, f in fields.items()}
    payload.update(axis_h=np.asarray(axis_h, np.float64),
                    axis_v=np.asarray(axis_v, np.float64),
                    h_label=np.array(h_label), v_label=np.array(v_label),
                    field_names=np.array(list(fields.keys())),
                    meta=np.array(str(meta)))
    np.savez_compressed(out_path, **payload)
    shp = next(iter(fields.values())).shape
    print(f"  wrote {out_path}  fields={list(fields.keys())}  ({shp[1]}×{shp[0]})")


# ─────────────────────────────────────────────────────────────────────────────
# Plot (LOCAL only — matplotlib imported here, never on the cluster path)
# ─────────────────────────────────────────────────────────────────────────────
def plot_npz(npz_path, out_png=None, field=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = np.load(npz_path, allow_pickle=True)
    ax_h, ax_v = d['axis_h'], d['axis_v']
    h_label, v_label = str(d['h_label']), str(d['v_label'])
    meta = str(d['meta'])
    names = [field] if field else (
        [str(n) for n in d['field_names']] if 'field_names' in d.files else ['gamma'])
    stem = os.path.splitext(out_png or npz_path)[0]
    for name in names:
        if name not in d.files:
            print(f"  [warn] field '{name}' not in {npz_path} "
                  f"(has: {list(d.files)}) — skipping.")
            continue
        fplane = d[name]
        is_gamma = name.startswith('gamma')
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
        cf = ax.contourf(ax_h, ax_v, fplane,
                         levels=np.linspace(0, 1, 21) if is_gamma else 21,
                         cmap='hot_r' if is_gamma else 'RdBu_r')
        if is_gamma:
            ax.contour(ax_h, ax_v, np.nan_to_num(fplane), levels=[0.5],
                       colors='cyan', linewidths=1.0)
        plt.colorbar(cf, ax=ax, label=name)
        ax.set_xlabel(h_label); ax.set_ylabel(v_label)
        ax.set_title(f"{name}  [{meta}]", fontsize=8)
        out = f"{stem}.png" if len(names) == 1 else f"{stem}_{name}.png"
        plt.tight_layout(); plt.savefig(out); plt.close(fig)
        print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────────────
def _parse_idx_list(s):
    """'10,15,20' -> [10, 15, 20]; None/'' -> []."""
    if not s:
        return []
    return [int(v) for v in s.split(',') if v.strip() != '']


# ─────────────────────────────────────────────────────────────────────────────
# Intermittency from the tlab plane files (planesK.* / planesJ.*)
# ─────────────────────────────────────────────────────────────────────────────
# A separate, cheaper data source than the scarce 3-D flow.*.1/2/3 triplets: the
# solver saves planesK.<it> (fixed-z, x–y slices) and planesJ.<it> (fixed-y, x–z
# slices) at MANY iterations, giving a long ensemble for the time-average in
# eq. 4.1.  Logic is the port of PhAvg_rotated.py's intermittency block — the
# high-pass (Reynolds-fluctuation) vorticity magnitude thresholded at
# ω₀ = factor·ω_rms(δ) — restricted to the ONE in-plane vorticity component each
# slice can resolve (a K-plane has no ∂/∂z, a J-plane no ∂/∂y):
#   • K-plane (x,y):  ω_z' = ∂v'/∂x − ∂u'/∂y,  high-pass = remove the x-row mean
#                     (exactly PhAvg_rotated.py; x is the in-plane homogeneous dir)
#   • J-plane (x,z):  ω_y' = ∂u'/∂z − ∂w'/∂x,  high-pass = remove the z (spanwise,
#                     homogeneous) mean
# NO destructive spanwise/field averaging is ever applied to the raw data: the
# only averages are (a) subtracting a mean to FORM the fluctuation and (b) the
# eq. 4.1 time-average of the pointwise Heaviside indicator — the definition of γ.
# The threshold ω₀ is fixed once from the K-plane at δ and REUSED for the
# J-planes (single consistent threshold; the components differ, so J is read
# against the K-plane's ω_z-based ω₀ by explicit choice).
_PLANE_ITER_RE = {'planesK': re.compile(r'^planesK\.(\d+)$'),
                  'planesJ': re.compile(r'^planesJ\.(\d+)$')}


def _glob_plane_files(workdir, kind):
    """Sorted list of real planesK.<it>/planesJ.<it> data files (numeric suffix
    only — drops planesK_animation.*, *.npy, *.gif and other glob debris)."""
    rex = _PLANE_ITER_RE[kind]
    hits = []
    for p in glob.glob(os.path.join(workdir, f'{kind}.*')):
        m = rex.match(os.path.basename(p))
        if m:
            hits.append((int(m.group(1)), p))
    return [p for _, p in sorted(hits)]


def _infer_nplanes(path, plane_cells, nvars, kind):
    """Solve n_planes from the file size: size = plane_cells·n_planes·nvars·4."""
    size = os.path.getsize(path)
    denom = plane_cells * nvars * 4
    if size % denom != 0:
        raise ValueError(
            f"{os.path.basename(path)}: size {size}B not divisible by "
            f"plane_cells·nvars·4 = {denom} — check --{'pk' if kind=='planesK' else 'pj'}-nvars.")
    return size // denom


def read_planesK_frame(path, nx, ny, nk, nvars, kslot):
    """(u, v, w) each (ny, nx) from ONE k-slot of a planesK file.  Layout
    (planes.f90): data_k(imax, jmax, nvars·n_kplanes), variable blocks slowest,
    so var vi / k-slot occupies 3rd-dim index vi·nk + kslot.  Read-only."""
    data = np.fromfile(path, dtype=np.float32)          # opens read-only
    expect = nx * ny * nk * nvars
    if data.size != expect:
        raise ValueError(f"{os.path.basename(path)}: {data.size} floats, "
                         f"expected {expect} (nx·ny·nk·nvars).")
    arr = data.reshape((nx, ny, nvars * nk), order='F')
    return (arr[:, :, 0 * nk + kslot].T,                # u  (ny,nx)
            arr[:, :, 1 * nk + kslot].T,                # v
            arr[:, :, 2 * nk + kslot].T)                # w


def read_planesJ_frame(path, nx, nz, nj, nvars, jslot):
    """(u, w) each (nx, nz) from ONE j-slot of a planesJ file.  Layout
    (planes.f90): data_j(imax, nvars·n_jplanes, kmax), variable blocks in the
    MIDDLE dim, so var vi / j-slot occupies 2nd-dim index vi·nj + jslot.
    Read-only."""
    data = np.fromfile(path, dtype=np.float32)
    expect = nx * nz * nj * nvars
    if data.size != expect:
        raise ValueError(f"{os.path.basename(path)}: {data.size} floats, "
                         f"expected {expect} (nx·nz·nj·nvars).")
    arr = data.reshape((nx, nvars * nj, nz), order='F')
    return (arr[:, 0 * nj + jslot, :],                  # u  (nx,nz)
            arr[:, 2 * nj + jslot, :])                  # w


def _ddp(f, d, axis):
    """Central periodic ∂/∂(uniform axis)."""
    return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) * (0.5 / d)


def _jplane_mask(eps, jnode, nz):
    """Interior-fluid mask (nx, nz) for a J-plane at wall-normal grid index
    jnode.  eps(ny,nx) is z-independent (2-D valley), so the solid footprint at
    that height is eps[jnode,:] over x, broadcast over z; eroded one cell in x
    (periodic) to drop the IBM interface ring."""
    m = (eps[jnode, :] < 0.5)                           # fluid = True
    er = m & np.roll(m, 1) & np.roll(m, -1)             # x-neighbours (periodic)
    return np.broadcast_to(er[:, None].astype(np.float64), (er.size, nz)).copy()


def gamma_from_planesK(files, x, y, kslot, nk, nvars, mask_er,
                       factor, delta, vel_max):
    """Two-pass γ(x,y) on one k-slot: pass 1 → ω_rms(y), δ, ω₀ = factor·ω_rms(δ);
    pass 2 → γ = ⟨H(|ω_z'| − ω₀)⟩_t.  Returns a result dict."""
    ny, nx = y.size, x.size
    dx = float(x[1] - x[0])
    mask = mask_er if mask_er is not None else np.ones((ny, nx))

    def _omega(fp):
        u, v, w = read_planesK_frame(fp, nx, ny, nk, nvars, kslot)
        if vel_max and (np.abs(u).max() > vel_max or np.abs(v).max() > vel_max):
            return None, None                           # diverged frame
        up = u - u.mean(axis=1, keepdims=True)          # remove x-row mean
        vp = v - v.mean(axis=1, keepdims=True)
        oz = _ddp(vp, dx, 1) - np.gradient(up, y, axis=0)   # ω_z' = ∂v'/∂x − ∂u'/∂y
        if not np.isfinite(oz).all():
            return None, None
        return oz, (u, w)

    # pass 1 — ω_rms(y) and the mean wind (for auto δ)
    sumsq = np.zeros((ny, nx)); n = 0
    uacc = np.zeros(ny); wacc = np.zeros(ny)
    omega_inst = None
    for fp in files:
        try:
            oz, uw = _omega(fp)
        except (ValueError, OSError):
            continue
        if oz is None:
            continue
        if n == 0:
            omega_inst = oz * mask                       # first clean frame (Fig-2 look)
        sumsq += oz * oz
        uacc += uw[0].mean(axis=1); wacc += uw[1].mean(axis=1)
        n += 1
    if n == 0:
        raise RuntimeError("no clean planesK frames (all unreadable/diverged/non-finite)")
    num = np.sum(sumsq * mask, axis=1)
    den = n * np.sum(mask, axis=1)
    omega_rms = np.sqrt(_safe_divide(num, den))

    if delta is not None:
        j_delta = int(np.searchsorted(y, delta))
    else:                                                # δ₉₅ from the ensemble mean wind
        umag = np.sqrt((uacc / n) ** 2 + (wacc / n) ** 2)
        j_delta = int(np.argmax(umag >= 0.95 * umag.max()))
    j_delta = min(max(j_delta, 0), ny - 1)
    e_omega = float(omega_rms[j_delta])
    omega0 = factor * e_omega

    # pass 2 — γ(x,y)
    acc = np.zeros((ny, nx)); n2 = 0
    for fp in files:
        try:
            oz, _ = _omega(fp)
        except (ValueError, OSError):
            continue
        if oz is None:
            continue
        acc += (np.abs(oz) > omega0).astype(np.float64)
        n2 += 1
    gamma_field = (acc / n2) * mask
    gden = np.sum(mask, axis=1)
    gamma_profile = _safe_divide(np.sum(gamma_field, axis=1), gden)
    print(f"    [planesK k={kslot}] {n2} frame(s); δ at j={j_delta}, "
          f"z={float(y[j_delta]):.4g}; e_ω=ω_rms(δ)={e_omega:.4g}; "
          f"ω₀={factor:g}·e_ω={omega0:.4g}; max γ={float(np.nanmax(gamma_profile)):.2f}")
    return {'gamma': gamma_field, 'gamma_profile': gamma_profile,
            'omega_hp': omega_inst, 'omega0': omega0, 'e_omega': e_omega,
            'j_delta': j_delta, 'omega_rms': omega_rms, 'n_used': n2}


def gamma_from_planesJ(files, x, z, jslot, nj, nvars, omega0, mask_row, vel_max):
    """Single-pass γ(x,z) on one j-slot with a REUSED ω₀ (from the K-plane).
    ω_y' = ∂u'/∂z − ∂w'/∂x, high-pass = remove the spanwise (z) mean."""
    nx, nz = x.size, z.size
    dx = float(x[1] - x[0]); dz = float(z[1] - z[0])
    mask = mask_row if mask_row is not None else np.ones((nx, nz))

    acc = np.zeros((nx, nz)); n = 0
    omega_inst = None
    for fp in files:
        try:
            u, w = read_planesJ_frame(fp, nx, nz, nj, nvars, jslot)
        except (ValueError, OSError):
            continue
        if vel_max and (np.abs(u).max() > vel_max or np.abs(w).max() > vel_max):
            continue
        up = u - u.mean(axis=1, keepdims=True)          # remove spanwise (z) mean
        wp = w - w.mean(axis=1, keepdims=True)
        oy = _ddp(up, dz, 1) - _ddp(wp, dx, 0)          # ω_y' = ∂u'/∂z − ∂w'/∂x
        if not np.isfinite(oy).all():
            continue
        if n == 0:
            omega_inst = oy * mask
        acc += (np.abs(oy) > omega0).astype(np.float64)
        n += 1
    if n == 0:
        raise RuntimeError("no clean planesJ frames (all unreadable/diverged/non-finite)")
    gamma_field = (acc / n) * mask
    print(f"    [planesJ j={jslot}] {n} frame(s); ω₀={omega0:.4g} (reused from K); "
          f"mean γ={float(np.nanmean(np.where(mask > 0, gamma_field, np.nan))):.2f}")
    return {'gamma': gamma_field, 'omega_hp': omega_inst, 'omega0': omega0,
            'n_used': n}


def run_plane_intermittency(args, workdir, x, y, z, eps, mask_er):
    """Compute γ on the requested K- and/or J-plane slots from planesK.*/planesJ.*.
    The K path sets ω₀; the J path reuses it (or an explicit --omega0)."""
    nx, ny, nz = x.size, y.size, z.size
    kslots = _parse_idx_list(args.from_planesK)
    jslots = _parse_idx_list(args.from_planesJ)
    omega0 = args.omega0                                 # explicit override (optional)
    prefix = args.out_prefix

    # ── K-planes (x, y) ──────────────────────────────────────────────────────
    if kslots:
        kfiles = _glob_plane_files(workdir, 'planesK')
        if not kfiles:
            sys.exit("ERROR: --from-planesK given but no planesK.<iter> files found.")
        nk = args.pk_nplanes or _infer_nplanes(kfiles[0], nx * ny, args.pk_nvars, 'planesK')
        print(f"  planesK: {len(kfiles)} file(s); layout nvars={args.pk_nvars}, "
              f"n_kplanes={nk}")
        for ks in kslots:
            if ks >= nk:
                print(f"  [skip] k-slot {ks} >= n_kplanes {nk}.")
                continue
            res = gamma_from_planesK(kfiles, x, y, ks, nk, args.pk_nvars, mask_er,
                                     args.factor, args.delta, args.vel_max)
            if omega0 is None:
                omega0 = res['omega0']                   # first K-slot fixes the reused ω₀
            meta = {'source': 'planesK', 'k_slot': ks, 'omega0': res['omega0'],
                    'e_omega': res['e_omega'], 'factor': args.factor,
                    'j_delta': res['j_delta'], 'n_used': res['n_used'],
                    'component': "omega_z' = dv'/dx - du'/dy"}
            out = os.path.join(workdir, f'{prefix}_planesK_k{ks:04d}.npz')
            fields = {'gamma': res['gamma']}
            if res['omega_hp'] is not None:
                fields['omega_hp'] = res['omega_hp']
            write_plane_npz(out, fields, x, y, 'x', 'z (wall-normal)', meta)

    # ── J-planes (x, z) ──────────────────────────────────────────────────────
    if jslots:
        if omega0 is None:
            sys.exit("ERROR: --from-planesJ needs a threshold — also request a "
                     "--from-planesK (to derive ω₀ at δ) or pass --omega0 explicitly.")
        jfiles = _glob_plane_files(workdir, 'planesJ')
        if not jfiles:
            sys.exit("ERROR: --from-planesJ given but no planesJ.<iter> files found.")
        nj = args.pj_nplanes or _infer_nplanes(jfiles[0], nx * nz, args.pj_nvars, 'planesJ')
        print(f"  planesJ: {len(jfiles)} file(s); layout nvars={args.pj_nvars}, "
              f"n_jplanes={nj}; ω₀={omega0:.4g} (reused from K).")
        ynodes = _parse_idx_list(args.pj_ynodes)
        for i, js in enumerate(jslots):
            if js >= nj:
                print(f"  [skip] j-slot {js} >= n_jplanes {nj}.")
                continue
            mask_row = None
            if ynodes and i < len(ynodes):
                if eps is None:
                    print(f"  [warn] --pj-ynodes given but no eps — j-slot {js} unmasked.")
                else:
                    mask_row = _jplane_mask(eps, ynodes[i], nz)
            elif eps is not None:
                print(f"  [note] no --pj-ynode for j-slot {js} — terrain mask skipped "
                      "(valid if this plane is above the valley crest).")
            res = gamma_from_planesJ(jfiles, x, z, js, nj, args.pj_nvars,
                                     omega0, mask_row, args.vel_max)
            meta = {'source': 'planesJ', 'j_slot': js, 'omega0': omega0,
                    'omega0_origin': 'planesK (reused)', 'n_used': res['n_used'],
                    'y_node': ynodes[i] if (ynodes and i < len(ynodes)) else None,
                    'component': "omega_y' = du'/dz - dw'/dx"}
            out = os.path.join(workdir, f'{prefix}_planesJ_j{js:04d}.npz')
            # store as (v=z_span, h=x): transpose the (nx,nz) fields to (nz,nx)
            fields = {'gamma': res['gamma'].T}
            if res['omega_hp'] is not None:
                fields['omega_hp'] = res['omega_hp'].T
            write_plane_npz(out, fields, x, z, 'x', 'z (spanwise)', meta)

    if not kslots and not jslots:
        print("  [note] neither --from-planesK nor --from-planesJ requested.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workdir', default='.', help='case dir (grid + flow.*/scal.*)')
    ap.add_argument('--grid', default=None, help='grid file [workdir/grid]')
    ap.add_argument('--eps', default=None,
                    help='eps_save.npy (ny,nx) [workdir/eps_save.npy]')
    ap.add_argument('--u', default=None, help='explicit u file (else glob flow.*.1)')
    ap.add_argument('--v', default=None, help='explicit v file')
    ap.add_argument('--w', default=None, help='explicit w file')
    ap.add_argument('--scalar', default=None,
                    help='explicit scalar (buoyancy) file, paired with --u/--v/--w')
    ap.add_argument('--skip-scalar', action='store_true',
                    help='ignore scal.<tag>.1 even if present (velocity gamma only)')
    ap.add_argument('--factor', type=float, default=1.0,
                    help='omega0 = factor * e_omega  (Ansorge sweeps 1/8...3; default 1)')
    ap.add_argument('--factor-b', type=float, default=1.0,
                    help='b0 = factor_b * e_b, the buoyancy threshold (own scale; default 1)')
    ap.add_argument('--delta', type=float, default=None,
                    help='BL-edge height (y units) for e_omega/e_b; default = auto delta_95')
    ap.add_argument('--planesK', default=None,
                    help='comma-separated z-index (K, spanwise) planes to write RAW '
                         '(no averaging), e.g. "0,10,20" -- each an (x, y-wall-normal) '
                         'plane, the same orientation as a raw planesK.<iter> file')
    ap.add_argument('--planesI', default=None,
                    help='comma-separated x-index (I, streamwise) planes to write RAW '
                         '-- each a (z-spanwise, y-wall-normal) plane')
    ap.add_argument('--planesJ', default=None,
                    help='comma-separated y-index (J, wall-normal) planes to write RAW '
                         '-- each an (x, z-spanwise) plane')
    # --- γ directly from the tlab plane files (planesK.*/planesJ.*) ----------
    ap.add_argument('--from-planesK', default=None,
                    help='compute γ(x,y) from planesK.<iter> files at the given '
                         'comma-separated k-slot index(es) — ω_z, x-mean high-pass '
                         '(PhAvg_rotated.py logic). Sets ω₀ for the J-planes.')
    ap.add_argument('--from-planesJ', default=None,
                    help='compute γ(x,z) from planesJ.<iter> files at the given '
                         'comma-separated j-slot index(es) — ω_y, spanwise-mean '
                         'high-pass; reuses the K-plane ω₀ (or --omega0).')
    ap.add_argument('--pk-nvars', type=int, default=5,
                    help='planesK variables per slot (u,v,w,s1,p) [5]')
    ap.add_argument('--pk-nplanes', type=int, default=None,
                    help='planesK k-slots per file [inferred from size]')
    ap.add_argument('--pj-nvars', type=int, default=5,
                    help='planesJ variables per slot [5]')
    ap.add_argument('--pj-nplanes', type=int, default=None,
                    help='planesJ j-slots per file [inferred from size]')
    ap.add_argument('--pj-ynodes', default=None,
                    help='wall-normal grid index of each --from-planesJ slot '
                         '(comma list, same order) for terrain masking; omit if '
                         'the plane is above the valley crest (all-fluid).')
    ap.add_argument('--omega0', type=float, default=None,
                    help='explicit threshold (bypasses δ); lets --from-planesJ run '
                         'without a --from-planesK in the same call.')
    ap.add_argument('--vel-max', type=float, default=1.5,
                    help='reject a plane frame whose |u|,|v|,|w| exceeds this '
                         '(diverged/unphysical) [1.5]')
    ap.add_argument('--save-full', action='store_true',
                    help='also save the whole 3-D field(s) (big; stays on cluster)')
    ap.add_argument('--from-full', default=None,
                    help='slice a previously saved 3-D gamma .npy (velocity only, no recompute)')
    ap.add_argument('--out-prefix', default='intermittency')
    ap.add_argument('--plot', default=None, help='LOCAL: plot a saved *.npz')
    ap.add_argument('--field', default=None,
                    help='LOCAL: which field to plot from --plot npz (default: all)')
    args = ap.parse_args()

    # ---- LOCAL plot mode ----------------------------------------------------
    if args.plot:
        plot_npz(args.plot, field=args.field)
        return

    workdir = args.workdir
    x, y, z = read_grid(args.grid or os.path.join(workdir, 'grid'))
    nx, ny, nz = x.size, y.size, z.size
    print(f"  grid: nx={nx} ny={ny} nz={nz}")
    eps = load_eps(args.eps or os.path.join(workdir, 'eps_save.npy'), ny, nx)
    mask_er = interior_fluid_mask(1.0 - eps) if eps is not None else None
    if eps is not None:
        print("  eps loaded → IBM interface ring eroded from stats/output.")

    # ---- γ from the tlab plane files (planesK.*/planesJ.*) ------------------
    # A distinct, cheaper source from the 3-D flow.* path below: it consumes the
    # frequently-saved plane slices directly, mirroring PhAvg_rotated.py.
    if args.from_planesK is not None or args.from_planesJ is not None:
        run_plane_intermittency(args, workdir, x, y, z, eps, mask_er)
        return

    # ---- field(s): recompute, or slice a previously saved velocity gamma ----
    if args.from_full:
        gamma3d = np.load(args.from_full)
        print(f"  loaded gamma3d {gamma3d.shape} from {args.from_full}")
        fields3d = {'gamma': gamma3d}
        meta = {'source': os.path.basename(args.from_full)}
    else:
        fields3d, meta, j_delta = compute_all(args, workdir, x, y, z, mask_er)
        if args.save_full:
            for name, f3d in fields3d.items():
                fp = os.path.join(workdir, f'{args.out_prefix}_{name}3d.npy')
                np.save(fp, f3d)
                print(f"  wrote {fp}  ({f3d.nbytes / 1e9:.2f} GB, stays on cluster)")

    # ---- always: spanwise-averaged plane(s) — quick-look / cross-case use
    # (e.g. results.py's Ri_B collapse view reads this). This is a time+space
    # AVERAGE, not a substitute for the RAW per-plane output below — spanwise-
    # averaging washes out real patchy turbulent/quiescent structure.
    planes = spanwise_mean_fields(fields3d, mask_er)
    fp = os.path.join(workdir, f'{args.out_prefix}_xy.npz')
    write_plane_npz(fp, planes, x, y, 'x', 'z (wall-normal)',
                    {**meta, 'reduction': 'spanwise-mean'})

    # ---- requested RAW planes (no averaging in the slicing direction) --------
    # Each is still the time-average over snapshots (eq 4.1 IS a time average —
    # that part is physical, not a flaw) at ONE fixed K/I/J index; unlike the
    # spanwise mean above, the plane's other two dimensions are untouched, so
    # real spatial structure survives. Same naming as the old single
    # --slice/--index (<prefix>_slice_<axis><idx>.npz) so existing consumers
    # (results.py) keep working unchanged — just loop over as many indices
    # as requested, across all three directions.
    _requests = ([('z', i) for i in _parse_idx_list(args.planesK)] +
                 [('x', i) for i in _parse_idx_list(args.planesI)] +
                 [('y', i) for i in _parse_idx_list(args.planesJ)])
    if not _requests:
        print("  [note] no --planesK/--planesI/--planesJ given — only the "
              "spanwise-mean plane was written.")
    for _direction, _index in _requests:
        planes, ax_h, ax_v, hl, vl, idx = extract_slice_fields(
            fields3d, _direction, _index, x, y, z, mask_er)
        out = os.path.join(workdir,
                           f'{args.out_prefix}_slice_{_direction}{idx:04d}.npz')
        write_plane_npz(out, planes, ax_h, ax_v, hl, vl,
                        {**meta, 'slice': f'{_direction}={idx}'})


if __name__ == '__main__':
    main()
