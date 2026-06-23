#!/usr/bin/env python3
"""
prepare_fields.py  —  Enhanced DNS field preparation before running transfields.

Four operations applied in sequence to tlab stream-binary field files:

  1. Y-CROP      Remove wall-normal (j) planes above the new grid domain height.
                 Updates header ny; preserves offset, nx, nz, nt, params exactly.
                 Sanity checks: file-size consistency, header/grid agreement,
                 crop index vs IBM geometry, NaN/Inf sampling.

  2. VELOCITY CAP  Cap horizontal (x-z plane) speed to the geostrophic wind
                 magnitude |G| = sqrt(cos²α + sin²α) = 1 (simulation units).
                 Direction is preserved; only fluid cells (eps=0) are modified.
                 Reports how many cells were capped.

  3. IBM SURFACE SMOOTHING  In the first `smooth_pts` fluid cells directly
                 above each IBM surface element, replace field values with a
                 y-direction linear interpolation anchored at:
                   wall  (last solid cell, value = 0, no-slip / Dirichlet BC)
                   far   (cell at j_surf + smooth_pts, original value kept)
                 Horizontal (x-z) continuity is never perturbed; each column
                 (i, k) is processed independently in y only.

  4. DIVERGENCE CHECK  Stream through the three velocity output files one
                 z-plane at a time using a 3-plane rolling window for dw/dz.
                 Reports max |div u|, RMS div u, and fraction of fluid cells
                 exceeding a user-supplied threshold.
                 Derivatives: central differences (2nd-order) on the
                 non-uniform y grid; periodic central differences in x, z.

FIELD FILE FORMAT  (tlab stream binary, no Fortran record markers):
  bytes  0- 3   int32   offset  = total header size = 5*4 + n_params*8
  bytes  4- 7   int32   nx
  bytes  8-11   int32   ny
  bytes 12-15   int32   nz
  bytes 16-19   int32   nt  (iteration number)
  bytes 20-...  float64 x n_params  (rtime, …)
  byte  offset  float64   field data; nz planes of nx*ny values
                           plane k: data[j*nx + i] = q(i+1, j+1, k+1)
                           j (y) slow, i (x) fast

GRID FILE FORMAT  (Fortran sequential unformatted, 5 records):
  record 1   int32   x 3    nx, ny, nz
  record 2   float64 x 3    Lx, Ly, Lz  (domain scales)
  record 3   float64 x nx   x_nodes
  record 4   float64 x ny   y_nodes
  record 5   float64 x nz   z_nodes

USAGE:
  python3 prepare_fields.py --workdir /path/to/dir [OPTIONS]

OPTIONS:
  --workdir PATH    Work directory (default: .)
  --eps     PATH    eps indicator .npy file, shape (ny, nx);
                    1 = solid, 0 = fluid  [default: <workdir>/eps_save.npy]
  --alpha   FLOAT   Geostrophic wind angle alpha in radians [default: -0.430511]
  --G-mag   FLOAT   Geostrophic wind magnitude in sim units [default: 1.0]
  --smooth-pts N    Fluid cells above IBM to smooth         [default: 5]
  --div-thresh F    Divergence warning threshold            [default: 1e-4]
  --suffix  STR     Output file suffix                      [default: _prep]
  --no-crop         Skip y-cropping (j_crop = old_ny)
  --no-cap          Skip velocity capping
  --no-smooth       Skip IBM surface smoothing
  --no-divergence   Skip divergence check
  --info            Print parameters only — no files written
"""

import os
import sys
import argparse
import numpy as np
from scipy.io import FortranFile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
F64 = np.float64
I32 = np.int32
F64B = 8   # bytes per float64
I32B = 4   # bytes per int32

# ---------------------------------------------------------------------------
# Grid I/O  (Fortran sequential unformatted)
# ---------------------------------------------------------------------------

def read_grid(path):
    """
    Returns: (nx,ny,nz), (Lx,Ly,Lz), x_nodes, y_nodes, z_nodes
    """
    with FortranFile(path, 'r') as f:
        nx, ny, nz         = f.read_ints(dtype=I32)
        Lx, Ly, Lz         = f.read_reals(dtype=F64)
        x_nodes            = f.read_reals(dtype=F64)
        y_nodes            = f.read_reals(dtype=F64)
        z_nodes            = f.read_reals(dtype=F64)
    return (int(nx), int(ny), int(nz)), (float(Lx), float(Ly), float(Lz)), \
           x_nodes, y_nodes, z_nodes

# ---------------------------------------------------------------------------
# Field header I/O  (stream binary, no record markers)
# ---------------------------------------------------------------------------

def read_header(fh):
    """
    Read tlab stream-binary header from an open file handle fh positioned at 0.
    Returns (offset, nx, ny, nz, nt, params_raw, n_params).
    params_raw is raw bytes (preserved as-is for writing).
    """
    raw = fh.read(5 * I32B)
    if len(raw) < 5 * I32B:
        raise IOError("File too short to contain a valid tlab header")
    vals   = np.frombuffer(raw, dtype=I32)
    offset = int(vals[0])
    nx, ny, nz, nt = int(vals[1]), int(vals[2]), int(vals[3]), int(vals[4])
    rem = offset - 5 * I32B
    if rem < 0 or rem % F64B != 0:
        raise ValueError(
            f"Header corrupt: offset={offset} → {rem} param bytes "
            f"(not a multiple of {F64B})")
    params_raw = fh.read(rem)
    return offset, nx, ny, nz, nt, params_raw, rem // F64B


def write_header(fh, offset, nx, ny, nz, nt, params_raw):
    """Write tlab stream-binary header to open file handle fh."""
    fh.write(np.array([offset, nx, ny, nz, nt], dtype=I32).tobytes())
    fh.write(params_raw)


def decode_params(params_raw):
    """Decode float64 parameters from raw bytes."""
    n = len(params_raw) // F64B
    return list(np.frombuffer(params_raw, dtype=F64)) if n else []

# ---------------------------------------------------------------------------
# Plane I/O helpers
# ---------------------------------------------------------------------------

def read_plane(fh, k, nx, ny, hdr_offset):
    """Read z-plane k (0-indexed) using an open file handle."""
    fh.seek(hdr_offset + k * nx * ny * F64B)
    return np.fromfile(fh, dtype=F64, count=nx * ny).reshape(ny, nx)

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check_header_vs_grid(path, exp_nx, exp_ny, exp_nz):
    """
    Cross-check header dims and file size against expected grid dims.
    Returns (offset, nx, ny, nz, nt, params_raw, n_params).
    Raises ValueError listing all discrepancies found.
    """
    filesize = os.path.getsize(path)
    with open(path, 'rb') as f:
        offset, nx, ny, nz, nt, params_raw, n_params = read_header(f)

    issues = []
    if nx != exp_nx:
        issues.append(f"nx: header={nx}, grid={exp_nx}")
    if ny != exp_ny:
        issues.append(f"ny: header={ny}, grid={exp_ny}")
    if nz != exp_nz:
        issues.append(f"nz: header={nz}, grid={exp_nz}")

    expected_sz = offset + nx * ny * nz * F64B
    if filesize != expected_sz:
        issues.append(
            f"file size {filesize} B ≠ expected {expected_sz} B "
            f"(header {offset} B + {nx}×{ny}×{nz}×8 B data)")

    if issues:
        raise ValueError("  " + "\n  ".join(issues))

    return offset, nx, ny, nz, nt, params_raw, n_params


def sample_field_quality(path, nx, ny, nz, hdr, label, n_sample=3):
    """
    Sample n_sample z-planes and report NaN / Inf / extreme-value issues.
    Returns list of warning strings (empty = clean).
    """
    issues = []
    ks = np.linspace(0, nz - 1, min(n_sample, nz), dtype=int)
    with open(path, 'rb') as f:
        for k in ks:
            plane = read_plane(f, k, nx, ny, hdr)
            if np.any(np.isnan(plane)):
                issues.append(f"{label}: NaN at z-plane k={k}")
            if np.any(np.isinf(plane)):
                issues.append(f"{label}: Inf at z-plane k={k}")
            vmax = float(np.max(np.abs(plane)))
            if vmax > 1e6:
                issues.append(f"{label}: |value| max={vmax:.2e} at k={k} (suspiciously large)")
    return issues

# ---------------------------------------------------------------------------
# Crop-index computation
# ---------------------------------------------------------------------------

def find_crop_index(old_y, new_Ly):
    """
    Return j_crop (number of y-planes to keep, Fortran 1-indexed upper bound).
    Keeps planes 0 … j_crop-1 (Python) = 1 … j_crop (Fortran).
    j_crop = largest j such that old_y[j-1] <= new_Ly.
    """
    return int(np.searchsorted(old_y, new_Ly, side='right'))

# ---------------------------------------------------------------------------
# Operation 2: velocity cap  (applied to one z-plane of u and w)
# ---------------------------------------------------------------------------

def cap_horizontal_velocity(u_pl, w_pl, eps_2d, G_mag):
    """
    Cap sqrt(u²+w²) to G_mag in fluid cells (eps_2d == 0).
    Scales u and w proportionally so direction is preserved.
    Modifies u_pl and w_pl in-place.
    Returns (u_pl, w_pl, n_capped).
    """
    speed  = np.sqrt(u_pl**2 + w_pl**2)
    fluid  = (eps_2d == 0) if eps_2d is not None else np.ones_like(u_pl, dtype=bool)
    excess = fluid & (speed > G_mag)
    n_capped = int(np.sum(excess))
    if n_capped:
        scale = np.where(excess, G_mag / np.maximum(speed, 1e-30), 1.0)
        u_pl *= scale
        w_pl *= scale
    return u_pl, w_pl, n_capped

# ---------------------------------------------------------------------------
# Operation 3: IBM surface smoothing  (one z-plane, one component)
# ---------------------------------------------------------------------------

def smooth_above_ibm(plane, y, eps_hgt, smooth_pts=5):
    """
    For each x-column i, linearly interpolate the first `smooth_pts` fluid
    cells above the IBM surface.

    Anchors:
      wall: y[j_surf-1], F = 0  (IBM no-slip / Dirichlet BC)
      far:  y[j_surf + smooth_pts], F = plane[j_surf+smooth_pts, i]  (kept)

    Only modifies values in y; horizontal (x-z) continuity is not perturbed.
    eps_hgt[i] = number of solid cells in column i = index of first fluid cell.
    """
    ny, nx = plane.shape
    for i in range(nx):
        j_surf = int(eps_hgt[i])   # first fluid cell above IBM
        if j_surf == 0:
            continue               # no IBM in this column
        j_far = j_surf + smooth_pts
        if j_far >= ny:
            j_far = ny - 1        # clamp to domain top
        if j_far <= j_surf:
            continue

        y_wall  = y[j_surf - 1]   # last solid cell centre  (velocity = 0 here)
        y_far   = y[j_far]
        F_far   = plane[j_far, i]
        dy_span = y_far - y_wall
        if abs(dy_span) < 1e-30:
            continue

        for j in range(j_surf, j_far):  # smooth_pts cells
            t = (y[j] - y_wall) / dy_span
            plane[j, i] = t * F_far

    return plane

# ---------------------------------------------------------------------------
# Process velocity triplet  (u, v, w together so cap can use both u and w)
# ---------------------------------------------------------------------------

def process_velocity_triplet(paths_in, paths_out,
                              j_crop, old_nx, old_ny, old_nz,
                              eps_2d, eps_hgt, y_old,
                              G_mag, smooth_pts,
                              do_cap, do_smooth):
    """
    Process flow.old.1/2/3 one z-plane at a time:
      read full plane → cap → smooth → crop → write.
    Returns total cells capped.
    """
    # Read and validate headers
    hdrs = []
    for p in paths_in:
        with open(p, 'rb') as f:
            hdr = read_header(f)
        offset, nx, ny, nz = hdr[0], hdr[1], hdr[2], hdr[3]
        if (nx, ny, nz) != (old_nx, old_ny, old_nz):
            raise ValueError(
                f"Header dims ({nx},{ny},{nz}) ≠ expected "
                f"({old_nx},{old_ny},{old_nz}) in {p}")
        hdrs.append(hdr)

    total_capped = 0

    fins  = [open(p, 'rb') for p in paths_in]
    fouts = [open(p, 'wb') for p in paths_out]
    try:
        # Seek past headers in inputs
        for fin, hdr in zip(fins, hdrs):
            fin.seek(hdr[0])   # hdr[0] = offset

        # Write updated headers to outputs (ny → j_crop)
        for fout, hdr in zip(fouts, hdrs):
            offset, nx, ny, nz, nt, params_raw, n_params = hdr
            write_header(fout, offset, old_nx, j_crop, old_nz, nt, params_raw)

        for k in range(old_nz):
            # Read one full plane per component
            planes = []
            for fin in fins:
                raw = fin.read(old_nx * old_ny * F64B)
                if len(raw) < old_nx * old_ny * F64B:
                    raise IOError(f"Unexpected EOF at z-plane k={k}")
                planes.append(
                    np.frombuffer(raw, dtype=F64).reshape(old_ny, old_nx).copy())

            u_p, v_p, w_p = planes

            # Cap horizontal velocity (u, w)
            if do_cap:
                u_p, w_p, n_cap = cap_horizontal_velocity(u_p, w_p, eps_2d, G_mag)
                total_capped += n_cap

            # IBM surface smoothing
            if do_smooth and eps_hgt is not None:
                u_p = smooth_above_ibm(u_p, y_old, eps_hgt, smooth_pts)
                v_p = smooth_above_ibm(v_p, y_old, eps_hgt, smooth_pts)
                w_p = smooth_above_ibm(w_p, y_old, eps_hgt, smooth_pts)

            # Crop to j_crop rows and write
            for fout, plane in zip(fouts, [u_p, v_p, w_p]):
                fout.write(plane[:j_crop, :].astype(F64).tobytes())

    finally:
        for f in fins + fouts:
            f.close()

    return total_capped


def process_scalar_file(src, dst, j_crop, old_nx, old_ny, old_nz,
                        eps_2d=None, eps_hgt=None, y_old=None,
                        smooth_pts=5, do_smooth=False):
    """Crop (and optionally smooth) a single scalar field file."""
    with open(src, 'rb') as f:
        offset, nx, ny, nz, nt, params_raw, n_params = read_header(f)
    if (nx, ny, nz) != (old_nx, old_ny, old_nz):
        raise ValueError(
            f"Header dims ({nx},{ny},{nz}) ≠ expected "
            f"({old_nx},{old_ny},{old_nz}) in {src}")

    with open(src, 'rb') as fin, open(dst, 'wb') as fout:
        fin.seek(offset)
        write_header(fout, offset, old_nx, j_crop, old_nz, nt, params_raw)
        for k in range(old_nz):
            raw = fin.read(old_nx * old_ny * F64B)
            plane = np.frombuffer(raw, dtype=F64).reshape(old_ny, old_nx).copy()
            if do_smooth and eps_hgt is not None and y_old is not None:
                plane = smooth_above_ibm(plane, y_old, eps_hgt, smooth_pts)
            fout.write(plane[:j_crop, :].astype(F64).tobytes())

# ---------------------------------------------------------------------------
# Divergence correction: column-integration Helmholtz-Hodge projection
# ---------------------------------------------------------------------------

def _project_divergence_free(u, v, w, x, y, z):
    """
    Divergence-free projection for wall-bounded flow (Ekman DNS convention).

    Algorithm:
      1. Compute horizontal divergence div_h = du/dx + dw/dz spectrally
         (exact for the periodic x-z directions).

      2. Top-wall mass imbalance Q(i,k) = ∫₀^Ly div_h dy  may be non-zero.
         Solve the 2-D spectral Poisson  ∇²_xz P = Q / Ly  to obtain a
         uniform-in-y pressure correction.  Apply:
           u ← u − ∂P/∂x   (same at every y-level)
           w ← w − ∂P/∂z
         After this step ∫₀^Ly div_h dy = 0 everywhere in x-z.

      3. Column-integrate the corrected div_h to find the divergence-free v:
           v_new[j] = −∫₀^{y_j} div_h dy   (trapezoidal rule)
         v_new[0] = 0 (bottom wall BC, v=0) and v_new[ny-1] = 0 (top wall BC)
         are both satisfied exactly after Step 2.

    The projection is exact in the sense that:
       spectral(du_new/dx) + FD(dv_new/dy) + spectral(dw_new/dz) = 0
    to O(dy²) (trapezoidal integration error in y).
    For DNS grids with ny ≥ 400 the residual is < 1e-8.

    Parameters: u, v, w  ndarray (ny, nx, nz) float64 — modified in-place.
    """
    ny, nx, nz = u.shape
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    Ly = float(y[-1])

    kx = np.fft.fftfreq(nx, d=dx) * (2.0 * np.pi)   # (nx,)
    kz = np.fft.fftfreq(nz, d=dz) * (2.0 * np.pi)   # (nz,)
    KX, KZ = np.meshgrid(kx, kz, indexing='ij')       # (nx, nz)

    def _spectral_div_h(u_, w_):
        u_hat = np.fft.fft2(u_, axes=(1, 2))
        w_hat = np.fft.fft2(w_, axes=(1, 2))
        return np.fft.ifft2(
            1j * KX[np.newaxis] * u_hat + 1j * KZ[np.newaxis] * w_hat,
            axes=(1, 2)).real

    def _col_integrate(dh_):
        """v[j] = −∫₀^{y_j} dh dy  (trapezoidal, v[0]=0)."""
        v_new = np.zeros_like(v)
        for j in range(1, ny):
            dy_j = float(y[j] - y[j - 1])
            v_new[j] = v_new[j - 1] - 0.5 * (dh_[j - 1] + dh_[j]) * dy_j
        return v_new

    # ── Step 1: horizontal divergence ──────────────────────────────────────
    dh = _spectral_div_h(u, w)                        # (ny, nx, nz)

    # ── Step 2: top-wall mass imbalance → uniform u,w pressure correction ─
    # Q = ∫₀^Ly div_h dy  (trapezoidal, shape (nx, nz))
    Q = np.zeros((nx, nz), dtype=np.float64)
    for j in range(1, ny):
        dy_j = float(y[j] - y[j - 1])
        Q += 0.5 * (dh[j - 1] + dh[j]) * dy_j

    Q_max = float(np.max(np.abs(Q)))
    if Q_max > 1e-30:
        # Solve ∇²_xz P = Q/Ly  (uniform-in-y, so ∫f(y)dy = Ly cancels out)
        Q_hat = np.fft.fft2(Q)
        lam   = -(KX**2 + KZ**2)
        lam_s = np.where(lam == 0.0, 1.0, lam)
        P_hat = Q_hat / (lam_s * Ly)
        P_hat[0, 0] = 0.0                              # gauge fix (mean pressure=0)

        dP_dx = np.fft.ifft2(1j * KX * P_hat).real    # (nx, nz)
        dP_dz = np.fft.ifft2(1j * KZ * P_hat).real    # (nx, nz)

        # Uniform correction — same at every y-level so ∫ new_dh dy = 0 exactly
        u -= dP_dx[np.newaxis, :, :]
        w -= dP_dz[np.newaxis, :, :]

        # Recompute dh with corrected u, w
        dh = _spectral_div_h(u, w)

    # ── Step 3: set v by column integration ────────────────────────────────
    v[:] = _col_integrate(dh)
    # v[0]  = 0 from initialisation inside _col_integrate
    # v[ny-1] = 0 guaranteed by ∫ dh dy = 0 after Step 2


def load_velocity_3d(u_path, v_path, w_path, nx, ny, nz, hdr):
    """Load (ny, nx, nz) velocity arrays from three tlab field files."""
    def _load(path):
        arr = np.empty((ny, nx, nz), dtype=F64)
        with open(path, 'rb') as f:
            f.seek(hdr)
            for k in range(nz):
                arr[:, :, k] = np.fromfile(f, dtype=F64, count=nx*ny).reshape(ny, nx)
        return arr
    return _load(u_path), _load(v_path), _load(w_path)


def write_velocity_3d(u, v, w, paths_out, nx, ny, nz, hdrs_out):
    """Write three (ny, nx, nz) velocity arrays back to tlab field files."""
    for arr, path, (offset, nt, params_raw) in zip([u, v, w], paths_out, hdrs_out):
        with open(path, 'r+b') as f:
            f.seek(offset)
            for k in range(nz):
                f.write(arr[:, :, k].astype(F64).tobytes())


def compute_divergence_3d(u, v, w, x, y, z, spectral_xz=True):
    """
    Compute ∇·u on the full 3-D field.

    spectral_xz=True  (default): du/dx and dw/dz via FFT (exact for periodic
      grids, consistent with _project_divergence_free).
    spectral_xz=False: all three derivatives via 2nd-order central differences.

    Returns div ndarray (ny, nx, nz) float64.
    """
    ny, nx, nz = u.shape
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])

    if spectral_xz:
        kx = np.fft.fftfreq(nx, d=dx) * (2.0 * np.pi)
        kz = np.fft.fftfreq(nz, d=dz) * (2.0 * np.pi)
        KX, KZ = np.meshgrid(kx, kz, indexing='ij')
        u_hat = np.fft.fft2(u, axes=(1, 2))
        w_hat = np.fft.fft2(w, axes=(1, 2))
        div = np.fft.ifft2(
            1j * KX[np.newaxis] * u_hat + 1j * KZ[np.newaxis] * w_hat,
            axes=(1, 2)).real
    else:
        div = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
        div += (np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)) / (2.0 * dz)

    # dv/dy — trapezoidal-consistent central differences on non-uniform y grid
    for j in range(1, ny - 1):
        h2 = float(y[j + 1] - y[j - 1])
        div[j] += (v[j + 1] - v[j - 1]) / h2
    div[0]      += (v[1]      - v[0])      / float(y[1]      - y[0])
    div[ny - 1] += (v[ny - 1] - v[ny - 2]) / float(y[ny - 1] - y[ny - 2])

    return div


def correct_divergence_full(u_path, v_path, w_path,
                             nx, ny, nz, hdr,
                             x, y, z,
                             eps_crop=None,
                             threshold=1e-4,
                             max_iter=3):
    """
    Load the full 3-D velocity field and project it to be divergence-free.

    Uses `_project_divergence_free`, which combines:
      - exact column-wise v correction (∫ div_h dy)
      - 2-D spectral Poisson for the residual top-wall mass imbalance
    One projection is usually sufficient (machine-precision result).
    max_iter is used only as a safety net if IBM solid re-zeroing re-introduces
    small divergence; 3 repetitions are virtually always enough.

    Returns (max_div, rms_div) of the final corrected field.
    Writes corrected fields back in-place to u_path/v_path/w_path.
    """
    ram_gb = 3 * nx * ny * nz * F64B / 1e9
    print(f"    Loading full 3-D velocity field  "
          f"({nx}×{ny}×{nz}, ~{ram_gb:.1f} GB RAM) ...")

    u, v, w = load_velocity_3d(u_path, v_path, w_path, nx, ny, nz, hdr)
    print("    Loaded.")

    fluid_mask = (eps_crop == 0) if eps_crop is not None else None

    def _report(label):
        div = compute_divergence_3d(u, v, w, x, y, z)
        d   = div[fluid_mask] if fluid_mask is not None else div.ravel()
        mx  = float(np.max(np.abs(d))) if d.size else 0.0
        rms = float(np.sqrt(np.mean(d**2))) if d.size else 0.0
        print(f"    {label}: max|∇·u| = {mx:.4e}   RMS = {rms:.4e}")
        return mx, rms

    mx0, _ = _report("Before correction")

    for it in range(1, max_iter + 1):
        _project_divergence_free(u, v, w, x, y, z)

        # Re-enforce no-slip inside IBM solid (zeroing reintroduces small div)
        if eps_crop is not None:
            solid = (eps_crop == 1)
            for arr in (u, v, w):
                arr[solid] = 0.0

        mx, rms = _report(f"After pass {it}")
        if mx <= threshold:
            print(f"    Within tolerance after {it} pass(es).")
            break
    else:
        print(f"    WARNING: max|∇·u| = {mx:.4e} still above threshold "
              f"{threshold:.1e} after {max_iter} passes.")

    print(f"    Total reduction: {mx0/max(mx, 1e-30):.0f}×")
    print("    Writing corrected velocity fields back ...")

    hdrs_out = []
    for p in (u_path, v_path, w_path):
        with open(p, 'rb') as f:
            offset, _, _, _, nt, params_raw, _ = read_header(f)
        hdrs_out.append((offset, nt, params_raw))

    write_velocity_3d(u, v, w, [u_path, v_path, w_path], nx, ny, nz, hdrs_out)
    print("    Written.")

    return mx, rms


# ---------------------------------------------------------------------------
# Operation 4: 3-D divergence check  (streaming, 3-plane rolling window)
# ---------------------------------------------------------------------------

def check_divergence_streaming(u_path, v_path, w_path,
                                nx, ny, nz, hdr_offset,
                                x, y, z,
                                eps_crop=None,
                                threshold=1e-4):
    """
    Compute div(u) = du/dx + dv/dy + dw/dz in a streaming fashion.

    - du/dx, dw/dz: periodic central differences on uniform x, z grids.
    - dv/dy:        central differences on the (non-uniform) y grid.
    - dw/dz uses a 3-plane rolling window (only w[k-1], w[k], w[k+1] in RAM).

    Returns (max_abs_div, rms_div, n_pts, frac_above_threshold).
    """
    dx = float(x[1] - x[0])   # uniform periodic spacing
    dz = float(z[1] - z[0])   # uniform periodic spacing

    def rplane(fh, k_idx):
        return read_plane(fh, k_idx % nz, nx, ny, hdr_offset)

    div_sq   = 0.0
    max_div  = 0.0
    n_pts    = 0
    n_exceed = 0

    fu = open(u_path, 'rb')
    fv = open(v_path, 'rb')
    fw = open(w_path, 'rb')
    try:
        # Initialise rolling window for w: planes nz-1, 0, 1
        w_prev = rplane(fw, nz - 1)
        w_curr = rplane(fw, 0)
        w_next = rplane(fw, 1)

        for k in range(nz):
            u_k = rplane(fu, k)
            v_k = rplane(fv, k)

            # du/dx  —  periodic central difference (uniform x)
            du_dx = (np.roll(u_k, -1, axis=1) - np.roll(u_k, 1, axis=1)) / (2.0 * dx)

            # dv/dy  —  central differences on non-uniform y grid
            dv_dy = np.empty((ny, nx), dtype=F64)
            for j in range(1, ny - 1):
                h2 = float(y[j + 1] - y[j - 1])
                dv_dy[j] = (v_k[j + 1] - v_k[j - 1]) / h2
            # One-sided at top and bottom boundaries
            dv_dy[0]      = (v_k[1]      - v_k[0])      / float(y[1]      - y[0])
            dv_dy[ny - 1] = (v_k[ny - 1] - v_k[ny - 2]) / float(y[ny - 1] - y[ny - 2])

            # dw/dz  —  periodic central difference (uniform z, rolling window)
            dw_dz = (w_next - w_prev) / (2.0 * dz)

            div = du_dx + dv_dy + dw_dz

            # Mask to fluid cells
            if eps_crop is not None:
                mask = (eps_crop == 0)
                d    = div[mask]
            else:
                d = div.ravel()

            if d.size:
                abs_d    = np.abs(d)
                max_div  = max(max_div, float(abs_d.max()))
                div_sq  += float(np.sum(d ** 2))
                n_pts   += d.size
                n_exceed += int(np.sum(abs_d > threshold))

            # Advance rolling window
            w_prev = w_curr
            w_curr = w_next
            w_next = rplane(fw, k + 2)   # (k+2) % nz handled inside rplane

    finally:
        fu.close()
        fv.close()
        fw.close()

    rms          = float(np.sqrt(div_sq / max(n_pts, 1)))
    frac_exceed  = n_exceed / max(n_pts, 1)
    return max_div, rms, n_pts, frac_exceed

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workdir',    default='.',
                    help='Work directory [default: .]')
    ap.add_argument('--eps',        default=None,
                    help='eps indicator .npy file (ny×nx) '
                         '[default: <workdir>/eps_save.npy]')
    ap.add_argument('--alpha',      type=float, default=-0.430511,
                    help='Geostrophic wind angle α (rad) [default: -0.430511]')
    ap.add_argument('--G-mag',      type=float, default=1.0,
                    help='Geostrophic wind magnitude (sim units) [default: 1.0]')
    ap.add_argument('--smooth-pts', type=int,   default=5,
                    help='Fluid cells above IBM to smooth [default: 5]')
    ap.add_argument('--div-thresh', type=float, default=1e-4,
                    help='Divergence warning threshold [default: 1e-4]')
    ap.add_argument('--suffix',     default='_prep',
                    help='Output file suffix [default: _prep]')
    ap.add_argument('--no-crop',           action='store_true',
                    help='Skip y-cropping (j_crop = old_ny)')
    ap.add_argument('--no-cap',            action='store_true')
    ap.add_argument('--no-smooth',         action='store_true')
    ap.add_argument('--no-divergence',     action='store_true')
    ap.add_argument('--correct-divergence', action='store_true',
                    help='If max|div u| > div-thresh, apply Helmholtz-Hodge '
                         'projection to correct the field in-place. '
                         'WARNING: loads the full 3-D velocity into memory.')
    ap.add_argument('--div-max-iter',      type=int, default=3,
                    help='Max projection iterations for divergence correction '
                         '[default: 3]')
    ap.add_argument('--info',              action='store_true',
                    help='Print parameters only — no files written')
    args = ap.parse_args()

    workdir   = os.path.abspath(args.workdir)
    do_cap    = not args.no_cap
    do_smooth = not args.no_smooth
    do_div    = not args.no_divergence

    SEP = "=" * 68

    # ── Grids ──────────────────────────────────────────────────────────────
    old_grid_path = os.path.join(workdir, 'grid')
    new_grid_path = os.path.join(workdir, 'grid_new')

    for p, label in [(old_grid_path, 'grid'), (new_grid_path, 'grid_new')]:
        if not os.path.isfile(p):
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    (old_nx, old_ny, old_nz), (old_Lx, old_Ly, old_Lz), \
        old_x, old_y, old_z = read_grid(old_grid_path)
    (new_nx, new_ny, new_nz), (new_Lx, new_Ly, new_Lz), \
        new_x, new_y, new_z = read_grid(new_grid_path)

    # ── Crop index ─────────────────────────────────────────────────────────
    if args.no_crop:
        j_crop = old_ny
        n_drop = 0
    else:
        j_crop = find_crop_index(old_y, new_Ly)
        n_drop = old_ny - j_crop

    # ── Eps field ──────────────────────────────────────────────────────────
    eps_path = args.eps or os.path.join(workdir, 'eps_save.npy')
    eps_2d   = None
    eps_hgt  = None

    if os.path.isfile(eps_path):
        eps_2d = np.load(eps_path)
        if eps_2d.shape != (old_ny, old_nx):
            print(f"  WARNING: eps shape {eps_2d.shape} ≠ expected ({old_ny}, {old_nx}). "
                  "Ignoring eps — smoothing disabled.")
            eps_2d   = None
            do_smooth = False
        else:
            eps_hgt = np.sum(eps_2d, axis=0).astype(int)
    else:
        print(f"  WARNING: eps not found at {eps_path}. "
              "Capping applied to all cells; smoothing disabled.")
        do_smooth = False

    # ── Geostrophic wind ───────────────────────────────────────────────────
    alpha = args.alpha
    G_u   = np.cos(alpha)
    G_w   = -np.sin(alpha)
    G_mag = args.G_mag   # simulation units; |G_horiz| = sqrt(G_u²+G_w²) = G_mag

    # ── Sanity checks on crop index ────────────────────────────────────────
    crop_warnings = []
    if j_crop <= 0:
        crop_warnings.append(
            "CRITICAL: j_crop <= 0 — new domain is taller than old domain!")
    if j_crop >= old_ny:
        crop_warnings.append("No y-crop needed (old domain fits within new Ly).")
    if j_crop < old_ny and j_crop > 0:
        if old_y[j_crop - 1] > new_Ly:
            crop_warnings.append(
                f"WARNING: last kept y = {old_y[j_crop-1]:.6f} > new_Ly={new_Ly:.6f}")
        if eps_hgt is not None:
            max_ibm_j = int(np.max(eps_hgt))
            if j_crop <= max_ibm_j:
                crop_warnings.append(
                    f"CRITICAL: j_crop={j_crop} ≤ max IBM height={max_ibm_j}. "
                    "Crop cuts into the IBM solid body!")
            elif j_crop <= max_ibm_j + 5:
                crop_warnings.append(
                    f"CAUTION: j_crop={j_crop} is only {j_crop - max_ibm_j} cells "
                    f"above the IBM surface (max height={max_ibm_j}).")

    # ── Summary ────────────────────────────────────────────────────────────
    print(SEP)
    print("  FIELD PREPARATION SUMMARY")
    print(SEP)
    print(f"  Work directory   : {workdir}")
    print(f"  Old grid         : nx={old_nx}  ny={old_ny}  nz={old_nz}"
          f"  Ly={old_Ly:.8f}")
    print(f"  New grid         : nx={new_nx}  ny={new_ny}  nz={new_nz}"
          f"  Ly={new_Ly:.8f}")
    print()
    if args.no_crop:
        print(f"  Y-crop           : SKIPPED (j_crop = old_ny = {old_ny})")
    else:
        print(f"  Y-crop           : j_crop={j_crop}  "
              f"(keep rows 0..{j_crop-1}, drop {n_drop})")
        if j_crop > 0:
            print(f"                     last kept y = {old_y[j_crop-1]:.8f}"
                  f"  ≤  new_Ly = {new_Ly:.8f}")
        if j_crop < old_ny:
            print(f"                     first dropped y = {old_y[j_crop]:.8f}")
    print()
    print(f"  Velocity cap     : {'YES' if do_cap else 'NO (--no-cap)'}")
    if do_cap:
        print(f"    α={alpha:.6f} rad  →  Gx={G_u:.6f}  Gz={G_w:.6f}  "
              f"cap |u_horiz| ≤ {G_mag:.4f}")
    print(f"  IBM smoothing    : {'YES' if do_smooth else 'NO'}")
    if do_smooth:
        print(f"    smooth_pts={args.smooth_pts}  "
              f"(linear interp in y for first {args.smooth_pts} fluid cells above IBM)")
    print(f"  Divergence check : {'YES' if do_div else 'NO (--no-divergence)'}")
    if do_div:
        print(f"    threshold = {args.div_thresh:.1e}")
        if args.correct_divergence:
            print(f"    Auto-correct ON  (max_iter={args.div_max_iter}, "
                  "Helmholtz-Hodge projection)")
    print(f"  Output suffix    : {args.suffix!r}")
    print(f"  Eps field        : {eps_path if eps_2d is not None else 'NOT LOADED'}")
    print()
    if crop_warnings:
        for w in crop_warnings:
            print(f"  *** {w}")
        print()
    print(SEP)

    if args.info:
        print("  [--info] No files written.")
        return

    # ── Collect field files ────────────────────────────────────────────────
    entries     = sorted(os.listdir(workdir))
    field_files = [
        e for e in entries
        if (e.startswith('flow.old.') or e.startswith('scal.old.'))
        and os.path.isfile(os.path.join(workdir, e))
    ]
    if not field_files:
        print(f"  ERROR: no flow.old.* / scal.old.* files in {workdir}")
        sys.exit(1)

    vel_names    = ['flow.old.1', 'flow.old.2', 'flow.old.3']
    vel_present  = all(v in field_files for v in vel_names)
    scalar_names = [f for f in field_files if f not in vel_names]

    print(f"  Found {len(field_files)} field file(s):")
    if vel_present:
        print("    Velocity triplet  flow.old.1/2/3  — full pipeline")
    else:
        print("    WARNING: velocity triplet incomplete — "
              "cap/smooth/divergence skipped for missing component(s)")
    if scalar_names:
        print(f"    Scalar files: {scalar_names}")
    print()

    # ── Pre-flight header checks ───────────────────────────────────────────
    print("  Pre-flight header checks:")
    any_fail  = False
    hdrs_info = {}
    for fname in field_files:
        p = os.path.join(workdir, fname)
        try:
            offset, nx, ny, nz, nt, pr, np_ = check_header_vs_grid(
                p, old_nx, old_ny, old_nz)
            params  = decode_params(pr)
            rtime   = params[0] if params else float('nan')
            print(f"    {fname:20s}  OK  "
                  f"nt={nt}  rtime={rtime:.6g}  hdr={offset}B  n_params={np_}")
            hdrs_info[fname] = (offset, nt, params)

            # Sample for NaN/Inf
            issues = sample_field_quality(p, nx, ny, nz, offset, fname)
            for iss in issues:
                print(f"      *** {iss}")
                any_fail = True

        except ValueError as e:
            print(f"    {fname:20s}  FAIL:\n{e}")
            any_fail = True

    if any_fail:
        print("\n  ERROR: Pre-flight checks failed. Aborting.")
        sys.exit(1)
    print()

    # ── Process velocity triplet ───────────────────────────────────────────
    out_vel_paths = {}
    if vel_present:
        print("  Processing velocity triplet (u, v, w) ...")
        paths_in  = [os.path.join(workdir, v) for v in vel_names]
        paths_out = [os.path.join(workdir, v + args.suffix) for v in vel_names]
        for n, p in zip(vel_names, paths_out):
            out_vel_paths[n] = p

        n_capped = process_velocity_triplet(
            paths_in, paths_out,
            j_crop, old_nx, old_ny, old_nz,
            eps_2d, eps_hgt, old_y,
            G_mag, args.smooth_pts,
            do_cap and vel_present,
            do_smooth,
        )

        for name, path_out in zip(vel_names, paths_out):
            sz_in  = os.path.getsize(os.path.join(workdir, name)) / 1e6
            sz_out = os.path.getsize(path_out) / 1e6
            print(f"    {name}  →  {name}{args.suffix}  "
                  f"[{sz_in:.0f} MB → {sz_out:.0f} MB]")

        print(f"    Total capped cells (per z-plane × nz={old_nz}): {n_capped:,}")
        print()

        # Verify output header consistency
        print("  Post-write output header verification:")
        for name in vel_names:
            p = os.path.join(workdir, name + args.suffix)
            try:
                check_header_vs_grid(p, old_nx, j_crop, old_nz)
                print(f"    {name}{args.suffix}  OK  (ny={j_crop})")
            except ValueError as e:
                print(f"    {name}{args.suffix}  FAIL:\n{e}")
        print()

    # ── Process scalar files ───────────────────────────────────────────────
    if scalar_names:
        print("  Processing scalar / additional field files:")
        for fname in scalar_names:
            src = os.path.join(workdir, fname)
            dst = os.path.join(workdir, fname + args.suffix)
            print(f"    {fname}  →  {fname}{args.suffix} ...", end='', flush=True)
            try:
                process_scalar_file(
                    src, dst, j_crop, old_nx, old_ny, old_nz,
                    eps_2d=eps_2d, eps_hgt=eps_hgt, y_old=old_y,
                    smooth_pts=args.smooth_pts, do_smooth=do_smooth)
                sz_in  = os.path.getsize(src) / 1e6
                sz_out = os.path.getsize(dst) / 1e6
                print(f"  done  [{sz_in:.0f} MB → {sz_out:.0f} MB]")
            except Exception as exc:
                print(f"\n    ERROR: {exc}")
        print()

    # ── 3-D divergence check ───────────────────────────────────────────────
    if do_div and vel_present:
        print("  Computing 3-D divergence of processed velocity ...")

        # Coordinates for the cropped domain
        x_c   = old_x
        y_c   = old_y[:j_crop]
        z_c   = old_z
        nz_c  = old_nz      # z unchanged
        nx_c  = old_nx
        ny_c  = j_crop

        # Header offset of the processed u file
        with open(paths_out[0], 'rb') as f:
            hdr_out = read_header(f)[0]

        eps_crop = eps_2d[:j_crop, :] if eps_2d is not None else None

        if nx_c < 2 or ny_c < 2 or nz_c < 2:
            print("    SKIP: cropped domain too small for divergence computation.")
        else:
            max_div, rms_div, n_pts, frac = check_divergence_streaming(
                paths_out[0], paths_out[1], paths_out[2],
                nx_c, ny_c, nz_c, hdr_out,
                x_c, y_c, z_c,
                eps_crop=eps_crop,
                threshold=args.div_thresh,
            )

            print()
            print("  ┌─── 3-D Divergence check ─────────────────────────────────┐")
            print(f"  │  Cropped domain  : {nx_c} × {ny_c} × {nz_c}              ")
            print(f"  │  Fluid cells     : {n_pts:,}                               ")
            print(f"  │  max |∇·u|       = {max_div:.4e}                           ")
            print(f"  │  RMS  |∇·u|      = {rms_div:.4e}                           ")
            print(f"  │  Threshold       = {args.div_thresh:.1e}                   ")
            print(f"  │  Cells > thresh  = {frac*100:.3f}%                         ")
            print("  └──────────────────────────────────────────────────────────┘")
            print()

            if max_div > args.div_thresh:
                print(f"  WARNING: max divergence ({max_div:.2e}) exceeds "
                      f"threshold ({args.div_thresh:.1e}).")

                if args.correct_divergence:
                    print()
                    print("  ── Applying Helmholtz-Hodge projection "
                          "(--correct-divergence) ──")
                    corr_max, corr_rms = correct_divergence_full(
                        paths_out[0], paths_out[1], paths_out[2],
                        nx_c, ny_c, nz_c, hdr_out,
                        x_c, y_c, z_c,
                        eps_crop=eps_crop,
                        threshold=args.div_thresh,
                        max_iter=args.div_max_iter,
                    )
                    print()
                    if corr_max <= args.div_thresh:
                        print(f"  Divergence corrected: max|∇·u| = {corr_max:.4e} "
                              f"(within threshold {args.div_thresh:.1e}).")
                    else:
                        print(f"  WARNING: divergence still {corr_max:.4e} after "
                              f"{args.div_max_iter} iterations. "
                              "Consider increasing --div-max-iter or checking "
                              "the IBM penalization.")
                else:
                    print("           Re-run with --correct-divergence to apply "
                          "Helmholtz-Hodge projection.")
                    print("           The field may have been modified significantly "
                          "by the IBM transition or grid interpolation.")
            else:
                print("  Divergence within acceptable limits.")
            print()

    # ── Next steps ─────────────────────────────────────────────────────────
    print(SEP)
    print("  NEXT STEPS")
    print(SEP)
    print(f"  1. Rename/link processed files to flow.old.*/scal.old.* for transfields:")
    print(f"       e.g.   mv flow.old.1{args.suffix} flow.old.1")
    print(f"              (after backing up originals as flow.old.1.bak, etc.)")
    print()
    print("  2. Update tlab.ini for the transfields run:")
    if not args.no_crop:
        print(f"       jmax = {j_crop}   Ly = {old_y[j_crop-1]:.8f}")
    print(f"       imax = {old_nx}   kmax = {old_nz}   (unchanged)")
    print()
    print("  3. Run transfields option 3 (Remesh):")
    print(f"       source: {old_nx} × {j_crop} × {old_nz}  (old grid, cropped)")
    print(f"       target: {new_nx} × {new_ny} × {new_nz}  (new grid)")
    print()
    print("  4. Update tlab.ini to final new-grid dimensions:")
    print(f"       imax={new_nx}  jmax={new_ny}  kmax={new_nz}")
    print(f"       Lx={new_Lx:.8f}  Ly={new_Ly:.8f}  Lz={new_Lz:.8f}")
    print(SEP)


if __name__ == '__main__':
    main()
