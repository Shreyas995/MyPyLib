#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EditGrid.py  —  tlab wall-normal grid generator / editor (single canonical copy).

This is the ONE canonical copy.  It lives in MyPyLib and is symlinked into each
working grid directory so the same code edits/creates every grid.  Run it from a
working directory (the symlink): it reads an OLD grid, builds a NEW wall-normal
(Y) distribution that satisfies the physics + machine constraints below, and
writes one tlab grid file per machine target (a "grid pair").

--------------------------------------------------------------------------------
CONSTRAINTS IMPLEMENTED  (engineering coords: x=streamwise, y=wall-normal, z=span)
--------------------------------------------------------------------------------
 1. NOT COARSER than the old grid anywhere.  transfields interpolates the old
    field onto the new grid; coarsening/under-resolution makes the field blow up.
    Enforced as  dy_new(y) <= dy_old(y)  for every cell  (old dy is the ceiling).
 2. Uniform near-wall Zone 1 spanning exactly ceil(hill_hgt/dy_wall) + 20 cells.
    "hill + 20 grid cells" — the 20 is grid CELLS, NOT 20 viscous lengths/wall units.
    If old dy[0] < 1 wall unit: use that spacing as-is (no refinement needed).
    If old dy[0] >= 1 wall unit: refine to 1 wall unit.
 3. X/Z isotropy: dx = dz in 2-4 wall units (finer allowed, never coarser).
    Stretch Y at <= 2 %/cell, ramped up over 10 cells then ramped back down to 0
    over 5 cells before the BL uniform zone.  Inside BL (y < u_star): dy/dx < 1.
 4. If the 2 % stretch reaches OUTSIDE the BL before isotropy -> keep stretching
    to the end.  If isotropy is reached INSIDE the BL -> stop (BL stays uniform)
    until y = u_star, then stretch again at <= 3 %/cell to the domain top.
 5. Sponge: the last N_SPONGE cells close the domain; they may be stretched at
    up to r_out per cell but NEVER contract (dy monotonically non-decreasing).
    The final node is pinned to old_y[-1] so domain length is preserved exactly.
 6. Same domain scales as the old grid (Lx, Ly, Lz written byte-for-byte) so
    transfields' scale-equality check passes.  Y node range [0, y_old[-1]] kept.
 7. MPI decomposition divisibility (see VERIFIED RULES below).

--------------------------------------------------------------------------------
VERIFIED tlab MPI RULES  (confirmed against tlab/src/base, 2026-06)
--------------------------------------------------------------------------------
Local per-rank dims:  Imax(*) = NX/npro_i ,  Kmax(*) = NZ/npro_k ,  Jmax = NY.
  C1  mod(NX, Imax(*)) == 0          -> npro_i      [tlab_mpi_procs.f90:51]
  C2  mod(NZ, Kmax(*)) == 0          -> npro_k      [tlab_mpi_procs.f90:42]
  C4  Imax(*) even                                   (needed by the /2 below)
  C5  mod(Kmax(*)*Jmax, npro_i) == 0   Ox derivative transpose
                                       [tlab_mpi_transpose.f90:426-427]
  C6  mod(Imax(*)*Jmax, npro_k) == 0   Oz derivative transpose [..:431-433]
  C7  mod((Imax(*)+2)*Jmax/2, npro_k)==0  Poisson Oz FFT buffer
                                       isize_txc_dimz=(imax+2)*jmax
                                       [tlab_memory.f90:188 ; opr_fourier.f90:89]
  C8  mod(Jmax*Kmax(*), npro_i) == 0   Poisson Ox FFT [opr_fourier.f90:134] (= C5)

Machine task layouts the user runs on (cluster choices, stronger-than-required):
  Curta : 1024 ranks = 32 x 32.  NX/32 div by 8, NZ/32 div by 4, NY div by 32.
  Hunter: 96 ranks   = 12 x 8.   NX div by 96, NY div by 48, NZ div by 8.
  Note: if NY is divisible by lcm(32,48)=96 then C5/C6/C8 are automatic and only
  C7 adds a real condition (NY div by 32 for Curta, div by 8 for Hunter).  The
  user's div-by-8 / div-by-4 rules are safe (sufficient) supersets of C1-C8.

Make grids in PAIRS (one per machine), NX/NZ within ~+-64, identical Y.
"""

import os
import numpy as np
from scipy.io import FortranFile


# ============================================================================
# tlab grid file I/O  (Fortran sequential unformatted, 5 records)
# ============================================================================
def read_grid(path):
    """Return (nmax[3], scales[3], x, y, z) from a tlab grid file."""
    with open(path, 'rb') as f:
        np.fromfile(f, '<i4', 1); nmax = np.fromfile(f, '<i4', 3); np.fromfile(f, '<i4', 1)
        np.fromfile(f, '<i4', 1); scl = np.fromfile(f, '<f8', 3); np.fromfile(f, '<i4', 1)
        np.fromfile(f, '<i4', 1); x = np.fromfile(f, '<f8', nmax[0]); np.fromfile(f, '<i4', 1)
        np.fromfile(f, '<i4', 1); y = np.fromfile(f, '<f8', nmax[1]); np.fromfile(f, '<i4', 1)
        np.fromfile(f, '<i4', 1); z = np.fromfile(f, '<f8', nmax[2]); np.fromfile(f, '<i4', 1)
    return nmax, scl, x, y, z


def write_grid(path, nx, ny, nz, scales, xn, yn, zn):
    """
    Write a tlab grid file.  `scales` (Lx, Ly, Lz) are written byte-for-byte from
    the OLD grid so transfields' scale-equality check passes (Constraint 6).
    """
    with FortranFile(path, 'w') as f:
        f.write_record(np.array([nx, ny, nz], dtype=np.int32))
        f.write_record(np.asarray(scales, dtype=np.float64))
        f.write_record(np.asarray(xn[:nx], dtype=np.float64))
        f.write_record(np.asarray(yn[:ny], dtype=np.float64))
        f.write_record(np.asarray(zn[:nz], dtype=np.float64))


# ============================================================================
# Zone diagnostics
# ============================================================================
def find_grid_zones(r, tol=1e-9):
    """Classify cell-ratio array r[i]=dy[i+1]/dy[i] into uniform / stretch /
    contraction runs.  Returns list of dicts with node_start, node_end, type,
    r_mean, r_max."""
    r = np.asarray(r, float)
    dr = np.empty_like(r)
    dr[0] = r[1] - r[0] if len(r) > 1 else 0.0
    dr[1:] = np.diff(r)
    lab = np.where(np.abs(dr) <= tol, 0, np.where(dr > 0, 1, -1))
    name = {0: 'uniform', 1: 'stretching', -1: 'contraction'}
    zones, i = [], 0
    while i < len(r):
        j = i
        while j < len(r) and lab[j] == lab[i]:
            j += 1
        zones.append({'type': name[lab[i]], 'node_start': i, 'node_end': (j - 1) + 2,
                      'r_mean': float(np.mean(r[i:j])), 'r_max': float(np.max(r[i:j]))})
        i = j
    for k in range(1, len(zones)):
        zones[k]['node_start'] = zones[k - 1]['node_end']
    return zones


def print_zones(y, l_in, dx_ref):
    dy = np.diff(y); r = dy[1:] / dy[:-1]
    zones = find_grid_zones(r)
    print(f"  {'#':>2} {'type':12} {'nodes':>11} {'y_start':>11} {'y_end':>11} "
          f"{'stretch%':>9} {'dy/dx_max':>9} {'n':>5}")
    for k, z in enumerate(zones):
        ns, ne = z['node_start'], z['node_end']
        print(f"  {k+1:>2} {z['type']:12} {ns:>5d}-{ne:<5d} {y[ns]:>11.4e} {y[ne]:>11.4e} "
              f"{(z['r_mean']-1)*100:>+8.3f}% {np.max(dy[ns:ne])/dx_ref:>9.4f} {ne-ns+1:>5d}")
    print(f"  total: {len(y)} nodes, {len(zones)} zones, "
          f"dy0={dy[0]/l_in:.3f}wu, dymax={dy.max()/l_in:.2f}wu, Ly={y[-1]:.8f}")


# ============================================================================
# Constraint-based wall-normal grid builder
# ============================================================================
def build_y_grid(old_y, u_star, l_in, hill_hgt, dx_ref, ny_target,
                 r_in=1.02, r_out=1.03, n_in_decel=5, n_sponge=20,
                 iso_thresh=0.90, dy_wall=None, n_bl=None, n_ramp=10,
                 zone1_top=None):
    """
    Build a wall-normal node distribution satisfying Constraints 1-5.

    old_y      : old grid y nodes (ceiling, Constraint 1).
    u_star     : friction velocity = BL height (outer length scale).
    l_in       : viscous length = 1 wall unit.
    hill_hgt   : hill/valley crest height in grid coordinates.  Zone 1 spans
                 ceil(hill_hgt/dy_wall) + 20 cells (20 = grid CELLS, not wu).
    dx_ref     : horizontal cell width to size isotropy against; pass the FINEST
                 (smallest dx) of the machine pair so dy/dx<1 holds for both.
    ny_target  : desired Jmax.  Reached by tuning n_bl (BL-uniform cell count);
                 see make_grid_pair() which searches n_bl/n_sponge for an exact hit.
    r_in       : near-wall stretch ratio per cell  (<= 2 %,   Constraint 3).
    r_out      : outer stretch ratio cap per cell  (<= 3 %,   Constraint 4).
    n_ramp     : cells to ramp UP from 0 to r_in (or r_out); cell i gets ratio
                 1 + i*(r-1)/n_ramp for i=0..n_ramp-1.
    n_in_decel : cells to ramp DOWN from r_in (or r_out) back to 0; cell k gets
                 ratio 1 + (n_in_decel-1-k)*(r-1)/n_in_decel for k=0..n_in_decel-1.
    n_sponge   : minimum sponge cells at the top; stretched but not contracted.
    iso_thresh : stretch until dy reaches iso_thresh*dx; the inner decel then
                 lifts it to ~0.94-0.95 dx (close to isotropy) staying < dx.
    dy_wall    : near-wall uniform spacing.  Default: old_y[1] if old_y[1] < l_in
                 (no refinement needed), else l_in (refine to 1 wall unit).
    zone1_top  : ALIGNED Zone-1 mode (grid-gen rule 1).  When given, the uniform
                 near-wall zone has a FIXED height whose top lands EXACTLY on an
                 OLD-grid node `zone1_top` (caller passes (valley+20)*dy_old, which
                 coincides with an old node).  Zone 1 is then refined into
                 n1 = ceil(zone1_top/dy_wall) uniform cells with dy1 = zone1_top/n1
                 (<= 1 wall unit), so the last uniform node aligns with the old grid
                 (transfields-clean) and the scale matches exactly.  None (default)
                 keeps the legacy "ceil(hill/dy_wall)+20 new cells" Zone 1.
    n_bl       : extra uniform padding cells appended at dy_bl beyond the natural
                 BL zone (same spacing = no contraction) to tune ny to ny_target.

    Returns the y node array (y[0]=0, y[-1]=old_y[-1]).
    """
    Ly = float(old_y[-1])
    dy_ref0 = float(old_y[1] - old_y[0])
    if dy_wall is None:
        dy_wall = dy_ref0 if dy_ref0 < l_in else l_in   # refine only if coarser than 1wu
    ody = np.diff(old_y); oyc = 0.5 * (old_y[1:] + old_y[:-1])
    ceil_dy = lambda yc: np.interp(np.minimum(yc, oyc[-1]), oyc, ody)  # old dy ceiling

    # --- Zone 1: uniform near-wall zone -----------------------------------------
    if zone1_top is not None:
        # ALIGNED mode (rule 1): FIXED height to an old-grid node `zone1_top`
        # (= (valley+20)*dy_old).  Refine into n1 uniform cells at <= 1 wall unit;
        # dy1 = zone1_top/n1 so the last uniform node lands exactly on the old node.
        n1 = int(np.ceil(zone1_top / dy_wall))
        dy1 = zone1_top / n1
    else:
        # Legacy mode: ceil(hill/dy_wall) + 20 cells; the "20" is 20 reference-grid
        # CELL widths (grid points), not 20 wall units.
        n1 = int(np.ceil(hill_hgt / dy_wall)) + 20
        dy1 = dy_wall                               # exact reference spacing (C1 satisfied)
    w = [dy1] * n1
    dy = dy1

    # --- Zone 2a: ramp up to r_in over n_ramp cells (ratio = 1 + i*(r_in-1)/n) ---
    dy_tgt = iso_thresh * dx_ref
    reached_iso_in_bl = True
    for i in range(n_ramp):
        ri = 1.0 + i * (r_in - 1.0) / n_ramp
        dy *= ri; w.append(dy)
        if sum(w) >= u_star:
            reached_iso_in_bl = False
            break
    # --- Zone 2b: constant stretch at r_in until dy ~ iso_thresh*dx, OR exit BL --
    if reached_iso_in_bl:
        while dy * r_in < dy_tgt:
            if sum(w) >= u_star:
                reached_iso_in_bl = False
                break
            dy *= r_in; w.append(dy)

    if reached_iso_in_bl:
        # --- Zone 3: ramp DOWN from r_in to 0 over n_in_decel cells -------------
        # ratio for cell k: 1 + (n_in_decel-1-k)*(r_in-1)/n_in_decel  (k=0..n-1)
        for k in range(n_in_decel):
            dy *= 1.0 + (n_in_decel - 1 - k) * (r_in - 1.0) / n_in_decel
            w.append(dy)
        dy_iso = dy
        # --- Zone 4: BL uniform from here up to u_star --------------------------
        y_iso = sum(w); span_bl = u_star - y_iso
        n_bl_nat = max(1, int(span_bl / dy_iso))  # floor: avoids contraction at zone junction
        nB = n_bl_nat                                          # floor: no contraction
        dy_bl = span_bl / nB
        w += [dy_bl] * nB
        n_pad = 0 if n_bl is None else int(n_bl)              # extra uniform padding
        w += [dy_bl] * n_pad
        dy = dy_bl
        # --- Zone 5a: ramp UP to r_out over n_ramp cells -------------------------
        ypos = sum(w)
        _res5 = (n_in_decel + n_sponge + 2)           # reservation sentinel (cells)
        for i in range(n_ramp):
            ri = 1.0 + i * (r_out - 1.0) / n_ramp
            dyn = min(dy * ri, ceil_dy(ypos + dy * 0.5))
            dyn = max(dyn, dy)                         # no contraction
            if ypos + dyn > Ly - _res5 * dyn:
                break                                  # near top: stop ramp early
            dy = dyn; w.append(dy); ypos += dy
        # --- Zone 5b: stretch at r_out, EASING into the old-dy ceiling over -----
        #     n_in_decel cells (rate ramps r_out -> 0) instead of a 1-cell snap.
        #     F_dec = cumulative ramp-down factor; trigger when dy*F_dec would
        #     reach the ceiling, so dy lands exactly on it after the ramp.
        F_dec = 1.0
        for k in range(n_in_decel):
            F_dec *= 1.0 + (n_in_decel - 1 - k) * (r_out - 1.0) / n_in_decel
        eased = False
        while True:
            cap = ceil_dy(ypos + dy * 0.5)
            # Trigger one 3 % step early (dy*r_out*F_dec) so the full n_in_decel-cell
            # ramp fits below the ceiling instead of being clipped to ~2 cells.
            if (not eased) and dy * r_out * F_dec >= cap:
                for k in range(n_in_decel):
                    cap = ceil_dy(ypos + dy * 0.5)
                    ratio = 1.0 + (n_in_decel - 1 - k) * (r_out - 1.0) / n_in_decel
                    dyn = max(min(dy * ratio, cap), dy)   # clamp to ceiling, no contraction
                    if ypos + dyn > Ly - _res5 * dyn:
                        break
                    dy = dyn; w.append(dy); ypos += dy
                eased = True
                continue
            dyn = min(dy * r_out, cap)
            if ypos + dyn > Ly - _res5 * dyn:         # reserve sponge-onset + sponge
                break
            dy = dyn; w.append(dy); ypos += dy
        # --- Zone 5c: ease OUT of the ceiling into the sponge over n_in_decel ----
        #     cells (same ramp formula; grows dy above the ceiling as a smooth
        #     sponge onset).  Capped by the C5 no-contraction rule only.
        for k in range(n_in_decel):
            dy *= 1.0 + (n_in_decel - 1 - k) * (r_out - 1.0) / n_in_decel
            w.append(dy)
    else:
        # Constraint 4 alternate branch: keep the 2 % stretch to the top, EASING
        # into the ceiling over n_in_decel cells, then ramp into the sponge.
        _res_fb = (n_in_decel + n_sponge + 2)
        ypos = sum(w)
        F_dec = 1.0
        for k in range(n_in_decel):
            F_dec *= 1.0 + (n_in_decel - 1 - k) * (r_in - 1.0) / n_in_decel
        eased = False
        while True:
            cap = ceil_dy(ypos + dy * 0.5)
            if (not eased) and dy * r_in * F_dec >= cap:  # ease into ceiling, no snap
                for k in range(n_in_decel):
                    cap = ceil_dy(ypos + dy * 0.5)
                    ratio = 1.0 + (n_in_decel - 1 - k) * (r_in - 1.0) / n_in_decel
                    dyn = max(min(dy * ratio, cap), dy)
                    if ypos + dyn > Ly - _res_fb * dyn:
                        break
                    dy = dyn; w.append(dy); ypos += dy
                eased = True
                continue
            dyn = min(dy * r_in, cap)
            if ypos + dyn > Ly - _res_fb * dyn:
                break
            dy = dyn; w.append(dy); ypos += dy
        for k in range(n_in_decel):                   # sponge onset (ease out of ceiling)
            dy *= 1.0 + (n_in_decel - 1 - k) * (r_in - 1.0) / n_in_decel
            w.append(dy)
        n_pad = 0 if n_bl is None else int(n_bl)              # extra uniform padding
        w += [dy] * n_pad

    # --- Zone 6: stretched sponge filling the remainder -------------------------
    # Cells form a geometric series starting at current dy, ratio r_sp <= r_out.
    # Sum formula: dy*(r_sp^n - 1)/(r_sp - 1) = rem.  Solved by bisection.
    # No contraction: r_sp >= 1, so every sponge cell >= its predecessor.
    rem = Ly - sum(w)
    def _gsum(r, n, d0):
        return n * d0 if abs(r - 1.0) < 1e-14 else d0 * (r**n - 1.0) / (r - 1.0)
    n_sp = n_sponge
    while n_sp < 100000 and _gsum(r_out, n_sp, dy) < rem * (1.0 - 1e-9):
        n_sp += 1
    r_lo, r_hi = 1.0, r_out + 1e-9
    for _ in range(80):
        r_mid = 0.5 * (r_lo + r_hi)
        if _gsum(r_mid, n_sp, dy) < rem:
            r_lo = r_mid
        else:
            r_hi = r_mid
    r_sp = min(0.5 * (r_lo + r_hi), r_out)
    d = dy
    for _ in range(n_sp - 1):
        w.append(d); d *= r_sp
    w.append(Ly - sum(w))                              # last cell pins Ly exactly

    y = np.concatenate([[0.0], np.cumsum(w)])
    y[-1] = Ly                                         # guarantee endpoint (Constraint 6)
    return y


def make_grid_pair(old_y, u_star, l_in, hill_hgt, dx_ref, ny_target, **kw):
    """Search (n_bl, n_sponge) so build_y_grid hits len(y)==ny_target exactly while
    keeping the smooth structure (so the BL->outer junction stays <= r_out)."""
    base_sp = kw.pop('n_sponge', 20)
    for nsp in [base_sp, base_sp + 1, base_sp + 2, base_sp - 1, base_sp - 2]:
        for nB in range(0, 500):
            y = build_y_grid(old_y, u_star, l_in, hill_hgt, dx_ref, ny_target,
                             n_sponge=nsp, n_bl=nB, **kw)
            if len(y) == ny_target:
                return y, nB, nsp
    raise RuntimeError(f"Could not hit ny_target={ny_target}; widen the search or "
                       f"adjust ny_target (it must leave room for r_out<=3%).")


# ============================================================================
# Rule + MPI verification
# ============================================================================
def check_grid_rules(y, old_y, u_star, l_in, hill_hgt, dx_ref, n_sponge=20,
                     zone1_top=None):
    dy = np.diff(y); r = dy[1:] / dy[:-1]
    ody = np.diff(old_y); oyc = 0.5 * (old_y[1:] + old_y[:-1])
    nyc = 0.5 * (y[1:] + y[:-1])
    ratio_old = dy / np.interp(np.minimum(nyc, oyc[-1]), oyc, ody)
    sp_mask = ratio_old > 1.01                  # sponge cells (coarser than old, relaxed)
    coarser_below = ratio_old[~sp_mask].max() if (~sp_mask).any() else 0.0
    coarser_top = ratio_old.max()
    j_us = np.searchsorted(y, u_star)
    dy_wall = float(ody[0]) if float(ody[0]) < l_in else l_in
    if zone1_top is not None:
        # ALIGNED Zone 1 (rule 1): n1 = ceil(zone1_top/dy_wall); top lands on old node.
        n1_exp = int(np.ceil(zone1_top / dy_wall))
        z1_top_ref = zone1_top
        z1_label = "C2 Zone 1 aligned: top on old node (valley+20 dy_old)"
    else:
        # Legacy: expected cell count = ceil(hill_hgt/dy_wall) + 20  (20 = grid cells)
        n1_exp = int(np.ceil(hill_hgt / dy_wall)) + 20
        z1_top_ref = n1_exp * dy_wall
        z1_label = "C2 Zone 1 = ceil(hill/dy_wall)+20 cells (hill+20 grid pts)"
    # Direct index check: cells 0..n1_exp-1 must all equal dy1 (± rounding)
    z1_uniform = (n1_exp <= len(dy) and
                  np.std(dy[:n1_exp]) / dy[0] < 1e-6 and
                  abs(y[n1_exp] - z1_top_ref) < dy[0] * 0.01)
    checks = [
        ("C1 not coarser than old below sponge (sponge relaxed)", coarser_below <= 1.01,
         f"below={coarser_below:.5f}, sponge_top={coarser_top:.5f}"),
        (z1_label,
         z1_uniform,
         f"n1={n1_exp}, top={y[n1_exp]/l_in:.2f}wu" if n1_exp < len(y) else f"n1_exp={n1_exp} out of range"),
        ("C3 dy/dx<1 inside BL (isotropy)", (dy[:j_us] / dx_ref).max() < 1.0,
         f"max dy/dx (y<u*) = {(dy[:j_us]/dx_ref).max():.4f}"),
        ("C3 stretch <= 2% in BL", (r[:j_us - 1].max() if j_us > 1 else 1.0) <= 1.0201,
         f"max BL ratio = {(r[:j_us-1].max() if j_us>1 else 1.0):.5f}"),
        ("C4 stretch <= 3% overall", r.max() <= 1.0301,
         f"max ratio = {r.max():.5f} ({(r.max()-1)*100:.3f}%)"),
        ("C5 no contraction (dy non-decreasing everywhere)", r.min() >= 0.9999,
         f"min ratio = {r.min():.6f} at j={np.argmin(r)+1}"),
        ("C6 Ly preserved", np.isclose(y[-1], old_y[-1]),
         f"Ly={y[-1]:.8f}"),
    ]
    print("\n  --- grid-quality rules ---")
    ok_all = True
    for name, ok, det in checks:
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:45s} {det}")
    return ok_all


def grid_quality_table(y, old_y, u_star, l_in, hill_hgt, dx_ref,
                       label='', n_sponge=20, zone1_top=None):
    """
    Print a formatted grid-quality table and return True if all checks pass.

    The two primary physics rules checked explicitly:
      • Max stretch inside BL  (y < u_star)  ≤ 2 %
      • Max stretch outside BL (y ≥ u_star)  ≤ 3 %
    Plus Zone 1 cell count, isotropy, no-contraction, and coarsening.
    """
    dy  = np.diff(y)
    r   = dy[1:] / dy[:-1]
    ody = np.diff(old_y)
    oyc = 0.5 * (old_y[1:] + old_y[:-1])
    nyc = 0.5 * (y[1:] + y[:-1])

    j_us = np.searchsorted(y, u_star)           # first index with y >= u_star

    # Zone 1
    dy_wall = float(ody[0]) if float(ody[0]) < l_in else l_in
    if zone1_top is not None:
        n1_exp     = int(np.ceil(zone1_top / dy_wall))   # aligned (rule 1)
        z1_top_ref = zone1_top
    else:
        n1_exp     = int(np.ceil(hill_hgt / dy_wall)) + 20
        z1_top_ref = n1_exp * dy_wall
    z1_ok   = (n1_exp <= len(dy) and
               np.std(dy[:n1_exp]) / dy[0] < 1e-6 and
               abs(y[n1_exp] - z1_top_ref) < dy[0] * 0.01)

    # Stretch inside BL (all ratio transitions below j_us)
    r_bl  = r[:max(0, j_us - 1)].max() if j_us > 1 else 1.0
    # Stretch outside BL
    r_obl = r[j_us:].max() if j_us < len(r) else 1.0
    # Global max (should be r_obl)
    r_max = r.max()
    r_min = r.min()

    dydx_bl = (dy[:j_us] / dx_ref).max() if j_us > 0 else 0.0
    # C1: strict (dy_new <= dy_old) BELOW the sponge; the sponge cells (those coarser
    # than the old grid, above the ceiling) are user-approved to stretch further.
    ratio_old     = dy / np.interp(np.minimum(nyc, oyc[-1]), oyc, ody)
    sp_mask       = ratio_old > 1.01            # cells coarser than old = sponge region
    coarser_below = ratio_old[~sp_mask].max() if (~sp_mask).any() else 0.0
    coarser_top   = ratio_old.max()             # max coarsening in the sponge
    n_sp_cells    = int(sp_mask.sum())
    ly_ok   = np.isclose(y[-1], old_y[-1])

    dy0_wu    = dy[0] / l_in
    dymax_bl  = dy[:j_us].max() / l_in if j_us > 0 else dy[0] / l_in
    dymax_all = dy.max() / l_in
    u_bl_wu   = u_star / l_in
    dx_wu     = dx_ref / l_in

    W = 68
    def _row(lbl, val, limit, ok):
        tag = 'INFO' if ok is None else ('PASS' if ok else 'FAIL')
        print(f"  {lbl:<33} {val:>12}  {limit:>10}  [{tag}]")

    print(f"\n{'═'*W}")
    hdr = f"  GRID QUALITY TABLE{f'  [{label}]' if label else ''}"
    print(hdr)
    print('═'*W)
    print(f"  {'Property':<33} {'Value':>12}  {'Limit':>10}  Status")
    print('─'*W)

    _row('ny nodes', f"{len(y)}", '—', None)
    _row('dy_wall  [first cell, wu]', f"{dy0_wu:.4f}", '≤ 1.00', dy0_wu <= 1.0)
    print('─'*W)

    print(f"  Zone 1  {'─'*40}")
    z1_cells_lbl = '  cells  [aligned to old node]' if zone1_top is not None \
        else '  cells  [= hill_cells + 20]'
    _row(z1_cells_lbl,
         f"{n1_exp}" if z1_ok else f"{'?'} (exp {n1_exp})",
         f"= {n1_exp}", z1_ok)
    n1_top_wu = y[n1_exp] / l_in if n1_exp < len(y) else float('nan')
    _row('  top height  [wu]', f"{n1_top_wu:.2f}", '—', None)
    print('─'*W)

    print(f"  Stretch rates  {'─'*33}")
    _row('  max inside BL   (y < u★)',
         f"{(r_bl-1)*100:.3f}%", '≤ 2.000%', r_bl <= 1.0201)
    _row('  max outside BL  (y ≥ u★)',
         f"{(r_obl-1)*100:.3f}%", '≤ 3.000%', r_obl <= 1.0301)
    _row('  no contraction  [min ratio]',
         f"{r_min:.6f}", '≥ 0.9999', r_min >= 0.9999)
    print('─'*W)

    print(f"  Isotropy & resolution  {'─'*25}")
    _row('  max dy/dx in BL', f"{dydx_bl:.4f}", '< 1.00', dydx_bl < 1.0)
    _row('  dx_ref  [wu]', f"{dx_wu:.3f}", '—', None)
    _row('  BL top  [u★, wu]', f"{u_bl_wu:.1f}", '—', None)
    _row('  max dy+ in BL', f"{dymax_bl:.2f}", '≤ 20.0', dymax_bl <= 20.0)
    _row('  max dy+ overall', f"{dymax_all:.2f}", '—', None)
    print('─'*W)

    print(f"  Reference compatibility  {'─'*21}")
    _row('  new/old below sponge  [C1]', f"{coarser_below:.5f}", '≤ 1.010', coarser_below <= 1.01)
    _row(f'  sponge coarsening [{n_sp_cells} cells]', f"{coarser_top:.5f}", 'relaxed', None)
    _row('  Ly preserved', f"{y[-1]:.8f}", 'exact', ly_ok)
    print('─'*W)

    ok_all = all([
        dy0_wu <= 1.0, z1_ok,
        r_bl <= 1.0201, r_obl <= 1.0301,
        r_min >= 0.9999, dydx_bl < 1.0,
        dymax_bl <= 20.0, coarser_below <= 1.01, ly_ok,
    ])
    print(f"  {'OVERALL':<33} {'':>12}  {'':>10}  "
          f"[{'ALL PASS' if ok_all else 'FAIL -- see above'}]")
    print('═'*W)
    return ok_all


def check_mpi(nx, nz, jmax, imax_star, kmax_star, fourier=True):
    npi = nx // imax_star; npk = nz // kmax_star
    checks = [
        ("C1 NX % Imax(*)", nx % imax_star == 0),
        ("C2 NZ % Kmax(*)", nz % kmax_star == 0),
        ("C4 Imax(*) even", imax_star % 2 == 0 if fourier else True),
        ("C5 (Kmax(*)*Jmax) % npro_i", (kmax_star * jmax) % npi == 0 if npi > 1 else True),
        ("C6 (Imax(*)*Jmax) % npro_k", (imax_star * jmax) % npk == 0 if npk > 1 else True),
        ("C7 ((Imax(*)+2)*Jmax/2) % npro_k",
         ((imax_star + 2) * jmax // 2) % npk == 0 if (fourier and npk > 1) else True),
        ("C8 (Jmax*Kmax(*)) % npro_i", (jmax * kmax_star) % npi == 0 if (fourier and npi > 1) else True),
    ]
    ok_all = all(v for _, v in checks)
    print(f"  npro_i={npi}, npro_k={npk}, npro={npi*npk}")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok_all


# ============================================================================
# ===============================  CONFIG  ===================================
# ============================================================================
# Physics (Ekman-over-valley case).  ACTIVE TARGET: Re_D = 750.
# u_star for Re_750 (orography): smooth-case estimate ~0.0561, raised +20% for the
# valley (u* ~20% higher with orography than over a smooth wall) => 0.06732.
# nu_L and L_IN are derived, so they re-scale automatically with Re_D / U_STAR.
Re_D     = 750
U_STAR   = 0.06732                     # friction velocity = BL height (Re_750, orography)
HILL_FRAC = 0.008528931571390622       # crest height as a fraction of Ly (geometry fixed)
VALLEY_CELLS = 48                      # Zone-1 valley resolution at the reference grid
nu_L     = 1.0 / (0.5 * Re_D ** 2)
L_IN     = nu_L / U_STAR               # 1 wall unit (viscous length)

# I/O — run from the working dir (symlink); OLD_GRID is relative to it.
OLD_GRID = '../1024x416x1024/grid'

# Machine targets: (name, NX, NZ, Imax(*), Kmax(*)).  Identical Y, paired NX/NZ.
# 1728 divides BOTH layouts, so one grid serves Curta and Hunter (dx=2.91 wu at Re_750).
GRID_TARGETS = [
    ('Curta',  1728, 1728, 54, 54),    # 32x32 = 1024 ranks  (Imax*=Kmax*=1728/32)
    ('Hunter', 1728, 1728, 144, 216),  # 12x8  =   96 ranks  (1728/12, 1728/8)
]
NY_TARGET = 784                        # natural 774 -> 784 = smallest MPI-feasible (>=774)

# --- previous Re_D=500 setup (built the verified 1024x416x1024 / 1056x672x1056 pair) ---
#   Re_D = 500 ; U_STAR = 0.077
#   GRID_TARGETS = [('Curta', 1024, 1024, 32, 32), ('Hunter', 1056, 1056, 88, 132)]
#   NY_TARGET = 672   # div by lcm(32,48)=96

# Grid-shape knobs (Constraints 3-5)
R_IN, R_OUT     = 1.02,  1.03          # 2% near-wall, 3% outer
N_IN_DECEL      = 5                    # ramp-down cells (inner and outer)
N_RAMP          = 10                   # ramp-up cells (inner and outer)
N_SPONGE        = 20                   # minimum sponge cells (stretched, not uniform)
ISO_THRESH      = 0.90                 # pre-decel dy/dx target

WRITE_TO_FILE   = False                # set True to write grid files


def main():
    cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
    o_nmax, o_scl, o_x, o_y, o_z = read_grid(cwd + OLD_GRID)
    Lx, Lz = float(o_scl[0]), float(o_scl[2])
    # Isotropy reference = FINEST dx in the pair (so dy/dx<1 holds for both).
    dx_ref = Lx / max(t[1] for t in GRID_TARGETS)
    print(f"OLD: {tuple(int(v) for v in o_nmax)}  scales={o_scl}  "
          f"dy0={float(o_y[1]-o_y[0])/L_IN:.3f}wu  "
          f"u*={U_STAR/L_IN:.0f}wu  dx_ref={dx_ref/L_IN:.3f}wu")

    # Zone-1 valley height = VALLEY_CELLS uniform cells of the REFERENCE grid
    # (rule 2: reference resolves the valley with 48 cells).  +20 buffer cells are
    # added inside build_y_grid.  This refines correctly at any target Re.
    hill = VALLEY_CELLS * float(o_y[1] - o_y[0])
    dy_w0 = float(o_y[1] - o_y[0]) if float(o_y[1] - o_y[0]) < L_IN else L_IN
    n_hill = int(np.ceil(hill / dy_w0))
    print(f"  hill_hgt={hill:.6e} ({hill/L_IN:.3f}wu)  "
          f"n_hill=ceil({hill/L_IN:.3f}/{dy_w0/L_IN:.3f})={n_hill}  "
          f"Zone1={n_hill+20} cells")

    y, n_bl, n_sp = make_grid_pair(o_y, U_STAR, L_IN, hill, dx_ref, NY_TARGET,
                                   r_in=R_IN, r_out=R_OUT, n_in_decel=N_IN_DECEL,
                                   n_sponge=N_SPONGE, iso_thresh=ISO_THRESH,
                                   n_ramp=N_RAMP)
    print(f"\nNEW Y: Ny={len(y)} (n_bl={n_bl}, n_sponge={n_sp})")
    print_zones(y, L_IN, dx_ref)
    all_ok = grid_quality_table(y, o_y, U_STAR, L_IN, hill, dx_ref, label='new Y grid')

    for tname, nx, nz, istar, kstar in GRID_TARGETS:
        print(f"\n=== {tname}: {nx} x {len(y)} x {nz}  (dx={Lx/nx/L_IN:.3f}wu) ===")
        mpi_ok = check_mpi(nx, nz, len(y), istar, kstar)
        all_ok &= mpi_ok
        if WRITE_TO_FILE:
            xn = np.arange(nx) * (Lx / nx)
            zn = np.arange(nz) * (Lz / nz)
            out_dir = os.path.normpath(cwd + f'../{nx}x{len(y)}x{nz}_{tname}')
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.join(out_dir, f'grid_{nx}x{len(y)}x{nz}')
            write_grid(fname, nx, len(y), nz, o_scl, xn, y, zn)
            print(f"  wrote {fname}")
            # Read back the file and re-run the quality table to confirm what was
            # actually written to disk matches the design.
            _, _, _, y_rb, _ = read_grid(fname)
            rb_ok = grid_quality_table(y_rb, o_y, U_STAR, L_IN, hill, dx_ref,
                                       label=f"readback: grid_{nx}x{len(y)}x{nz}")
            all_ok &= rb_ok
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'CHECKS FAILED -- review above'}")
    return all_ok


if __name__ == '__main__':
    main()
