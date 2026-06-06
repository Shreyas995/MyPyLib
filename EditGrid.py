#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EditGrid.py — Wall-normal grid post-processing and quality enforcement for tlab DNS.

PURPOSE
-------
This script reads an EXISTING tlab binary grid file, diagnoses the wall-normal (Y)
spacing, applies a sequence of automatic fixes to enforce six grid-quality rules,
and optionally writes the corrected grid back to disk.  It does NOT create a grid
from scratch — the input grid must already have a reasonable structure.

WORKFLOW
--------
  1. Read grid  →  print zone table + sanity checks on the original grid.
  2. Manual Y reshape  (optional, activated by RESHAPE_XYZ[1] = True):
       call remesh_uniform() / remesh_geometric() in the "Manual Interventions"
       section below to coarsen, refine, or re-space a specific node range.
  3. Auto-fix isotropy  — if any cell below y = u★ has dy > dx, the grid is
       made uniform at 0.9 × dx from that point to u★ (Rule 3).
  4. Auto-fix contraction tails  — every significant stretch zone is followed
       by at least N_CONTRACTION cells of smooth ratio decrease (Rule 2).
  5. Auto-fix short-stretch violations  — single/double-cell jumps > 10 % are
       smoothed with a geometric-progression fill (Rule 6).
  6. Fit to NY_TARGET  — remove or add nodes from the largest non-protected
       uniform zone to hit an exact Jmax needed for the MPI decomposition.
  7. MPI decomposition check  — verifies C1–C8 constraints so the grid is
       compatible with tlab's pencil decomposition and Fourier solver.
  8. Write  (only when WRITE_TO_FILE = True).

SIX GRID-QUALITY RULES
-----------------------
  Rule 1  First zone must be uniform (near-wall resolution preserved).
  Rule 2  Every stretch zone (≥ 3 cells) is followed by ≥ N_CONTRACTION
          cells of contraction that gradually return the ratio to 1.
  Rule 3  dy / dx < 1 for all cells below y = u★ (BL isotropy).
  Rule 4  Domain endpoints Lx, Ly, Lz are never changed.
  Rule 5  At least one cell above y = u★ has dy / dx > 1 (anisotropy).
  Rule 6  No 1- or 2-cell stretch zone may exceed 10 % growth rate.

KEY CONVENTIONS
---------------
  • "stretching"  — find_grid_zones type where r = dy[i+1]/dy[i] is increasing.
  • "contraction" — r is decreasing (cells still grow, just more slowly).
  • "uniform"     — r is constant (includes regions growing at a fixed %).
  • All node indices are 0-based (Python convention).
  • y is the wall-normal direction (engineering: v, z in met. convention).
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import FortranFile


def read_grid(path):
    # Each Fortran record is bookended by a 4-byte int32 record-length marker.
    # Pattern: [marker][data][marker]. 'h' holds the byte count — read but only
    # printed for verification; a mismatch would indicate a corrupt file.
    f = open(path, 'rb')

    # header - number of nodes
    print("--------------------------------------------------")
    h = np.fromfile(f, '<i4', 1)
    print('iheader length = ', h)
    nmax = np.fromfile(f, '<i4', 3)
    h = np.fromfile(f, '<i4', 1)
    print('check iheader  = ', h)

    # header - grid scales
    print("--------------------------------------------------")
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    scales = np.fromfile(f, '<f8', 3)
    print('scales         = ', scales)
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)

    # x - nodes
    print("--------------------------------------------------")
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    x = np.fromfile(f, '<f8', nmax[0])
    print('x-nodes       =  ', x[:5])
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)

    # y - nodes
    print("--------------------------------------------------")
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    y = np.fromfile(f, '<f8', nmax[1])
    print('y-nodes       =  ', y[:5])
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)

    # z - nodes
    print("--------------------------------------------------")
    h = np.fromfile(f, '<i4', 1)
    print('fheader length  = ', h)
    z = np.fromfile(f, '<f8', nmax[2])
    print('z-nodes       =  ', z[:5])
    h = np.fromfile(f, '<i4', 1)
    print('check fheader  = ', h)
    print("--------------------------------------------------")

    f.close()

    return nmax, scales, x, y, z

# Print these params
def print_table(title, param_dict):
    print("##################")
    print(f"{title}")
    print("".ljust(25, '-'))
    for key, val in param_dict.items():
        print(f"{key:<15} : {val}")
    print()
    
def write_grid(path, nx, ny, nz, x_scale, y_scale, z_scale,
               x_nodes, y_nodes, z_nodes,
               int_dtype=np.int32, float_dtype=np.float64):
    """
    Write a tlab Fortran-format grid file (identical layout to read_grid).

    Scale conventions
    -----------------
    Periodic directions (X, Z):  scale = nodes[-1] + nodes[1]
                                        = last node + first spacing = Lx / Lz
    Wall-normal direction (Y):   scale = nodes[-1]  (= Ly, since y[0] = 0)
    """
    with FortranFile(path, mode='w') as f:
        f.write_record(np.array([nx, ny, nz],                         dtype=int_dtype))
        f.write_record(np.array([x_scale, y_scale, z_scale],          dtype=float_dtype))
        f.write_record(x_nodes[:nx].astype(float_dtype))
        f.write_record(y_nodes[:ny].astype(float_dtype))
        f.write_record(z_nodes[:nz].astype(float_dtype))


def find_grid_zones(r, tol=1e-10):
    """
    Classify contiguous regions of the grid by how the ratio r[i] changes.

    r[i] = dy[i+1] / dy[i]  where  dy = np.diff(o_Y).
    If o_Y has N nodes: dy has N-1 cells, r has N-2 elements.

    Classification uses a backward finite difference at each position
    (forward difference at i=0, since there is no left neighbour):
        dr[i] = r[i] - r[i-1]

        |dr[i]| <= tol  →  'uniform'     constant ratio (flat spacing or steady-% stretch)
         dr[i]  >  tol  →  'stretching'  ratio is increasing (ramp-up transition)
         dr[i]  < -tol  →  'contraction' ratio is decreasing (ramp-down transition)

    Parameters
    ----------
    r   : 1-D ndarray, shape (N-2,)
    tol : float  — tolerance on |dr| for the 'uniform' test.
                   Default 1e-10 is safe for double-precision grids.

    Returns
    -------
    zones : list of dict with keys
        'type'       – 'uniform', 'stretching', or 'contraction'
        'node_start' – first o_Y node index in this zone  (0-based, inclusive)
        'node_end'   – last  o_Y node index in this zone  (0-based, inclusive)
        'r_start'    – first r-array index in this zone   (0-based)
        'r_end'      – last  r-array index in this zone   (0-based, inclusive)
        'r_mean'     – mean ratio value within the zone
        'r_min'      – min  ratio value within the zone
        'r_max'      – max  ratio value within the zone

    Index mapping
    -------------
    r[k] = dy[k+1] / dy[k]:
        left  cell  dy[k]   spans  o_Y[k]   → o_Y[k+1]
        right cell  dy[k+1] spans  o_Y[k+1] → o_Y[k+2]
    So r[k] "lives between" o_Y nodes k and k+2.
    A zone that spans r[a..b] therefore maps to o_Y nodes [a, b+2].
    This means the first zone always starts at node 0 and the last zone
    always ends at node N-1, covering the full grid with no gaps.

    Note: each ratio straddles two adjacent cells, so zone boundaries carry
    an inherent ±1 node ambiguity at phase transitions.
    """
    r = np.asarray(r, dtype=float)
    n_r = len(r)
    if n_r < 2:
        raise ValueError(f"r needs at least 2 elements (o_Y needs ≥ 4 nodes); got {n_r}.")

    diff_r = np.diff(r)       # diff_r[i] = r[i+1] - r[i],  length n_r - 1

    # Build dr[i] = backward diff at r[i]; i=0 uses forward diff (no left neighbour).
    dr = np.empty(n_r)
    dr[0]  = diff_r[0]        # forward diff at the left boundary
    dr[1:] = diff_r           # dr[i] = r[i] - r[i-1]  for i >= 1

    label = np.where(np.abs(dr) <= tol,  0,     # uniform
             np.where(dr > 0,             1,     # stretching
                                         -1))    # contraction

    label_name = {0: 'uniform', 1: 'stretching', -1: 'contraction'}

    zones = []
    i = 0
    while i < n_r:
        j = i
        while j < n_r and label[j] == label[i]:
            j += 1
        a, b = i, j - 1          # run covers r[a..b]
        zones.append({
            'type'       : label_name[label[i]],
            'node_start' : a,         # o_Y[a]   = left  edge of dy[a]
            'node_end'   : b + 2,     # o_Y[b+2] = right edge of dy[b+1]
            'r_start'    : a,
            'r_end'      : b,
            'r_mean'     : float(np.mean(r[a : b + 1])),
            'r_min'      : float(np.min( r[a : b + 1])),
            'r_max'      : float(np.max( r[a : b + 1])),
        })
        i = j

    # Each r[k] straddles two adjacent cells, so the raw node_start formula
    # (= a, the left edge of dy[a]) lands one node before the shared boundary
    # between zones.  Force continuity: every zone after the first starts
    # exactly at the node where the previous zone ended.
    for k in range(1, len(zones)):
        zones[k]['node_start'] = zones[k - 1]['node_end']

    return zones


def print_zones(zones, o_Y, dy, dx_ref=None):
    """Print a formatted zone summary table with physical y-positions.
    Node numbers are displayed 0-based (Python convention).
    dy_min / dy_max are the smallest and largest cell widths within each zone.
    If dx_ref is given, dy/dx_min and dy/dx_max are also printed (isotropy check).
    """
    iso_hdr = f"  {'dy/dx_min':>10}  {'dy/dx_max':>10}" if dx_ref is not None else ""
    hdr = (f"{'#':>3}  {'type':12}  {'node_start':>10}  {'node_end':>8}  "
           f"{'y_start':>11}  {'y_end':>11}  "
           f"{'r_mean':>8}  {'stretch%':>9}  {'n_nodes':>7}  "
           f"{'dy_min':>10}  {'dy_max':>10}" + iso_hdr)
    sep = '-' * len(hdr)
    print(hdr)
    print(sep)
    for k, z in enumerate(zones):
        ns, ne  = z['node_start'], z['node_end']
        r_pct   = (z['r_mean'] - 1.0) * 100.0
        dy_zone = dy[ns:ne]          # cells dy[ns], dy[ns+1], …, dy[ne-1]
        dy_min  = float(np.min(dy_zone))
        dy_max  = float(np.max(dy_zone))
        iso_val = (f"  {dy_min/dx_ref:>10.4f}  {dy_max/dx_ref:>10.4f}"
                   if dx_ref is not None else "")
        print(f"{k+1:>3}  {z['type']:12}  {ns:>10d}  {ne:>8d}  "
              f"{o_Y[ns]:>11.4e}  {o_Y[ne]:>11.4e}  "
              f"{z['r_mean']:>8.5f}  {r_pct:>+9.4f}%  {ne - ns + 1:>7d}  "
              f"{dy_min:>10.4e}  {dy_max:>10.4e}" + iso_val)
    print(sep)
    last = zones[-1]['node_end']
    print(f"  Total zones: {len(zones)}   nodes: 0 – {last}  ({last + 1} total)")


def run_sanity_checks(o_Y, dy, dx_ref, u_star, l_in, n_contraction=5,
                      nx=None, nz=None, old_nz=None,
                      nx_divisor=None, nz_divisor=None, dx_wall_units=None):
    """
    Run all 6 grid sanity checks and print PASS / WARN for each.

    Rule 1 — If dy[0] > l_in the first zone MUST be refined to meet the wall-unit
              requirement.  If dy[0] ≤ l_in the first zone is already adequate and
              must be kept uniform (protected from modification).
    Rule 2 — every significant stretch zone (>= 3 nodes) must be followed by
              at least n_contraction contraction nodes.
    Rule 3 — dy/dx < 1 for every cell below y = u_star (BL isotropy).
    Rule 4 — informational: print current Ly (endpoint; cannot auto-verify).
    Rule 5 — necessity: at least one cell above y = u_star has dy/dx > 1.
              rule:      max cell-to-cell ratio anywhere above u_star ≤ 3 %.
    Rule 6 — no 1- or 2-cell stretch zone may exceed 3 % growth rate.
    """
    results = []

    # Pre-compute zones and cell-width ratios
    if len(dy) >= 3:
        r_tmp = dy[1:] / dy[:-1]
        zones = find_grid_zones(r_tmp)
    else:
        r_tmp = np.array([])
        zones = []

    # Boundary indices
    j_ustar_cell = int(np.clip(np.searchsorted(o_Y, u_star, side='right') - 1,
                               0, len(dy) - 1))
    j_ustar_node = int(np.clip(np.searchsorted(o_Y, u_star, side='right'),
                               0, len(dy)))

    # --- Rule 1: zone 1 is ALWAYS protected — may only be refined, NEVER stretched.
    #            Additionally, dy[0] must be ≤ l_in (wall-unit resolution). -----------
    if zones:
        zone1_uniform = zones[0]['type'] == 'uniform'
        needs_refine  = float(dy[0]) > l_in
        _z1 = zones[0]
        if not zone1_uniform:
            results.append((
                False, "Rule 1 — zone 1 stretched (never allowed)",
                f"dy[0] = {float(dy[0]):.4e},  first zone type = '{_z1['type']}'"
                f"  [FAIL: zone 1 may only be refined, never stretched]",
            ))
        elif needs_refine:
            results.append((
                False, "Rule 1 — zone 1 uniform but dy[0] > l_in (must be refined)",
                f"dy[0] = {float(dy[0]):.4e} > l_in = {l_in:.4e}"
                f"  [FAIL: refine zone 1 to finer spacing — never stretch it]",
            ))
        else:
            results.append((
                True, "Rule 1 — zone 1 uniform, dy[0] ≤ l_in (protected)",
                f"dy[0] = {float(dy[0]):.4e} ≤ l_in = {l_in:.4e},  "
                f"nodes {_z1['node_start']}–{_z1['node_end']}  [OK]",
            ))

    # --- Rule 2: stretch → ≥ n_contraction contraction nodes --------------------
    if zones:
        viol2 = check_stretch_contraction(zones, min_points=n_contraction)
        passed = len(viol2) == 0
        if passed:
            detail2 = f"all significant stretch zones followed by ≥{n_contraction} contraction nodes  [OK]"
        else:
            parts = [f"zone {v['zone_idx']} (node {v['node_stretch_end']},"
                     f" found {v['n_found']})" for v in viol2]
            detail2 = f"{len(viol2)} violation(s): " + ", ".join(parts) + "  [FAIL]"
        results.append((passed, f"Rule 2 — stretch → ≥{n_contraction} contraction nodes", detail2))

    # --- Rule 3: dy/dx < 1 for all cells below u_star ---------------------------
    max_ratio_bl  = max((float(dy[j]) / dx_ref for j in range(j_ustar_cell + 1)),
                        default=0.0)
    j_max_bl      = int(np.argmax([dy[j] / dx_ref for j in range(j_ustar_cell + 1)]))
    passed = max_ratio_bl <= 1.0
    results.append((
        passed, "Rule 3 — dy/dx < 1 for y < u_star",
        f"max dy/dx = {max_ratio_bl:.4f} at cell j={j_max_bl}, y={o_Y[j_max_bl]:.4e}"
        + ("  [OK]" if passed else "  [FAIL] isotropy violated in BL"),
    ))

    # --- Rule 4: Ly endpoint (informational) ------------------------------------
    Ly_current = float(o_Y[-1])
    results.append((
        True, "Rule 4 — Ly endpoint (informational)",
        f"Ly = {Ly_current:.8f}  [INFO]",
    ))

    # --- Rule 5: outer-region dy/dx > 1 (necessity) + max ratio ≤ 3 % (rule) ----
    MAX_OUTER_RATIO = 1.03
    if j_ustar_node < len(dy):
        max_ratio_above = max((float(dy[j]) / dx_ref
                               for j in range(j_ustar_node, len(dy))), default=0.0)
        ok_necessity = max_ratio_above > 1.0
        max_r_above = float(np.max(r_tmp[j_ustar_node:])) if j_ustar_node < len(r_tmp) else 1.0
        ok_rule5 = max_r_above <= MAX_OUTER_RATIO * (1.0 + 1e-8)
        passed = ok_necessity and ok_rule5
        results.append((
            passed, "Rule 5 — outer stretch ≤ 3 % and dy/dx > 1 above u★",
            f"max dy/dx above u★ = {max_ratio_above:.4f}"
            + ("  [necessity OK]" if ok_necessity else "  [FAIL: no anisotropic cells]")
            + f",  max cell ratio above u★ = {max_r_above:.5f}"
            + ("  [OK]" if ok_rule5 else f"  [FAIL: ratio > {MAX_OUTER_RATIO:.2f}]"),
        ))
    else:
        results.append((False, "Rule 5 — outer stretch ≤ 3 % and dy/dx > 1 above u★",
                        "  [FAIL] no nodes above u_star found"))

    # --- Rule 6: short stretch zones (1–2 cells) must not exceed 3 % ------------
    MAX_SHORT_RATIO = 1.03
    short_viol = [(z['node_start'], z['node_end'], z['r_max'])
                  for z in zones
                  if z['type'] == 'stretching'
                  and z['node_end'] - z['node_start'] <= 2
                  and z['r_max'] > MAX_SHORT_RATIO * (1.0 + 1e-8)]
    passed = len(short_viol) == 0
    if passed:
        detail6 = "no short stretch zones (1–2 cells) exceed 3 %  [OK]"
    else:
        parts6 = [f"nodes {ns}–{ne}, r_max={rm:.4f} (+{(rm-1)*100:.1f}%)"
                  for ns, ne, rm in short_viol]
        detail6 = f"{len(short_viol)} violation(s):  " + ";  ".join(parts6) + "  [FAIL]"
    results.append((passed, "Rule 6 — short stretch (1–2 cells) ≤ 3 %", detail6))

    # --- XZ rules (optional — only run when params are provided) -----------------
    if nx is not None and dx_wall_units is not None:
        _dx = dx_ref
        _dx_lin = _dx / l_in
        ok = abs(_dx_lin - dx_wall_units) / dx_wall_units <= 0.25
        results.append((ok, f"XZ-1 — dx ≈ {dx_wall_units:.0f} wall units",
            f"dx = {_dx:.4e} = {_dx_lin:.2f} l_in  "
            f"(target {dx_wall_units:.1f} l_in,  NX = {nx})  [{'OK' if ok else 'FAIL'}]"))
    if nx is not None and nx_divisor is not None:
        ok = (nx % nx_divisor == 0)
        results.append((ok, f"XZ-2 — NX divisible by {nx_divisor}",
            f"NX = {nx},  {nx} mod {nx_divisor} = {nx % nx_divisor}  [{'OK' if ok else 'FAIL'}]"))
    if nz is not None and old_nz is not None:
        ok = (nz >= old_nz)
        results.append((ok, "XZ-3 — NZ ≥ old NZ",
            f"NZ = {nz}  ≥  old NZ = {old_nz}  [{'OK' if ok else 'FAIL'}]"))
    if nz is not None and nz_divisor is not None:
        ok = (nz % nz_divisor == 0)
        results.append((ok, f"XZ-4 — NZ divisible by {nz_divisor}",
            f"NZ = {nz},  {nz} mod {nz_divisor} = {nz % nz_divisor}  [{'OK' if ok else 'FAIL'}]"))

    # --- Print ------------------------------------------------------------------
    print("\n--- Sanity checks ---")
    n_fail = sum(1 for ok, *_ in results if not ok)
    for ok, label, detail in results:
        tag = "PASS" if ok else "WARN"
        print(f"  [{tag}] {label}")
        print(f"         {detail}")
    if n_fail:
        print(f"\n  *** {n_fail} warning(s) — review grid before running the simulation ***")
    else:
        print("\n  All checks passed.")
    return n_fail == 0


def check_stretch_contraction(zones, min_points=5, min_zone_nodes=3):
    """
    Check that every sufficiently large stretching zone is followed by a
    contraction zone of at least min_points nodes.

    min_zone_nodes : skip stretching zones with fewer nodes than this (avoids
                     flagging single-cell boundary transitions like the
                     initial fine→coarse jump).

    Violation dict keys:
        zone_idx          – index k into zones of the stretching zone
        node_stretch_end  – zones[k]['node_end']
        r_peak            – zones[k]['r_max']
        n_found           – actual contraction node count following the zone
    """
    violations = []
    for k, z in enumerate(zones):
        if z['type'] != 'stretching':
            continue
        # Skip short single-cell transitions (Rule 1 protection)
        if z['node_end'] - z['node_start'] < min_zone_nodes:
            continue
        if k + 1 >= len(zones):
            n_found = 0
        elif zones[k + 1]['type'] == 'contraction':
            nxt = zones[k + 1]
            n_found = nxt['node_end'] - nxt['node_start']
        else:
            n_found = 0   # uniform or another stretching zone follows directly

        if n_found < min_points:
            violations.append({
                'zone_idx'         : k,
                'node_stretch_end' : z['node_end'],
                'r_peak'           : z['r_max'],
                'n_found'          : n_found,
            })
    return violations


def remesh_contraction_tail(y, node_stretch_end, r_peak, n_contraction):
    """
    Insert a smooth contraction zone of n_contraction cells after a stretching region.

    Cell ratios decrease linearly: r_c[k] = r_peak - (r_peak-1)*(k+1)/n
        r_c[0] = r_peak - delta  < r_peak  →  dr < 0 at the boundary  →  classified
                                               as contraction immediately (no uniform gap)
        r_c[-1] = 1.0            →  uniform from here onwards

    The replacement is size-neutral: y[ne : ne+n+1] (n+1 nodes) is replaced with
    n+1 new nodes.  All nodes from y[ne+n+1] onward are UNCHANGED, so Ly = y[-1]
    is preserved exactly and the total node count does not change.

    The only side-effect is that y[ne+n] moves slightly, creating a 1-cell ratio
    discontinuity at the boundary (too short for check_stretch_contraction to flag).

    Parameters
    ----------
    y                : 1-D ndarray — current node positions.
    node_stretch_end : int — last node of the stretching zone (0-based).
    r_peak           : float — peak stretch ratio at the end of the stretch zone.
    n_contraction    : int — number of cells in the contraction zone (≥ 2).

    Returns
    -------
    new_y, new_dy, new_r
    """
    y  = np.asarray(y, dtype=float)
    ne = node_stretch_end

    if ne + n_contraction + 1 >= len(y):
        raise ValueError(
            f"node_stretch_end={ne} + n_contraction={n_contraction} would exceed "
            f"grid length {len(y) - 1}.")

    dy_prev = float(y[ne] - y[ne - 1])

    # Ratios: start one step below r_peak, decrease linearly to 1.0
    k_arr    = np.arange(1, n_contraction + 1, dtype=float)
    ratios_c = r_peak - (r_peak - 1.0) * k_arr / n_contraction
    # ratios_c[0] < r_peak  →  dr = ratios_c[0] - r_peak < 0  →  contraction ✓
    # ratios_c[-1] = 1.0    →  uniform after the tail

    # Build cell widths: each cell is ratio-times the previous
    dy_c    = np.empty(n_contraction)
    dy_c[0] = dy_prev * ratios_c[0]
    for k in range(1, n_contraction):
        dy_c[k] = dy_c[k - 1] * ratios_c[k]

    # Replace y[ne : ne+n+1] with n+1 new nodes; y[ne+n+1:] unchanged → Ly intact
    new_nodes    = np.empty(n_contraction + 1)
    new_nodes[0] = y[ne]
    new_nodes[1:] = y[ne] + np.cumsum(dy_c)

    new_y  = np.concatenate([y[:ne], new_nodes, y[ne + n_contraction + 1:]])
    new_dy = np.diff(new_y)
    new_r  = new_dy[1:] / new_dy[:-1]

    print(f"\n[remesh_contraction_tail]  node {ne},  r_peak={r_peak:.5f}  →  1.0"
          f"  over {n_contraction} cells  (dy_prev={dy_prev:.4e},"
          f"  dy_final={float(dy_c[-1]):.4e},  y={y[ne]:.4e})")
    zones = find_grid_zones(new_r)
    print_zones(zones, new_y, new_dy)

    return new_y, new_dy, new_r


def remesh_uniform(y, node_start, node_end, n_div):
    """
    Replace a region of the grid with uniform spacing and return the updated grid.

    Nodes outside [node_start, node_end] are left unchanged.
    The physical endpoints y[node_start] and y[node_end] are preserved exactly;
    the interior is replaced by np.linspace(y[node_start], y[node_end], n_div).

    Parameters
    ----------
    y          : 1-D ndarray — current wall-normal node positions (n_Y or o_Y).
    node_start : int — first node of the region to remesh  (0-based, inclusive).
    node_end   : int — last  node of the region to remesh  (0-based, inclusive).
    n_div      : int — number of nodes in the remeshed region, including both
                       endpoints.  n_div=2 gives a single cell; n_div must be ≥ 2.

    Returns
    -------
    new_y  : ndarray — full updated node-position array.
    new_dy : ndarray — cell widths  np.diff(new_y).
    new_r  : ndarray — cell-width ratios  new_dy[1:] / new_dy[:-1].

    Side-effects
    ------------
    Calls find_grid_zones + print_zones on the updated grid so the zone
    structure is printed immediately after the remesh.

    Index notes
    -----------
    Original node count in region : node_end - node_start + 1
    New      node count in region : n_div
    Net change in total node count: n_div - (node_end - node_start + 1)
    Uniform cell size in region   : (y[node_end] - y[node_start]) / (n_div - 1)
    """
    y = np.asarray(y, dtype=float)
    N = len(y)

    a = node_start
    b = node_end

    if not (0 <= a < b < N):
        raise ValueError(
            f"node_start={node_start} and node_end={node_end} must satisfy "
            f"0 <= node_start < node_end <= {N - 1}.")
    if n_div < 2:
        raise ValueError(f"n_div must be ≥ 2 (got {n_div}).")

    region = np.linspace(y[a], y[b], n_div)

    # Stitch: keep everything before a, the new uniform region, then
    # everything after b.  Endpoints are shared — no gap, no duplicate:
    # y[:a] ends at y[a-1]; region starts at y[a].
    # region ends at y[b];  y[b+1:] starts at y[b+1].
    new_y = np.concatenate([y[:a], region, y[b + 1:]])

    new_dy = np.diff(new_y)
    new_r  = new_dy[1:] / new_dy[:-1]

    dy_uniform = (y[b] - y[a]) / (n_div - 1)
    print(f"\n[remesh_uniform]  nodes {node_start} – {node_end}  →  {n_div} divisions"
          f"  |  Δy = {dy_uniform:.4e}"
          f"  |  total nodes: {N} → {len(new_y)}")

    zones = find_grid_zones(new_r)
    print_zones(zones, new_y, new_dy)

    return new_y, new_dy, new_r


def remesh_geometric(y, node_start, node_end, mode, n_div=None, r=None):
    """
    Replace a region of the grid with a geometric (GP) or arithmetic (AP) progression.

    Parameters
    ----------
    y          : 1-D ndarray — current node positions.
    node_start : int — first node of the region (0-based, inclusive).
    node_end   : int — last  node of the region (0-based, inclusive).
    mode       : 'GP' or 'AP'
        'GP' — geometric progression of node positions:  y_i = a * r^i
               where  a = y[node_start],  L = y[node_end],  k = n_div - 1.
               Requires a > 0 (region must not start at wall).
               Give n_div  →  r = (L/a)^(1/k)
               Give r      →  k = round(ln(L/a) / ln(r)),  n_div = k + 1
        'AP' — arithmetic progression of cell widths: dy_i = dy_0 + i*delta
               dy_0 matched to the adjacent cell; n_div required; r not applicable.
    n_div      : int, optional — total number of nodes including both endpoints.
    r          : float, optional — common ratio (GP only).
                 Cannot be combined with n_div.

    Returns
    -------
    new_y  : ndarray — updated node-position array.
    new_dy : ndarray — cell widths.
    new_r  : ndarray — cell-width ratios.
    """
    y = np.asarray(y, dtype=float)
    N = len(y)

    if mode not in ('GP', 'AP'):
        raise ValueError(f"mode must be 'GP' or 'AP', got '{mode}'.")

    a_idx = node_start
    b_idx = node_end

    if not (0 <= a_idx < b_idx < N):
        raise ValueError(
            f"node_start={node_start} and node_end={node_end} must satisfy "
            f"0 <= node_start < node_end <= {N - 1}.")

    a = float(y[a_idx])   # physical start position
    L = float(y[b_idx])   # physical end position

    if mode == 'GP':
        if a <= 0:
            raise ValueError(
                f"GP requires y[node_start] > 0; got y[{node_start}] = {a:.4e}.")
        if n_div is not None and r is not None:
            raise ValueError("GP: specify exactly one of n_div or r, not both.")
        if n_div is None and r is None:
            raise ValueError("GP: specify one of n_div or r.")

        if n_div is not None:
            k = int(n_div) - 1
            r = (L / a) ** (1.0 / k)
        else:
            r = float(r)
            k = int(round(np.log(L / a) / np.log(r)))
            n_div = k + 1

        region     = a * r ** np.arange(n_div, dtype=float)
        region[-1] = L   # pin right endpoint against float drift

        print(f"\n[remesh_geometric GP]  nodes {node_start}–{node_end}  →  {n_div} nodes"
              f"  |  r = {r:.6f}"
              f"  |  total nodes: {N} → {N - (b_idx - a_idx + 1) + n_div}")

    else:  # AP — arithmetic progression of cell widths: dy_i = dy_0 + i*delta
        if r is not None:
            raise ValueError("AP mode uses only n_div; r is not applicable.")
        if n_div is None:
            raise ValueError("AP: n_div must be specified.")
        n_div = int(n_div)
        if n_div < 3:
            raise ValueError("AP requires n_div ≥ 3 (need at least 2 cells to form a progression).")

        k    = n_div - 1          # number of cells
        span = L - a

        # First cell width matched to adjacent cell for a smooth transition.
        if a_idx > 0:
            dy0 = float(y[a_idx] - y[a_idx - 1])
        elif b_idx + 1 < N:
            dy0 = float(y[b_idx + 1] - y[b_idx])
        else:
            raise ValueError("Region spans entire grid; cannot infer dy_0 from a neighbour.")

        # Sum of AP cell widths = span:
        #   sum_{i=0}^{k-1} (dy0 + i*delta) = k*dy0 + delta*k*(k-1)/2 = span
        delta  = 2.0 * (span - k * dy0) / (k * (k - 1))
        widths = dy0 + delta * np.arange(k, dtype=float)
        region = a + np.concatenate([[0.0], np.cumsum(widths)])
        region[-1] = L   # pin right endpoint against float drift

        dy_last = float(widths[-1])
        print(f"\n[remesh_geometric AP]  nodes {node_start}–{node_end}  →  {n_div} nodes"
              f"  |  dy_0 = {dy0:.4e}  dy_last = {dy_last:.4e}  delta = {delta:.4e}"
              f"  |  total nodes: {N} → {N - (b_idx - a_idx + 1) + n_div}")

    new_y  = np.concatenate([y[:a_idx], region, y[b_idx + 1:]])
    new_dy = np.diff(new_y)
    new_r  = new_dy[1:] / new_dy[:-1]

    zones = find_grid_zones(new_r)
    print_zones(zones, new_y, new_dy)

    return new_y, new_dy, new_r


def remesh_stretch_max_ratio(y, node_start, node_end, max_ratio=1.05, dy0=None):
    """
    Redistribute nodes[node_start..node_end] using a geometric cell-width
    progression that respects max_ratio — without adding any new nodes.

    The cell widths form a GP: dy[i] = dy0 * r^i, where r is the constant ratio
    solved so that the total span [y[node_start], y[node_end]] is covered in
    exactly (node_end - node_start) cells.

    If r ≤ max_ratio the redistribution is valid.
    If r > max_ratio the caller must add nodes first (this function only redistributes).

    node_start : first node of the region (zone1 end — preserved as endpoint).
    node_end   : last  node of the region (last grid node — Ly preserved).
    dy0        : first cell width for the GP; defaults to y[node_start]-y[node_start-1].

    Returns (new_y, new_dy, new_r, achieved_ratio)
    """
    from scipy.optimize import brentq

    y       = np.asarray(y, dtype=float)
    a, b    = node_start, node_end
    n_cells = b - a
    span    = float(y[b] - y[a])

    if n_cells < 1:
        raise ValueError("node_end must be > node_start")
    if dy0 is None:
        dy0 = float(y[a] - y[a - 1])

    def _span(r):
        if abs(r - 1.0) < 1e-12:
            return dy0 * n_cells
        return dy0 * (r ** n_cells - 1.0) / (r - 1.0)

    # Solve for constant ratio r that spans [y[a], y[b]] with n_cells cells
    if abs(_span(1.0) - span) < 1e-10 * span:
        r = 1.0
    else:
        # Find a safe upper bound that avoids r**n overflow
        _r_hi = 5.0
        while True:
            try:
                if _span(_r_hi) > span:
                    break
            except (OverflowError, FloatingPointError):
                pass
            _r_hi = max(1.001, (_r_hi - 1.0) * 0.5 + 1.0)
        r = brentq(lambda r_: _span(r_) - span, 1.0, _r_hi,
                   xtol=1e-14, rtol=1e-12)

    # Build cell widths; rescale to remove float residual (Ly preserved exactly)
    widths    = dy0 * r ** np.arange(n_cells, dtype=float)
    widths   *= span / widths.sum()

    # Node positions
    new_nodes     = np.empty(n_cells + 1)
    new_nodes[0]  = y[a]
    new_nodes[1:] = y[a] + np.cumsum(widths)
    new_nodes[-1] = y[b]   # pin endpoint → Ly preserved

    achieved_ratio = float(np.max(widths[1:] / widths[:-1])) if n_cells > 1 else 1.0

    new_y  = np.concatenate([y[:a], new_nodes, y[b + 1:]])
    new_dy = np.diff(new_y)
    new_r  = new_dy[1:] / new_dy[:-1]

    print(f"\n[remesh_stretch_max_ratio]"
          f"  nodes {a}–{b}  |  n_cells={n_cells}"
          f"  |  dy0={dy0:.4e}  |  r={r:.6f} ({(r-1)*100:.4f}%)"
          f"  |  max ratio in region = {achieved_ratio:.6f}"
          f"  |  dy_first={float(widths[0]):.4e}  dy_last={float(widths[-1]):.4e}")

    zones = find_grid_zones(new_r)
    print_zones(zones, new_y, new_dy)

    return new_y, new_dy, new_r, r


def build_recipe_above_zone1(y, zone1_node, dx, y_BL, Ly,
                              r_trans, n_inner_decel, n_outer_decel, n_sponge,
                              iso_thresh, ny_target):
    """
    Rebuild all nodes ABOVE zone1 according to the 7-zone physics recipe.
    zone1 (nodes 0..zone1_node) is NEVER modified.

    Zones:
      2  Transition  : constant ratio r_trans (≤1.02) until dy ≈ iso_thresh × dx
      3  Inner decel : linear ratio ramp from r_trans to 1.0 over n_inner_decel cells
      4  BL uniform  : constant dy_iso from end-of-decel to y_BL
      5  Outer stretch: constant ratio r_outer (solved) from y_BL to near Ly
      6  Outer decel : linear ramp from r_outer to 1.0 over n_outer_decel cells
      7  Sponge      : uniform n_sponge cells at dy_sponge

    r_outer is solved so that zones 5+6+7 exactly span (Ly − y_BL_end).
    Total node count equals ny_target exactly.

    Returns (new_y, r_outer).
    """
    from scipy.optimize import brentq

    dy0  = float(y[zone1_node] - y[zone1_node - 1])
    y_z1 = float(y[zone1_node])

    # Zone 2 – transition stretch
    dy_tgt  = iso_thresh * dx
    n_trans = int(np.ceil(np.log(dy_tgt / dy0) / np.log(r_trans)))
    widths_trans = dy0 * r_trans ** np.arange(n_trans, dtype=float)

    # Zone 3 – inner deceleration (linear ratio ramp r_trans → 1.0)
    # Use actual last zone2 cell as base (not the hypothetical next cell),
    # so the zone2→zone3 junction ratio = ratios_inner[0] < r_trans ≤ 1.02 < 1.03.
    dy_s = dy0 * r_trans ** (n_trans - 1)
    k_arr = np.arange(1, n_inner_decel + 1, dtype=float)
    ratios_inner = r_trans - (r_trans - 1.0) * k_arr / n_inner_decel
    widths_inner = np.empty(n_inner_decel)
    widths_inner[0] = dy_s * ratios_inner[0]
    for k in range(1, n_inner_decel):
        widths_inner[k] = widths_inner[k - 1] * ratios_inner[k]
    dy_iso = float(widths_inner[-1])
    y_iso  = y_z1 + widths_trans.sum() + widths_inner.sum()

    # Zone 4 – BL uniform
    n_BL      = int(np.ceil((y_BL - y_iso) / dy_iso))
    widths_BL = np.full(n_BL, dy_iso)
    y_BL_end  = y_iso + widths_BL.sum()

    # Available cells for zones 5+6+7
    n_fixed    = n_trans + n_inner_decel + n_BL
    n_above_BL = ny_target - (zone1_node + 1) - n_fixed - n_outer_decel - n_sponge
    if n_above_BL < 1:
        raise ValueError(
            f"No cells left for outer stretch (n_above_BL={n_above_BL}). "
            f"Reduce n_inner_decel, n_outer_decel or n_sponge.")

    span_outer = Ly - y_BL_end

    # Helper: compute span of zones 5+6+7 for a given r
    def _outer_span(r):
        if abs(r - 1.0) < 1e-12:
            s5 = dy_iso * n_above_BL
            dy_last5 = dy_iso
        else:
            s5      = dy_iso * (r ** n_above_BL - 1.0) / (r - 1.0)
            dy_last5 = dy_iso * r ** (n_above_BL - 1)   # actual last zone5 cell
        # Zone 6: linear ramp r → 1.0; base = last zone5 cell so junction ratio ≤ r < 1.03
        kk = np.arange(1, n_outer_decel + 1, dtype=float)
        rat_o = r - (r - 1.0) * kk / n_outer_decel
        wo = np.empty(n_outer_decel)
        wo[0] = dy_last5 * rat_o[0]
        for j in range(1, n_outer_decel):
            wo[j] = wo[j - 1] * rat_o[j]
        s6 = wo.sum()
        # Zone 7: sponge
        s7 = n_sponge * float(wo[-1])
        return s5 + s6 + s7

    # Solve for r_outer — capped at 1.03 (Rule 5: outer stretch ≤ 3 %)
    MAX_R_OUTER = 1.03
    if _outer_span(MAX_R_OUTER) < span_outer:
        raise ValueError(
            f"Cannot span outer region ({span_outer:.4f}) with r_outer ≤ {MAX_R_OUTER:.2f}. "
            f"Increase NY_TARGET (or reduce n_sponge/n_outer_decel) to add more cells.")
    r_outer = brentq(lambda r: _outer_span(r) - span_outer,
                     1.0001, MAX_R_OUTER, xtol=1e-14, rtol=1e-12)

    # Build zones 5, 6, 7
    widths_outer = dy_iso * r_outer ** np.arange(n_above_BL, dtype=float)
    dy_last5 = dy_iso * r_outer ** (n_above_BL - 1)   # actual last zone5 cell
    kk = np.arange(1, n_outer_decel + 1, dtype=float)
    rat_o = r_outer - (r_outer - 1.0) * kk / n_outer_decel
    widths_od = np.empty(n_outer_decel)
    widths_od[0] = dy_last5 * rat_o[0]   # junction ratio = rat_o[0] < r_outer < 1.03
    for j in range(1, n_outer_decel):
        widths_od[j] = widths_od[j - 1] * rat_o[j]
    widths_sponge = np.full(n_sponge, float(widths_od[-1]))

    # Assemble and rescale to exact Ly
    all_widths  = np.concatenate([widths_trans, widths_inner,
                                   widths_BL, widths_outer,
                                   widths_od, widths_sponge])
    all_widths *= (Ly - y_z1) / all_widths.sum()

    new_above    = y_z1 + np.concatenate([[0.0], np.cumsum(all_widths)])
    new_above[-1] = Ly
    new_y = np.concatenate([y[:zone1_node + 1], new_above[1:]])
    return new_y, r_outer


def remesh_fix_isotropy(y, dx_ref, u_star, threshold=0.9):
    """
    Prevent dy > dx for y < u_star by making the grid uniform at the cell
    width where dy first reaches `threshold * dx_ref`.

    threshold=0.9 (not 1.0) accounts for the overshoot caused by contraction
    zones that follow a stretching zone: during contraction the ratio r can
    still be > 1, so cells keep growing.  Stopping at 0.9*dx lets that tail
    land near dx rather than above it.

    Parameters
    ----------
    y         : 1-D ndarray — current node positions.
    dx_ref    : float — representative horizontal cell width (isotropy reference).
    u_star    : float — wall-unit BL height; scan stops here.
    threshold : float — dy/dx ratio at which stretching is halted (default 0.9).

    Returns
    -------
    new_y, new_dy, new_r  (same convention as remesh_uniform)
    """
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    target_dy = threshold * dx_ref

    # Index of the first node at or beyond u_star
    j_ustar = int(np.clip(np.searchsorted(y, u_star, side='right'), 1, len(y) - 1))

    # Find the first cell below j_ustar where dy >= target_dy
    fix_node = None
    for j in range(j_ustar):
        if dy[j] >= target_dy:
            fix_node = j
            break

    if fix_node is None:
        print(f"\n[remesh_fix_isotropy]  dy < {threshold:.2f}*dx everywhere below"
              f" y=u_star={u_star:.4e}.  No change needed.")
        new_r = dy[1:] / dy[:-1]
        return y, dy, new_r

    dy_fix    = float(dy[fix_node])
    y_a       = float(y[fix_node])
    y_b       = float(y[j_ustar])
    n_div     = max(2, round((y_b - y_a) / dy_fix) + 1)
    dy_actual = (y_b - y_a) / (n_div - 1)

    print(f"\n[remesh_fix_isotropy]  threshold dy/dx = {threshold:.2f}"
          f"  →  target dy = {target_dy:.4e}")
    print(f"  Trigger: node j={fix_node},  y={y_a:.4e},"
          f"  dy={dy_fix:.4e}  (dy/dx = {dy_fix/dx_ref:.4f})")
    print(f"  Making uniform from node {fix_node} to {j_ustar}:"
          f"  {n_div} nodes,  Δy = {dy_actual:.4e}")

    return remesh_uniform(y, fix_node, j_ustar, n_div)


def fix_rule6_violations(y, ny_limit, max_r=1.03, protect_end=None):
    """
    Detect and fix Rule 6 violations: short stretch zones (1–2 cells) whose
    ratio r_max exceeds max_r (default 1.03 = 3 %).

    For each violation at node_start=ns with r_peak:
      1. Compute n_needed = ceil(log(r_peak)/log(max_r)) — minimum cells to
         smooth the transition so each ratio step ≤ max_r.
      2. Find ne_ext: the first existing node at or beyond the ideal GP endpoint
         y[ns] + dy_prev * max_r * (max_r^n_needed - 1) / (max_r - 1).
      3. Replace y[ns : ne_ext+1] with n_needed+1 nodes whose cell widths form
         a geometric progression starting at dy_prev * max_r, scaled to fit
         exactly in [y[ns], y[ne_ext]] — preserving Ly and the rest of the grid.

    Option A: used when len(y) + delta_nodes ≤ ny_limit (just inserts nodes).
    Option B: when ny_limit is tight, first removes the needed nodes from the
              largest suitable uniform zone (fewest structural side-effects),
              then applies Option A.

    Rule 1 protection:
      protect_end = None  → all violations fixed (ALLOW_RULE1_MODIFICATION=True).
      protect_end = k     → violations where ns ≤ k are skipped with a notice.

    Returns (new_y, new_dy, new_r, any_fixed_bool).
    """
    y = np.asarray(y, dtype=float)
    any_fixed  = False
    max_iters  = 30
    _notified  = set()   # track which protected violations have been printed

    for _it in range(max_iters):
        dy    = np.diff(y)
        r_arr = dy[1:] / dy[:-1]
        zones = find_grid_zones(r_arr)

        # Collect all Rule 6 violations (tiny tolerance avoids fp re-triggering)
        viols = [z for z in zones
                 if z['type'] == 'stretching'
                 and z['node_end'] - z['node_start'] <= 2
                 and z['r_max'] > max_r * (1.0 + 1e-8)]

        # Split into protected (skip) and active (fix)
        if protect_end is not None:
            active = [v for v in viols if v['node_start'] > protect_end]
            for v in viols:
                if v['node_start'] <= protect_end and v['node_start'] not in _notified:
                    _notified.add(v['node_start'])
                    print(f"  [Rule 6] Skipping nodes {v['node_start']}–{v['node_end']}"
                          f" (r={v['r_max']:.4f}): within Rule 1 protected zone."
                          f" Set ALLOW_RULE1_MODIFICATION=True to fix.")
        else:
            active = viols

        if not active:
            break

        # Fix one at a time (re-detect zones after each fix because indices shift)
        z      = active[0]
        ns     = z['node_start']
        r_peak = z['r_max']
        dy_prev = float(y[ns] - y[ns - 1])

        # Minimum cells for max ratio ≤ max_r per step
        n_min = max(2, int(np.ceil(np.log(r_peak) / np.log(max_r))))

        # Find smallest n_needed (≥ n_min) such that scale = span_actual/span_ideal ≥ MIN_SCALE.
        # This prevents extreme cell compression when span_actual << span_ideal.
        MIN_SCALE = 0.70
        n_needed = None
        ne_ext   = None
        span_ideal_found = None
        for _n in range(n_min, n_min + 20):
            _span = dy_prev * max_r * (max_r**_n - 1.0) / (max_r - 1.0)
            _ne   = int(np.clip(
                np.searchsorted(y, float(y[ns]) + _span, side='right') - 1,
                ns + 1, len(y) - 1))
            _scale = (float(y[_ne]) - float(y[ns])) / _span
            if _scale >= MIN_SCALE:
                n_needed, ne_ext, span_ideal_found = _n, _ne, _span
                break

        if n_needed is None:
            print(f"  [Rule 6 WARN] nodes {ns}–{z['node_end']}: "
                  f"cannot find valid fix window (all scales < {MIN_SCALE:.2f}). Skipping.")
            break

        delta_n = (n_needed + 1) - (ne_ext - ns + 1)   # node count change

        opt = 'A'
        if delta_n > 0 and ny_limit is not None and len(y) + delta_n > ny_limit:
            opt = 'B'
            # Steal from the largest uniform zone that has room and is unprotected
            candidates = [z2 for z2 in zones
                          if z2['type'] == 'uniform'
                          and z2['node_end'] - z2['node_start'] > delta_n + 2
                          and (protect_end is None or z2['node_start'] > protect_end)]
            if not candidates:
                print(f"\n  [Rule 6 WARN] Cannot fix nodes {ns}: "
                      f"ny_limit={ny_limit} exceeded and no free uniform zone. Skipping.")
                break
            steal   = max(candidates,
                          key=lambda z2: z2['node_end'] - z2['node_start'])
            new_cnt = steal['node_end'] - steal['node_start'] + 1 - delta_n
            print(f"\n  [Rule 6 - Option B] Stealing {delta_n} nodes from "
                  f"uniform zone {steal['node_start']}–{steal['node_end']} "
                  f"({steal['node_end']-steal['node_start']+1} → {new_cnt} nodes).")
            y, _, _ = remesh_uniform(y, steal['node_start'], steal['node_end'],
                                     max(2, new_cnt))
            # Recompute dy_prev and ne_ext after steal (indices before ns unchanged)
            dy_prev = float(y[ns] - y[ns - 1])
            ne_ext  = int(np.clip(
                np.searchsorted(y, float(y[ns]) + span_ideal_found, side='right') - 1,
                ns + 1, len(y) - 1))

        # Build GP-of-cell-widths: ideal cells scaled to fit [y[ns], y[ne_ext]]
        span_actual  = float(y[ne_ext] - y[ns])
        ideal_cells  = dy_prev * max_r ** np.arange(1, n_needed + 1, dtype=float)
        scale        = span_actual / float(np.sum(ideal_cells))
        scaled_cells = ideal_cells * scale

        new_nodes     = np.empty(n_needed + 1)
        new_nodes[0]  = float(y[ns])
        new_nodes[1:] = float(y[ns]) + np.cumsum(scaled_cells)
        new_nodes[-1] = float(y[ne_ext])   # pin endpoint → Ly intact

        y = np.concatenate([y[:ns], new_nodes, y[ne_ext + 1:]])
        any_fixed = True

        _dy_new = np.diff(y)
        _r_new  = _dy_new[1:] / _dy_new[:-1]
        print(f"\n  [Rule 6 - Option {opt}] nodes {ns}–{ne_ext}:"
              f"  r_peak={r_peak:.4f}, n_cells={n_needed},"
              f"  eff_r_start={scale*max_r:.4f}, scale={scale:.4f},"
              f"  Δnodes={delta_n}, total={len(y)}")
        print_zones(find_grid_zones(_r_new), y, _dy_new)
    else:
        print(f"\n[Rule 6 Fix] Max iterations ({max_iters}) reached — "
              f"check remaining violations manually.")

    new_dy = np.diff(y)
    new_r  = new_dy[1:] / new_dy[:-1]
    return y, new_dy, new_r, any_fixed


def check_parallel_decomposition(Imax, Kmax, Jmax, Imax_star, Kmax_star,
                                 fourier=True, label=''):
    """
    Verify that the grid dimensions are compatible with tlab's MPI decomposition.

    Imax, Kmax     – total grid points in X and Z (NEW_NX, NEW_NZ).
    Jmax           – total grid points in Y (n_NMAX[1] after all modifications).
    Imax_star      – local X points per rank, Imax(*) = NX_pro.
    Kmax_star      – local Z points per rank, Kmax(*) = NZ_pro.
    fourier        – True when running Fourier/Poisson (applies C4, C7, C8).

    Derived quantities
    ------------------
    npro_i = Imax / Imax_star   (ranks splitting X)
    npro_k = Kmax / Kmax_star   (ranks splitting Z)
    npro   = npro_i × npro_k    (total MPI ranks)

    Constraint table
    ----------------
    C1  Imax(*) > 0  and  Imax mod Imax(*) == 0            always
    C2  Kmax(*) > 0  and  Kmax mod Kmax(*) == 0            always
    C3  npro_i × npro_k  (informational — total rank count) always
    C4  Imax(*) mod 2 == 0                                  Fourier/Poisson
    C5  (Kmax(*) × Jmax) mod npro_i == 0                   npro_i > 1
    C6  (Imax(*) × Jmax) mod npro_k == 0                   npro_k > 1
    C7  ((Imax(*)+2) × Jmax) / 2  mod npro_k == 0          Fourier, npro_k > 1
    C8  (Jmax × Kmax(*)) mod npro_i == 0                   Fourier, npro_i > 1
        (same arithmetic as C5 — two code sites, same constraint)
    """
    results = []

    npro_i = Imax // Imax_star if Imax_star > 0 else 0
    npro_k = Kmax // Kmax_star if Kmax_star > 0 else 0
    npro   = npro_i * npro_k

    # C1
    ok = Imax_star > 0 and Imax % Imax_star == 0
    results.append((ok, "C1 — Imax divisible by Imax(*)",
        f"Imax={Imax}, Imax(*)={Imax_star},  {Imax} mod {Imax_star} = {Imax % Imax_star if Imax_star else 'N/A'}"
        + ("  [OK]" if ok else "  [FAIL]")))

    # C2
    ok = Kmax_star > 0 and Kmax % Kmax_star == 0
    results.append((ok, "C2 — Kmax divisible by Kmax(*)",
        f"Kmax={Kmax}, Kmax(*)={Kmax_star},  {Kmax} mod {Kmax_star} = {Kmax % Kmax_star if Kmax_star else 'N/A'}"
        + ("  [OK]" if ok else "  [FAIL]")))

    # C3 — informational
    results.append((True, "C3 — total MPI rank count (informational)",
        f"npro_i={npro_i}, npro_k={npro_k},  npro = {npro}  [INFO]"))

    # C4 — Imax(*) must be even for Fourier/Poisson
    if fourier:
        ok = Imax_star % 2 == 0
        results.append((ok, "C4 — Imax(*) even  (Fourier/Poisson)",
            f"Imax(*) = {Imax_star},  {Imax_star} mod 2 = {Imax_star % 2}"
            + ("  [OK]" if ok else "  [FAIL]")))

    # C5 — X-transpose: Kmax(*) × Jmax divisible by npro_i
    if npro_i > 1:
        val = Kmax_star * Jmax
        ok = val % npro_i == 0
        results.append((ok, "C5 — (Kmax(*) × Jmax) mod npro_i == 0",
            f"Kmax(*)×Jmax = {Kmax_star}×{Jmax} = {val},  npro_i = {npro_i},"
            f"  remainder = {val % npro_i}"
            + ("  [OK]" if ok else "  [FAIL]")))

    # C6 — Z-transpose: Imax(*) × Jmax divisible by npro_k
    if npro_k > 1:
        val = Imax_star * Jmax
        ok = val % npro_k == 0
        results.append((ok, "C6 — (Imax(*) × Jmax) mod npro_k == 0",
            f"Imax(*)×Jmax = {Imax_star}×{Jmax} = {val},  npro_k = {npro_k},"
            f"  remainder = {val % npro_k}"
            + ("  [OK]" if ok else "  [FAIL]")))

    # C7 — Fourier X-FFT buffer: ((Imax(*)+2) × Jmax) / 2 divisible by npro_k
    if fourier and npro_k > 1:
        val = (Imax_star + 2) * Jmax
        ok = (val // 2) % npro_k == 0
        results.append((ok, "C7 — ((Imax(*)+2) × Jmax) / 2  mod npro_k == 0  (Fourier buffer)",
            f"(Imax(*)+2)×Jmax/2 = ({Imax_star}+2)×{Jmax}/2 = {val//2},"
            f"  npro_k = {npro_k},  remainder = {(val // 2) % npro_k}"
            + ("  [OK]" if ok else "  [FAIL]")))

    # C8 — same constraint as C5 at a second code site
    if fourier and npro_i > 1:
        val = Jmax * Kmax_star
        ok = val % npro_i == 0
        results.append((ok, "C8 — (Jmax × Kmax(*)) mod npro_i == 0  (≡ C5)",
            f"Jmax×Kmax(*) = {Jmax}×{Kmax_star} = {val},  npro_i = {npro_i},"
            f"  remainder = {val % npro_i}"
            + ("  [OK]" if ok else "  [FAIL]")))

    # Print
    title = f"MPI decomposition checks — {label}" if label else "MPI decomposition checks"
    print(f"\n--- {title} ---")
    print(f"  Grid:    Imax={Imax},  Jmax={Jmax},  Kmax={Kmax}")
    print(f"  Local:   Imax(*)={Imax_star},  Kmax(*)={Kmax_star}")
    print(f"  Ranks:   npro_i={npro_i},  npro_k={npro_k},  npro={npro}")
    print(f"  Fourier: {fourier}")
    print()
    n_fail = sum(1 for ok2, *_ in results if not ok2)
    for ok2, label, detail in results:
        tag = "PASS" if ok2 else "FAIL"
        print(f"  [{tag}] {label}")
        print(f"         {detail}")
    if n_fail:
        print(f"\n  *** {n_fail} MPI constraint(s) FAILED — Jmax={Jmax} is incompatible "
              f"with the current decomposition.  Adjust Jmax or Imax(*)/Kmax(*). ***")
        # Suggest nearest valid Jmax values
        lcm_req = 1
        if npro_i > 1 and Kmax_star > 0:
            from math import gcd
            lcm_req = (lcm_req * npro_i) // gcd(lcm_req * Kmax_star, npro_i)
        if npro_k > 1 and Imax_star > 0:
            from math import gcd
            lcm_req_k = npro_k // gcd(Imax_star, npro_k)
            lcm_req = (lcm_req * lcm_req_k) // gcd(lcm_req, lcm_req_k)
        if fourier and npro_k > 1:
            from math import gcd
            fft_factor = npro_k // gcd((Imax_star + 2) // 2 if (Imax_star + 2) % 2 == 0 else 1, npro_k)
            lcm_req = (lcm_req * fft_factor) // gcd(lcm_req, fft_factor)
        if lcm_req > 1:
            prev_valid = (Jmax // lcm_req) * lcm_req
            next_valid = prev_valid + lcm_req if prev_valid < Jmax else prev_valid
            print(f"  Hint: Jmax must be a multiple of {lcm_req}."
                  f"  Nearest valid values: {prev_valid} (below) or {next_valid} (above).")
    else:
        print("\n  All MPI decomposition checks passed.")
    return n_fail == 0


def print_grid_summary(n_NMAX, n_SCALE, n_Y, dx_ref, u_star, l_in,
                       NX_pro, NZ_pro, mpi_ok):
    """
    Print a single-screen summary of the final grid: shape, domain, MPI layout,
    wall-normal spacing statistics, and a rule-compliance snapshot.
    """
    Imax = int(n_NMAX[0])
    Jmax = int(n_NMAX[1])
    Kmax = int(n_NMAX[2])
    Lx = float(n_SCALE[0])
    Ly = float(n_SCALE[1])   # original tlab.ini Ly (stored as grid scale, ≠ y[-1])
    Lz = float(n_SCALE[2])
    npro_i = Imax // NX_pro
    npro_k = Kmax // NZ_pro
    npro   = npro_i * npro_k

    dy     = np.diff(n_Y)
    dy_min = float(np.min(dy))
    dy_max = float(np.max(dy))

    j_bl        = int(np.clip(np.searchsorted(n_Y, u_star, side='right') - 1, 0, len(dy) - 1))
    max_ratio_bl    = float(np.max(dy[:j_bl + 1] / dx_ref))
    max_ratio_outer = float(np.max(dy[j_bl + 1:] / dx_ref)) if j_bl + 1 < len(dy) else 0.0

    # Quick rule compliance (silent recompute — no print side-effects)
    r_arr  = dy[1:] / dy[:-1]
    zones  = find_grid_zones(r_arr)
    r1_ok  = zones[0]['type'] == 'uniform'
    r2_ok  = len(check_stretch_contraction(zones)) == 0
    r3_ok  = max_ratio_bl <= 1.0
    _j_us  = int(np.clip(np.searchsorted(n_Y, u_star, side='right'), 1, len(n_Y) - 1))
    r5_ok  = (max_ratio_outer > 1.0 and
              (float(np.max(r_arr[_j_us:])) <= 1.03 * (1.0 + 1e-8) if _j_us < len(r_arr) else True))
    r6_ok  = not any(z['r_max'] > 1.03 * (1.0 + 1e-8)
                     for z in zones
                     if z['type'] == 'stretching'
                     and z['node_end'] - z['node_start'] <= 2)

    W   = 66
    bar = '═' * W
    mid = '─' * (W - 4)

    def row(lbl, val, note=''):
        line = f"  {lbl:<28} {val}"
        if note:
            line += f"   {note}"
        print(line)

    print(f"\n{bar}")
    print(f"{'FINAL GRID SUMMARY':^{W}}")
    print(bar)

    print(f"\n  {'Grid shape and domain':}")
    print(f"  {mid}")
    row("Shape  (Imax × Jmax × Kmax)", f"{Imax} × {Jmax} × {Kmax}")
    row("Total cells", f"{Imax * Jmax * Kmax:,}")
    row("Lx  (streamwise)", f"{Lx:.6f}")
    row("Ly  (wall-normal)", f"{Ly:.8f}")
    row("Lz  (spanwise)", f"{Lz:.6f}")

    print(f"\n  {'MPI decomposition':}")
    print(f"  {mid}")
    row("Imax(*) / npro_i", f"{NX_pro} / {npro_i}", "ranks in X")
    row("Kmax(*) / npro_k", f"{NZ_pro} / {npro_k}", "ranks in Z")
    row("Total MPI ranks", f"{npro}")

    print(f"\n  {'Wall-normal (Y) grid statistics':}")
    print(f"  {mid}")
    row("dy_min  (near-wall cell)", f"{dy_min:.4e}", f"= {dy_min / l_in:.2f} δ_ν  (wall units)")
    row("dy_max  (outermost cell)", f"{dy_max:.4e}", f"= {dy_max / l_in:.1f} δ_ν")
    row("dx_ref  (X cell, uniform)", f"{dx_ref:.4e}", f"= {dx_ref / l_in:.2f} δ_ν")
    row("dx / dy_min", f"{dx_ref / dy_min:.2f}", "(≥ 1 → X coarser than near-wall Y)")
    row("dy/dx max  (y < u★)", f"{max_ratio_bl:.4f}", f"u★ = {u_star:.3e}  — Rule 3 boundary")
    row("dy/dx max  (y > u★)", f"{max_ratio_outer:.4f}", "outer anisotropic region — Rule 5")
    row("Number of Y zones", f"{len(zones)}")

    print(f"\n  {'Grid-quality rules':}")
    print(f"  {mid}")
    for label, ok in [
        ("Rule 1  first zone uniform",       r1_ok),
        ("Rule 2  stretch → contraction",    r2_ok),
        ("Rule 3  dy/dx < 1 in BL",          r3_ok),
        ("Rule 4  Ly preserved",             True),
        ("Rule 5  dy/dx > 1 above BL",       r5_ok),
        ("Rule 6  short stretch ≤ 3 %",      r6_ok),
        ("MPI     C1–C8 decomposition",      mpi_ok),
    ]:
        tag = "PASS" if ok else "WARN"
        print(f"  [{tag}] {label}")

    print(f"\n{bar}\n")


###############################################################################
# ═══════════════════════════ USER SETTINGS ═══════════════════════════════════
# All parameters that may need to be changed are collected here.
# Everything below the "── Derived quantities ──" line is computed automatically.
###############################################################################

# ── 1. Simulation physics ─────────────────────────────────────────────────────
Re_D   = 500    # Reynolds number based on Ekman depth D  (Re_D = G·D/ν).
u_star = 0.077  # Friction velocity u★ for Re_D=500 orography case.
hill_hgt = 0.00311
# ── 2. I/O ────────────────────────────────────────────────────────────────────
OLD_GRID = 'grid_1280x576x1280'
NEW_NX   = 1280
NEW_NZ   = 1280

# ── 3. Two-system MPI decomposition targets ───────────────────────────────────
GRID_TARGETS = [
    # (system_name, NX,   NZ,   NX_pro, NZ_pro)
    ('Curta',       1280, 1280,    80,    320),   # 16×4=64 tasks
    ('Hunter',      1272, 1272,   212,    159),   # 6×8=48 tasks
]

# Keep these for backward compatibility
NX_pro   = GRID_TARGETS[0][3]
NZ_pro   = GRID_TARGETS[0][4]
NX_pro_H = GRID_TARGETS[1][3]
NZ_pro_H = GRID_TARGETS[1][4]

# ── 4. Optional manual Y reshape ──────────────────────────────────────────────
RESHAPE_XYZ = [False, True, False]
# False/True/False: X and Z are already at 1280.  Y is reshaped below.

SMOOTH_STRETCH = False
# True  → constant-ratio GP above zone1 (max ratio ≤ MAX_STRETCH_RATIO).
# False → use RECIPE_GRID or IBM zone1 refinement (see below).

MAX_STRETCH_RATIO = 1.05

RECIPE_GRID = True
# True  → physics-based 7-zone recipe (Rule 3 compliant):
#   zone1 unchanged → ≤2% transition → inner decel → BL uniform → outer stretch
#   → outer decel → sponge.  zone1 is NEVER modified.
# False → SMOOTH_STRETCH or IBM branch.

RECIPE_R_TRANS      = 1.02   # transition stretch rate (≤1.02 per cell)
RECIPE_N_INNER_DECEL = 7     # inner decel cells (transition→BL-uniform)
RECIPE_N_OUTER_DECEL = 7     # outer decel cells (outer-stretch→sponge)
RECIPE_N_SPONGE      = 18    # uniform sponge cells at the domain top
RECIPE_ISO_THRESH    = 0.9   # target dy/dx at end of transition zone (≈0.9)

# ── 5. Auto-fix parameters ────────────────────────────────────────────────────
N_CONTRACTION = 5
NY_LIMIT = 650
NY_TARGET = 576
# NY=576 satisfies all MPI C1–C8 for both Curta (NX=1280) and Hunter (NX=1272).

ALLOW_RULE1_MODIFICATION = True

# ── 6. Output ─────────────────────────────────────────────────────────────────
WRITE_TO_FILE = True
# Two grid files are written (one per entry in GRID_TARGETS):
#   grid_1280x<NY>x1280  — Curta
#   grid_1272x<NY>x1272  — Hunter

# ── Derived quantities (do not edit) ─────────────────────────────────────────
# Re_L = Re_D²/2 is the reference Reynolds number (length scale L = Re_D/(2G)).
Re_L      = 0.5 * Re_D**2
nu_L      = 1.0 / Re_L           # non-dimensional kinematic viscosity
BL_y_plus = u_star**2 / nu_L     # BL thickness in wall units (= Re_tau)
l_in      = nu_L / u_star        # viscous length scale δ_ν  (1 wall unit)
l_out     = u_star                # outer length scale (BL thickness, f = 1)

# Resolve script directory — all grid files are read/written relative to it
cwd = os.path.dirname(os.path.abspath(__file__)) + '/'

# Reading old grid
o_NMAX, o_SCALE, o_X, o_Y, o_Z = read_grid(cwd + OLD_GRID)

# Grid diagnostics — cell widths along wall-normal direction
dy = np.diff(o_Y)

# r[i] = dy[i+1]/dy[i]: ratio of consecutive cell widths.
# Uniform zones have r ≈ constant; ramp-up/down transitions show r changing.
r = dy[1:] / dy[:-1]

dx_ref = float(np.mean(np.diff(o_X)))   # representative x cell width for isotropy check

print("\n--- Old grid zone analysis ---")
zones = find_grid_zones(r)
print_zones(zones, o_Y, dy, dx_ref=dx_ref)
run_sanity_checks(o_Y, dy, dx_ref, u_star, l_in)

# Override dx_ref to the finest target grid spacing so that all new-grid
# isotropy checks (Rule 3) and auto-fixes use the actual new cell width.
# Finest dx = Lx / largest NX among GRID_TARGETS.
dx_ref = (float(o_X[-1]) + float(o_X[1])) / max(t[1] for t in GRID_TARGETS)
print(f"\n  [dx_ref] updated to finest target dx = {dx_ref:.4e}  "
      f"({dx_ref/l_in:.2f} y+)  (NX={max(t[1] for t in GRID_TARGETS)})")

# ---------------------------------------------------------------------------
# zone1 and IBM hill refinement check
#
# zone1 = y coordinate at the end of the first uniform zone (the non-stretched
#         near-wall region where cell height = l_in = 1 wall unit).
#
# IBM hill region = y=0 to y=hill_hgt (physical orography crest height).
#
# Decision:
#   dy[0] > l_in → cells coarser than 1 wall unit → refine zone1 to 1 y+ cells
#   dy[0] ≤ l_in → cells already ≤ 1 wall unit   → add 4 safety points
#
# _ibm_zone1_n    = new total node count (0..zone1) after refinement
# _ibm_extra_pts  = extra nodes added relative to old grid
# _trans_node_end = end of the AP transition above zone1 (bridges cell-size jump)
# ---------------------------------------------------------------------------
_zone1_node  = zones[0]['node_end']               # last node of first uniform zone
zone1        = float(o_Y[_zone1_node])            # physical y position of zone1
_dnu_old     = float(dy[0])                       # current near-wall cell width
_hill_node_old = int(np.searchsorted(o_Y, hill_hgt, side='right'))  # old hill pts

print("\n--- IBM hill refinement check ---")
print(f"  hill_hgt                : {hill_hgt:.6e}  ({hill_hgt/l_in:.2f} y+)")
print(f"  zone1 (1st uniform end) : {zone1:.6e}  ({zone1/l_in:.2f} y+)  node {_zone1_node}")
print(f"  Old hill pts ≤ hill_hgt : {_hill_node_old}")
print(f"  Old near-wall Δy        : {_dnu_old:.4e}  ({_dnu_old/l_in:.3f} y+)")
print(f"  1 wall unit  (l_in)     : {l_in:.4e}")

if _dnu_old > l_in:
    # Coarser than 1 y+: refine so every cell in zone1 ≤ 1 wall unit
    _ibm_zone1_n   = int(np.ceil(zone1 / l_in)) + 1   # +1 to include node 0
    _ibm_extra_pts = _ibm_zone1_n - (_zone1_node + 1)
    _ibm_reason    = (f"Δy ({_dnu_old/l_in:.3f} y+) > 1 y+ "
                      f"→ refining zone1 to {_ibm_zone1_n} nodes (≤1 y+ each)")
else:
    # Already ≤ 1 y+: add 4 safety points uniformly across zone1
    _ibm_zone1_n   = _zone1_node + 1 + 4
    _ibm_extra_pts = 4
    _ibm_reason    = f"Δy ({_dnu_old/l_in:.3f} y+) ≤ 1 y+ → adding 4 safety pts"

_dnu_new_zone1  = zone1 / (_ibm_zone1_n - 1) if _ibm_zone1_n > 1 else _dnu_old
_trans_node_end = _ibm_zone1_n - 1 + 12    # AP transition window: 12 nodes above zone1

print(f"  Decision                : {_ibm_reason}")
print(f"  New zone1 nodes         : {_ibm_zone1_n}  (+{_ibm_extra_pts})")
print(f"  New Δy in zone1         : {_dnu_new_zone1:.4e}  ({_dnu_new_zone1/l_in:.3f} y+)")

#------------------------------------------------------------------------------
# New grid — starts as an exact copy of the old grid.
# Modify n_Y incrementally with remesh_uniform(), then write with write_grid().
#------------------------------------------------------------------------------
n_Y    = o_Y.copy()
n_X    = o_X.copy()
n_Z    = o_Z.copy()
n_NMAX  = np.array([NEW_NX, o_NMAX[1], NEW_NZ]) # Update Ny before saving the grid
n_SCALE = o_SCALE.copy()

# Remesh X and Z
# X and Z are PERIODIC directions.  The stored nodes cover 0 … Lx×(N-1)/N;
# the implied repeated node at Lx is NOT stored.  The correct domain length is
#   Lx = o_X[-1] + o_X[1]   (last node + one spacing = full period)
# When changing the node count we must regenerate nodes from scratch using Lx,
# NOT interpolate between existing nodes — that would shift Lx slightly.
if RESHAPE_XYZ[0]:
    _Lx = float(o_X[-1]) + float(o_X[1])     # preserve original domain length
    n_X = np.arange(NEW_NX, dtype=np.float64) * (_Lx / NEW_NX)
    print(f"[Remesh X]  {o_NMAX[0]} → {NEW_NX} nodes,  dx = {_Lx/NEW_NX:.6e},  Lx = {_Lx:.8f}")
if RESHAPE_XYZ[2]:
    _Lz = float(o_Z[-1]) + float(o_Z[1])     # preserve original domain length
    n_Z = np.arange(NEW_NZ, dtype=np.float64) * (_Lz / NEW_NZ)
    print(f"[Remesh Z]  {o_NMAX[2]} → {NEW_NZ} nodes,  dz = {_Lz/NEW_NZ:.6e},  Lz = {_Lz:.8f}")

if RESHAPE_XYZ[1]:
  if RECIPE_GRID:
    # ── Physics-based 7-zone recipe ─────────────────────────────────────────
    # zone1 (nodes 0.._zone1_node) is NEVER modified.
    _dx_recipe = (float(o_X[-1]) + float(o_X[1])) / max(t[1] for t in GRID_TARGETS)
    _Ly_recipe  = float(n_Y[-1])
    print(f"\n[Recipe grid]  zone1_node={_zone1_node}  "
          f"dx={_dx_recipe:.4e}  dy_target={RECIPE_ISO_THRESH*_dx_recipe:.4e}  "
          f"y_BL={u_star:.4f}  Ly={_Ly_recipe:.6f}")
    n_Y, _r_outer = build_recipe_above_zone1(
        n_Y, _zone1_node, _dx_recipe, u_star, _Ly_recipe,
        RECIPE_R_TRANS, RECIPE_N_INNER_DECEL, RECIPE_N_OUTER_DECEL,
        RECIPE_N_SPONGE, RECIPE_ISO_THRESH, NY_TARGET)
    n_dy = np.diff(n_Y)
    n_r  = n_dy[1:] / n_dy[:-1]
    print(f"  r_outer (solved) = {_r_outer:.6f} ({(_r_outer - 1)*100:.3f}%)")
    print_zones(find_grid_zones(n_r), n_Y, n_dy, dx_ref=_dx_recipe)
    _grid_ok = run_sanity_checks(n_Y, n_dy, _dx_recipe, u_star, l_in,
                                  n_contraction=N_CONTRACTION)

  elif SMOOTH_STRETCH:
    # ── Smooth stretch: redistribute all nodes ABOVE zone1 ──────────────────
    # zone1 (nodes 0.._zone1_node) is NEVER touched.
    # All cells above zone1 are redistributed using a constant-ratio GP so that
    # the max cell-to-cell growth rate ≤ MAX_STRETCH_RATIO everywhere.
    # No new nodes are added unless strictly necessary.
    _dy0_above   = float(np.diff(n_Y[:_zone1_node + 1])[-1])  # last zone1 cell width
    _n_above     = len(n_Y) - 1 - _zone1_node                  # cells above zone1
    _span_above  = float(n_Y[-1] - n_Y[_zone1_node])           # physical span to cover

    # Compute required constant ratio r for the existing node count
    from scipy.optimize import brentq as _brentq
    def _span_fn(rr):
        if abs(rr - 1.0) < 1e-12:
            return _dy0_above * _n_above
        return _dy0_above * (rr**_n_above - 1.0) / (rr - 1.0)
    # Upper bound for brentq: use a conservative value (r=5.0 is far more than needed)
    # Avoid r**n_above overflow for large r by capping at a safe value.
    _r_hi = 5.0
    while True:
        try:
            _test = _span_fn(_r_hi)
            if _test > _span_above:
                break
        except (OverflowError, FloatingPointError):
            pass
        _r_hi = max(1.001, (_r_hi - 1.0) * 0.5 + 1.0)  # bisect toward 1
    _r_needed = _brentq(lambda rr: _span_fn(rr) - _span_above, 1.0, _r_hi,
                        xtol=1e-14, rtol=1e-12)
    print(f"\n[Smooth stretch]  nodes {_zone1_node}–{len(n_Y)-1}  |  "
          f"n_cells={_n_above}  |  dy0={_dy0_above:.4e}  |  span={_span_above:.6f}")
    print(f"  Required constant ratio = {_r_needed:.6f} ({(_r_needed-1)*100:.4f}%)  "
          f"— needs extra nodes? {_r_needed > MAX_STRETCH_RATIO}")

    if _r_needed <= MAX_STRETCH_RATIO:
        # Redistribute without adding nodes
        print(f"  Redistributing {_n_above} existing cells (no new nodes needed).")
        n_Y, n_dy, n_r, _ = remesh_stretch_max_ratio(
            n_Y, node_start=_zone1_node, node_end=len(n_Y) - 1,
            max_ratio=MAX_STRETCH_RATIO, dy0=_dy0_above)
    else:
        # Last resort: add the minimum number of cells to satisfy max ratio
        _k = 0
        while _span_fn(MAX_STRETCH_RATIO) < _span_above:
            _k += 1
            _n_above_new = _n_above + _k
            def _span_fn_new(rr):
                if abs(rr - 1.0) < 1e-12:
                    return _dy0_above * _n_above_new
                return _dy0_above * (rr**_n_above_new - 1.0) / (rr - 1.0)
            if _span_fn_new(MAX_STRETCH_RATIO) >= _span_above:
                break
        _n_extra = _k
        print(f"  Adding {_n_extra} extra cell(s) to satisfy max ratio {MAX_STRETCH_RATIO:.2f}.")
        # Extend grid by appending _n_extra uniform cells at the top with the last dy
        _last_dy = float(np.diff(n_Y)[-1])
        for _ in range(_n_extra):
            n_Y = np.append(n_Y, n_Y[-1] + _last_dy)
        n_Y, n_dy, n_r, _ = remesh_stretch_max_ratio(
            n_Y, node_start=_zone1_node, node_end=len(n_Y) - 1,
            max_ratio=MAX_STRETCH_RATIO, dy0=_dy0_above)
    n_zones = find_grid_zones(n_r)
    # Immediate sanity check after smooth stretch
    print("\n--- Grid quality after smooth stretch ---")
    _grid_ok = run_sanity_checks(n_Y, n_dy, dx_ref, u_star, l_in, n_contraction=N_CONTRACTION)

  else:
    # ── IBM zone1 refinement (original behaviour) ────────────────────────────
    print(f"\n[IBM remesh] zone1 refinement: nodes 0–{_zone1_node} → {_ibm_zone1_n} nodes")
    n_Y, n_dy, n_r = remesh_uniform(n_Y, node_start=0, node_end=_zone1_node,
                                    n_div=_ibm_zone1_n)
    n_zones = find_grid_zones(n_r)
    _jz1_new = _ibm_zone1_n - 1
    _te      = min(_trans_node_end, len(n_Y) - 2)
    print(f"[IBM remesh] AP transition: nodes {_jz1_new}–{_te} → 20 nodes")
    n_Y, n_dy, n_r = remesh_geometric(n_Y, node_start=_jz1_new,
                                       node_end=_te, mode='AP', n_div=20)

# Tracks grid-rule pass/fail through the fix pipeline; updated by every
# run_sanity_checks call on n_Y — the last assignment is the final verdict.
# RECIPE_GRID already ran its own sanity check inside the RESHAPE block; don't reset.
if not RECIPE_GRID:
    _grid_ok = False

# Auto-fix: if any cell below y=u_star has dy > dx, apply isotropy correction.
# Skipped in SMOOTH_STRETCH mode: the monotonic geometric-cell-width progression
# grows beyond dx inside the BL by design; adding uniform nodes here would
# undo the smooth stretch and cause NY_TARGET oscillation.
_dy_check = np.diff(n_Y)
_j_ustar  = int(np.clip(np.searchsorted(n_Y, u_star, side='right'), 1, len(n_Y) - 1))
if (not SMOOTH_STRETCH) and (not RECIPE_GRID) and any(_dy_check[j] > dx_ref for j in range(min(_j_ustar, len(_dy_check)))):
    print("\n[Auto-fix] dy > dx detected below y = u_star — applying isotropy correction.")
    n_Y, n_dy, n_r = remesh_fix_isotropy(n_Y, dx_ref, u_star, threshold=0.9)
    print("\n--- Updated zone analysis after isotropy fix ---")
    print_zones(find_grid_zones(n_r), n_Y, n_dy, dx_ref=dx_ref)
    _grid_ok = run_sanity_checks(n_Y, n_dy, dx_ref, u_star, l_in)

# Auto-fix: ensure every significant stretching zone is followed by
# >= N_CONTRACTION contraction nodes.  AP remesh on a fixed [ne, ne+N] window
# leaves all nodes outside that window unchanged (Ly preserved, no infinite loop).
_contraction_fixed = False
_max_iters = 20
for _iter in range(_max_iters):
    _n_dy_tmp  = np.diff(n_Y)
    _n_r_tmp   = _n_dy_tmp[1:] / _n_dy_tmp[:-1]
    _zones_tmp = find_grid_zones(_n_r_tmp)
    _violations = check_stretch_contraction(_zones_tmp, min_points=N_CONTRACTION)
    if not _violations:
        break
    _contraction_fixed = True
    v = _violations[0]
    print(f"\n[Auto-fix iter {_iter+1}] Stretching zone {v['zone_idx']} "
          f"ends at node {v['node_stretch_end']}, r_peak={v['r_peak']:.5f}: "
          f"only {v['n_found']} contraction node(s) follow (need >= {N_CONTRACTION}).")
    n_Y, n_dy, n_r = remesh_contraction_tail(
        n_Y, v['node_stretch_end'], v['r_peak'], N_CONTRACTION)
else:
    print(f"\n[Auto-fix] Reached max iterations ({_max_iters}); check remaining violations manually.")

if _contraction_fixed:
    print("\n--- Zone analysis after contraction fix ---")
    print_zones(find_grid_zones(n_r), n_Y, n_dy, dx_ref=dx_ref)
    _grid_ok = run_sanity_checks(n_Y, n_dy, dx_ref, u_star, l_in, n_contraction=N_CONTRACTION)

# Auto-fix: Rule 6 — smooth out short stretch zones (1–2 cells) with ratio > 10%.
# Skipped in SMOOTH_STRETCH mode: the geometric-cell-width progression has no
# short-stretch zones; the fix would modify cells above zone1 and break smoothness.
_n_dy_r6  = np.diff(n_Y)
_n_r_r6   = _n_dy_r6[1:] / _n_dy_r6[:-1]
_zones_r6 = find_grid_zones(_n_r_r6)
# Zone 1 is always protected from stretching — Rule 6 fix never touches it.
_protect  = _zones_r6[0]['node_end']
_r6_fixed = False

if not SMOOTH_STRETCH and not RECIPE_GRID:
    print(f"\n[Rule 6 auto-fix]  ny_limit={NY_LIMIT}, protect_end={_protect}  (zone1 always protected)")
    n_Y, n_dy, n_r, _r6_fixed = fix_rule6_violations(
        n_Y, NY_LIMIT, max_r=1.03, protect_end=_protect)

if _r6_fixed:
    print("\n--- Final zone analysis after Rule 6 fix ---")
    print_zones(find_grid_zones(n_r), n_Y, n_dy, dx_ref=dx_ref)
    _grid_ok = run_sanity_checks(n_Y, n_dy, dx_ref, u_star, l_in, n_contraction=N_CONTRACTION)
elif not any(
    z['r_max'] > 1.03
    for z in _zones_r6
    if z['type'] == 'stretching' and z['node_end'] - z['node_start'] <= 2
       and (_protect is None or z['node_start'] > _protect)
):
    print("  [Rule 6] No fixable violations found.")

# Fit to exact target node count (NY_TARGET > 0).
# Nodes are taken from / added to the largest non-protected uniform zone so that
# both endpoints of that zone are preserved → Ly and Rule 4 remain satisfied.
if NY_TARGET > 0 and n_Y.size != NY_TARGET:
    _excess    = n_Y.size - NY_TARGET       # > 0 remove; < 0 add
    _dy_fit    = np.diff(n_Y)
    _r_fit     = _dy_fit[1:] / _dy_fit[:-1]
    _zones_fit = find_grid_zones(_r_fit)
    _prot_end  = _zones_fit[0]['node_end']  # Rule 1 boundary (end of first uniform zone)
    # Also protect the BL region (below u_star) so Rule 3 survives the adjustment
    _j_ustar_prot = int(np.searchsorted(n_Y, u_star, side='right'))

    _candidates = [z for z in _zones_fit
                   if z['type'] == 'uniform'
                   and z['node_start'] >= max(_prot_end, _j_ustar_prot)]
    if not _candidates:
        print(f"[NY_TARGET] No suitable uniform zone found; keeping {n_Y.size} nodes.")
    else:
        _zone  = max(_candidates, key=lambda z: z['node_end'] - z['node_start'])
        _old_n = _zone['node_end'] - _zone['node_start'] + 1
        _new_n = _old_n - _excess
        if _new_n < 3:
            print(f"[NY_TARGET] Cannot adjust zone {_zone['node_start']}–{_zone['node_end']}: "
                  f"would leave only {_new_n} nodes.")
        else:
            print(f"\n[NY_TARGET={NY_TARGET}]  "
                  f"{'Removing' if _excess > 0 else 'Adding'} {abs(_excess)} node(s) — "
                  f"uniform zone {_zone['node_start']}–{_zone['node_end']} "
                  f"({_old_n} → {_new_n} nodes).")
            n_Y, n_dy, n_r = remesh_uniform(
                n_Y, _zone['node_start'], _zone['node_end'], _new_n)
            print(f"  New Jmax = {n_Y.size}  |  Ly = {n_Y[-1]:.8f} (preserved)")
            print("\n--- Zone analysis after NY_TARGET fit ---")
            print_zones(find_grid_zones(n_r), n_Y, n_dy, dx_ref=dx_ref)
            _grid_ok = run_sanity_checks(n_Y, n_dy, dx_ref, u_star, l_in, n_contraction=N_CONTRACTION)

            # NY_TARGET may introduce new Rule 6 violations at zone boundaries.
            # Run a final Rule 6 pass to clean them up.
            _n_dy_post = np.diff(n_Y)
            _n_r_post  = _n_dy_post[1:] / _n_dy_post[:-1]
            _viols_post = [z for z in find_grid_zones(_n_r_post)
                           if z['type'] == 'stretching'
                           and z['node_end'] - z['node_start'] <= 2
                           and z['r_max'] > 1.03 * (1 + 1e-8)]
            if _viols_post:
                print(f"\n[Post-NY_TARGET Rule 6 fix]  {len(_viols_post)} violation(s) to clean up.")
                n_Y, n_dy, n_r, _ = fix_rule6_violations(n_Y, NY_LIMIT + 20, max_r=1.03)
                print("\n--- Zone analysis after post-NY_TARGET Rule 6 fix ---")
                print_zones(find_grid_zones(n_r), n_Y, n_dy, dx_ref=dx_ref)
                _grid_ok = run_sanity_checks(n_Y, n_dy, dx_ref, u_star, l_in, n_contraction=N_CONTRACTION)

                # Rule 6 fix may have added nodes; re-fit to NY_TARGET so MPI stays valid.
                if NY_TARGET > 0 and n_Y.size != NY_TARGET:
                    _ex2 = n_Y.size - NY_TARGET
                    _df2 = np.diff(n_Y)
                    _rf2 = _df2[1:] / _df2[:-1]
                    _zf2 = find_grid_zones(_rf2)
                    _pe2 = _zf2[0]['node_end']
                    _ju2 = int(np.searchsorted(n_Y, u_star, side='right'))
                    _c2  = [z for z in _zf2
                            if z['type'] == 'uniform'
                            and z['node_start'] >= max(_pe2, _ju2)]
                    if _c2:
                        _z2   = max(_c2, key=lambda z: z['node_end'] - z['node_start'])
                        _on2  = _z2['node_end'] - _z2['node_start'] + 1
                        _nn2  = _on2 - _ex2
                        if _nn2 >= 3:
                            print(f"\n[Re-fit NY_TARGET={NY_TARGET}]  "
                                  f"{'Removing' if _ex2>0 else 'Adding'} {abs(_ex2)} node(s) "
                                  f"from zone {_z2['node_start']}–{_z2['node_end']} "
                                  f"({_on2}→{_nn2})")
                            n_Y, n_dy, n_r = remesh_uniform(
                                n_Y, _z2['node_start'], _z2['node_end'], _nn2)
                            print(f"  New Jmax = {n_Y.size}  |  Ly = {n_Y[-1]:.8f}")
                            _grid_ok = run_sanity_checks(n_Y, n_dy, dx_ref, u_star, l_in,
                                                         n_contraction=N_CONTRACTION)

# Update NY
# After all modifications, update n_NMAX[1] = len(n_Y):
n_NMAX[1] = n_Y.size

# ── IBM hill and zone1 diagnostics (new Y grid, same for both systems) ───────
_new_hill_pts   = int(np.sum(n_Y <= hill_hgt))
_new_dy_near    = np.diff(n_Y)
_new_r_near     = _new_dy_near[1:] / _new_dy_near[:-1]
_new_zones      = find_grid_zones(_new_r_near)
_new_zone1_node = _new_zones[0]['node_end']
_new_zone1      = float(n_Y[_new_zone1_node])
_Jmax           = int(n_Y.size)

print("\n--- IBM hill and zone1 diagnostics (new Y grid) ---")
print(f"  hill_hgt ({hill_hgt:.4e}) in new grid : {_new_hill_pts} pts  "
      f"(prev: {_hill_node_old} pts,  +{_new_hill_pts - _hill_node_old})")
print(f"  hill_hgt in wall units              : ≈ {hill_hgt/l_in:.2f} y+")
print(f"  zone1 new grid                      : node {_new_zone1_node}  "
      f"y = {_new_zone1:.4e}  ({_new_zone1/l_in:.2f} y+)")
print(f"  New near-wall Δy                    : {_new_dy_near[0]:.4e}  "
      f"({_new_dy_near[0]/l_in:.3f} y+)")

# ── Per-system MPI check, summary, and optional write ────────────────────────
# The Y grid is identical for both systems.  X and Z are regenerated
# from the original domain length (Lx/Lz) for each system's NX/NZ.
_Lx_src = float(o_X[-1]) + float(o_X[1])   # domain length for node generation
_Lz_src = float(o_Z[-1]) + float(o_Z[1])
# Scales written to grid file: preserve EXACT binary values from the original grid.
# This avoids a ULP-level mismatch (e.g., Lx/1272 × 1272 ≠ Lx in IEEE 754)
# that causes tlab's transfields.x Error 103 ("Ox scales are not equal at the end").
_x_scale_write = float(o_SCALE[0])   # original tlab.ini Lx (NOT recomputed from nodes)
_y_scale        = float(o_SCALE[1])  # original tlab.ini Ly (NOT y[-1])
_z_scale_write  = float(o_SCALE[2])  # original tlab.ini Lz
_W = 66

all_ready = True
for _sys_name, _nx, _nz, _nx_pro, _nz_pro in GRID_TARGETS:
    _sys_X  = np.arange(_nx, dtype=np.float64) * (_Lx_src / _nx)
    _sys_Z  = np.arange(_nz, dtype=np.float64) * (_Lz_src / _nz)
    _sys_dx = _Lx_src / _nx

    print(f"\n{'═'*_W}")
    print(f"  {_sys_name} grid :  NX={_nx}  NY={_Jmax}  NZ={_nz}")
    print(f"  dx = {_sys_dx:.4e}  ({_sys_dx/l_in:.2f} y+)  "
          f"MPI ranks = ({_nx//_nx_pro}) × ({_nz//_nz_pro}) = {(_nx//_nx_pro)*(_nz//_nz_pro)}")

    _mpi = check_parallel_decomposition(
        Imax=_nx, Kmax=_nz, Jmax=_Jmax,
        Imax_star=_nx_pro, Kmax_star=_nz_pro,
        fourier=True, label=_sys_name)

    print_grid_summary(
        np.array([_nx, _Jmax, _nz]),
        n_SCALE, n_Y, _sys_dx, u_star, l_in,
        _nx_pro, _nz_pro, mpi_ok=_mpi)

    _ready = _grid_ok and _mpi
    all_ready = all_ready and _ready
    print('═' * _W)
    if _ready:
        print(f"{'SIMULATION READY — all grid and MPI checks passed.':^{_W}}")
    else:
        _issues = []
        if not _grid_ok:
            _issues.append('grid quality (Rules 1–6)')
        if not _mpi:
            _issues.append(f'MPI (NX_pro={_nx_pro}, NZ_pro={_nz_pro})')
        print(f"{'NOT READY — fix the following:':^{_W}}")
        for _iss in _issues:
            print(f"{'  •  ' + _iss:^{_W}}")
    print('═' * _W)

    if WRITE_TO_FILE:
        _fname = f"grid_{_nx}x{_Jmax}x{_nz}"
        write_grid(cwd + _fname, _nx, _Jmax, _nz,
                   _x_scale_write, _y_scale, _z_scale_write,
                   _sys_X, n_Y, _sys_Z)
        print(f"\n  Wrote → {_fname}  "
              f"(Lx={_x_scale_write:.20f}  Ly={_y_scale:.8f}  Lz={_z_scale_write:.20f})")











