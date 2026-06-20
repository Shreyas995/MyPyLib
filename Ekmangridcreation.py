#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ekman layer DNS grid generation with vertical stretching and iterative solver.

Workflow
--------
1. Optionally read an existing 'grid' file (USE_EXISTING_GRID_SCALE = True)
   to lock the horizontal domain scales (Lx, Lz) and the domain-height ceiling.

2. Resolve horizontal isotropy:
   • New grid : iso_factor (user-set, 2.5–4.0) → dx = iso_factor × wall_units
   • Old grid : iso_factor is fixed by old_Lx / nx / wall_units (no user choice).

3. Build the vertical (y) grid through Phases 1–7 via build_grid_y().
   Phase 1 – Uniform near_wall_dnu wall-units through the hill crest + buffer.
   Phase 2 – Quadratic ramp-up to stretch_per % (stretch_pts nodes).
   Phase 3 – Constant stretch_per % until deltaz >= iso_factor × wall_units.
   Phase 4 – Same stretch_per % until BL height (BL_y_plus wall-units).
   Phase 5 – Quadratic ramp-up to post_BL_stretch_per % (post_BL_stretch_pts nodes).
   Phase 6 – Constant post_BL_stretch_per % until free-stream (FS_y_plus wall-units).
   Phase 7 – Quadratic ramp-down (contraction, post_BL_stretch_pts nodes).

4. Iterative solver (ENABLE_SOLVER = True):
   Sweeps (stretch_per, post_BL_stretch_per) and n_hill_buffer_pts within
   user-specified ranges to find a solution where:
     • ny_final is divisible by block_size (96 for Hunter, 64 otherwise)
     • grid_y[-1] ≤ max_Ly  (old domain ceiling)
     • grid_y[-1] ≥ min_Ly  (minimum domain height)
   Parameters are swept from nominal values outward for fastest convergence.

5. Generate uniform x/z arrays, write 'grid_new', save diagnostic plots.

Edit only the USER PARAMETERS section.
"""

import numpy as np
import os
import math
from math import gcd
import matplotlib.pyplot as plt
from scipy.io import FortranFile


# ===========================================================================
# I/O helpers
# ===========================================================================

def read_grid(path, int_dtype=np.int32, float_dtype=np.float64):
    """
    Read a Fortran-format DNS grid file (five Fortran records).

    Returns
    -------
    nodes  : (nx, ny, nz)
    scales : (x_scale, y_scale, z_scale)  – physical domain extents
    coords : (x_nodes, y_nodes, z_nodes)  – node position arrays
    """
    with FortranFile(path, mode='r') as f:
        nx, ny, nz                = f.read_ints(dtype=int_dtype)
        x_scale, y_scale, z_scale = f.read_reals(dtype=float_dtype)
        x_nodes                   = f.read_reals(dtype=float_dtype)
        y_nodes                   = f.read_reals(dtype=float_dtype)
        z_nodes                   = f.read_reals(dtype=float_dtype)
    return (nx, ny, nz), (x_scale, y_scale, z_scale), (x_nodes, y_nodes, z_nodes)


def write_grid(path, nx, ny, nz, x_scale, y_scale, z_scale,
               x_nodes, y_nodes, z_nodes,
               int_dtype=np.int32, float_dtype=np.float64):
    """
    Write a Fortran-format DNS grid file (same layout as read_grid).

    Scales for periodic directions:  x_scale = x_nodes[-1] + x_nodes[1]  (= Lx)
    Scale for wall-normal direction:  y_scale = y_nodes[-1]               (= Ly)
    """
    with FortranFile(path, mode='w') as f:
        f.write_record(np.array([nx, ny, nz], dtype=int_dtype))
        f.write_record(np.array([x_scale, y_scale, z_scale], dtype=float_dtype))
        f.write_record(x_nodes[:nx].astype(float_dtype))
        f.write_record(y_nodes[:ny].astype(float_dtype))
        f.write_record(z_nodes[:nz].astype(float_dtype))


# ===========================================================================
# Grid stretching / contraction helpers
# ===========================================================================

def stretchingfn(current_node, str_nodes, str_percent, old_deltaz, z_last):
    """
    Quadratic ramp-up: stretching % grows from ~0 to str_percent over str_nodes cells.
    Returns (deltaz, stretching, grid) for the new nodes.
    """
    deltaz    = np.empty(str_nodes)
    grid      = np.empty(str_nodes)
    node_idx  = np.linspace(current_node + 1, current_node + str_nodes, str_nodes)
    stretching = str_percent * ((node_idx - current_node) / str_nodes) ** 2
    for k in range(str_nodes):
        if k == 0:
            deltaz[k] = old_deltaz * (1 + stretching[k] / 100)
            grid[k]   = deltaz[k] + z_last
        else:
            deltaz[k] = deltaz[k - 1] * (1 + stretching[k] / 100)
            grid[k]   = deltaz[k] + grid[k - 1]
    return deltaz, stretching, grid


def contractionfn(current_node, str_nodes, str_percent, old_deltaz, z_last):
    """
    Quadratic ramp-down: stretching % decreases from str_percent to ~0 over str_nodes cells.
    Returns (deltaz, stretching, grid) for the new nodes.
    """
    deltaz    = np.empty(str_nodes)
    grid      = np.empty(str_nodes)
    node_idx  = np.linspace(current_node + 1, current_node + str_nodes, str_nodes)
    stretching = str_percent * \
        ((node_idx - (current_node + str_nodes)) / str_nodes) ** 2
    for k in range(str_nodes):
        if k == 0:
            deltaz[k] = old_deltaz * (1 + stretching[k] / 100)
            grid[k]   = deltaz[k] + z_last
        else:
            deltaz[k] = deltaz[k - 1] * (1 + stretching[k] / 100)
            grid[k]   = deltaz[k] + grid[k - 1]
    return deltaz, stretching, grid


# ===========================================================================
# Post-creation sanity checker
# ===========================================================================

def sanity_check(grid_y, phase_ends,
                 dnu, valley_height, n_hill_buffer_pts,
                 wall_units, BL_y_plus, FS_y_plus,
                 min_Ly, max_Ly, block_size, nx, nz,
                 stretch_pts, post_BL_stretch_pts,
                 use_existing_scale=False,
                 old_Lx=None, old_Lz=None,
                 x_scale_new=None, z_scale_new=None,
                 ims_npro=0, ibm_refine_pts=None):
    """
    Verify all hardbound non-negotiable grid conditions.

    Returns a list of failure strings; empty list means every condition is met.

    Hardbound conditions
    --------------------
    C1  Phase-1 uniform : when ibm_refine_pts is None: Δy = dnu from wall to zone1
                          (= valley_height + n_hill_buffer_pts*dnu).
                          when ibm_refine_pts is set: Phase-0 (y=0..valley_height)
                          uniform at dnu_ibm; Phase-1b (valley_height..zone1)
                          uniform at dnu.
    C2  Divisibility    : ny, nx, nz each divisible by block_size
    C3  Scale match     : new Lx, Lz == old scales to single-precision (< 1e-6 relative)
    C4  Buffer zones    : Phase-2 ramp = stretch_pts nodes exactly;
                          Phase-5 ramp = Phase-7 contraction = post_BL_stretch_pts each
    C5  BL reached      : grid at Phase-4 end >= BL_y_plus * wall_units
    C6  FS reached      : grid at Phase-6 end >= FS_y_plus * wall_units
    C7  Domain height   : min_Ly <= y_max <= max_Ly
    C8  Monotonicity    : all cell widths strictly > 0
    C9  Near-wall cell  : first Δy == dnu (or dnu_ibm when IBM refinement active)
    C10 MPI transpose   : (nx × ny) and (nz × ny) each divisible by ims_npro
                          (ims_npro = ims_npro_i × ims_npro_k from tlab.ini; 0 = skip)
    """
    fails = []
    ny  = len(grid_y)
    dy  = np.diff(grid_y)       # cell widths derived directly from node positions

    ph0 = phase_ends.get('ph0')
    ph1 = phase_ends.get('ph1')
    ph2 = phase_ends.get('ph2')
    ph4 = phase_ends.get('ph4')
    ph5 = phase_ends.get('ph5')
    ph6 = phase_ends.get('ph6')
    ph7 = phase_ends.get('ph7')

    # Compute dnu_ibm when IBM refinement is active (Phase-0 cell size inside body)
    if ibm_refine_pts is not None and valley_height > 0:
        hill_pts_base = int(math.ceil(valley_height / dnu))
        total_ibm_pts = hill_pts_base + int(ibm_refine_pts)
        dnu_ibm       = valley_height / total_ibm_pts
    else:
        dnu_ibm = None

    # ------------------------------------------------------------------
    # C8 – Monotonicity (check first; other checks assume positive dy)
    # ------------------------------------------------------------------
    bad_mono = np.where(dy <= 0)[0]
    if bad_mono.size:
        fails.append(
            f"C8 FAIL: non-positive cell width at node(s) {bad_mono[:5].tolist()} "
            f"(min Δy = {dy.min():.4e})")

    # ------------------------------------------------------------------
    # C9 – Near-wall first cell  (dnu, or dnu_ibm when IBM refinement active)
    # ------------------------------------------------------------------
    expected_first_cell = dnu_ibm if dnu_ibm is not None else dnu
    if abs(dy[0] - expected_first_cell) / expected_first_cell > 1e-9:
        fails.append(
            f"C9 FAIL: first cell Δy = {dy[0]:.6e}, expected {expected_first_cell:.6e} "
            f"({'dnu_ibm' if dnu_ibm is not None else 'dnu'}) "
            f"(rel error {abs(dy[0]-expected_first_cell)/expected_first_cell:.2e})")

    # ------------------------------------------------------------------
    # C1 – Phase-1 uniform zone (single zone, or two-part when IBM active)
    # ------------------------------------------------------------------
    if ph1 is None:
        fails.append("C1 FAIL: Phase-1 end index not recorded (builder error)")
    else:
        uniform_target = valley_height + n_hill_buffer_pts * dnu

        if dnu_ibm is None:
            # Original: all cells 0..ph1 must be uniform at dnu
            phase1_dy    = dy[:ph1]
            tol_uniform  = 1e-8
            bad_uniform  = np.where(np.abs(phase1_dy - dnu) / dnu > tol_uniform)[0]
            if bad_uniform.size:
                fails.append(
                    f"C1 FAIL: {bad_uniform.size} cells in Phase-1 zone have Δy ≠ dnu "
                    f"(first at node {bad_uniform[0]})")
        else:
            # Two-part: Phase-0 cells at dnu_ibm, Phase-1b cells at dnu
            if ph0 is None:
                fails.append("C1 FAIL: ph0 not recorded but ibm_refine_pts is set")
            else:
                ph0_dy  = dy[:ph0]
                bad_ibm = np.where(np.abs(ph0_dy - dnu_ibm) / dnu_ibm > 1e-8)[0]
                if bad_ibm.size:
                    fails.append(
                        f"C1 FAIL: {bad_ibm.size} cells in Phase-0 (IBM body) have "
                        f"Δy ≠ dnu_ibm={dnu_ibm:.4e}  (first at node {bad_ibm[0]})")
                ph1b_dy = dy[ph0:ph1]
                bad_buf = np.where(np.abs(ph1b_dy - dnu) / dnu > 1e-8)[0]
                if bad_buf.size:
                    fails.append(
                        f"C1 FAIL: {bad_buf.size} cells in Phase-1b (buffer) have "
                        f"Δy ≠ dnu={dnu:.4e}  (first at node {ph0 + bad_buf[0]})")

        if grid_y[ph1] < uniform_target - 1e-12:
            fails.append(
                f"C1 FAIL: uniform zone ends at y = {grid_y[ph1]:.6e}, "
                f"but must reach >= {uniform_target:.6e} "
                f"(valley_height={valley_height:.4e} + {n_hill_buffer_pts} buf × dnu={dnu:.4e})")

    # ------------------------------------------------------------------
    # C2 – Block-size divisibility
    # ------------------------------------------------------------------
    if ny % block_size != 0:
        fails.append(
            f"C2 FAIL: ny = {ny} not divisible by {block_size} "
            f"(remainder {ny % block_size})")
    if nx % block_size != 0:
        fails.append(f"C2 FAIL: nx = {nx} not divisible by {block_size}")
    if nz % block_size != 0:
        fails.append(f"C2 FAIL: nz = {nz} not divisible by {block_size}")

    # ------------------------------------------------------------------
    # C3 – Scale matching to single precision
    # ------------------------------------------------------------------
    if use_existing_scale and old_Lx is not None:
        for label, new_val, old_val in (('Lx', x_scale_new, old_Lx),
                                         ('Lz', z_scale_new, old_Lz)):
            if new_val is None or old_val is None:
                continue
            rel_err = abs(new_val - old_val) / old_val
            if rel_err > 1e-6:
                fails.append(
                    f"C3 FAIL: {label} mismatch — new={new_val:.10f}, "
                    f"old={old_val:.10f} (rel err {rel_err:.2e} > 1e-6)")

    # ------------------------------------------------------------------
    # C4 – Buffer-zone node counts (exact)
    # ------------------------------------------------------------------
    if ph1 is None or ph2 is None:
        fails.append("C4 FAIL: Phase-2 boundary indices missing")
    else:
        n_ph2 = ph2 - ph1
        if n_ph2 != stretch_pts:
            fails.append(
                f"C4 FAIL: Phase-2 ramp has {n_ph2} nodes, expected {stretch_pts}")

    if ph4 is None or ph5 is None:
        fails.append("C4 FAIL: Phase-5 boundary indices missing")
    else:
        n_ph5 = ph5 - ph4
        if n_ph5 != post_BL_stretch_pts:
            fails.append(
                f"C4 FAIL: Phase-5 ramp has {n_ph5} nodes, expected {post_BL_stretch_pts}")

    if ph6 is None or ph7 is None:
        fails.append("C4 FAIL: Phase-7 boundary indices missing")
    else:
        n_ph7 = ph7 - ph6
        if n_ph7 != post_BL_stretch_pts:
            fails.append(
                f"C4 FAIL: Phase-7 contraction has {n_ph7} nodes, "
                f"expected {post_BL_stretch_pts}")

    # ------------------------------------------------------------------
    # C5 – BL height reached before Phase 5
    # ------------------------------------------------------------------
    BL_phys = BL_y_plus * wall_units
    if ph4 is None:
        fails.append("C5 FAIL: Phase-4 end not recorded")
    elif grid_y[ph4] < BL_phys:
        fails.append(
            f"C5 FAIL: grid at Phase-4 end = {grid_y[ph4]:.6e} "
            f"< BL height {BL_phys:.6e} ({BL_y_plus:.1f} y+)")

    # ------------------------------------------------------------------
    # C6 – FS height reached before Phase 7
    # ------------------------------------------------------------------
    FS_phys = FS_y_plus * wall_units
    if ph6 is None:
        fails.append("C6 FAIL: Phase-6 end not recorded")
    elif grid_y[ph6] < FS_phys:
        fails.append(
            f"C6 FAIL: grid at Phase-6 end = {grid_y[ph6]:.6e} "
            f"< FS height {FS_phys:.6e} ({FS_y_plus:.1f} y+)")

    # ------------------------------------------------------------------
    # C7 – Domain height bounds
    # ------------------------------------------------------------------
    if grid_y[-1] < min_Ly:
        fails.append(
            f"C7 FAIL: y_max = {grid_y[-1]:.6f} < min_Ly = {min_Ly:.6f}")
    if grid_y[-1] > max_Ly:
        fails.append(
            f"C7 FAIL: y_max = {grid_y[-1]:.6f} > max_Ly = {max_Ly:.6f}")

    # ------------------------------------------------------------------
    # C4 (cont.) – Phase-8 top-padding cells must be uniform
    # ------------------------------------------------------------------
    ph8 = phase_ends.get('ph8')
    ph7 = phase_ends.get('ph7')
    if ph8 is not None and ph7 is not None and ph8 > ph7:
        ph8_dy = dy[ph7:ph8]
        if ph8_dy.size > 0 and not np.allclose(ph8_dy, ph8_dy[0], rtol=1e-9):
            fails.append(
                f"C4 FAIL: Phase-8 top-padding is not uniform "
                f"(max deviation {np.max(np.abs(ph8_dy - ph8_dy[0])):.2e})")

    # ------------------------------------------------------------------
    # C10 – MPI transpose divisibility
    # TLabMPI_Trp_PlanI checks (kmax_local × ny) % ims_npro_i == 0
    # TLabMPI_Trp_PlanK checks (imax_local × ny) % ims_npro_k == 0
    # Both reduce to: (nx × ny) % ims_npro_total == 0  (and same for nz)
    # where ims_npro_total = ims_npro_i × ims_npro_k (from tlab.ini).
    # ------------------------------------------------------------------
    if ims_npro > 0:
        for label, dim in (('nx', nx), ('nz', nz)):
            product = dim * ny
            if product % ims_npro != 0:
                g = math.gcd(dim, ims_npro)
                required = ims_npro // g
                fails.append(
                    f"C10 FAIL: ({label} × ny) = {dim} × {ny} = {product} "
                    f"not divisible by ims_npro = {ims_npro} "
                    f"(remainder {product % ims_npro}). "
                    f"ny must be divisible by {required} "
                    f"(= ims_npro / gcd({label}, ims_npro) = {ims_npro}/{g}).")

    return fails


# ===========================================================================
# Core vertical-grid builder  (Phases 1–7, no top padding)
# ===========================================================================

def build_grid_y(dnu, valley_height, n_hill_buffer_pts,
                 stretch_per, stretch_pts, deltax, wall_units,
                 BL_y_plus, post_BL_stretch_per, post_BL_stretch_pts, FS_y_plus,
                 ibm_refine_pts=None,
                 verbose=False):
    """
    Build the vertical grid through Phases 0/1–7 (wall → contraction complete).

    Parameters
    ----------
    dnu                 : Phase-1 uniform cell width (physical); Phase 2+ start size
    valley_height       : physical hill crest height
    n_hill_buffer_pts   : extra uniform cells above the crest before stretching
    stretch_per         : first stretching rate [%]  (Phases 2–4)
    stretch_pts         : ramp-up nodes for Phase 2 (hardbound: must be exact)
    deltax              : isotropic cell-size target (Phase 3 end criterion)
    wall_units          : viscous length scale (converts y+ thresholds to physical)
    BL_y_plus           : BL height [wall-units] (Phase 4 end criterion)
    post_BL_stretch_per : post-BL stretching rate [%]  (Phases 5–6)
    post_BL_stretch_pts : nodes for Phase-5 ramp-up and Phase-7 contraction (hardbound)
    FS_y_plus           : free-stream height [wall-units] (Phase 6 end criterion)
    ibm_refine_pts      : int or None.  When not None, Phase 1 is split into:
                          Phase 0 – IBM hill body y=0..valley_height with
                            (hill_pts_base + ibm_refine_pts) cells at
                            dnu_ibm = valley_height / (hill_pts_base + ibm_refine_pts).
                          Phase 1b – buffer valley_height..zone1 with
                            n_hill_buffer_pts cells at dnu.
                          Phase 2+ starts from dnu exactly as before.
    verbose             : print phase completion messages if True

    Returns
    -------
    grid_y     : ndarray – physical y-positions (Phases 0/1–7)
    deltaz     : ndarray – cell widths (same length as grid_y)
    stretching : ndarray – local stretching % at each node
    phase_ends : dict    – node index at the END of each phase
                          {'ph0':…,   # IBM body end (None when ibm_refine_pts is None)
                           'ph1':…, 'ph2':…, 'ph3':…, 'ph4':…,
                           'ph5':…, 'ph6':…, 'ph7':…}
                          Used by sanity_check() to verify buffer zones.
    """
    # Phase flags
    stretching1        = False
    isotropic_grid     = False
    BL_height          = False
    post_BL_stretch    = False
    max_vertical_resol = False
    contraction        = False

    # Phase-boundary indices recorded as each phase completes
    phase_ends = {k: None for k in ('ph0', 'ph1', 'ph2', 'ph3', 'ph4', 'ph5', 'ph6', 'ph7')}

    # Phase 0 – IBM hill body refinement (optional)
    # Built analytically (np.linspace) to avoid floating-point accumulation.
    # Phase 2+ still starts from dnu: the buffer (Phase 1b) bridges dnu_ibm → dnu.
    if ibm_refine_pts is not None and valley_height > 0:
        hill_pts_base = int(math.ceil(valley_height / dnu))
        total_ibm_pts = hill_pts_base + int(ibm_refine_pts)
        dnu_ibm       = valley_height / total_ibm_pts
        ph0_nodes     = np.linspace(0.0, valley_height, total_ibm_pts + 1)
        grid_y        = ph0_nodes.copy()
        i             = total_ibm_pts
        phase_ends['ph0'] = i
        if verbose:
            print(f"    Ph0 (IBM body) end: node={i:4d}  y={grid_y[i]:.4e}  "
                  f"dnu_ibm={dnu_ibm:.4e}  ({dnu_ibm/wall_units:.3f} y+)  "
                  f"cells={total_ibm_pts} ({hill_pts_base}+{ibm_refine_pts})")
    else:
        dnu_ibm       = None
        total_ibm_pts = None
        # Original initialisation: wall + two nodes at uniform dnu spacing
        grid_y = np.array([0.0, dnu, 2.0 * dnu])
        i      = 2

    deltaz     = np.array([])
    stretching = np.array([])

    while True:

        # Phase 1 / Phase 1b – Uniform dnu from (valley_height or wall) through
        # hill crest + n_hill_buffer_pts cells.
        # HARDBOUND: no stretching permitted below valley_height + buf*dnu
        if grid_y[i] < (valley_height + n_hill_buffer_pts * dnu):
            grid_y = np.append(grid_y, grid_y[i] + dnu)
            i += 1

        elif grid_y[i] >= valley_height:

            # Phase 2 – Quadratic ramp-up (once, exactly stretch_pts nodes — hardbound)
            if not stretching1:
                phase_ends['ph1'] = i
                # deltaz: Phase-0 cells at dnu_ibm (if active), Phase-1b cells at dnu
                if dnu_ibm is not None and phase_ends['ph0'] is not None:
                    ph0_end = phase_ends['ph0']
                    deltaz  = np.empty(i + 1)
                    deltaz[:ph0_end + 1] = dnu_ibm
                    deltaz[ph0_end + 1:] = dnu
                else:
                    deltaz = dnu * np.ones(i + 1)
                stretching = np.zeros(i + 1)
                tmp_dz, tmp_str, tmp_y = stretchingfn(
                    i, stretch_pts, stretch_per, dnu, grid_y[i])
                deltaz     = np.append(deltaz, tmp_dz)
                stretching = np.append(stretching, tmp_str)
                grid_y     = np.append(grid_y, tmp_y)
                i         += stretch_pts
                phase_ends['ph2'] = i
                stretching1 = True
                if verbose:
                    print(f"    Ph2 node={i:4d}  y={grid_y[i]:.4e}  "
                          f"dz={deltaz[i]:.4e}  ({deltaz[i]/wall_units:.2f} y+)")

            # Phase 3 – Constant stretch_per % until isotropic cell size reached
            elif not isotropic_grid:
                new_dz = deltaz[i] * (1 + stretch_per / 100)
                deltaz     = np.append(deltaz, new_dz)
                stretching = np.append(stretching, stretch_per)
                grid_y     = np.append(grid_y, new_dz + grid_y[i])
                i += 1
                if new_dz >= deltax:
                    isotropic_grid = True
                    phase_ends['ph3'] = i
                    if verbose:
                        print(f"    Ph3 node={i:4d}  y={grid_y[i]:.4e}  "
                              f"dz={deltaz[i]:.4e}  ({deltaz[i]/wall_units:.2f} y+)")

            # Phase 4 – Same stretch_per % until BL height is reached (hardbound C5)
            # BUG FIX: was `<= BL_y_plus * wall_units` which terminated after 1 node.
            # Correct condition: continue until grid_y[i] >= BL height.
            elif not BL_height:
                new_dz = deltaz[i] * (1 + stretch_per / 100)
                deltaz     = np.append(deltaz, new_dz)
                stretching = np.append(stretching, 0)   # Phase-4 marker
                grid_y     = np.append(grid_y, new_dz + grid_y[i])
                i += 1
                if grid_y[i] >= BL_y_plus * wall_units:
                    BL_height = True
                    phase_ends['ph4'] = i
                    if verbose:
                        print(f"    Ph4 node={i:4d}  y+={grid_y[i]/wall_units:.1f}"
                              f"  (BL {BL_y_plus:.1f} y+)")

            # Phase 5 – Post-BL ramp-up (once, exactly post_BL_stretch_pts nodes — hardbound)
            elif not post_BL_stretch:
                tmp_dz, tmp_str, tmp_y = stretchingfn(
                    i, post_BL_stretch_pts, post_BL_stretch_per,
                    deltaz[i], grid_y[i])
                deltaz     = np.append(deltaz, tmp_dz)
                stretching = np.append(stretching, tmp_str)
                grid_y     = np.append(grid_y, tmp_y)
                i         += post_BL_stretch_pts
                phase_ends['ph5'] = i
                post_BL_stretch = True
                if verbose:
                    print(f"    Ph5 node={i:4d}  y={grid_y[i]:.4e}")

            # Phase 6 – Constant post_BL_stretch_per % until free-stream height (hardbound C6)
            elif not max_vertical_resol:
                new_dz = deltaz[i] * (1 + post_BL_stretch_per / 100)
                deltaz     = np.append(deltaz, new_dz)
                stretching = np.append(stretching, post_BL_stretch_per)
                grid_y     = np.append(grid_y, new_dz + grid_y[i])
                i += 1
                if grid_y[i] >= FS_y_plus * wall_units:
                    max_vertical_resol = True
                    phase_ends['ph6'] = i
                    if verbose:
                        print(f"    Ph6 node={i:4d}  y+={grid_y[i]/wall_units:.1f}"
                              f"  (FS {FS_y_plus:.1f} y+)")

            # Phase 7 – Quadratic ramp-down / contraction (once, exactly post_BL_stretch_pts — hardbound)
            elif not contraction:
                tmp_dz, tmp_str, tmp_y = contractionfn(
                    i, post_BL_stretch_pts, post_BL_stretch_per,
                    deltaz[i], grid_y[i])
                deltaz     = np.append(deltaz, tmp_dz)
                stretching = np.append(stretching, tmp_str)
                grid_y     = np.append(grid_y, tmp_y)
                i         += post_BL_stretch_pts
                phase_ends['ph7'] = i
                contraction = True
                if verbose:
                    print(f"    Ph7 node={i:4d}  y={grid_y[i]:.4e}")

            # Phases 1–7 all complete – exit loop
            else:
                break

    return grid_y, deltaz, stretching, phase_ends


# ===========================================================================
# Iterative solver
# ===========================================================================

def _append_phase8(gy17, dz17, st17, ph17, n_ph8):
    """
    Append Phase-8 uniform top-padding cells to a Phase-1..7 grid.

    Phase 8 uses the final Phase-7 cell size (last_dz) for all n_ph8 cells,
    extending the grid smoothly from the Phase-7 top toward max_Ly.
    Returns (grid_y, deltaz, stretching, phase_ends) for the full grid.
    """
    last_dz = dz17[-1]
    last_y  = gy17[-1]
    if n_ph8 > 0:
        ph8_y  = last_y + np.arange(1, n_ph8 + 1) * last_dz
        gy     = np.append(gy17, ph8_y)
        dz     = np.append(dz17, np.full(n_ph8, last_dz))
        st     = np.append(st17, np.zeros(n_ph8))
    else:
        gy, dz, st = gy17.copy(), dz17.copy(), st17.copy()
    ph        = dict(ph17)
    ph['ph8'] = gy.size - 1
    return gy, dz, st, ph


def solve_grid(block_size, max_Ly, min_Ly,
               nominal_buf, stretch_per_range, post_BL_range,
               solver_step_sp, solver_step_psp, buffer_range,
               # Fixed build_grid_y parameters
               dnu, valley_height, stretch_pts, deltax, wall_units,
               BL_y_plus, post_BL_stretch_pts, FS_y_plus,
               # MPI constraint (optional)
               nx=0, nz=0, ims_npro=0,
               # IBM hill-body refinement (optional)
               ibm_refine_pts=None):
    """
    Find (stretch_per, post_BL_stretch_per, n_hill_buffer_pts) such that the
    7-phase grid + Phase-8 top-padding satisfies:

      C2:  ny_final % block_size == 0
      C7:  min_Ly <= grid_y[-1] <= max_Ly
      C10: (nx × ny_final) % ims_npro == 0  (when ims_npro > 0)
           (nz × ny_final) % ims_npro == 0

    Phase-8 design
    --------------
    After Phases 1–7 finish at height grid_y17[-1] (< max_Ly in general), Phase 8
    pads the domain with uniform cells at the Phase-7 final cell size (last_dz) up
    to max_Ly.  The exact cell count n_ph8 is chosen analytically so that
    (ny17 + n_ph8) % eff_block == 0  and  min_Ly <= grid_y17[-1] + n_ph8*last_dz <= max_Ly,
    where eff_block = lcm(block_size, mpi_ny_factor) combines both divisibility requirements.

    Sweep strategy  (sorted nearest-nominal first for fastest convergence)
    -----------------------------------------------------------------------
    Pass 1 (fast, 176 calls):   sweep (sp, psp) at nominal_buf
    Pass 2 (extended):          sweep (sp, psp, buf) for all buf in buffer_range
    Returns the first valid solution dict, or None.
    """
    # Effective block size: lcm(block_size, mpi_ny_factor) where
    # mpi_ny_factor is the factor by which ny must be divisible to satisfy C10.
    # For each dimension d: mpi_ny_factor_d = ims_npro // gcd(d, ims_npro).
    # Combined: lcm over both dimensions, then lcm with block_size.
    if ims_npro > 0 and nx > 0 and nz > 0:
        def _lcm(a, b):
            return a * b // gcd(a, b)
        mpi_req_x = ims_npro // gcd(nx, ims_npro)
        mpi_req_z = ims_npro // gcd(nz, ims_npro)
        mpi_req   = _lcm(mpi_req_x, mpi_req_z)
        eff_block = _lcm(block_size, mpi_req)
    else:
        eff_block = block_size
    buf_min, buf_max = buffer_range
    sp_nom  = (stretch_per_range[0] + stretch_per_range[1]) / 2
    psp_nom = (post_BL_range[0]     + post_BL_range[1])     / 2

    sp_vals  = sorted(np.round(np.arange(stretch_per_range[0],
                                         stretch_per_range[1] + 1e-9,
                                         solver_step_sp), 6),
                      key=lambda x: abs(x - sp_nom))
    psp_vals = sorted(np.round(np.arange(post_BL_range[0],
                                         post_BL_range[1] + 1e-9,
                                         solver_step_psp), 6),
                      key=lambda x: abs(x - psp_nom))

    def _try(sp, psp, buf):
        """Return solution dict for (sp, psp, buf), or None if infeasible."""
        gy17, dz17, st17, ph17 = build_grid_y(
            dnu, valley_height, buf,
            sp, stretch_pts, deltax, wall_units,
            BL_y_plus, psp, post_BL_stretch_pts, FS_y_plus,
            ibm_refine_pts=ibm_refine_pts,
            verbose=False)

        last_y  = gy17[-1]
        last_dz = dz17[-1]
        ny17    = gy17.size

        if last_y > max_Ly:
            return None   # Phase 7 already overshot max_Ly

        # Phase-8 cell-count bounds
        n8_lo = max(0, math.ceil((min_Ly - last_y) / last_dz))
        n8_hi = math.floor((max_Ly - last_y) / last_dz)
        if n8_lo > n8_hi:
            return None   # min_Ly not reachable without exceeding max_Ly

        # Smallest n_ph8 such that (ny17 + n_ph8) % eff_block == 0, in [n8_lo, n8_hi]
        # eff_block = lcm(block_size, mpi_ny_factor) enforces both C2 and C10.
        r     = (-ny17) % eff_block           # required residue (0 if ny17 already divisible)
        k_min = math.ceil(max(0, n8_lo - r) / eff_block)
        n_ph8 = r + k_min * eff_block
        if n_ph8 > n8_hi:
            return None   # no multiple lands inside the height window

        gy, dz, st, ph = _append_phase8(gy17, dz17, st17, ph17, n_ph8)
        return {
            'grid_y'             : gy,
            'deltaz'             : dz,
            'stretching'         : st,
            'phase_ends'         : ph,
            'ny'                 : gy.size,
            'y_max'              : gy[-1],
            'n_ph8'              : n_ph8,
            'stretch_per'        : float(sp),
            'post_BL_stretch_per': float(psp),
            'n_hill_buffer_pts'  : buf,
            'buf_delta'          : buf - nominal_buf,
            'ibm_refine_pts'     : ibm_refine_pts,
        }

    # Pass 1 – nominal buffer only (fast)
    for sp in sp_vals:
        for psp in psp_vals:
            r = _try(sp, psp, nominal_buf)
            if r is not None:
                return r

    # Pass 2 – vary buffer as well
    buf_vals = sorted(range(buf_min, buf_max + 1),
                      key=lambda x: abs(x - nominal_buf))
    for sp in sp_vals:
        for psp in psp_vals:
            for buf in buf_vals:
                if buf == nominal_buf:
                    continue   # already tried in Pass 1
                r = _try(sp, psp, buf)
                if r is not None:
                    return r

    return None   # no valid solution found in the parameter space


# ===========================================================================
# USER PARAMETERS  –  edit only this section
# ===========================================================================

# ---------------------------------------------------------------------------
# Target compute system  (controls block-size divisibility requirement)
# ---------------------------------------------------------------------------
# 'Hunter' → ny, nx, nz must each be divisible by 96
# any other string → divisible by 64
SYSTEM = 'Hunter'

# ---------------------------------------------------------------------------
# MPI process count on the target system  (C10 divisibility check)
#
# tlab requires (nx × ny) % ims_npro == 0  AND  (nz × ny) % ims_npro == 0,
# where  ims_npro = ims_npro_i × ims_npro_k  (tlab.ini [Grid] Imax(*) × Kmax(*)).
#
# This is the constraint in TLabMPI_Trp_PlanI / TLabMPI_Trp_PlanK:
#   npage = (local imax or kmax) × ny   must be divisible by ims_npro_k or ims_npro_i.
#
# Set to 0 to skip this check (not recommended).
# Example: Curta with 192 total MPI processes → ims_npro_curta = 192
# ---------------------------------------------------------------------------
ims_npro_curta = 0   # ← fill in your cluster's total MPI process count

# ---------------------------------------------------------------------------
# Option flags
# ---------------------------------------------------------------------------
USE_EXISTING_GRID_SCALE = True    # True: lock Lx, Lz (and max_Ly) from old grid
USE_KNOWN_GEOMETRY      = True    # True: use known_hill_height/pts and ustar for BL
ENABLE_SOLVER           = True    # True: iterate to satisfy block_size divisibility

# ---------------------------------------------------------------------------
# New grid dimensions  [nx, ny_hint, nz]
# ny_hint is a guidance target for ny; the solver will find the closest
# feasible value divisible by block_size.  nx and nz must already be
# divisible by block_size (checked below with a warning).
# ---------------------------------------------------------------------------
NewNodes  = np.array([1056, 864, 1056], dtype=int)
nx        = int(NewNodes[0])
ny_hint   = int(NewNodes[1])   # approximate target for wall-normal nodes
nz        = int(NewNodes[2])

# ---------------------------------------------------------------------------
# Flow / physical parameters
# ---------------------------------------------------------------------------
ustar = 0.066   # friction velocity (non-dimensional)
Re_D  = 500     # Reynolds number based on Ekman-layer depth
G     = 1       # geostrophic wind magnitude
Bo    = 1       # bulk Richardson number

# ---------------------------------------------------------------------------
# Horizontal isotropy factor
#
# The horizontal cell size is:  dx = dz = iso_factor × wall_units
#
# • USE_EXISTING_GRID_SCALE = False (new grid from scratch):
#     iso_factor is user-controlled.  Typical range: 2.5–4.0.
#     2.0–2.5 → finer horizontal resolution, smaller domain for same nx.
#     4.0     → standard DNS isotropic target (coarser, larger domain).
#     The vertical isotropic target (Phase 3 end) also uses iso_factor.
#
# • USE_EXISTING_GRID_SCALE = True (old grid):
#     iso_factor is COMPUTED from old_Lx / nx / wall_units (not user-chosen).
#     The old domain footprint is a hard boundary; no flexibility here.
# ---------------------------------------------------------------------------
iso_factor = 4.0   # only used when USE_EXISTING_GRID_SCALE = False

# ---------------------------------------------------------------------------
# Near-wall resolution  (Phase 1)
#
# Δy+ = near_wall_dnu.  Recommendation:
#   1.0 → standard DNS (5 pts in viscous sub-layer y+ < 5).  Use this.
#   0.5 → doubles Phase-1 nodes; rarely justified at Re_D = 500.
# ---------------------------------------------------------------------------
near_wall_dnu = 1.0

# ---------------------------------------------------------------------------
# Hill height  (Phase 1 extent)
#
# Specify EXACTLY ONE:
#   known_hill_pts    (int)  – cell count from wall to crest; physical = pts × dnu.
#   known_hill_height (float) – physical crest height; cell count is auto-computed.
# If both are set, known_hill_pts takes precedence.
# If USE_KNOWN_GEOMETRY = False, both are ignored → hill = h_by_L × BL.
# ---------------------------------------------------------------------------
known_hill_pts    = None          # e.g. 52  (None → use physical height)
known_hill_height = 3.10656e-4   # physical crest height [non-dim]

# Buffer: uniform cells above the crest before stretching starts.
# The solver adjusts this to hit block_size divisibility.
n_hill_buffer_pts = 15   # nominal; solver range: buffer_range below

# ---------------------------------------------------------------------------
# IBM hill-body refinement (optional — Phase 0)
#   None  → Phase 1 is a single uniform zone at dnu from the wall (default).
#   int N → add N extra cells inside the orography body (y=0..valley_height),
#           giving dnu_ibm = valley_height/(ceil(valley_height/dnu)+N) there.
#           A Phase-1b buffer (n_hill_buffer_pts cells at dnu) then bridges up
#           to the stretching zone, so Phase 2+ is unchanged.
# Use this to better resolve the immersed boundary inside the hill without
# refining the whole near-wall block.
# ---------------------------------------------------------------------------
ibm_refine_pts = None    # e.g. 30  (None → no extra IBM-body refinement)

# ---------------------------------------------------------------------------
# Boundary-layer thickness
#   δ_BL = u*/f = u*  (since f = 1 in these units).
#   For a new simulation use the neutral-run value; update when DNS converges.
# ---------------------------------------------------------------------------
known_BL_thickness = ustar   # programmatic assignment – stays in sync with ustar

# ---------------------------------------------------------------------------
# Free-stream height  (Phase 6 end – computed, not a user input)
#
# FS_y_plus = FS_BL_multiple × BL / wall_units
# Set FS_BL_multiple = 1.2–1.5 if the velocity profile is already flat at 1.5 δ_BL.
# Increase toward 2.0 if still transitioning.
# ---------------------------------------------------------------------------
FS_BL_multiple = 1.5

# ---------------------------------------------------------------------------
# Fallback geometry  (only used when USE_KNOWN_GEOMETRY = False)
# ---------------------------------------------------------------------------
h_by_L = 0.1   # valley height-to-wavelength ratio  H/λ

# ---------------------------------------------------------------------------
# Stretching parameters  (Phases 2–4)
# The solver sweeps stretch_per within stretch_per_range.
# stretch_pts = number of transition nodes for a smooth cell-size change.
# ---------------------------------------------------------------------------
stretch_per = 1.5   # nominal first stretching rate [%]
stretch_pts = 10    # ramp-up transition nodes (continuity buffer)

# ---------------------------------------------------------------------------
# Post-BL stretching parameters  (Phases 5–7)
# The solver sweeps post_BL_stretch_per within post_BL_stretch_range.
# post_BL_stretch_pts = transition nodes for smooth change at Phases 5 and 7.
# ---------------------------------------------------------------------------
post_BL_stretch_per  = 2.9    # nominal post-BL stretching rate [%]
post_BL_stretch_pts  = 10     # ramp-up / ramp-down transition nodes

# ---------------------------------------------------------------------------
# Domain height bounds
#
# min_Ly : grid must reach at least this physical height.
# max_Ly : hard ceiling (set automatically from old_Ly when USE_EXISTING_GRID_SCALE=True).
#          Override with max_Ly_user when building without an old grid.
# ---------------------------------------------------------------------------
min_Ly      = 0.3    # minimum physical domain height
max_Ly_user = None   # override ceiling; None → use old_Ly (or 5 × min_Ly as fallback)

# ---------------------------------------------------------------------------
# Solver settings
#
# stretch_per_range     : [min, max] first stretching rate [%]  (Phase 2–4)
# post_BL_stretch_range : [min, max] post-BL stretching rate [%] (Phase 5–6)
# buffer_range          : [min, max] n_hill_buffer_pts
# solver_step_sp / _psp : sweep step size for each parameter
# ---------------------------------------------------------------------------
stretch_per_range     = (1.0, 2.0)    # first stretch bounds [%]
post_BL_stretch_range = (2.0, 3.5)    # post-BL stretch bounds [%]
buffer_range          = (5,   40)     # hill buffer bounds [cells]
solver_step_sp        = 0.1           # sweep step for stretch_per
solver_step_psp       = 0.1           # sweep step for post_BL_stretch_per

# ===========================================================================
# END OF USER PARAMETERS
# ===========================================================================


# ---------------------------------------------------------------------------
# Derived block size
# ---------------------------------------------------------------------------
block_size = 96 if SYSTEM.lower() == 'hunter' else 64

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
errors = []

if nx % block_size != 0:
    errors.append(f"nx={nx} is not divisible by block_size={block_size} ({SYSTEM})")
if nz % block_size != 0:
    errors.append(f"nz={nz} is not divisible by block_size={block_size} ({SYSTEM})")
if ny_hint % block_size != 0:
    print(f"  WARNING: ny_hint={ny_hint} is not divisible by {block_size}; "
          f"solver will find nearest valid value.")

if errors:
    for e in errors:
        print(f"  ERROR: {e}")
    raise SystemExit(1)

print(f"System: {SYSTEM}  (block_size={block_size})")
print(f"Target grid: nx={nx}, ny≈{ny_hint}, nz={nz}")


# ---------------------------------------------------------------------------
# Derived flow scales
# ---------------------------------------------------------------------------
f_cor       = 1                            # Coriolis parameter (normalised)
re_lam      = (Re_D ** 2) / 2
nu_D        = 2 / ((Re_D ** 2) * f_cor)   # kinematic viscosity
delta_ekman = (2 * nu_D / f_cor) ** 0.5   # Ekman-layer depth

Ro = G / (delta_ekman * f_cor)
Fr = G ** 2 / (Bo * delta_ekman)

wall_units = nu_D / ustar    # viscous length scale (1 wall unit)
delta_nu   = wall_units      # alias for plotting

dnu    = near_wall_dnu * wall_units   # Phase-1 cell width (physical)


# ---------------------------------------------------------------------------
# Geometry – from known data OR derived from flow parameters
# ---------------------------------------------------------------------------
if USE_KNOWN_GEOMETRY:
    BL    = known_BL_thickness   # = ustar  (δ_BL = u*/f = u* since f = 1)
    delta = BL / 3

    if known_hill_pts is not None:
        hill_pts      = int(known_hill_pts)
        valley_height = hill_pts * dnu
    else:
        valley_height = known_hill_height
        hill_pts      = int(math.ceil(valley_height / dnu))

    print("\nGeometry from known simulation data:")
    print(f"  Hill input mode   : "
          f"{'grid points' if known_hill_pts is not None else 'physical units'}")
    print(f"  Hill crest        : {hill_pts} cells × {dnu:.4e} = "
          f"{valley_height:.4e}  ({valley_height/wall_units:.1f} y+)")
    print(f"  Phase-1 total     : {hill_pts} (hill) + "
          f"{n_hill_buffer_pts} (buffer) = {hill_pts + n_hill_buffer_pts} cells  [nominal]")
    print(f"  BL thickness (u*) : {BL:.4e}")
else:
    delta         = ustar / f_cor
    BL            = 3 * delta
    valley_height = h_by_L * BL
    hill_pts      = int(math.ceil(valley_height / dnu))

    print("\nGeometry derived from flow parameters:")
    print(f"  Hill crest        : {hill_pts} cells × {dnu:.4e} = "
          f"{valley_height:.4e}  ({valley_height/wall_units:.1f} y+)")
    print(f"  Phase-1 total     : {hill_pts} (hill) + "
          f"{n_hill_buffer_pts} (buffer) = {hill_pts + n_hill_buffer_pts} cells  [nominal]")
    print(f"  BL thickness (3δ) : {BL:.4e}")

# Thresholds (same formula regardless of geometry source)
BL_y_plus = BL / wall_units
FS_y_plus = FS_BL_multiple * BL / wall_units
print(f"  BL_y_plus (auto)  : {BL_y_plus:.1f} y+")
print(f"  FS_y_plus (auto)  : {FS_y_plus:.1f} y+  (×{FS_BL_multiple} BL)")


# ---------------------------------------------------------------------------
# Load old grid – lock Lx/Lz, compute iso_factor and max_Ly
# ---------------------------------------------------------------------------
old_Lx = old_Lz = None   # set below if USE_EXISTING_GRID_SCALE

if USE_EXISTING_GRID_SCALE:
    print("\nReading existing grid file ...")
    old_nodes, old_scales, old_coords = read_grid(cwd + 'grid')
    old_nx, old_ny, old_nz            = old_nodes
    old_Lx, old_Ly, old_Lz            = old_scales
    old_x,  old_y,  old_z             = old_coords

    # Horizontal extents locked to old grid
    sizex = old_Lx
    sizez = old_Lz

    # iso_factor is FIXED by the old scale and the chosen nx
    # (no user control – changing nx changes dx; changing iso_factor is impossible)
    deltax     = old_Lx / nx           # horizontal cell width (physical)
    iso_factor = deltax / wall_units   # implied isotropy factor [wall units]

    # Domain height ceiling (hard limit from old grid, overridable)
    max_Ly = old_Ly if max_Ly_user is None else max_Ly_user

    # Diagnostic info from old grid
    old_dnu_phys = old_y[1] - old_y[0]
    old_dnu_plus = old_dnu_phys / wall_units
    old_hill_node = int(np.searchsorted(old_y, valley_height))
    old_BL_node   = int(np.searchsorted(old_y, BL))

    print(f"  Old grid dims             : nx={old_nx}, ny={old_ny}, nz={old_nz}")
    print(f"  Old scales                : Lx={old_Lx:.6f}, Ly={old_Ly:.6f}, "
          f"Lz={old_Lz:.6f}")
    print(f"  Old y_max                 : {old_y[-1]:.6f}  ({old_y[-1]/wall_units:.1f} y+)")
    print(f"  Max domain height (new)   : {max_Ly:.6f}  "
          f"({'from old grid' if max_Ly_user is None else 'user override'})")
    print(f"  Old near-wall Δy          : {old_dnu_phys:.4e}  ({old_dnu_plus:.3f} y+)")
    print(f"  New near-wall Δy          : {dnu:.4e}  ({near_wall_dnu:.2f} y+)  "
          f"[{'finer' if near_wall_dnu < old_dnu_plus else 'coarser'} than old]")
    print(f"  Old node near hill crest  : #{old_hill_node}  "
          f"y={old_y[old_hill_node]:.4e}  [new hill_pts={hill_pts}]")
    print(f"  Old node near BL top      : #{old_BL_node}  "
          f"y={old_y[old_BL_node]:.4e}  ({old_y[old_BL_node]/wall_units:.1f} y+)")
    print(f"  iso_factor (fixed)        : {iso_factor:.3f}  "
          f"(dx = {deltax:.4e} = {iso_factor:.2f} y+)")
    print(f"  Preserving Lx={sizex:.6f}, Lz={sizez:.6f}")

else:
    # New grid from scratch: iso_factor is user-controlled
    deltax = iso_factor * wall_units   # horizontal and isotropic target cell width
    sizex  = deltax * nx
    sizez  = deltax * nz
    max_Ly = max_Ly_user if max_Ly_user is not None else min_Ly * 5

    print("\nNew grid from scratch:")
    print(f"  iso_factor (user)         : {iso_factor:.2f}  "
          f"(dx = {deltax:.4e} = {iso_factor:.2f} y+)")
    print(f"  Domain Lx                 : {sizex:.6f}")
    print(f"  Domain Lz                 : {sizez:.6f}")
    print(f"  Max domain height         : {max_Ly:.6f}  "
          f"({'user' if max_Ly_user is not None else 'fallback 5×min_Ly'})")


# ===========================================================================
# Build / solve vertical (y) grid
# ===========================================================================

if ENABLE_SOLVER:
    print(f"\nRunning solver  (block_size={block_size}) ...")
    print(f"  stretch_per      : [{stretch_per_range[0]}, {stretch_per_range[1]}] %  "
          f"step {solver_step_sp}")
    print(f"  post_BL_stretch  : [{post_BL_stretch_range[0]}, "
          f"{post_BL_stretch_range[1]}] %  step {solver_step_psp}")
    print(f"  buffer_range     : {buffer_range}  nominal={n_hill_buffer_pts}")

    solution = solve_grid(
        block_size           = block_size,
        max_Ly               = max_Ly,
        min_Ly               = min_Ly,
        nominal_buf          = n_hill_buffer_pts,
        stretch_per_range    = stretch_per_range,
        post_BL_range        = post_BL_stretch_range,
        solver_step_sp       = solver_step_sp,
        solver_step_psp      = solver_step_psp,
        buffer_range         = buffer_range,
        # Fixed physics parameters passed through
        dnu                  = dnu,
        valley_height        = valley_height,
        stretch_pts          = stretch_pts,
        deltax               = deltax,
        wall_units           = wall_units,
        BL_y_plus            = BL_y_plus,
        post_BL_stretch_pts  = post_BL_stretch_pts,
        FS_y_plus            = FS_y_plus,
        # MPI constraint
        nx                   = nx,
        nz                   = nz,
        ims_npro             = ims_npro_curta,
        # IBM hill-body refinement
        ibm_refine_pts       = ibm_refine_pts,
    )

    if solution is None:
        print("\n  WARNING: solver found no valid solution in the parameter space.")
        print("  Falling back to nominal parameters + Phase-8 padding to min_Ly.")
        gy17, dz17, st17, ph17 = build_grid_y(
            dnu, valley_height, n_hill_buffer_pts,
            stretch_per, stretch_pts, deltax, wall_units,
            BL_y_plus, post_BL_stretch_per, post_BL_stretch_pts, FS_y_plus,
            ibm_refine_pts=ibm_refine_pts,
            verbose=True)
        # Pad to at least min_Ly (no divisibility guarantee in fallback)
        last_dz = dz17[-1]
        n_ph8_fb = max(0, math.ceil((min_Ly - gy17[-1]) / last_dz))
        grid_y, deltaz, stretching, phase_ends_final = _append_phase8(
            gy17, dz17, st17, ph17, n_ph8_fb)
        solved_stretch_per = stretch_per
        solved_post_BL     = post_BL_stretch_per
        solved_buf         = n_hill_buffer_pts
        solver_converged   = False
    else:
        grid_y           = solution['grid_y']
        deltaz           = solution['deltaz']
        stretching       = solution['stretching']
        phase_ends_final = solution['phase_ends']
        solved_stretch_per = solution['stretch_per']
        solved_post_BL     = solution['post_BL_stretch_per']
        solved_buf         = solution['n_hill_buffer_pts']
        solver_converged   = True
        print("  Solver converged:")
        print(f"    stretch_per        : {solved_stretch_per:.2f} %  "
              f"(nominal {stretch_per:.2f} %)")
        print(f"    post_BL_stretch_per: {solved_post_BL:.2f} %  "
              f"(nominal {post_BL_stretch_per:.2f} %)")
        print(f"    n_hill_buffer_pts  : {solved_buf}  "
              f"(nominal {n_hill_buffer_pts}, Δ={solution['buf_delta']:+d})")
        print(f"    n_ph8 (top pad)    : {solution['n_ph8']} cells at "
              f"Δy={deltaz[-1]:.4e}")
        print(f"    ny_final           : {solution['ny']}  "
              f"(divisible by {block_size}: "
              f"{'YES' if solution['ny'] % block_size == 0 else 'NO'})")
        print(f"    y_max              : {solution['y_max']:.6f}  "
              f"(max_Ly={max_Ly:.6f})")

else:
    # Single run with nominal parameters (no iteration)
    print("\nBuilding vertical grid (solver disabled) ...")
    gy17, dz17, st17, ph17 = build_grid_y(
        dnu, valley_height, n_hill_buffer_pts,
        stretch_per, stretch_pts, deltax, wall_units,
        BL_y_plus, post_BL_stretch_per, post_BL_stretch_pts, FS_y_plus,
        ibm_refine_pts=ibm_refine_pts,
        verbose=True)
    # Phase 8: pad to min_Ly (no divisibility guarantee when solver is disabled)
    last_dz = dz17[-1]
    n_ph8_manual = max(0, math.ceil((min_Ly - gy17[-1]) / last_dz))
    grid_y, deltaz, stretching, phase_ends_final = _append_phase8(
        gy17, dz17, st17, ph17, n_ph8_manual)
    print(f"  Phase-8 top-pad: {n_ph8_manual} cells at Δy={last_dz:.4e}")
    solved_stretch_per = stretch_per
    solved_post_BL     = post_BL_stretch_per
    solved_buf         = n_hill_buffer_pts
    solver_converged   = False

ny_final = grid_y.size
print(f"\nVertical grid: {ny_final} nodes, y_max = {grid_y[-1]:.6f}  "
      f"(ny % {block_size} = {ny_final % block_size})")


# ===========================================================================
# Build uniform horizontal (x, z) node arrays
#
# Periodic grid: x = [0, dx, 2dx, …, (n-1)dx]
# Scale = x[-1] + x[1] = (n-1)dx + dx = L  ✓
# ===========================================================================

dx_new  = sizex / nx
dz_new  = sizez / nz
x_nodes = np.arange(nx) * dx_new
z_nodes = np.arange(nz) * dz_new

x_scale_new = x_nodes[-1] + x_nodes[1]   # Lx (includes periodic wrap-around)
y_scale_new = grid_y[-1]                  # Ly (domain top, non-periodic)
z_scale_new = z_nodes[-1] + z_nodes[1]   # Lz (includes periodic wrap-around)


# ===========================================================================
# Sanity check  –  verify every hardbound condition on the final grid
# If any condition fails, the solver is retried with expanded ranges.
# The grid is NOT written until all conditions pass.
# ===========================================================================

def _run_sanity(gy, ph, buf):
    """Helper: call sanity_check with all current physics params."""
    return sanity_check(
        gy, ph,
        dnu            = dnu,
        valley_height  = valley_height,
        n_hill_buffer_pts = buf,
        wall_units     = wall_units,
        BL_y_plus      = BL_y_plus,
        FS_y_plus      = FS_y_plus,
        min_Ly         = min_Ly,
        max_Ly         = max_Ly,
        block_size     = block_size,
        nx             = nx,
        nz             = nz,
        stretch_pts          = stretch_pts,
        post_BL_stretch_pts  = post_BL_stretch_pts,
        use_existing_scale   = USE_EXISTING_GRID_SCALE,
        old_Lx         = old_Lx,
        old_Lz         = old_Lz,
        x_scale_new    = x_scale_new,
        z_scale_new    = z_scale_new,
        ims_npro       = ims_npro_curta,
        ibm_refine_pts = ibm_refine_pts,
    )

print("\n--- Sanity check ---")
sc_fails = _run_sanity(grid_y, phase_ends_final, solved_buf)

if sc_fails:
    print(f"  {len(sc_fails)} condition(s) FAILED:")
    for f in sc_fails:
        print(f"    {f}")

    # -----------------------------------------------------------------------
    # Retry with expanded parameter ranges (2× width on all axes)
    # -----------------------------------------------------------------------
    print("\n  Retrying solver with expanded parameter ranges ...")
    sp_lo   = max(0.1,  stretch_per_range[0]     * 0.5)
    sp_hi   =           stretch_per_range[1]     * 2.0
    psp_lo  = max(0.1,  post_BL_stretch_range[0] * 0.5)
    psp_hi  =           post_BL_stretch_range[1] * 2.0
    buf_lo  = max(1,    buffer_range[0] - 15)
    buf_hi  =           buffer_range[1] + 30

    print(f"    stretch_per      : [{sp_lo:.2f}, {sp_hi:.2f}] %")
    print(f"    post_BL_stretch  : [{psp_lo:.2f}, {psp_hi:.2f}] %")
    print(f"    buffer_range     : [{buf_lo}, {buf_hi}]")

    solution2 = solve_grid(
        block_size           = block_size,
        max_Ly               = max_Ly,
        min_Ly               = min_Ly,
        nominal_buf          = solved_buf,
        stretch_per_range    = (sp_lo, sp_hi),
        post_BL_range        = (psp_lo, psp_hi),
        solver_step_sp       = solver_step_sp,
        solver_step_psp      = solver_step_psp,
        buffer_range         = (buf_lo, buf_hi),
        dnu                  = dnu,
        valley_height        = valley_height,
        stretch_pts          = stretch_pts,
        deltax               = deltax,
        wall_units           = wall_units,
        BL_y_plus            = BL_y_plus,
        post_BL_stretch_pts  = post_BL_stretch_pts,
        FS_y_plus            = FS_y_plus,
        nx                   = nx,
        nz                   = nz,
        ims_npro             = ims_npro_curta,
        ibm_refine_pts       = ibm_refine_pts,
    )

    if solution2 is not None:
        grid_y           = solution2['grid_y']
        deltaz           = solution2['deltaz']
        stretching       = solution2['stretching']
        phase_ends_final = solution2['phase_ends']
        solved_buf       = solution2['n_hill_buffer_pts']
        solved_stretch_per = solution2['stretch_per']
        solved_post_BL     = solution2['post_BL_stretch_per']
        ny_final           = grid_y.size
        y_scale_new        = grid_y[-1]
        print(f"  Retry solver converged: ny={ny_final}, y_max={grid_y[-1]:.6f}")

        sc_fails2 = _run_sanity(grid_y, phase_ends_final, solved_buf)
    else:
        sc_fails2 = ["Expanded solver found no valid solution in the parameter space"]

    if sc_fails2:
        print("\n  FATAL — grid does NOT satisfy all hardbound conditions after retry:")
        for f in sc_fails2:
            print(f"    {f}")
        print("\n  Grid NOT written. Adjust USER PARAMETERS and re-run.")
        raise SystemExit(1)
    else:
        print("  All conditions PASSED after retry.")

else:
    n_checked = 10 if ims_npro_curta > 0 else 9
    print(f"  All {n_checked} hardbound conditions PASSED.")


# ===========================================================================
# Write new grid file
# ===========================================================================

file_out = cwd + 'grid_new'
print(f"\nWriting {file_out} ...")
write_grid(file_out,
           nx, ny_final, nz,
           x_scale_new, y_scale_new, z_scale_new,
           x_nodes, grid_y, z_nodes)
print("  Done.")


# ===========================================================================
# Diagnostic plots
# ===========================================================================

nodes      = np.arange(ny_final)
y_plus     = grid_y / wall_units
y_by_delta = grid_y / delta

# Plot 1: physical y positions
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(nodes, grid_y, lw=1)
ax.axhline(valley_height, color='red',    linestyle='--',
           label=f'Hill crest  H = {valley_height:.3e}')
ax.axhline(BL,            color='orange', linestyle='--',
           label=f'BL thickness = {BL:.3e}')
ax.axhline(max_Ly,        color='purple', linestyle=':',
           label=f'max_Ly = {max_Ly:.4f}')
ax.set_xlabel('Node index')
ax.set_ylabel('y  (physical)')
ax.set_title(f'Vertical grid – node positions  (ny={ny_final}, '
             f'ny÷{block_size}={ny_final//block_size} r{ny_final%block_size})')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(cwd + 'grid_y_positions.png', dpi=150)
plt.show()

# Plot 2: cell width
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(nodes, deltaz, lw=1)
ax.axhline(deltax, color='red', linestyle='--',
           label=f'Isotropic target  Δ = {deltax:.3e}  ({iso_factor:.1f} y+)')
ax.set_xlabel('Node index')
ax.set_ylabel('Δy  (physical)')
ax.set_title('Vertical grid – cell width per node')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(cwd + 'grid_deltaz.png', dpi=150)
plt.show()

# Plot 3: wall-unit spacing vs y+
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(grid_y[:-1] / delta_nu, np.diff(grid_y) / delta_nu, lw=1)
ax.axhline(near_wall_dnu, color='red', linestyle='--',
           label=f'Phase-1 target  Δy+ = {near_wall_dnu}')
ax.set_xlabel('y+')
ax.set_ylabel('Δy+')
ax.set_title('Grid spacing in wall units')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(cwd + 'grid_spacing_yplus.png', dpi=150)
plt.show()


# ===========================================================================
# Grid summary
# ===========================================================================

print("\n--- Grid summary ---")
print(f"  System / block_size        : {SYSTEM} / {block_size}")
print(f"  USE_EXISTING_GRID_SCALE    : {USE_EXISTING_GRID_SCALE}")
print(f"  USE_KNOWN_GEOMETRY         : {USE_KNOWN_GEOMETRY}")
print(f"  ENABLE_SOLVER              : {ENABLE_SOLVER}  "
      f"({'converged' if solver_converged else 'not used / no solution'})")
print(f"  Final grid dims            : nx={nx}, ny={ny_final}, nz={nz}")
print(f"  ny divisible by {block_size}         : "
      f"{'YES' if ny_final % block_size == 0 else 'NO  ← SOLVER DID NOT CONVERGE'}")
print(f"  nx divisible by {block_size}         : "
      f"{'YES' if nx % block_size == 0 else 'NO'}")
print(f"  nz divisible by {block_size}         : "
      f"{'YES' if nz % block_size == 0 else 'NO'}")
print(f"  y_max                      : {grid_y[-1]:.6f}  (max_Ly={max_Ly:.6f})")
print(f"  wall_units (δ_ν)           : {wall_units:.4e}")
print(f"  Near-wall Δy (Phase 1)     : {dnu:.4e}  ({near_wall_dnu:.2f} y+)")
print(f"  iso_factor                 : {iso_factor:.3f}  (dx={dx_new:.4e}  = "
      f"{dx_new/wall_units:.2f} y+)")
print(f"  Isotropic cell target      : {deltax:.4e}  ({iso_factor:.2f} y+)")
print(f"  Hill crest                 : {valley_height:.4e}  "
      f"({valley_height/wall_units:.1f} y+)  [{hill_pts} cells]")
print(f"  BL thickness               : {BL:.4e}  ({BL_y_plus:.1f} y+)")
print(f"  Free-stream threshold      : {FS_y_plus:.1f} y+  "
      f"(×{FS_BL_multiple} BL)")
print(f"  Solved stretch_per         : {solved_stretch_per:.2f} %")
print(f"  Solved post_BL_stretch_per : {solved_post_BL:.2f} %")
print(f"  Solved n_hill_buffer_pts   : {solved_buf}")
print(f"  Rossby number              : {Ro:.3f}")
print(f"  Froude number              : {Fr:.3f}")
print(f"  Domain  Lx                 : {x_scale_new:.6f}")
print(f"  Domain  Ly                 : {y_scale_new:.6f}")
print(f"  Domain  Lz                 : {z_scale_new:.6f}")
print("  Output files               : grid_new, grid_y_positions.png, "
      "grid_deltaz.png, grid_spacing_yplus.png")
