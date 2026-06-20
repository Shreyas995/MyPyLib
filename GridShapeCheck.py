#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GridShapeCheck.py  —  MPI-shape feasibility checker for tlab DNS grids.

Run this BEFORE EditGrid.py.  It answers one question only:
    "Does a grid of shape (NX, NY, NZ) run on Curta and/or Hunter?"
It knows nothing about physics/Reynolds number — it only checks the parallel
decomposition rules verified against tlab/src/base (see EditGrid.py header).

Workflow
--------
  1. Pick a desired resolution (NX/NZ from dx in wall units, NY from EditGrid's
     physical construction estimate).
  2. Run this script: it tells you the nearest feasible NX per machine and the
     smallest shared NY, and prints names carrying a machine hint, e.g.
         1024x672x1024_Curta      1056x672x1056_Hunter
  3. Paste the feasible GRID_TARGETS / NY_TARGET into EditGrid.py and generate.

Machines (rank layouts the user runs):
    Curta : 1024 ranks = npro_i 32  x npro_k 32
    Hunter:   96 ranks = npro_i 12  x npro_k  8

Verified per-rank constraints (Imax*=NX/npro_i, Kmax*=NZ/npro_k, Jmax=NY):
    C1 NX % npro_i == 0     C2 NZ % npro_k == 0     C4 Imax* even
    C5 (Kmax*·Jmax) % npro_i == 0     C6 (Imax*·Jmax) % npro_k == 0
    C7 ((Imax*+2)·Jmax/2) % npro_k == 0   C8 (Jmax·Kmax*) % npro_i == 0
"""

SYSTEMS = {
    'Curta':  dict(npro_i=32, npro_k=32),   # 32*32 = 1024 ranks
    'Hunter': dict(npro_i=12, npro_k=8),    # 12* 8 =   96 ranks
}


def check_one(nx, ny, nz, system):
    """Return (ok, info_dict, list_of_failures) for one machine."""
    s = SYSTEMS[system]
    npi, npk = s['npro_i'], s['npro_k']
    fails = []
    if nx % npi:
        fails.append(f"C1 NX%{npi}={nx % npi}")
    if nz % npk:
        fails.append(f"C2 NZ%{npk}={nz % npk}")
    if fails:                                   # cannot form Imax*/Kmax*
        return False, dict(npro_i=npi, npro_k=npk, ranks=npi * npk), fails
    Is, Ks = nx // npi, nz // npk
    rules = {
        "C4 Imax* even":          Is % 2 == 0,
        "C5 (Ks*J)%npi":          (Ks * ny) % npi == 0,
        "C6 (Is*J)%npk":          (Is * ny) % npk == 0,
        "C7 ((Is+2)*J/2)%npk":    ((Is + 2) * ny // 2) % npk == 0,
        "C8 (J*Ks)%npi":          (ny * Ks) % npi == 0,
    }
    fails = [k for k, v in rules.items() if not v]
    info = dict(npro_i=npi, npro_k=npk, ranks=npi * npk, Imax_star=Is, Kmax_star=Ks)
    return (len(fails) == 0), info, fails


def classify(nx, ny, nz):
    """Return (works_list, hinted_name, per_system_results)."""
    res = {name: check_one(nx, ny, nz, name) for name in SYSTEMS}
    works = [n for n, (ok, *_), in res.items() if ok]
    hint = ''.join(works) if works else 'NONE'
    return works, f"{nx}x{ny}x{nz}_{hint}", res


def _nx_step(system, square=True):
    """Divisibility step for a SQUARE (NZ=NX) grid: needs NX % npro_i == 0 with
    Imax* even (NX % 2*npro_i) AND NX % npro_k == 0  ->  lcm of both."""
    from math import gcd
    npi, npk = SYSTEMS[system]['npro_i'], SYSTEMS[system]['npro_k']
    a = 2 * npi
    if not square:
        return a
    return a * npk // gcd(a, npk)


def nearest_nx(nx0, system, square=True, direction='nearest'):
    """Feasible NX near nx0 for `system` (square grid by default)."""
    step = _nx_step(system, square)
    lo = (nx0 // step) * step
    hi = lo + step
    if direction == 'up':
        return hi if lo < nx0 else lo
    if direction == 'down':
        return lo
    return lo if (nx0 - lo) <= (hi - nx0) else hi


def shared_ny(targets, ny0, search=4096):
    """Smallest NY >= ny0 feasible for every (system, nx, nz) in `targets`."""
    for ny in range(ny0, ny0 + search):
        if all(check_one(nx, ny, nz, sys)[0] for sys, nx, nz in targets):
            return ny
    return None


def propose_pair(nx_target, ny_phys, direction='up'):
    """
    Build a Curta+Hunter SQUARE pair near nx_target with one shared NY >= ny_phys.
    direction='up' keeps NX >= nx_target (do not coarsen below the request).

    Returns dict ready to paste into EditGrid.py's GRID_TARGETS / NY_TARGET.
    """
    out, targets = {}, []
    for sysname in ('Curta', 'Hunter'):
        nx = nearest_nx(nx_target, sysname, square=True, direction=direction)
        targets.append((sysname, nx, nx))
        out[sysname] = dict(nx=nx, nz=nx)
    ny = shared_ny(targets, ny_phys)
    out['NY'] = ny
    for sysname, nx, nz in targets:
        ok, info, _ = check_one(nx, ny, nz, sysname)
        out[sysname].update(Imax_star=info['Imax_star'], Kmax_star=info['Kmax_star'],
                            ranks=info['ranks'],
                            name=f"{nx}x{ny}x{nz}_{sysname}", ok=ok)
    return out


def estimate_nx(Lx, l_in, dx_wall_units=2.5):
    """NX needed to hold dx ~ dx_wall_units (wall units) for domain length Lx.
    l_in = nu/u_star (1 wall unit).  Re-scaling lives here: feed the NEW Re's
    l_in to size a higher-Re grid (see Re-scaling note in __main__)."""
    return Lx / (dx_wall_units * l_in)


# ----------------------------------------------------------------------------
def _print_check(nx, ny, nz):
    works, name, res = classify(nx, ny, nz)
    print(f"\n{nx} x {ny} x {nz}  ->  runs on: {works or 'NONE'}   (name: {name})")
    for sysname, (ok, info, fails) in res.items():
        tag = 'OK ' if ok else 'NO '
        det = (f"Imax*={info.get('Imax_star')}, Kmax*={info.get('Kmax_star')}, "
               f"{info['npro_i']}x{info['npro_k']}={info['ranks']} ranks")
        if not ok:
            det += f"  FAIL: {', '.join(fails)}"
        print(f"   [{tag}] {sysname:7s} {det}")


if __name__ == '__main__':
    print("=" * 70)
    print("CURRENT PAIR (Re_D=500, built from 1024x416x1024):")
    print("=" * 70)
    _print_check(1024, 672, 1024)     # Curta member
    _print_check(1056, 672, 1056)     # Hunter member

    print("\n" + "=" * 70)
    print("PROPOSE A PAIR near NX~1040, NY>=672 (shared NY):")
    print("=" * 70)
    p = propose_pair(1040, 672)
    print(f"  shared NY = {p['NY']}")
    for s in ('Curta', 'Hunter'):
        d = p[s]
        print(f"  {s:7s}: {d['name']:24s} Imax*={d['Imax_star']:3d} Kmax*={d['Kmax_star']:3d} "
              f"ranks={d['ranks']}  feasible={d['ok']}")
    print("\n  -> EditGrid.py GRID_TARGETS:")
    for s in ('Curta', 'Hunter'):
        d = p[s]
        print(f"       ('{s}', {d['nx']}, {d['nz']}, {d['Imax_star']}, {d['Kmax_star']}),")
    print(f"     NY_TARGET = {p['NY']}")

    # ---- Re-scaling sketch for the END OBJECTIVE (Re 750 / 1000) -------------
    # NX scales to keep dx in wall units fixed: NX ~ Lx / (dx_wu * l_in),
    # with l_in = nu/u_star and nu = 1/(0.5*Re_D**2).  u_star must come from the
    # flow setup (precursor / a-posteriori) — it is NOT a pure function of Re.
    print("\n" + "=" * 70)
    print("Re-SCALING SKETCH (refine u_star with the real value per Re):")
    print("=" * 70)
    Lx = 0.26537242
    for Re, u_star in [(500, 0.077), (750, 0.077), (1000, 0.077)]:
        nu = 1.0 / (0.5 * Re ** 2)
        l_in = nu / u_star
        nx_est = estimate_nx(Lx, l_in, dx_wall_units=2.5)
        nxC = nearest_nx(nx_est, 'Curta'); nxH = nearest_nx(nx_est, 'Hunter')
        print(f"  Re_D={Re:4d} (u*={u_star}): l_in={l_in:.3e}  Re_tau~{u_star/l_in:6.0f}  "
              f"NX_est~{nx_est:6.0f}  -> Curta {nxC}, Hunter {nxH}  (NY grows similarly)")
    print("\n  NOTE: u_star=0.077 reused as a placeholder for Re>500.  Replace it with"
          "\n        the actual friction velocity for Re 750/1000, then re-run this"
          "\n        script to lock NX/NZ, and let EditGrid.py size NY physically.")
