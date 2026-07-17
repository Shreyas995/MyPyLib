#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_pickles.py  (v2 — diagnostic) — acceptance test for the PhAvg_rotated.py
cleanup refactor.

Run ONE case end-to-end BEFORE and AFTER the refactor, then diff the two
sim1_results.pkl key-by-key.  For every key this compares APPLES TO APPLES:

  * same type / same shape is checked first (a type or shape mismatch is reported
    verbatim — never silently reduced to a number);
  * scalars print the actual  old vs new  values;
  * arrays print WHERE they differ — how many / what fraction of elements, the
    location & values of the single largest difference, and the NaN pattern on
    each side — so you can localise the problem instead of reading one max|Δ|;
  * dicts / lists recurse element-by-element (so phi_m_st = {'windward':…} is
    compared per station, not by object identity).

The ONLY difference expected from a correct refactor is `dx` (grid spacing now,
was the wavenumber 2*pi/x[-1] — a results.py bug fix), listed under
KNOWN_INTENTIONAL_DIFFS.  A STALE old.pkl (from an earlier code/config state) can
show unrelated diffs (e.g. the modified-law fit outputs); the values printed here
make the cause obvious (e.g. old=nan vs new=finite ⇒ the fit block changed, not
the core).  To prove the refactor itself is clean, run the CURRENT code twice and
diff the two — they must match on every key.

Usage:
    python3 compare_pickles.py old.pkl new.pkl [--rtol 1e-9] [--atol 1e-12] [--top 5]
"""
import argparse
import pickle
import sys

import numpy as np

KNOWN_INTENTIONAL_DIFFS = {
    'dx',   # grid spacing now, was the wavenumber 2*pi/x[-1] (results.py bug fix)
}


def _load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _fmt(v):
    """Compact scalar formatter that makes nan/inf/sign obvious."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if np.isnan(fv):
        return 'nan'
    if np.isinf(fv):
        return 'inf' if fv > 0 else '-inf'
    return f'{fv:.6g}'


def _compare(a, b, rtol, atol, top):
    """Return (match, kind, detail) comparing one old/new value apples-to-apples.

    kind is a short tag ('scalar' | 'array' | 'dict' | 'list' | 'str' | 'none' |
    'type' | 'shape') so the caller can see what was actually compared.
    """
    # ---- type mismatch is a hard, explicit failure (no silent coercion) --------
    if type(a) is not type(b):
        # allow int/float and numpy/python numeric mixing (same numeric value)
        num = (int, float, np.integer, np.floating)
        if not (isinstance(a, num) and isinstance(b, num)):
            return False, 'type', f'TYPE {type(a).__name__} vs {type(b).__name__}'

    # ---- None -----------------------------------------------------------------
    if a is None or b is None:
        ok = (a is None and b is None)
        return ok, 'none', ('both None' if ok else f'{a!r} vs {b!r}')

    # ---- dict: recurse per shared key -----------------------------------------
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False, 'dict', f'KEYS {sorted(a)} vs {sorted(b)}'
        bad = []
        for k in a:
            ok, _, d = _compare(a[k], b[k], rtol, atol, top)
            if not ok:
                bad.append(f'{k}: {d}')
        return (not bad), 'dict', ('ok' if not bad else ' | '.join(bad))

    # ---- list / tuple: recurse elementwise ------------------------------------
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False, 'list', f'LEN {len(a)} vs {len(b)}'
        bad = []
        for i, (ai, bi) in enumerate(zip(a, b)):
            ok, _, d = _compare(ai, bi, rtol, atol, top)
            if not ok:
                bad.append(f'[{i}] {d}')
        return (not bad), 'list', ('ok' if not bad else ' | '.join(bad[:top]))

    # ---- string / bytes -------------------------------------------------------
    if isinstance(a, (str, bytes)):
        ok = (a == b)
        return ok, 'str', ('equal' if ok else f'{a!r} vs {b!r}')

    # ---- numeric (scalar or array) --------------------------------------------
    try:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        try:
            ok = bool(a == b)
        except Exception:
            ok = (a is b)
        return ok, 'obj', ('equal' if ok else f'{a!r} vs {b!r}')

    if aa.shape != bb.shape:
        return False, 'shape', f'SHAPE {aa.shape} vs {bb.shape}'

    if aa.size == 0:
        return True, 'array', 'empty'

    close = np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True)

    # scalar → just show the two values
    if aa.ndim == 0:
        with np.errstate(invalid='ignore', divide='ignore'):
            diff = float(abs(aa - bb))
            denom = max(abs(float(aa)), abs(float(bb)))
            rel = diff / denom if denom > 0 else 0.0
        return close, 'scalar', (f'old={_fmt(aa)}  new={_fmt(bb)}  '
                                 f'|Δ|={diff:.3e}  relΔ={rel:.3e}')

    # array → localise the difference
    nan_a, nan_b = np.isnan(aa), np.isnan(bb)
    with np.errstate(invalid='ignore'):
        diff = np.abs(aa - bb)
        diff_finite = np.where(nan_a | nan_b, 0.0, diff)
    differ = (~np.isclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True))
    ndiff = int(np.count_nonzero(differ))
    parts = [f'{ndiff}/{aa.size} elems differ ({100.0*ndiff/aa.size:.2f}%)']
    if ndiff:
        flat = np.argmax(diff_finite)
        idx = np.unravel_index(flat, aa.shape)
        parts.append(f'max|Δ|={diff_finite.flat[flat]:.3e} @ {idx} '
                     f'(old={_fmt(aa[idx])} new={_fmt(bb[idx])})')
        # show the first few differing indices for locality
        diff_idx = np.argwhere(differ)[:top]
        loc = ', '.join(f'{tuple(ix)}:{_fmt(aa[tuple(ix)])}→{_fmt(bb[tuple(ix)])}'
                        for ix in diff_idx)
        parts.append('first: ' + loc)
    if int(nan_a.sum()) != int(nan_b.sum()):
        parts.append(f'NaN old={int(nan_a.sum())} new={int(nan_b.sum())}')
    return close, 'array', '; '.join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('--rtol', type=float, default=1e-9)
    ap.add_argument('--atol', type=float, default=1e-12)
    ap.add_argument('--top', type=int, default=5, help='how many differing elements to list')
    args = ap.parse_args()

    print('== compare_pickles v2 (diagnostic) ==')
    old, new = _load(args.old), _load(args.new)
    ko, kn = set(old), set(new)

    only_old = sorted(ko - kn)
    only_new = sorted(kn - ko)
    shared = sorted(ko & kn)

    if only_old:
        print(f'[keys] {len(only_old)} in OLD but MISSING from NEW: {only_old}')
    if only_new:
        print(f'[keys] {len(only_new)} in NEW but MISSING from OLD: {only_new}')

    failures, intentional = [], []
    for k in shared:
        ok, kind, detail = _compare(old[k], new[k], args.rtol, args.atol, args.top)
        if ok:
            continue
        line = f'  {k:<20} [{kind}]  {detail}'
        (intentional if k in KNOWN_INTENTIONAL_DIFFS else failures).append(line)

    if intentional:
        print('\n[intentional] differ BY DESIGN (not a regression):')
        print('\n'.join(intentional))

    if failures:
        print(f'\n[FAIL] {len(failures)} shared key(s) differ beyond tol '
              f'(rtol={args.rtol}, atol={args.atol}):')
        print('\n'.join(failures))
    else:
        print(f'\n[PASS] all {len(shared)} shared keys match '
              f'(rtol={args.rtol}, atol={args.atol}).')

    ok = (not failures) and (not only_old)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
