#!/usr/bin/env bash
# =============================================================================
# setup.sh — wire a simulation/data directory to the master MyPyLib code.
#
# Keeps ONE master copy of the code (this directory) and makes every script
# runnable from the folder where the DNS data lives, by creating symlinks back
# to the master.  Each data directory keeps its OWN editable `config.py`, so
# different simulations stay independent while sharing identical code.
#
# Usage
# -----
#   cd /path/to/simulation_data
#   bash /path/to/MyPyLib/setup.sh                # set up the current directory
#   # or, target another directory explicitly:
#   bash /path/to/MyPyLib/setup.sh /path/to/simulation_data
#
#   # results-only: link JUST results.py + its dependencies (for the central
#   # post-processing folder that holds all the per-case data and from which you
#   # run only results.py — no PhAvg pipeline, no config.py):
#   bash /path/to/MyPyLib/setup.sh --results-only /path/to/examples_root
#
# What it does
# ------------
#   * Full mode (default): symlinks every master *.py into the target dir (so
#     `python3 PhAvg.py` there runs the master code; data is read from the target
#     via __file__) and manages a per-sim config.py (copied, not linked).
#   * --results-only mode: symlinks ONLY results.py and the modules it imports
#     (PlotField.py, functions.py) and does NOT touch config.py — results.py
#     defines its own scalars and never imports config.  This is the minimal set
#     needed to run `python3 results.py` from the central examples root.
#   * config.py (full mode): COPIED as a local template, not symlinked.  If a
#     local config.py already exists and DIFFERS from the master, it is replaced
#     with the master copy AFTER backing the old one up to config.py.bak.<ts>
#     (so per-sim edits are recoverable).  An identical local config.py is left
#     untouched.  PhAvg.py prepends the data dir to sys.path, so the LOCAL config
#     wins.
#   * Re-running is safe and idempotent.
# =============================================================================
set -euo pipefail

# results.py and the modules it imports (PlotField, functions).  Python puts the
# script's own directory on sys.path[0], so linking these three into the target
# is enough for `python3 results.py` to find its imports AND read data via
# __file__ (= the symlink's directory).
# results.py = Froude ladder (Re=500 fixed); results_Re.py = Reynolds ladder
# (neutral, Re=500 vs 750).  Both are stage-c plotting scripts run from the
# central examples root and share the same two imports.  A missing file is
# warned about and skipped, so listing both is safe.
RESULTS_DEPS=(results.py results_Re.py PlotField.py functions.py)

RESULTS_ONLY=0
if [ "${1:-}" = "--results-only" ]; then
    RESULTS_ONLY=1
    shift
fi

# Master = directory containing this script (resolve symlinks).
MASTER="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

# Target = first argument, or the current working directory.
TARGET="${1:-$PWD}"
TARGET="$(cd "$TARGET" && pwd)"

if [ "$TARGET" = "$MASTER" ]; then
    echo "Refusing to set up the master directory itself ($MASTER)." >&2
    exit 1
fi

echo "Master : $MASTER"
echo "Target : $TARGET"
echo "Mode   : $([ "$RESULTS_ONLY" -eq 1 ] && echo 'results-only' || echo 'full')"
echo

# --- results-only: link results.py + its imports, then stop -------------------
if [ "$RESULTS_ONLY" -eq 1 ]; then
    linked=0
    for base in "${RESULTS_DEPS[@]}"; do
        if [ ! -e "$MASTER/$base" ]; then
            echo "Warning: $MASTER/$base not found — skipping." >&2
            continue
        fi
        ln -sfn "$MASTER/$base" "$TARGET/$base"
        linked=$((linked + 1))
    done
    echo "Linked $linked file(s): ${RESULTS_DEPS[*]}"
    cat <<EOF

Done. Next steps:
  cd "$TARGET"
  python3 results.py        # post-process every case found under this folder

Re-run with --results-only any time results.py's dependencies change.
EOF
    exit 0
fi

linked=0
for src in "$MASTER"/*.py; do
    base="$(basename "$src")"
    # config.py is per-simulation: handled separately (copied, not linked).
    if [ "$base" = "config.py" ]; then
        continue
    fi
    ln -sfn "$src" "$TARGET/$base"
    linked=$((linked + 1))
done
echo "Linked $linked master modules/scripts into the target."

# Per-simulation config.py: copy the master template.  Replace an existing local
# config.py when it differs from the master (backing the old one up first); leave
# an identical one untouched.
if [ ! -e "$TARGET/config.py" ]; then
    cp "$MASTER/config.py" "$TARGET/config.py"
    echo "Created local config.py from the master template — EDIT its per-sim values."
elif cmp -s "$MASTER/config.py" "$TARGET/config.py"; then
    echo "Local config.py already matches the master — left unchanged."
else
    backup="$TARGET/config.py.bak.$(date +%Y%m%d_%H%M%S)"
    cp "$TARGET/config.py" "$backup"
    cp "$MASTER/config.py" "$TARGET/config.py"
    echo "Replaced differing local config.py with the master copy."
    echo "  Old version backed up to: $backup"
    echo "  Re-apply this sim's per-sim values (Re, Fr, u_star, paths) if needed."
fi

cat <<EOF

Done. Next steps:
  cd "$TARGET"
  \$EDITOR config.py        # set Re, u_star, grid sizes, paths for THIS simulation
  python3 PhAvg.py          # run the pipeline (reads data here, code from master)

Re-run this script any time the master gains new scripts.
EOF
