#!/usr/bin/env bash
# =============================================================================
# setup_sim.sh — wire a simulation/data directory to the master MyPyLib code.
#
# Keeps ONE master copy of the code (this directory) and makes every script
# runnable from the folder where the DNS data lives, by creating symlinks back
# to the master.  Each data directory keeps its OWN editable `config.py`, so
# different simulations stay independent while sharing identical code.
#
# Usage
# -----
#   cd /path/to/simulation_data
#   bash /path/to/MyPyLib/setup_sim.sh            # set up the current directory
#   # or, target another directory explicitly:
#   bash /path/to/MyPyLib/setup_sim.sh /path/to/simulation_data
#
# What it does
# ------------
#   * Symlinks every master *.py into the target dir (so `python3 PhAvg.py`
#     there runs the master code; data is read from the target via __file__).
#   * Does NOT symlink config.py.  Instead it COPIES the master config.py as a
#     local template.  If a local config.py already exists and DIFFERS from the
#     master, it is replaced with the master copy AFTER backing the old one up to
#     config.py.bak.<timestamp> (so per-sim edits are recoverable).  An identical
#     local config.py is left untouched.
#     PhAvg.py prepends the data dir to sys.path, so this LOCAL config wins.
#   * Re-running is safe and idempotent.
# =============================================================================
set -euo pipefail

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
echo

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
