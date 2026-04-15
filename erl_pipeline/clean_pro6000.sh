#!/bin/bash
# Clean up artifacts from a previous Pro 6000 ERL pipeline run.
# Run from the llm_scaffold/ directory, or set SCAFFOLD_DIR.
#
# Usage:
#   bash erl_pipeline/clean_pro6000.sh          # delete artifacts
#   bash erl_pipeline/clean_pro6000.sh --dry-run # show what would be deleted

set -euo pipefail

# ── Paths (match run_erl_pro6000.sh) ─────────────────────────
PROJ_DIR="${PROJ_DIR:-$HOME/auto_proj}"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-${PROJ_DIR}/llm_scaffold}"
ERL_REPO="${SCAFFOLD_DIR}/autoresearch_erl"
ERL_LOG="${SCAFFOLD_DIR}/erl_pipeline/erl_log_pro6000"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/raye_${USER}}"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Would delete:"
fi

delete() {
    local target="$1"
    if [ ! -e "$target" ]; then
        return
    fi
    if $DRY_RUN; then
        echo "  $target"
        return
    fi
    echo "Removing $target"
    if [ -d "$target" ]; then
        local unlink_bin="rm -f"
        command -v munlink >/dev/null 2>&1 && unlink_bin="munlink"
        find "$target" -type f -print0 2>/dev/null \
            | xargs -0 -r -n 1000 -P 8 $unlink_bin 2>/dev/null || true
        find "$target" -depth -type d -empty -delete 2>/dev/null || true
    fi
    rm -rf "$target"
}

# ── Worker dirs first (symlinks point into ERL_REPO) ────────
for d in "${SCAFFOLD_DIR}"/autoresearch_erl_worker_*; do
    delete "$d"
done

# ── Copied repo ─────────────────────────────────────────────
delete "$ERL_REPO"

# ── ERL logs + checkpoints (includes results.tsv) ───────────
delete "$ERL_LOG"

# ── Ray plasma sockets / spill files ────────────────────────
delete "$RAY_TMPDIR"

if ! $DRY_RUN; then
    echo "Done. Ready for a fresh Pro 6000 ERL run."
fi
