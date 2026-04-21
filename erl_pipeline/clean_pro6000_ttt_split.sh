#!/bin/bash
# Clean up artifacts from a previous Pro 6000 ERL TTT+split run.
#
# Usage:
#   bash erl_pipeline/clean_pro6000_ttt_split.sh           # delete artifacts
#   bash erl_pipeline/clean_pro6000_ttt_split.sh --dry-run # show what would be deleted

set -euo pipefail

# ── Paths (match run_erl_pro6000_ttt_split.sh) ───────────────
PROJ_DIR="${PROJ_DIR:-$HOME/auto_proj}"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-${PROJ_DIR}/llm_scaffold}"
ERL_REPO="${SCAFFOLD_DIR}/autoresearch_erl_pro6000_ttt_split"
ERL_LOG="${SCAFFOLD_DIR}/erl_pipeline/erl_log_pro6000_ttt_split"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/raye_ttt_split_${USER}}"

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

for d in "${SCAFFOLD_DIR}"/autoresearch_erl_pro6000_ttt_split_worker_*; do
    delete "$d"
done
delete "$ERL_REPO"
delete "$ERL_LOG"
delete "$RAY_TMPDIR"

if ! $DRY_RUN; then
    echo "Done. Ready for a fresh Pro 6000 ERL TTT+split run."
fi
