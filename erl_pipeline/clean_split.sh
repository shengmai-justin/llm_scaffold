#!/bin/bash
# Clean up artifacts from a previous B200 ERL GRPO+split pipeline run.
#
# Usage:
#   bash erl_pipeline/clean_split.sh          # delete artifacts
#   bash erl_pipeline/clean_split.sh --dry-run # show what would be deleted

set -euo pipefail

# ── Paths (match run_erl_4gpu_split.sh) ──────────────────────
PROJ_DIR="${PROJ_DIR:-/blue/buyuheng/li_an.ucsb/proj_yepeng}"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-${PROJ_DIR}/llm_scaffold_rl/llm_scaffold}"
ERL_REPO="${SCAFFOLD_DIR}/autoresearch_erl_split"
ERL_LOG="${SCAFFOLD_DIR}/erl_pipeline/erl_log_split"

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

for d in "${SCAFFOLD_DIR}"/autoresearch_erl_split_worker_*; do
    delete "$d"
done
delete "$ERL_REPO"
delete "$ERL_LOG"

if ! $DRY_RUN; then
    echo "Done. Ready for a fresh ERL GRPO+split run."
fi
