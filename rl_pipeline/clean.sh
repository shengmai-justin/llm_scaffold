#!/bin/bash
# Clean up artifacts from a previous RL pipeline run.
# Run from the llm_scaffold/ directory, or set SCAFFOLD_DIR.
#
# Usage:
#   bash rl_pipeline/clean.sh          # uses paths from run_rl.sh
#   bash rl_pipeline/clean.sh --dry-run # show what would be deleted

set -euo pipefail

# ── Paths (match run_rl.sh) ──────────────────────────────────
PROJ_DIR="${PROJ_DIR:-/blue/buyuheng/li_an.ucsb/proj_yepeng}"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-${PROJ_DIR}/llm_scaffold_rl/llm_scaffold}"
RL_REPO="${SCAFFOLD_DIR}/autoresearch_rl"
RL_LOG="${SCAFFOLD_DIR}/rl_pipeline/rl_log"
RESULTS_TSV="${SCAFFOLD_DIR}/results.tsv"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Would delete:"
fi

delete() {
    local target="$1"
    if [ -e "$target" ]; then
        if $DRY_RUN; then
            echo "  $target"
        else
            echo "Removing $target"
            rm -rf "$target"
        fi
    fi
}

# ── Copied repo ──────────────────────────────────────────────
delete "$RL_REPO"

# ── Ray eval worker copies (eval_worker_0, eval_worker_1, ...) ─
for d in "${SCAFFOLD_DIR}"/eval_worker_*; do
    delete "$d"
done

# ── RL logs + checkpoints ────────────────────────────────────
delete "$RL_LOG"

# ── Shared results ───────────────────────────────────────────
delete "$RESULTS_TSV"

if ! $DRY_RUN; then
    echo "Done. Ready for a fresh run."
fi
