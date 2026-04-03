#!/bin/bash
# Clean up artifacts from a previous ERL pipeline run.
# Run from the llm_scaffold/ directory, or set SCAFFOLD_DIR.
#
# Usage:
#   bash erl_pipeline/clean.sh          # delete artifacts
#   bash erl_pipeline/clean.sh --dry-run # show what would be deleted

set -euo pipefail

# ── Paths (match run_erl.sh) ─────────────────────────────────
PROJ_DIR="${PROJ_DIR:-/blue/buyuheng/li_an.ucsb/proj_yepeng}"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-${PROJ_DIR}/llm_scaffold_rl/llm_scaffold}"
ERL_REPO="${SCAFFOLD_DIR}/autoresearch_erl"
ERL_LOG="${SCAFFOLD_DIR}/erl_pipeline/erl_log"

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
delete "$ERL_REPO"

# ── Ray eval worker copies (autoresearch_erl_worker_0, ...) ──
for d in "${SCAFFOLD_DIR}"/autoresearch_erl_worker_*; do
    delete "$d"
done

# ── ERL logs + checkpoints (includes results.tsv) ────────────
delete "$ERL_LOG"

if ! $DRY_RUN; then
    echo "Done. Ready for a fresh ERL run."
fi
