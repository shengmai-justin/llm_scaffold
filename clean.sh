#!/bin/bash
# Clean up artifacts from a previous frozen pipeline run.
# Run from the llm_scaffold/ directory, or set SCAFFOLD_DIR.
#
# Usage:
#   bash clean.sh          # delete artifacts
#   bash clean.sh --dry-run # show what would be deleted

set -euo pipefail

# ── Paths (match run.sh) ─────────────────────────────────────
PROJ_DIR="${PROJ_DIR:-/blue/buyuheng/li_an.ucsb/proj_yepeng}"
SCAFFOLD_DIR="${SCAFFOLD_DIR:-${PROJ_DIR}/llm_scaffold_rl/llm_scaffold}"
FROZEN_REPO="${SCAFFOLD_DIR}/autoresearch_frozen"
FROZEN_LOG="${SCAFFOLD_DIR}/frozen_log"

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

# ── Copied autoresearch repo (autoresearch_frozen/) ──────────
delete "$FROZEN_REPO"

# ── Frozen logs: results.tsv, run.log, state.json ────────────
delete "$FROZEN_LOG"

# ── SGLang server log ────────────────────────────────────────
delete "${SCAFFOLD_DIR}/sglang_server.log"

if ! $DRY_RUN; then
    echo "Done. Ready to resubmit: sbatch run.sh"
fi
