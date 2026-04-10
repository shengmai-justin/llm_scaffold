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
    if [ ! -e "$target" ]; then
        return
    fi
    if $DRY_RUN; then
        echo "  $target"
        return
    fi
    echo "Removing $target"
    # Lustre fast-delete: parallel `munlink` (skips the per-file stat() that
    # `rm -rf` does).  Falls back to parallel `rm -f` on non-Lustre hosts.
    # Final `rm -rf` is a safety net for anything the fast path missed
    # (empty dirs, symlinks, special files).
    if [ -d "$target" ]; then
        local unlink_bin="rm -f"
        command -v munlink >/dev/null 2>&1 && unlink_bin="munlink"
        find "$target" -type f -print0 2>/dev/null \
            | xargs -0 -r -n 1000 -P 8 $unlink_bin 2>/dev/null || true
        find "$target" -depth -type d -empty -delete 2>/dev/null || true
    fi
    rm -rf "$target"
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
