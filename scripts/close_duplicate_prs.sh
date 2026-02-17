#!/usr/bin/env bash
# PR Dedup Script — agno-agi/agno
# Run this locally with `gh` authenticated to close duplicate PRs with comments.
#
# Usage: bash scripts/close_duplicate_prs.sh
# Requires: gh CLI authenticated with repo write access

set -euo pipefail

REPO="agno-agi/agno"

comment_and_close() {
    local pr_num=$1
    local comment=$2
    echo "Commenting on PR #${pr_num}..."
    gh pr comment "$pr_num" --repo "$REPO" --body "$comment"
    echo "Closing PR #${pr_num}..."
    gh pr close "$pr_num" --repo "$REPO"
    echo "  Done."
    echo ""
}

echo "=== Closing duplicate PRs ==="
echo ""

# --- Cluster 1: get_member_id UUID priority ---
# Keep #6336 (kausmeows), close #6434 (themavik)
comment_and_close 6434 "Closing as duplicate of #6336 which addresses the same \`get_member_id\` UUID priority bug with a more complete fix (cleaner rewrite, updated tests, \`Optional[str]\` return type)."

# --- Cluster 2: Datetime serialization in Postgres session storage ---
# Keep #6363 (kausmeows), close #6436 and #6437 (themavik)
comment_and_close 6436 "Closing as duplicate of #6363 which fixes datetime serialization at the SQLAlchemy engine level — this covers all serialization paths (not just \`to_dict()\`) and includes comprehensive tests."

comment_and_close 6437 "Closing as duplicate of #6363 which fixes datetime serialization at the SQLAlchemy engine level — this covers all serialization paths (not just \`to_dict()\`) and includes comprehensive tests."

echo ""
echo "=== Closing PRs already fixed on main ==="
echo ""

# --- Already fixed: metadata on run_context ---
comment_and_close 6340 "Closing — this bug has been fixed on \`main\` via the refactor into \`_run_options.py\`. The \`resolve_run_options()\` function now correctly handles metadata merging and assignment to \`run_context\`."

# --- Already fixed: reasoning attribute in OpenAI chat ---
comment_and_close 5332 "Closing — this bug has been fixed on \`main\`. The code at \`models/openai/chat.py\` now handles both \`reasoning_content\` and \`reasoning\` attributes."

comment_and_close 4177 "Closing — this bug has been fixed on \`main\`. The code at \`models/openai/chat.py\` now handles both \`reasoning_content\` and \`reasoning\` attributes."

# --- Already fixed: MCP header_provider validation ---
comment_and_close 5988 "Closing — the incompatible transport case is already handled on \`main\` (logs a warning and ignores the \`header_provider\`). The proposed \`ValueError\` approach would be a breaking behavior change that should be a separate deliberate decision."

echo ""
echo "=== Summary ==="
echo "Closed 7 PRs:"
echo "  #6434 — duplicate of #6336 (get_member_id)"
echo "  #6436 — duplicate of #6363 (datetime serialization)"
echo "  #6437 — duplicate of #6363 (datetime serialization)"
echo "  #6340 — already fixed on main (metadata on run_context)"
echo "  #5332 — already fixed on main (reasoning attribute)"
echo "  #4177 — already fixed on main (reasoning attribute)"
echo "  #5988 — already fixed on main (header_provider)"
echo ""
echo "Done."
