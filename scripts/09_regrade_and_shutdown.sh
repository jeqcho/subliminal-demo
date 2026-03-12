#!/bin/bash
set -e

cd /workspace/subliminal-demo

echo "=== Waiting for pipeline tmux session to finish ==="
while tmux has-session -t pipeline 2>/dev/null; do
    sleep 30
done
echo "=== Pipeline session done ==="

# Commit current state
echo "=== Committing training + pattern-based eval results ==="
git add -A
git commit -m "Complete pipeline: training + pattern-based eval

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" || echo "Nothing to commit"

# Regrade all eval results with LLM judge
echo "=== Regrading eval results with GPT-5-mini LLM judge ==="
uv run python scripts/06b_regrade_eval.py 2>&1 | tee logs/regrade_2026-03-12.log

# Commit regraded results
echo "=== Committing regraded results ==="
git add -A
git commit -m "Regrade eval results with GPT-5-mini LLM judge

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" || echo "Nothing to commit"

echo "=== All done. Stopping pod ==="
runpodctl stop pod $RUNPOD_POD_ID
