#!/usr/bin/env bash
# Exécute un notebook en MODE_DEMO sur une COPIE (les originaux restent propres).
# Usage: run_nb.sh <notebook.ipynb> [timeout_s]
set -u
export PATH="$HOME/.local/bin:$PATH"
export ATELIER_DEMO=1
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

NB="$1"
TO="${2:-900}"
WORK="/tmp/claude-1000/-home-hmb-Desktop/1f30d318-57c7-4b08-83bc-2954a63cd75a/scratchpad/nbwork"
mkdir -p "$WORK"
BASE="$(basename "$NB")"
cp "$NB" "$WORK/$BASE"

echo ">>> Exécution: $BASE (timeout ${TO}s/cellule)"
cd "$WORK"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout="$TO" \
  --ExecutePreprocessor.kernel_name=python3 \
  "$BASE" 2>&1
RC=$?
echo ">>> RC=$RC pour $BASE"
exit $RC
