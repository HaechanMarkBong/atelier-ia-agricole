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
WORK="$(mktemp -d)"
RESULTS="$(cd "$(dirname "$0")/.." && pwd)/.nbruns"
mkdir -p "$RESULTS"
trap 'cp -f "$WORK/$BASE" "$RESULTS/$BASE" 2>/dev/null; rm -rf "$WORK"' EXIT
BASE="$(basename "$NB")"
cp "$NB" "$WORK/$BASE"

echo ">>> Exécution: $BASE (timeout ${TO}s/cellule)"
echo ">>> Résultat (avec sorties) conservé dans: $RESULTS/$BASE"
cd "$WORK"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout="$TO" \
  --ExecutePreprocessor.kernel_name=python3 \
  "$BASE" 2>&1
RC=$?
echo ">>> RC=$RC pour $BASE"
exit $RC
