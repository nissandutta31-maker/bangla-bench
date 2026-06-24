#!/usr/bin/env bash
# Generate a deterministic N-item prefix subset from the full Belebele Bengali split.
# Usage: ./scripts/make_subset.sh [N] [output_path]
# Default: 100 items -> belebele_ben_100.jsonl
set -euo pipefail
N="${1:-100}"
OUT="${2:-belebele_ben_100.jsonl}"
head -n "$N" belebele_ben_full.jsonl > "$OUT"
echo "Wrote $OUT ($N items from belebele_ben_full.jsonl)"
