#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: util/animal_frequency.sh <results-jsonl>

Build a sorted frequency table of chosen animals from eval JSONL output.
EOF
  exit 0
fi

RESULTS_FILE="${1:-}"
if [[ -z "$RESULTS_FILE" ]]; then
  echo "error: missing results file path" >&2
  echo "usage: util/animal_frequency.sh <results-jsonl>" >&2
  exit 1
fi

if [[ ! -f "$RESULTS_FILE" ]]; then
  echo "error: file not found: $RESULTS_FILE" >&2
  exit 1
fi

printf "count\tanimal\n"
jq -sr '
  map(.animal)
  | map(select(type == "string" and length > 0))
  | group_by(.)
  | map({animal: .[0], count: length})
  | sort_by(-.count, .animal)
  | .[]
  | "\(.count)\t\(.animal)"
' "$RESULTS_FILE"
