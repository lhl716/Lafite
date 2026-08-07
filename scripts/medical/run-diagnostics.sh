#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 EXP1_CONFIG EXP2_CONFIG EXP3_CONFIG" >&2
  exit 2
fi

LAFITE_PYTHON="${LAFITE_PYTHON:-python3}"

"${LAFITE_PYTHON}" -m medical_experiments detail-probe --config "$1"
"${LAFITE_PYTHON}" -m medical_experiments shortcut-eval --config "$2"
"${LAFITE_PYTHON}" -m medical_experiments llm-necessity --config "$3"
