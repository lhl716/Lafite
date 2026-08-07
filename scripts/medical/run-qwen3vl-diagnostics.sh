#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
LAFITE_PYTHON="${LAFITE_PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${REPO_DIR}/experiments/medical-diagnostics/qwen3-vl-8b"
export MEDICAL_OUTPUT_ROOT="${MEDICAL_OUTPUT_ROOT:-${REPO_DIR}/outputs/medical-real}"
cd "${REPO_DIR}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "error: environment variable ${name} is required" >&2
    exit 2
  fi
}

run_exp1() {
  require_var MEDICAL_MANIFEST
  require_var LAFITE_CHECKPOINT
  "${LAFITE_PYTHON}" -m medical_experiments validate-manifest "${MEDICAL_MANIFEST}" --check-paths --require-mask
  "${LAFITE_PYTHON}" -m medical_experiments export-lafite --config "${CONFIG_DIR}/exp1-export-lafite.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments export-mllm --config "${CONFIG_DIR}/exp1-export-qwen.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments merge-features --config "${CONFIG_DIR}/exp1-merge.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments detail-probe --config "${CONFIG_DIR}/exp1-probe.yaml"
}

run_exp2() {
  require_var MEDICAL_MANIFEST
  require_var LAFITE_CHECKPOINT
  require_var LAFITE_MASK_THRESHOLD
  require_var LAFITE_SCORE_THRESHOLD
  "${LAFITE_PYTHON}" -m medical_experiments make-interventions --config "${CONFIG_DIR}/exp2-positive.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments make-interventions --config "${CONFIG_DIR}/exp2-normal.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments shortcut-eval --config "${CONFIG_DIR}/exp2-evaluate.yaml"
}

run_exp3() {
  export SEMANTIC_MANIFEST="${SEMANTIC_MANIFEST:-${MEDICAL_MANIFEST:-}}"
  require_var SEMANTIC_MANIFEST
  "${LAFITE_PYTHON}" -m medical_experiments validate-manifest "${SEMANTIC_MANIFEST}" --check-paths --require-semantics
  "${LAFITE_PYTHON}" -m medical_experiments export-semantic-scores --config "${CONFIG_DIR}/exp3-export-scores.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments llm-necessity --config "${CONFIG_DIR}/exp3-evaluate.yaml"
}

case "${MODE}" in
  exp1) run_exp1 ;;
  exp2) run_exp2 ;;
  exp3) run_exp3 ;;
  all) run_exp1; run_exp2; run_exp3 ;;
  results) "${LAFITE_PYTHON}" "${SCRIPT_DIR}/show-qwen3vl-results.py" "${MEDICAL_OUTPUT_ROOT}" ;;
  *) echo "usage: $0 {exp1|exp2|exp3|all|results}" >&2; exit 2 ;;
esac
