#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# User settings: edit this block once on the GPU server.
# Existing exported environment variables still take precedence over these
# values, so cluster jobs can override them without modifying this file.
# -----------------------------------------------------------------------------
USER_BMAD_DATA_ROOT="/data/oss_bucket_0/yunlin/dataset/bmad"
# Leave these empty to use manifests made by prepare-bmad-hf-mirror.sh.
USER_MEDICAL_MANIFEST=""
USER_SEMANTIC_MANIFEST=""
USER_LAFITE_CHECKPOINT="/path/to/ckpt_best.pth.tar"
# Fill both thresholds with values selected on calibration/validation data.
# They intentionally have no unsafe test-set defaults.
USER_LAFITE_MASK_THRESHOLD=""
USER_LAFITE_SCORE_THRESHOLD=""
USER_MEDICAL_OUTPUT_ROOT=""
USER_MLLM_MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
USER_SINGLE_GPU="cuda:0"
USER_QWEN_ATTENTION="sdpa"
USER_LAFITE_PYTHON="python3"
# -----------------------------------------------------------------------------

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${REPO_DIR}/experiments/medical-diagnostics/qwen3-vl-8b"

BMAD_DATA_ROOT="${BMAD_DATA_ROOT:-${USER_BMAD_DATA_ROOT:-${REPO_DIR}/data/bmad}}"
DEFAULT_MEDICAL_MANIFEST="${BMAD_DATA_ROOT}/prepared/medical-manifest.jsonl"
DEFAULT_SEMANTIC_MANIFEST="${BMAD_DATA_ROOT}/prepared/semantic-manifest.jsonl"

export MEDICAL_MANIFEST="${MEDICAL_MANIFEST:-${USER_MEDICAL_MANIFEST:-${DEFAULT_MEDICAL_MANIFEST}}}"
export SEMANTIC_MANIFEST="${SEMANTIC_MANIFEST:-${USER_SEMANTIC_MANIFEST:-${DEFAULT_SEMANTIC_MANIFEST}}}"
export LAFITE_CHECKPOINT="${LAFITE_CHECKPOINT:-${USER_LAFITE_CHECKPOINT}}"
export LAFITE_MASK_THRESHOLD="${LAFITE_MASK_THRESHOLD:-${USER_LAFITE_MASK_THRESHOLD}}"
export LAFITE_SCORE_THRESHOLD="${LAFITE_SCORE_THRESHOLD:-${USER_LAFITE_SCORE_THRESHOLD}}"
export MEDICAL_OUTPUT_ROOT="${MEDICAL_OUTPUT_ROOT:-${USER_MEDICAL_OUTPUT_ROOT:-${REPO_DIR}/outputs/medical-real}}"
export MLLM_MODEL_ID="${MLLM_MODEL_ID:-${USER_MLLM_MODEL_ID}}"
export QWEN_ATTENTION="${QWEN_ATTENTION:-${USER_QWEN_ATTENTION}}"
LAFITE_PYTHON="${LAFITE_PYTHON:-${USER_LAFITE_PYTHON}}"

# Single-GPU default. Any individual EXP*_DEVICE variable may still override it.
SINGLE_GPU="${SINGLE_GPU:-${USER_SINGLE_GPU}}"
export EXP1_LAFITE_DEVICE="${EXP1_LAFITE_DEVICE:-${SINGLE_GPU}}"
export EXP1_QWEN_DEVICE="${EXP1_QWEN_DEVICE:-${SINGLE_GPU}}"
export EXP1_PROBE_DEVICE="${EXP1_PROBE_DEVICE:-${SINGLE_GPU}}"
export EXP2_LAFITE_DEVICE="${EXP2_LAFITE_DEVICE:-${SINGLE_GPU}}"
export EXP3_QWEN_DEVICE="${EXP3_QWEN_DEVICE:-${SINGLE_GPU}}"

cd "${REPO_DIR}"

require_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" || "${value}" == /path/to/* ]]; then
    echo "error: set ${name} in the user-settings block at the top of $0" >&2
    exit 2
  fi
}

run_exp1() {
  local started="${SECONDS}"
  require_var MEDICAL_MANIFEST
  require_var LAFITE_CHECKPOINT
  "${LAFITE_PYTHON}" -m medical_experiments validate-manifest "${MEDICAL_MANIFEST}" --check-paths --require-mask
  "${LAFITE_PYTHON}" -m medical_experiments export-lafite --config "${CONFIG_DIR}/exp1-export-lafite.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments export-mllm --config "${CONFIG_DIR}/exp1-export-qwen.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments merge-features --config "${CONFIG_DIR}/exp1-merge.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments detail-probe --config "${CONFIG_DIR}/exp1-probe.yaml"
  echo "experiment 1 completed in $((SECONDS - started)) seconds"
}

run_exp2() {
  local started="${SECONDS}"
  require_var MEDICAL_MANIFEST
  require_var LAFITE_CHECKPOINT
  require_var LAFITE_MASK_THRESHOLD
  require_var LAFITE_SCORE_THRESHOLD
  "${LAFITE_PYTHON}" -m medical_experiments make-interventions --config "${CONFIG_DIR}/exp2-positive.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments make-interventions --config "${CONFIG_DIR}/exp2-normal.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments shortcut-eval --config "${CONFIG_DIR}/exp2-evaluate.yaml"
  echo "experiment 2 completed in $((SECONDS - started)) seconds"
}

run_exp3() {
  local started="${SECONDS}"
  require_var SEMANTIC_MANIFEST
  "${LAFITE_PYTHON}" -m medical_experiments validate-manifest "${SEMANTIC_MANIFEST}" --check-paths --require-semantics
  "${LAFITE_PYTHON}" -m medical_experiments export-semantic-scores --config "${CONFIG_DIR}/exp3-export-scores.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments llm-necessity --config "${CONFIG_DIR}/exp3-evaluate.yaml"
  echo "experiment 3 completed in $((SECONDS - started)) seconds"
}

case "${MODE}" in
  exp1) run_exp1 ;;
  exp2) run_exp2 ;;
  exp3) run_exp3 ;;
  all) run_exp1; run_exp2; run_exp3 ;;
  results) "${LAFITE_PYTHON}" "${SCRIPT_DIR}/show-qwen3vl-results.py" "${MEDICAL_OUTPUT_ROOT}" ;;
  *) echo "usage: $0 {exp1|exp2|exp3|all|results}" >&2; exit 2 ;;
esac
