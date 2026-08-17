#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# User settings: edit this block once on the GPU server. Exported environment
# variables with the same names take precedence over these defaults.
# -----------------------------------------------------------------------------
USER_BMAD_DATA_ROOT="/data/oss_bucket_0/yunlin/dataset/bmad"
USER_SEMANTIC_MANIFEST=""
USER_LAFITE_RUN_ROOT="/data/oss_bucket_0/yunlin/dataset/bmad/lafite-runs"
USER_BRAIN_LAFITE_CHECKPOINT=""
USER_LIVER_LAFITE_CHECKPOINT=""
# Leave thresholds empty to read each model's validation calibration JSON.
USER_BRAIN_LAFITE_MASK_THRESHOLD=""
USER_BRAIN_LAFITE_SCORE_THRESHOLD=""
USER_LIVER_LAFITE_MASK_THRESHOLD=""
USER_LIVER_LAFITE_SCORE_THRESHOLD=""
USER_MEDICAL_OUTPUT_ROOT="/data/oss_bucket_0/yunlin/dataset/bmad/medical-experiment-outputs"
USER_MLLM_MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
USER_SINGLE_GPU="cuda:0"
USER_QWEN_ATTENTION="sdpa"
USER_LAFITE_PYTHON="python3"
USER_HF_ENDPOINT="https://hf-mirror.com"
USER_HF_HOME="/data/oss_bucket_0/yunlin/dataset/bmad/huggingface-cache"
# -----------------------------------------------------------------------------

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${REPO_DIR}/experiments/medical-diagnostics/qwen3-vl-8b"

BMAD_DATA_ROOT="${BMAD_DATA_ROOT:-${USER_BMAD_DATA_ROOT}}"
LAFITE_DATA_DIR="${LAFITE_DATA_DIR:-${BMAD_DATA_ROOT}/prepared/lafite}"
LAFITE_RUN_ROOT="${LAFITE_RUN_ROOT:-${USER_LAFITE_RUN_ROOT}}"
export SEMANTIC_MANIFEST="${SEMANTIC_MANIFEST:-${USER_SEMANTIC_MANIFEST:-${BMAD_DATA_ROOT}/prepared/semantic-manifest.jsonl}}"
export MEDICAL_OUTPUT_ROOT="${MEDICAL_OUTPUT_ROOT:-${USER_MEDICAL_OUTPUT_ROOT:-${REPO_DIR}/outputs/medical-real}}"
export MLLM_MODEL_ID="${MLLM_MODEL_ID:-${USER_MLLM_MODEL_ID}}"
export QWEN_ATTENTION="${QWEN_ATTENTION:-${USER_QWEN_ATTENTION}}"
export HF_ENDPOINT="${HF_ENDPOINT:-${USER_HF_ENDPOINT}}"
export HF_HOME="${HF_HOME:-${USER_HF_HOME}}"
LAFITE_PYTHON="${LAFITE_PYTHON:-${USER_LAFITE_PYTHON}}"

BRAIN_LAFITE_CHECKPOINT="${BRAIN_LAFITE_CHECKPOINT:-${USER_BRAIN_LAFITE_CHECKPOINT:-${LAFITE_RUN_ROOT}/brain/checkpoints/ckpt_best.pth.tar}}"
LIVER_LAFITE_CHECKPOINT="${LIVER_LAFITE_CHECKPOINT:-${USER_LIVER_LAFITE_CHECKPOINT:-${LAFITE_RUN_ROOT}/liver/checkpoints/ckpt_best.pth.tar}}"
BRAIN_LAFITE_MASK_THRESHOLD="${BRAIN_LAFITE_MASK_THRESHOLD:-${USER_BRAIN_LAFITE_MASK_THRESHOLD}}"
BRAIN_LAFITE_SCORE_THRESHOLD="${BRAIN_LAFITE_SCORE_THRESHOLD:-${USER_BRAIN_LAFITE_SCORE_THRESHOLD}}"
LIVER_LAFITE_MASK_THRESHOLD="${LIVER_LAFITE_MASK_THRESHOLD:-${USER_LIVER_LAFITE_MASK_THRESHOLD}}"
LIVER_LAFITE_SCORE_THRESHOLD="${LIVER_LAFITE_SCORE_THRESHOLD:-${USER_LIVER_LAFITE_SCORE_THRESHOLD}}"

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

read_calibration_value() {
  local path="$1"
  local field="$2"
  if [[ ! -f "${path}" ]]; then
    echo "error: calibration file not found: ${path}" >&2
    echo "run: bash scripts/medical/train-bmad-lafite.sh calibrate" >&2
    exit 2
  fi
  "${LAFITE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "${path}" "${field}"
}

activate_modality() {
  local name="$1"
  local need_thresholds="${2:-false}"
  export ACTIVE_EXPERIMENT_ROOT="${MEDICAL_OUTPUT_ROOT}/${name}"
  export ACTIVE_MEDICAL_MANIFEST="${LAFITE_DATA_DIR}/${name}-manifest.jsonl"
  if [[ "${name}" == "brain" ]]; then
    export ACTIVE_LAFITE_CONFIG="${REPO_DIR}/experiments/BMAD-Brain-LafitE/config.yaml"
    export ACTIVE_LAFITE_CHECKPOINT="${BRAIN_LAFITE_CHECKPOINT}"
    export ACTIVE_LAFITE_MASK_THRESHOLD="${BRAIN_LAFITE_MASK_THRESHOLD}"
    export ACTIVE_LAFITE_SCORE_THRESHOLD="${BRAIN_LAFITE_SCORE_THRESHOLD}"
  else
    export ACTIVE_LAFITE_CONFIG="${REPO_DIR}/experiments/BMAD-Liver-LafitE/config.yaml"
    export ACTIVE_LAFITE_CHECKPOINT="${LIVER_LAFITE_CHECKPOINT}"
    export ACTIVE_LAFITE_MASK_THRESHOLD="${LIVER_LAFITE_MASK_THRESHOLD}"
    export ACTIVE_LAFITE_SCORE_THRESHOLD="${LIVER_LAFITE_SCORE_THRESHOLD}"
  fi
  require_var ACTIVE_MEDICAL_MANIFEST
  require_var ACTIVE_LAFITE_CHECKPOINT
  if [[ ! -f "${ACTIVE_MEDICAL_MANIFEST}" ]]; then
    echo "error: modality manifest not found: ${ACTIVE_MEDICAL_MANIFEST}" >&2
    echo "run: bash scripts/medical/train-bmad-lafite.sh prepare" >&2
    exit 2
  fi
  if [[ ! -f "${ACTIVE_LAFITE_CHECKPOINT}" ]]; then
    echo "error: modality checkpoint not found: ${ACTIVE_LAFITE_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ "${need_thresholds}" == "true" ]]; then
    local calibration="${LAFITE_RUN_ROOT}/${name}/calibration.json"
    if [[ -z "${ACTIVE_LAFITE_MASK_THRESHOLD}" ]]; then
      ACTIVE_LAFITE_MASK_THRESHOLD="$(read_calibration_value "${calibration}" mask_threshold)"
      export ACTIVE_LAFITE_MASK_THRESHOLD
    fi
    if [[ -z "${ACTIVE_LAFITE_SCORE_THRESHOLD}" ]]; then
      ACTIVE_LAFITE_SCORE_THRESHOLD="$(read_calibration_value "${calibration}" score_threshold)"
      export ACTIVE_LAFITE_SCORE_THRESHOLD
    fi
  fi
}

run_exp1_one() {
  local name="$1"
  activate_modality "${name}" false
  "${LAFITE_PYTHON}" -m medical_experiments validate-manifest "${ACTIVE_MEDICAL_MANIFEST}" --check-paths --require-mask
  "${LAFITE_PYTHON}" -m medical_experiments export-lafite --config "${CONFIG_DIR}/exp1-export-lafite.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments export-mllm --config "${CONFIG_DIR}/exp1-export-qwen.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments merge-features --config "${CONFIG_DIR}/exp1-merge.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments detail-probe --config "${CONFIG_DIR}/exp1-probe.yaml"
}

run_exp1() {
  local started="${SECONDS}"
  run_exp1_one brain
  run_exp1_one liver
  echo "experiment 1 completed in $((SECONDS - started)) seconds"
}

run_exp2_one() {
  local name="$1"
  activate_modality "${name}" true
  "${LAFITE_PYTHON}" -m medical_experiments make-interventions --config "${CONFIG_DIR}/exp2-positive.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments make-interventions --config "${CONFIG_DIR}/exp2-normal.yaml"
  "${LAFITE_PYTHON}" -m medical_experiments shortcut-eval --config "${CONFIG_DIR}/exp2-evaluate.yaml"
}

run_exp2() {
  local started="${SECONDS}"
  run_exp2_one brain
  run_exp2_one liver
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
