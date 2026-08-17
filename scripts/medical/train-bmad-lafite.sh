#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# User settings: edit this block once on the GPU server. Exported variables with
# the same names still take precedence, which keeps Slurm jobs reproducible.
# -----------------------------------------------------------------------------
USER_BMAD_DATA_ROOT="/data/oss_bucket_0/yunlin/dataset/bmad"
USER_LAFITE_RUN_ROOT="/data/oss_bucket_0/yunlin/dataset/bmad/lafite-runs"
USER_EFFICIENTNET_B4_WEIGHTS=""
USER_GPU="0"
USER_BATCH_SIZE="64"
USER_WORKERS="8"
USER_MAX_EPOCHS="1000"
USER_VAL_FREQ="50"
USER_SAVE_STEP="50"
USER_LAFITE_PYTHON="python3"
# -----------------------------------------------------------------------------

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BMAD_DATA_ROOT="${BMAD_DATA_ROOT:-${USER_BMAD_DATA_ROOT}}"
export LAFITE_DATA_DIR="${LAFITE_DATA_DIR:-${BMAD_DATA_ROOT}/prepared/lafite}"
export LAFITE_RUN_ROOT="${LAFITE_RUN_ROOT:-${USER_LAFITE_RUN_ROOT}}"
EFFICIENTNET_B4_WEIGHTS="${EFFICIENTNET_B4_WEIGHTS:-${USER_EFFICIENTNET_B4_WEIGHTS}}"
GPU="${GPU:-${USER_GPU}}"
BATCH_SIZE="${BATCH_SIZE:-${USER_BATCH_SIZE}}"
WORKERS="${WORKERS:-${USER_WORKERS}}"
MAX_EPOCHS="${MAX_EPOCHS:-${USER_MAX_EPOCHS}}"
VAL_FREQ="${VAL_FREQ:-${USER_VAL_FREQ}}"
SAVE_STEP="${SAVE_STEP:-${USER_SAVE_STEP}}"
LAFITE_PYTHON="${LAFITE_PYTHON:-${USER_LAFITE_PYTHON}}"
MEDICAL_MANIFEST="${MEDICAL_MANIFEST:-${BMAD_DATA_ROOT}/prepared/medical-manifest.jsonl}"
export LAFITE_CALIBRATION_DEVICE="${LAFITE_CALIBRATION_DEVICE:-cuda:0}"

cd "${REPO_DIR}"

prepare_data() {
  echo "[prepare] Creating Brain/Liver LafitE metadata from ${MEDICAL_MANIFEST}"
  "${LAFITE_PYTHON}" -m medical_experiments prepare-lafite-data \
    --manifest "${MEDICAL_MANIFEST}" \
    --output-dir "${LAFITE_DATA_DIR}" \
    --output "${LAFITE_DATA_DIR}/preparation-summary.json"
  echo "[prepare] Metadata ready in ${LAFITE_DATA_DIR}"
}

ensure_data() {
  if [[ ! -f "${LAFITE_DATA_DIR}/brain-train.jsonl" || ! -f "${LAFITE_DATA_DIR}/liver-train.jsonl" ]]; then
    prepare_data
  else
    echo "[prepare] Reusing existing LafitE metadata in ${LAFITE_DATA_DIR}"
  fi
}

train_one() {
  local name="$1"
  local title
  title="$(tr '[:lower:]' '[:upper:]' <<< "${name:0:1}")${name:1}"
  local config="${REPO_DIR}/experiments/BMAD-${title}-LafitE/config.yaml"
  local run_dir="${LAFITE_RUN_ROOT}/${name}"
  echo "[train] modality=${name} gpu=${GPU} epochs=${MAX_EPOCHS} batch=${BATCH_SIZE} workers=${WORKERS}"
  echo "[train] train metadata: ${LAFITE_DATA_DIR}/${name}-train.jsonl"
  echo "[train] validation metadata: ${LAFITE_DATA_DIR}/${name}-val.jsonl"
  if [[ -z "${EFFICIENTNET_B4_WEIGHTS}" ]]; then
    echo "[train] EfficientNet-B4 weights: automatic torch cache lookup/download"
  else
    echo "[train] EfficientNet-B4 weights: ${EFFICIENTNET_B4_WEIGHTS}"
  fi
  local command=(
    "${LAFITE_PYTHON}" main/train_val_diff.py
    -c "${config}"
    -g "${GPU}"
    --train-meta "${LAFITE_DATA_DIR}/${name}-train.jsonl"
    --val-meta "${LAFITE_DATA_DIR}/${name}-val.jsonl"
    --output-dir "${run_dir}"
    --epochs "${MAX_EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --val-freq "${VAL_FREQ}"
    --save-step "${SAVE_STEP}"
  )
  if [[ -n "${EFFICIENTNET_B4_WEIGHTS}" ]]; then
    if [[ ! -f "${EFFICIENTNET_B4_WEIGHTS}" ]]; then
      echo "error: EfficientNet-B4 weights not found: ${EFFICIENTNET_B4_WEIGHTS}" >&2
      exit 2
    fi
    command+=(--pretrained-model "${EFFICIENTNET_B4_WEIGHTS}")
  fi
  "${command[@]}"
}

calibrate_one() {
  local name="$1"
  local title
  title="$(tr '[:lower:]' '[:upper:]' <<< "${name:0:1}")${name:1}"
  local checkpoint="${LAFITE_RUN_ROOT}/${name}/checkpoints/ckpt_best.pth.tar"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "error: best checkpoint not found: ${checkpoint}" >&2
    exit 2
  fi
  if [[ "${name}" == "brain" ]]; then
    export BRAIN_LAFITE_CHECKPOINT="${checkpoint}"
  else
    export LIVER_LAFITE_CHECKPOINT="${checkpoint}"
  fi
  "${LAFITE_PYTHON}" -m medical_experiments calibrate-lafite \
    --config "${REPO_DIR}/experiments/BMAD-${title}-LafitE/calibrate.yaml"
}

show_status() {
  for name in brain liver; do
    echo "${name}:"
    echo "  train metadata: ${LAFITE_DATA_DIR}/${name}-train.jsonl"
    echo "  latest:         ${LAFITE_RUN_ROOT}/${name}/checkpoints/ckpt.pth.tar"
    echo "  best:           ${LAFITE_RUN_ROOT}/${name}/checkpoints/ckpt_best.pth.tar"
    echo "  calibration:    ${LAFITE_RUN_ROOT}/${name}/calibration.json"
  done
}

case "${MODE}" in
  prepare) prepare_data ;;
  brain) ensure_data; train_one brain ;;
  liver) ensure_data; train_one liver ;;
  calibrate-brain) ensure_data; calibrate_one brain ;;
  calibrate-liver) ensure_data; calibrate_one liver ;;
  calibrate) ensure_data; calibrate_one brain; calibrate_one liver ;;
  all)
    ensure_data
    train_one brain
    calibrate_one brain
    train_one liver
    calibrate_one liver
    ;;
  status) show_status ;;
  *)
    echo "usage: $0 {prepare|brain|liver|calibrate-brain|calibrate-liver|calibrate|all|status}" >&2
    exit 2
    ;;
esac
