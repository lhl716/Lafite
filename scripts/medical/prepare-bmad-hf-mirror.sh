#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# User settings. Review BMAD/source-data terms first, then change acceptance to
# "true". Environment variables with the same names still take precedence.
# -----------------------------------------------------------------------------
USER_ACCEPT_BMAD_LICENSE="false"
USER_BMAD_DATA_ROOT=""
USER_HF_ENDPOINT="https://hf-mirror.com"
USER_HF_REPO_ID="code-dev05/BMAD"
USER_HF_REVISION="22cb97e1cfeeb9f677c6c57387347ce727625610"
# Optional explicit path/name for `hf` or `huggingface-cli`; empty auto-detects.
USER_HF_CLI=""
# Hugging Face Hub defaults to a short timeout; large mirror downloads need more.
USER_HF_HUB_DOWNLOAD_TIMEOUT="60"
# Leave empty to use data/bmad/downloads/bmad.zip.
USER_BMAD_ARCHIVE=""
# 0 keeps every slice. For a quick single-GPU pilot, try 4 or 8.
USER_MAX_SLICES_PER_PATIENT="0"
# 0 keeps one semantic sample per patient. For a pilot, try 25 per edge.
USER_SEMANTIC_PATIENTS_PER_EDGE="0"
USER_PREPARE_PYTHON="python3"
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODE="${1:-all}"

ACCEPT_BMAD_LICENSE="${ACCEPT_BMAD_LICENSE:-${USER_ACCEPT_BMAD_LICENSE}}"
BMAD_DATA_ROOT="${BMAD_DATA_ROOT:-${USER_BMAD_DATA_ROOT:-${REPO_DIR}/data/bmad}}"
HF_ENDPOINT="${HF_ENDPOINT:-${USER_HF_ENDPOINT}}"
HF_REPO_ID="${HF_REPO_ID:-${USER_HF_REPO_ID}}"
HF_REVISION="${HF_REVISION:-${USER_HF_REVISION}}"
HF_CLI="${HF_CLI:-${USER_HF_CLI}}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-${USER_HF_HUB_DOWNLOAD_TIMEOUT}}"
BMAD_ARCHIVE="${BMAD_ARCHIVE:-${USER_BMAD_ARCHIVE:-${BMAD_DATA_ROOT}/downloads/bmad.zip}}"
MAX_SLICES_PER_PATIENT="${MAX_SLICES_PER_PATIENT:-${USER_MAX_SLICES_PER_PATIENT}}"
SEMANTIC_PATIENTS_PER_EDGE="${SEMANTIC_PATIENTS_PER_EDGE:-${USER_SEMANTIC_PATIENTS_PER_EDGE}}"
PREPARE_PYTHON="${PREPARE_PYTHON:-${USER_PREPARE_PYTHON}}"
EXPECTED_BMAD_ARCHIVE_BYTES="21840892398"
export HF_ENDPOINT HF_HUB_DOWNLOAD_TIMEOUT

if [[ "${ACCEPT_BMAD_LICENSE}" != "true" ]]; then
  echo "error: review https://github.com/DorisBao/BMAD and set USER_ACCEPT_BMAD_LICENSE=\"true\" in $0" >&2
  exit 2
fi

cd "${REPO_DIR}"

download_archive() {
  local current_bytes="0"
  local download_dir
  local selected_cli="${HF_CLI}"
  download_dir="$(dirname "${BMAD_ARCHIVE}")"
  if [[ "$(basename "${BMAD_ARCHIVE}")" != "bmad.zip" ]]; then
    echo "error: BMAD_ARCHIVE must end with bmad.zip for Hugging Face CLI download" >&2
    exit 2
  fi
  mkdir -p "${download_dir}"
  if [[ -f "${BMAD_ARCHIVE}" ]]; then
    current_bytes="$("${PREPARE_PYTHON}" -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "${BMAD_ARCHIVE}")"
    if [[ "${current_bytes}" == "${EXPECTED_BMAD_ARCHIVE_BYTES}" ]]; then
      echo "BMAD archive is already complete: ${BMAD_ARCHIVE}"
      return
    fi
    echo "Resuming partial archive (${current_bytes}/${EXPECTED_BMAD_ARCHIVE_BYTES} bytes)"
  fi
  if [[ -z "${selected_cli}" ]]; then
    if command -v hf >/dev/null 2>&1; then
      selected_cli="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
      selected_cli="huggingface-cli"
    else
      echo "error: neither 'hf' nor 'huggingface-cli' is available" >&2
      echo "install/activate huggingface_hub, or set USER_HF_CLI to its executable" >&2
      exit 2
    fi
  fi
  if ! command -v "${selected_cli}" >/dev/null 2>&1; then
    echo "error: Hugging Face CLI is not executable: ${selected_cli}" >&2
    exit 2
  fi
  echo "Downloading ${HF_REPO_ID}/bmad.zip through ${HF_ENDPOINT} with ${selected_cli}"
  "${selected_cli}" download "${HF_REPO_ID}" bmad.zip \
    --repo-type dataset \
    --revision "${HF_REVISION}" \
    --local-dir "${download_dir}"
  current_bytes="$("${PREPARE_PYTHON}" -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "${BMAD_ARCHIVE}")"
  if [[ "${current_bytes}" != "${EXPECTED_BMAD_ARCHIVE_BYTES}" ]]; then
    echo "error: downloaded archive has ${current_bytes} bytes; expected ${EXPECTED_BMAD_ARCHIVE_BYTES}" >&2
    echo "rerun '$0 download' so Hugging Face CLI can resume" >&2
    exit 2
  fi
}

prepare_archive() {
  if [[ ! -f "${BMAD_ARCHIVE}" ]]; then
    echo "error: archive not found: ${BMAD_ARCHIVE}" >&2
    echo "run '$0 download' first, or set BMAD_ARCHIVE to an existing bmad.zip" >&2
    exit 2
  fi
  "${PREPARE_PYTHON}" -m medical_experiments prepare-bmad \
    --data-root "${BMAD_DATA_ROOT}" \
    --endpoint "${HF_ENDPOINT}" \
    --repo-id "${HF_REPO_ID}" \
    --revision "${HF_REVISION}" \
    --archive "${BMAD_ARCHIVE}" \
    --max-slices-per-patient "${MAX_SLICES_PER_PATIENT}" \
    --semantic-patients-per-edge "${SEMANTIC_PATIENTS_PER_EDGE}" \
    --accept-license
}

case "${MODE}" in
  download)
    download_archive
    ;;
  prepare)
    prepare_archive
    ;;
  all)
    download_archive
    prepare_archive
    ;;
  *)
    echo "usage: $0 {download|prepare|all}" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" != "download" ]]; then
  echo "medical manifest:  ${BMAD_DATA_ROOT}/prepared/medical-manifest.jsonl"
  echo "semantic manifest: ${BMAD_DATA_ROOT}/prepared/semantic-manifest.jsonl"
  echo "preparation report: ${BMAD_DATA_ROOT}/prepared/preparation-summary.json"
fi
