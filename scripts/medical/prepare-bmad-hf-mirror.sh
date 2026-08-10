#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# User settings. Review BMAD/source-data terms first, then change acceptance to
# "true". Environment variables with the same names still take precedence.
# -----------------------------------------------------------------------------
USER_ACCEPT_BMAD_LICENSE="false"
USER_BMAD_DATA_ROOT="/data/oss_bucket_0/yunlin/dataset/bmad"
USER_HF_ENDPOINT="https://hf-mirror.com"
USER_HF_REPO_ID="code-dev05/BMAD"
USER_HF_REVISION="22cb97e1cfeeb9f677c6c57387347ce727625610"
# Already downloaded BMAD archive; `prepare` never downloads it again.
USER_BMAD_ARCHIVE="/home/yunlin.lhl/dataset/bmad.zip"
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
BMAD_ARCHIVE="${BMAD_ARCHIVE:-${USER_BMAD_ARCHIVE:-${BMAD_DATA_ROOT}/downloads/bmad.zip}}"
MAX_SLICES_PER_PATIENT="${MAX_SLICES_PER_PATIENT:-${USER_MAX_SLICES_PER_PATIENT}}"
SEMANTIC_PATIENTS_PER_EDGE="${SEMANTIC_PATIENTS_PER_EDGE:-${USER_SEMANTIC_PATIENTS_PER_EDGE}}"
PREPARE_PYTHON="${PREPARE_PYTHON:-${USER_PREPARE_PYTHON}}"
EXPECTED_BMAD_ARCHIVE_BYTES="21840892398"
export HF_ENDPOINT

if [[ "${ACCEPT_BMAD_LICENSE}" != "true" ]]; then
  echo "error: review https://github.com/DorisBao/BMAD and set USER_ACCEPT_BMAD_LICENSE=\"true\" in $0" >&2
  exit 2
fi

cd "${REPO_DIR}"

download_archive() {
  local download_url="${HF_ENDPOINT}/datasets/${HF_REPO_ID}/resolve/${HF_REVISION}/bmad.zip?download=true"
  local current_bytes="0"
  mkdir -p "$(dirname "${BMAD_ARCHIVE}")"
  if [[ -f "${BMAD_ARCHIVE}" ]]; then
    current_bytes="$("${PREPARE_PYTHON}" -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "${BMAD_ARCHIVE}")"
    if [[ "${current_bytes}" == "${EXPECTED_BMAD_ARCHIVE_BYTES}" ]]; then
      echo "BMAD archive is already complete: ${BMAD_ARCHIVE}"
      return
    fi
    echo "Resuming partial archive (${current_bytes}/${EXPECTED_BMAD_ARCHIVE_BYTES} bytes)"
  fi
  echo "Downloading BMAD archive with resume support:"
  echo "  ${download_url}"
  echo "  -> ${BMAD_ARCHIVE}"
  if command -v wget >/dev/null 2>&1; then
    wget --continue --tries=20 --timeout=30 "${download_url}" -O "${BMAD_ARCHIVE}"
  elif command -v curl >/dev/null 2>&1; then
    curl --location --continue-at - --retry 20 --retry-all-errors \
      --connect-timeout 30 --output "${BMAD_ARCHIVE}" "${download_url}"
  else
    echo "error: wget or curl is required for direct download" >&2
    exit 2
  fi
  current_bytes="$("${PREPARE_PYTHON}" -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "${BMAD_ARCHIVE}")"
  if [[ "${current_bytes}" != "${EXPECTED_BMAD_ARCHIVE_BYTES}" ]]; then
    echo "error: downloaded archive has ${current_bytes} bytes; expected ${EXPECTED_BMAD_ARCHIVE_BYTES}" >&2
    echo "rerun '$0 download' to resume" >&2
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
