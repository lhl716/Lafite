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
# 0 keeps every slice. For a quick single-GPU pilot, try 4 or 8.
USER_MAX_SLICES_PER_PATIENT="0"
# 0 keeps one semantic sample per patient. For a pilot, try 25 per edge.
USER_SEMANTIC_PATIENTS_PER_EDGE="0"
USER_PREPARE_PYTHON="python3"
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACCEPT_BMAD_LICENSE="${ACCEPT_BMAD_LICENSE:-${USER_ACCEPT_BMAD_LICENSE}}"
BMAD_DATA_ROOT="${BMAD_DATA_ROOT:-${USER_BMAD_DATA_ROOT:-${REPO_DIR}/data/bmad}}"
HF_ENDPOINT="${HF_ENDPOINT:-${USER_HF_ENDPOINT}}"
HF_REPO_ID="${HF_REPO_ID:-${USER_HF_REPO_ID}}"
HF_REVISION="${HF_REVISION:-${USER_HF_REVISION}}"
MAX_SLICES_PER_PATIENT="${MAX_SLICES_PER_PATIENT:-${USER_MAX_SLICES_PER_PATIENT}}"
SEMANTIC_PATIENTS_PER_EDGE="${SEMANTIC_PATIENTS_PER_EDGE:-${USER_SEMANTIC_PATIENTS_PER_EDGE}}"
PREPARE_PYTHON="${PREPARE_PYTHON:-${USER_PREPARE_PYTHON}}"
export HF_ENDPOINT

if [[ "${ACCEPT_BMAD_LICENSE}" != "true" ]]; then
  echo "error: review https://github.com/DorisBao/BMAD and set USER_ACCEPT_BMAD_LICENSE=\"true\" in $0" >&2
  exit 2
fi

cd "${REPO_DIR}"
"${PREPARE_PYTHON}" -m medical_experiments prepare-bmad \
  --data-root "${BMAD_DATA_ROOT}" \
  --endpoint "${HF_ENDPOINT}" \
  --repo-id "${HF_REPO_ID}" \
  --revision "${HF_REVISION}" \
  --max-slices-per-patient "${MAX_SLICES_PER_PATIENT}" \
  --semantic-patients-per-edge "${SEMANTIC_PATIENTS_PER_EDGE}" \
  --accept-license

echo "medical manifest:  ${BMAD_DATA_ROOT}/prepared/medical-manifest.jsonl"
echo "semantic manifest: ${BMAD_DATA_ROOT}/prepared/semantic-manifest.jsonl"
echo "preparation report: ${BMAD_DATA_ROOT}/prepared/preparation-summary.json"
