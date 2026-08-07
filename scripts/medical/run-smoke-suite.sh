#!/usr/bin/env bash
set -euo pipefail

LAFITE_PYTHON="${LAFITE_PYTHON:-python3}"

"${LAFITE_PYTHON}" -m medical_experiments make-smoke-data \
  --config experiments/medical-diagnostics/smoke-data.yaml
"${LAFITE_PYTHON}" -m medical_experiments make-interventions \
  --config experiments/medical-diagnostics/make-interventions-smoke.yaml
"${LAFITE_PYTHON}" -m medical_experiments make-interventions \
  --config experiments/medical-diagnostics/make-normal-interventions-smoke.yaml
"${LAFITE_PYTHON}" -m medical_experiments detail-probe \
  --config experiments/medical-diagnostics/exp1-detail-probe-smoke.yaml
"${LAFITE_PYTHON}" -m medical_experiments shortcut-eval \
  --config experiments/medical-diagnostics/exp2-shortcut-smoke.yaml
"${LAFITE_PYTHON}" -m medical_experiments llm-necessity \
  --config experiments/medical-diagnostics/exp3-llm-necessity-smoke.yaml
