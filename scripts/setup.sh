#!/usr/bin/env bash
set -euo pipefail
export DO_NOT_TRACK=true
export DBT_SEND_ANONYMOUS_USAGE_STATS=false
export HF_HUB_DISABLE_TELEMETRY=1
export MLFLOW_DISABLE_TELEMETRY=true
cd "$(dirname "$0")/.."
"${PYTHON_BIN:-python3}" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
if [[ "$(uname -s)" == "Linux" ]]; then
  python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
fi
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "x86_64" ]]; then
  echo "Intel Mac profile: scikit-learn neural baseline; PyTorch remains available on supported platforms."
fi
python -m pip install -r requirements-portfolio.txt
python -m pip check
