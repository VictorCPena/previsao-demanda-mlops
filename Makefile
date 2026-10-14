PY ?= .venv/bin/python
.PHONY: setup test train monitor smoke api mlflow
setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt
test:
	$(PY) -m pytest tests -q -W error::DeprecationWarning
train:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MLFLOW_ENABLE_TELEMETRY=false $(PY) src/train.py
monitor:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(PY) src/monitor.py
smoke:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $(PY) src/smoke.py
api:
	$(PY) -m uvicorn serve:app --app-dir src --host 127.0.0.1 --port 8000
mlflow:
	$(PY) -m mlflow server --backend-store-uri sqlite:///$(CURDIR)/work/mlflow.db --host 127.0.0.1 --port 5000
