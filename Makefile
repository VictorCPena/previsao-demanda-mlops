PY ?= .venv/bin/python
.PHONY: backtest backtest-smoke load-study load-smoke setup test train monitor smoke api mlflow evaluate
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

evaluate:
	$(PY) src/evaluate.py

backtest:
	$(PY) src/backtest.py

load-study:
	$(PY) src/load_study.py --requests 60 --concurrency 1 4 8

load-smoke:
	$(PY) src/load_study.py --requests 8 --concurrency 1 2

backtest-smoke:
	$(PY) src/backtest.py --stride 14

# Execute depois de make train + make backtest (stride padrão 7).
calibrate:
	$(PY) src/uncertainty.py

# Build local: 2 workers, artefatos montados somente leitura. Requer Docker.
container:
	docker compose up --build
