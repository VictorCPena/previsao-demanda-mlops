"""Promove uma versão local existente; mantém alias MLflow e permite rollback."""
import argparse
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from data import ROOT


def promote(name, work=None):
    work = Path(work or ROOT / "work")
    path = work / "deployment.json"
    state = json.loads(path.read_text())
    if name not in state["versions"]:
        raise ValueError(f"Escolha: {list(state['versions'])}")
    chosen = state["versions"][name]
    if not (work / chosen["file"]).is_file():
        raise ValueError("Artefato local ausente")
    mlflow.set_tracking_uri(state["tracking_uri"])
    mlflow.set_registry_uri(state["tracking_uri"])
    MlflowClient().set_registered_model_alias("demanda-diaria", "champion", chosen["version"])
    previous = state["active"]
    state["active"] = chosen
    state["previous"] = previous
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)
    # A API usa o arquivo local como fonte de verdade; alias serve como rastreabilidade.
    return {"previous": previous, "active": chosen}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=["seasonal_naive", "ridge", "hist_gradient_boosting"])
    args = p.parse_args()
    print(json.dumps(promote(args.model), indent=2))
