"""Rollback with a validation gate; drift triggers review, not automatic promotion."""

import json
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
from forecasting.train import atomic_json, metrics


def deploy(root: Path, candidate: dict):
    current = json.loads((root / "deployment.json").read_text())
    if set(candidate) != set(current["active"]):
        raise ValueError("Candidate must include every active horizon")
    for h, new in candidate.items():
        if not np.isfinite(new["cv_mae"]) or new["cv_mae"] > current["active"][h]["cv_mae"] * 1.05:
            raise ValueError("Validation regression")
        import hashlib

        path = root / new["file"]
        if path.parent.resolve() != root.resolve() or not path.is_file():
            raise ValueError("Missing or invalid candidate artifact")
        if hashlib.sha256(path.read_bytes()).hexdigest() != new.get("sha256"):
            raise ValueError("Candidate artifact integrity failure")
    atomic_json(root / "deployment.json", {"active": candidate, "previous": current["active"]})


def rollback(root: Path):
    d = json.loads((root / "deployment.json").read_text())
    if not d.get("previous"):
        raise ValueError("No previous deployment")
    atomic_json(root / "deployment.json", {"active": d["previous"], "previous": d["active"]})


def drift(reference, current):
    result = {}
    for c in reference:
        stat, p = ks_2samp(reference[c], current[c])
        result[c] = {"ks": float(stat), "p_value": float(p), "review": bool(stat > 0.2)}
    return {
        "features": result,
        "retraining": "Review if KS>.2, source freshness fails, or matured rolling MAE exceeds validation MAE by 25%; backtest candidate and retain rollback before promotion",
    }


def matured_performance(actual, prediction, reference_mae):
    m = metrics(actual, prediction)
    return {**m, "review": m["mae"] > 1.25 * reference_mae}
