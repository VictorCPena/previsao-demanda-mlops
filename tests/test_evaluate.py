import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import evaluate


def fixture():
    return pd.DataFrame({"day": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
                         "product_id": ["A", "A", "B", "B"],
                         "actual": [10, 20, 0, 0],
                         "prediction": [12, 18, 0, 0],
                         "baseline": [20, 10, 1, 1]})


def test_evaluate_detects_segments_and_zero_volume():
    result = evaluate(fixture())
    assert result["overall"]["observations"] == 4
    assert result["by_product"]["A"]["mae"] == 2
    assert result["by_product"]["A"]["improvement_vs_baseline_pct"] == 80
    assert result["by_product"]["B"]["wape"] is None
    assert result["products_worse_than_baseline"] == []


def test_evaluate_identifies_local_regression():
    frame = fixture()
    frame.loc[frame.product_id == "B", "prediction"] = 4
    result = evaluate(frame)
    assert result["products_worse_than_baseline"] == ["B"]


def test_evaluate_rejects_duplicate_product_day():
    frame = fixture()
    frame.loc[1, "day"] = frame.loc[0, "day"]
    with pytest.raises(ValueError, match="repetido"):
        evaluate(frame)


def test_evaluate_rejects_invalid_data():
    frame = fixture()
    frame.loc[0, "actual"] = -1
    with pytest.raises(ValueError, match="não negativo"):
        evaluate(frame)
