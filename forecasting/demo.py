import argparse
import json
from pathlib import Path
from forecasting.data import ingest, download_months
from forecasting.train import train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    paths = (
        download_months(a.raw_dir)
        if a.download
        else [a.raw_dir / f"green_2025-{m:02}.parquet" for m in (1, 2, 3)]
    )
    print(json.dumps(ingest(paths, Path("data/tlc_hourly.parquet")), indent=2), flush=True)
    r = train(Path("data/tlc_hourly.parquet"), Path("work/tlc"))
    import shutil

    Path("evidence").mkdir(exist_ok=True)
    for name in ["training.json", "model_metrics.csv", "holdout_predictions.csv"]:
        shutil.copy(Path("work/tlc") / name, Path("evidence") / name)
    print(json.dumps(r["selection"], indent=2))
