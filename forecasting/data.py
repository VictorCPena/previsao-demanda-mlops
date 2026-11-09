"""TLC green taxi counts: UTC index, local calendar, origin-safe features."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from forecasting.fetch import download

FEATURES = ["zone", "hour", "weekday", "lag_origin", "lag_day", "lag_week", "mean24", "mean168"]


def ingest(paths, output: Path):
    frames = []
    sources = []
    invalid = 0
    for p in paths:
        p = Path(p)
        d = pd.read_parquet(p, columns=["lpep_pickup_datetime", "PULocationID"])
        ts = (
            pd.to_datetime(d.lpep_pickup_datetime)
            .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
            .dt.tz_convert("UTC")
        )
        keep = (
            ts.notna()
            & d.PULocationID.between(1, 263)
            & ts.between(
                pd.Timestamp("2025-01-01", tz="America/New_York"),
                pd.Timestamp("2025-04-01", tz="America/New_York"),
                inclusive="left",
            )
        )
        invalid += int((~keep).sum())
        frames.append(
            pd.DataFrame({"time": ts[keep].dt.floor("h"), "zone": d.PULocationID[keep].astype(int)})
        )
        sources.append({"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": len(d)})
    trips = pd.concat(frames, ignore_index=True)
    # Region selection uses only the first 14 days, never validation/holdout volume.
    start = pd.Timestamp("2025-01-01", tz="America/New_York").tz_convert("UTC")
    end = pd.Timestamp("2025-04-01", tz="America/New_York").tz_convert("UTC")
    zones = trips[trips.time < start + pd.Timedelta(days=14)].zone.value_counts().head(3).index.tolist()
    counts = trips.groupby(["zone", "time"]).size().rename("count")
    grid = pd.MultiIndex.from_product(
        [zones, pd.date_range(start, end, freq="h", inclusive="left")], names=["zone", "time"]
    )
    frame = counts.reindex(grid, fill_value=0).reset_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    manifest = {
        "source": "NYC TLC green taxi Jan–Mar 2025",
        "sources": sources,
        "discarded_invalid_or_out_of_window": invalid,
        "selected_zones": zones,
        "grid_rows": len(frame),
        "trips_selected": int(frame["count"].sum()),
        "aggregation": "UTC hour; all files Jan–Mar required; zero = no published trip",
        "limitations": "Published pickup demand, not unmet demand. Complete file assumed; source reporting gaps can mimic zero demand.",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return manifest


def features(frame, horizon):
    if horizon not in (1, 6, 24):
        raise ValueError("Supported direct horizons: 1,6,24 hours")
    f = frame.sort_values(["zone", "time"]).copy()
    f["time"] = pd.to_datetime(f.time, utc=True)
    if f.duplicated(["zone", "time"]).any() or not np.isfinite(f["count"]).all() or (f["count"] < 0).any():
        raise ValueError("Invalid hourly counts")
    for _, g in f.groupby("zone"):
        if not (g.time.diff().dropna() == pd.Timedelta(hours=1)).all():
            raise ValueError("Noncontiguous UTC hourly grid")
    local = f.time.dt.tz_convert("America/New_York")
    f["hour"] = local.dt.hour
    f["weekday"] = local.dt.dayofweek
    group = f.groupby("zone")["count"]
    for name, lag in [("lag_origin", horizon), ("lag_day", horizon + 23), ("lag_week", 168)]:
        f[name] = group.shift(lag)
    for size in (24, 168):
        f[f"mean{size}"] = group.transform(lambda s, window=size: s.shift(horizon).rolling(window).mean())
    f["origin"] = f.time - pd.Timedelta(hours=horizon)
    return f.dropna(subset=FEATURES).reset_index(drop=True)


def download_months(directory):
    paths = []
    for month in (1, 2, 3):
        p = directory / f"green_2025-{month:02}.parquet"
        download(f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-{month:02}.parquet", p)
        paths.append(p)
    return paths
