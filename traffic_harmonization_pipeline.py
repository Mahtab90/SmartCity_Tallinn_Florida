#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic Dataset Harmonization Pipeline (with timestamps)
=======================================================

Outputs:
- data_out/florida_harmonized.csv
- data_out/tallinn_harmonized.csv

Now preserves timestamp columns when available:
- `timestamp_utc` for Florida
- `measurement_time_utc` for Tallinn (renamed to `timestamp_utc` in output for parity)
"""

from __future__ import annotations
import os, math, warnings
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

@dataclass
class Config:
    florida_input_csv: str = "data_in/florida_tmc.csv"
    tallinn_detectors_csv: str = "data_in/tallinn_detectors.csv"
    tallinn_restrictions_geojson: str = "data_in/tallinn_restrictions.geojson"
    tz: str = "UTC"
    tallinn_metric_crs: str = "EPSG:3301"
    out_dir: str = "data_out"

CFG = Config()

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_ts(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def cyclical_time_features(ts: pd.Series, period_minutes: int = 1440):
    tod = ts.dt.hour * 60 + ts.dt.minute
    rad = 2 * math.pi * tod / period_minutes
    return np.sin(rad), np.cos(rad)

def derive_weekday(ts: pd.Series):
    return ts.dt.weekday

def etl_florida(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    def get(colname, fallback=None):
        if colname in df.columns: return df[colname]
        for k, v in cols.items():
            if k == colname.lower(): return df[v]
        return fallback

    ts = parse_ts(get("timestamp_utc"))
    speed = get("speed")
    ref = get("reference_speed")
    seglen_km = get("segment_length_km")
    tmc_id = get("tmc_segment_id")
    incident_flag = get("incident_flag", pd.Series(np.nan, index=df.index))
    travel_time = get("travel_time_minutes")
    if travel_time is None or travel_time.isna().all():
        travel_time = (seglen_km / speed) * 60.0

    # time features (safe if ts exists)
    if ts is not None and not ts.isna().all():
        time_sin, time_cos = cyclical_time_features(ts)
        weekday = derive_weekday(ts)
    else:
        # fallback zeros
        time_sin = pd.Series(0.0, index=df.index)
        time_cos = pd.Series(1.0, index=df.index)
        weekday = pd.Series(pd.NA, index=df.index, dtype="Int64")

    out = pd.DataFrame({
        "segment_id": tmc_id.astype(str),
        "speed": speed.astype(float),
        "reference_speed": ref.astype(float) if ref is not None else np.nan,
        "travel_time_minutes": travel_time.astype(float),
        "incident": incident_flag.fillna(0).astype(int),
        "weekday": weekday.astype("Int64"),
        "time_sin": time_sin.astype(float),
        "time_cos": time_cos.astype(float),
        "source": "FLORIDA_TMC",
    })
    # Preserve timestamp where available
    if ts is not None:
        out["timestamp_utc"] = ts.astype("datetime64[ns]")
    return out.dropna(subset=["segment_id", "speed", "travel_time_minutes"])

def tallinn_reference_speed(avg_speed_kmh: pd.Series, relative_pct: pd.Series) -> pd.Series:
    rel = relative_pct.astype(float) / 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = avg_speed_kmh.astype(float) / rel.replace(0, np.nan)
    return ref

def to_gdf_points(df: pd.DataFrame, lon_col="lon", lat_col="lat", crs="EPSG:4326"):
    return gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=crs)

def load_tallinn_detectors_csv(path: str) -> gpd.GeoDataFrame:
    d = pd.read_csv(path)
    if "measurement_time_utc" in d.columns:
        d["measurement_time_utc"] = parse_ts(d["measurement_time_utc"])
    return to_gdf_points(d, "lon", "lat", "EPSG:4326")

def load_tallinn_restrictions_geojson(path: str) -> gpd.GeoDataFrame:
    gj = gpd.read_file(path)
    for c in ["date_from_utc", "date_to_utc"]:
        if c in gj.columns:
            gj[c] = parse_ts(gj[c])
    return gj.to_crs("EPSG:4326")

def spatial_temporal_join_tallinn(det_gdf: gpd.GeoDataFrame, rest_gdf: gpd.GeoDataFrame, metric_crs: str = "EPSG:3301", max_dist_m: float = 100.0) -> pd.DataFrame:
    det_m = det_gdf.to_crs(metric_crs).copy()
    rest_m = rest_gdf.to_crs(metric_crs).copy()
    rest_m["geom_buff"] = rest_m.geometry.buffer(max_dist_m)
    rest_sindex = rest_m.sindex

    incident_flags, segment_ids, matched_rest_ids = [], [], []
    for idx, row in det_m.iterrows():
        geom = row.geometry
        ts = row.get("measurement_time_utc", pd.NaT)
        cand_idx = list(rest_sindex.intersection(geom.bounds))
        hit, rid = 0, None
        for j in cand_idx:
            rr = rest_m.iloc[j]
            df, dt = rr.get("date_from_utc", pd.NaT), rr.get("date_to_utc", pd.NaT)
            if pd.isna(ts) or pd.isna(df) or pd.isna(dt): 
                continue
            if not (df <= ts <= dt): 
                continue
            if rr["geom_buff"].contains(geom):
                hit, rid = 1, rr.get("restriction_id", f"R{j}")
                break
        incident_flags.append(hit)
        det_id = row.get("detector_id", f"D{idx}")
        direction = row.get("direction", "NA")
        sid = f"{rid}|{det_id}|{direction}" if rid else f"{det_id}|{direction}"
        segment_ids.append(sid)
        matched_rest_ids.append(rid if rid else "")
    out = det_gdf.copy()
    out["incident"] = incident_flags
    out["segment_id"] = segment_ids
    out["matched_restriction_id"] = matched_rest_ids
    return out

def etl_tallinn(detectors_csv: str, restrictions_geojson: str, metric_crs: str = "EPSG:3301") -> pd.DataFrame:
    det = load_tallinn_detectors_csv(detectors_csv)
    rest = load_tallinn_restrictions_geojson(restrictions_geojson)
    det["reference_speed"] = tallinn_reference_speed(det["average_speed_kmh"], det["relative_speed_pct"])
    det_aug = spatial_temporal_join_tallinn(det, rest, metric_crs=metric_crs, max_dist_m=100.0)

    seglen_map = {}
    if "restriction_id" in rest.columns and "segment_length_m" in rest.columns:
        tmp = rest[["restriction_id", "segment_length_m"]].dropna().groupby("restriction_id")["segment_length_m"].mean()
        seglen_map = tmp.to_dict()
    det_aug["segment_length_km"] = det_aug["matched_restriction_id"].map(lambda rid: seglen_map.get(rid, np.nan)/1000.0 if rid else np.nan)

    det_aug["speed"] = det_aug["average_speed_kmh"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        det_aug["travel_time_minutes"] = (det_aug["segment_length_km"] / det_aug["speed"]) * 60.0

    # Timestamps and time features
    if "measurement_time_utc" in det_aug.columns:
        ts = det_aug["measurement_time_utc"]
        time_sin, time_cos = cyclical_time_features(ts)
        det_aug["weekday"] = derive_weekday(ts).astype("Int64")
        det_aug["time_sin"] = time_sin
        det_aug["time_cos"] = time_cos
        det_aug["timestamp_utc"] = ts
    else:
        det_aug["weekday"] = pd.Series(pd.NA, index=det_aug.index, dtype="Int64")
        det_aug["time_sin"] = 0.0
        det_aug["time_cos"] = 1.0

    out = det_aug[["segment_id","speed","reference_speed","travel_time_minutes",
                   "incident","weekday","time_sin","time_cos"]].copy()
    if "timestamp_utc" in det_aug.columns:
        out["timestamp_utc"] = det_aug["timestamp_utc"].astype("datetime64[ns]")
    out["source"] = "TALLINN_DATEX"
    return out

def main():
    os.makedirs(CFG.out_dir, exist_ok=True)
    if os.path.exists(CFG.florida_input_csv):
        df_fl = etl_florida(CFG.florida_input_csv)
        df_fl.to_csv(os.path.join(CFG.out_dir, "florida_harmonized.csv"), index=False)
        print("[OK] data_out/florida_harmonized.csv", len(df_fl))
    else:
        print("[WARN] Missing", CFG.florida_input_csv)

    if os.path.exists(CFG.tallinn_detectors_csv) and os.path.exists(CFG.tallinn_restrictions_geojson):
        df_ta = etl_tallinn(CFG.tallinn_detectors_csv, CFG.tallinn_restrictions_geojson, metric_crs=CFG.tallinn_metric_crs)
        df_ta.to_csv(os.path.join(CFG.out_dir, "tallinn_harmonized.csv"), index=False)
        print("[OK] data_out/tallinn_harmonized.csv", len(df_ta))
    else:
        print("[WARN] Missing Tallinn inputs")

if __name__ == "__main__":
    main()
