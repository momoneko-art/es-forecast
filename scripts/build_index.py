"""Main pipeline: fetch live data, update history, (re)train per-station models, write data.json."""
import csv
import json
import os
import sys
from datetime import datetime

from stations import STATIONS
from climatology import jst_now, day_of_year, climatology_index
import fetch_pskreporter
import fetch_nict
import fetch_noaa
import heatmap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH = os.path.join(ROOT, "history.csv")
DATA_JSON_PATH = os.path.join(ROOT, "data.json")
DEBUG_JSON_PATH = os.path.join(ROOT, "debug.json")

MAX_HISTORY_ROWS = 3000
MIN_ROWS_FOR_MODEL = 60

HISTORY_FIELDS = ["timestamp_utc", "kp"] + [f"{s['id']}_6m" for s in STATIONS] + [f"{s['id']}_10m" for s in STATIONS]


def read_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_history(rows):
    rows = rows[-MAX_HISTORY_ROWS:]
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def try_train_models(history_rows):
    """Train a tiny Ridge regressor per station predicting 6m spot count from time/kp features.
    Returns dict station_id -> {"model": fitted_estimator, "n_samples": int} or {} if unavailable."""
    if len(history_rows) < MIN_ROWS_FOR_MODEL:
        return {}
    try:
        import numpy as np
        from sklearn.linear_model import Ridge
    except ImportError:
        return {}

    X = []
    targets = {s["id"]: [] for s in STATIONS}
    for row in history_rows:
        try:
            ts = datetime.fromisoformat(row["timestamp_utc"])
        except (KeyError, ValueError):
            continue
        doy = day_of_year(ts)
        hour = ts.hour + ts.minute / 60
        import math
        feat = [
            math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365),
            math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
            float(row.get("kp") or 0),
        ]
        X.append(feat)
        for s in STATIONS:
            try:
                targets[s["id"]].append(float(row.get(f"{s['id']}_6m") or 0))
            except ValueError:
                targets[s["id"]].append(0.0)

    if len(X) < MIN_ROWS_FOR_MODEL:
        return {}

    X = np.array(X)
    models = {}
    for s in STATIONS:
        y = np.array(targets[s["id"]])
        if y.std() < 1e-6:
            continue
        model = Ridge(alpha=2.0)
        model.fit(X, y)
        models[s["id"]] = {"model": model, "n_samples": len(X)}
    return models


def predict_expected_count(model_info, dt, kp):
    import math
    doy = day_of_year(dt)
    hour = dt.hour + dt.minute / 60
    feat = [[
        math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365),
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        float(kp or 0),
    ]]
    pred = model_info["model"].predict(feat)[0]
    return max(0.0, float(pred))


TREND_THRESHOLD = 3.0  # points; smaller deltas are shown as "flat" to avoid noisy flicker


def read_prev_es_index():
    """Best-effort read of the es_index each station had last cycle, from the
    data.json we are about to overwrite. Used only to compute up/down/flat trend
    arrows; any failure (missing file, first run, schema drift) degrades to no
    trend info rather than breaking the pipeline."""
    try:
        with open(DATA_JSON_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        return {s["id"]: s["es_index"] for s in prev.get("stations", []) if "id" in s and "es_index" in s}
    except Exception:  # noqa: BLE001 - degrade gracefully
        return {}


def run():
    now_jst = jst_now()
    now_utc_iso = datetime.utcnow().isoformat()

    prev_es_index = read_prev_es_index()

    noaa = fetch_noaa.run()
    kp = noaa.get("kp") if noaa.get("status") == "ok" else None

    psk_result, psk_debug = fetch_pskreporter.run()
    nict_result = fetch_nict.run()

    # --- append this sample to history ---
    row = {"timestamp_utc": now_utc_iso, "kp": kp if kp is not None else ""}
    for s in STATIONS:
        band6 = psk_result.get("6m", {})
        band10 = psk_result.get("10m", {})
        row[f"{s['id']}_6m"] = band6.get("region_counts", {}).get(s["id"], 0) if band6.get("status") == "ok" else 0
        row[f"{s['id']}_10m"] = band10.get("region_counts", {}).get(s["id"], 0) if band10.get("status") == "ok" else 0

    history_rows = read_history()
    history_rows.append(row)
    write_history(history_rows)

    models = try_train_models(history_rows)

    nict_stations = nict_result.get("stations", {}) if nict_result.get("status") == "ok" else {}

    stations_out = []
    for s in STATIONS:
        clima = climatology_index(now_jst, s, kp)
        count6 = int(row[f"{s['id']}_6m"])
        count10 = int(row[f"{s['id']}_10m"])
        evidence = count6 + 0.5 * count10
        evidence_boost = min(1.0, evidence / 8.0)

        model_used = False
        model_note = "climatology_only"
        if s["id"] in models:
            expected = predict_expected_count(models[s["id"]], now_jst, kp)
            model_used = True
            model_note = f"ridge(n={models[s['id']]['n_samples']})"
            if expected > 0.5:
                surprise = min(2.0, count6 / expected)
                evidence_boost = max(evidence_boost, min(1.0, (surprise - 0.5)))

        # NICT ionosonde reading (ground-truth measurement) - the strongest single
        # signal when available. foEs (MHz) is normalized into a 0-1 boost; a
        # confirmed "Qui." (no Es echo) is treated as confident zero, and "unknown"
        # (site unreachable / '?' on the page) simply falls back to the PSKReporter
        # evidence computed above.
        nict_station = nict_stations.get(s["id"])
        nict_boost = 0.0
        foes_mhz = None
        nict_status = "unavailable"
        if nict_station:
            nict_status = nict_station["esp_status"]
            foes_mhz = nict_station["foes_mhz"]
            if nict_status == "quiet":
                nict_boost = 0.0
            elif nict_status == "ok" and foes_mhz is not None:
                nict_boost = max(0.0, min(1.0, (foes_mhz - 2.0) / 8.0))

        # Combine PSKReporter-derived evidence and NICT evidence with a noisy-OR:
        # either strong real propagation reports OR a strong measured foEs should
        # independently be able to push the index up, without double-counting when
        # both agree.
        combined_boost = 1 - (1 - evidence_boost) * (1 - nict_boost)

        es_index = clima * (1 + 0.6 * combined_boost)
        es_index = max(0.0, min(100.0, es_index))

        prev_index = prev_es_index.get(s["id"])
        trend = "unknown"
        trend_delta = None
        if prev_index is not None:
            trend_delta = round(es_index - prev_index, 1)
            if trend_delta > TREND_THRESHOLD:
                trend = "up"
            elif trend_delta < -TREND_THRESHOLD:
                trend = "down"
            else:
                trend = "flat"

        stations_out.append({
            "id": s["id"], "name": s["name"], "loc": s["loc"], "lat": s["lat"], "lon": s["lon"],
            "es_index": round(es_index, 1),
            "trend": trend, "trend_delta": trend_delta,
            "climatology_index": round(clima, 1),
            "live_evidence": {"ft8_6m_spots_15min": count6, "ft8_10m_spots_15min": count10},
            "nict": {
                "status": nict_status,
                "foes_mhz": foes_mhz,
                "storm": nict_station["storm"] if nict_station else None,
                "dis": nict_station["dis"] if nict_station else None,
            },
            "model": model_note,
        })

    kp_history = []
    for r in history_rows[-150:]:
        if r.get("kp") not in (None, ""):
            try:
                kp_history.append({"t": r["timestamp_utc"], "kp": float(r["kp"])})
            except ValueError:
                pass

    # Nationwide heatmap: spatially interpolates the same climatology + evidence
    # model used for the 4 stations above across a grid covering all of Japan,
    # using every individual PSKReporter receiver location (not just the 4-bucket
    # counts) plus the NICT foEs readings as anchors.
    all_psk_points = []
    for band in psk_result.values():
        if band.get("status") == "ok":
            all_psk_points.extend(band.get("heatmap_points", []))
    heatmap_grid = heatmap.compute(now_jst, kp, all_psk_points, nict_stations)

    EXCLUDE_KEYS = {"region_counts_raw", "heatmap_points"}
    data = {
        "generated_at": now_utc_iso + "Z",
        "generated_at_jst": now_jst.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "kp": {"status": noaa.get("status"), "value": kp, "time_tag": noaa.get("time_tag")},
        "kp_history": kp_history,
        "pskreporter": {k: {kk: vv for kk, vv in v.items() if kk not in EXCLUDE_KEYS} for k, v in psk_result.items()},
        "nict": {"status": nict_result.get("status"), "stations": nict_stations},
        "stations": stations_out,
        "heatmap": heatmap_grid,
        "history_rows": len(history_rows),
        "models_trained": sorted(models.keys()),
    }

    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    with open(DEBUG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"pskreporter_debug": psk_debug, "nict_findings": nict_result}, f, ensure_ascii=False, indent=2)

    print(json.dumps({"stations": [(s["id"], s["es_index"]) for s in stations_out], "kp": kp,
                       "history_rows": len(history_rows), "models_trained": list(models.keys())}, ensure_ascii=False))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run()
