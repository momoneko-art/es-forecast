"""Main pipeline: fetch live data, update history, (re)train per-station models, write data.json."""
import csv
import json
import os
import sys
import time
from datetime import datetime

from stations import STATIONS
from climatology import jst_now, day_of_year, climatology_index, nict_floor_from_foes
import fetch_pskreporter
import fetch_nict
import fetch_noaa
import fetch_tropo
import fetch_jetstream
import heatmap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_PATH = os.path.join(ROOT, "history.csv")
DATA_JSON_PATH = os.path.join(ROOT, "data.json")
DEBUG_JSON_PATH = os.path.join(ROOT, "debug.json")

MAX_HISTORY_ROWS = 3000
MIN_ROWS_FOR_MODEL = 60

# jet250_kmh: EXPERIMENTAL shadow column (see fetch_jetstream.py) - appended
# at the end so old rows (written before this column existed) stay readable;
# csv.DictWriter fills missing keys in old rows with '' automatically.
HISTORY_FIELDS = (
    ["timestamp_utc", "kp"] + [f"{s['id']}_6m" for s in STATIONS] + [f"{s['id']}_10m" for s in STATIONS]
    + ["jet250_kmh"]
)


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
TREND_HISTORY_LEN = 4  # how many recent per-cycle trend steps to keep for the arrow-sequence UI

# NICT's ionosonde scrape occasionally comes back "unknown" for a single station
# for just one cycle (a transient sounding/parsing gap - confirmed 2026-09-02:
# Wakkanai showed "unknown" for one cycle while foEs had just been 11-12MHz
# moments earlier per NICT's own page, and a WebFetch of the live page seconds
# later showed a perfectly normal row). Rather than discarding that still-fresh
# evidence and reverting straight to climatology-only, carry the last known-good
# "ok" reading forward for a bounded grace window so a single blip doesn't hide a
# real, ongoing Es event from the dashboard.
NICT_CARRY_FORWARD_MAX_SECONDS = 45 * 60


def read_prev_stations():
    """Best-effort read of each station's last cycle es_index, short
    trend_history, and NICT reading (+ the timestamp that reading was actually
    obtained), from the data.json we are about to overwrite. Used to compute the
    up/down/flat trend arrow, to carry forward the rolling 4-entry trend history
    (see TREND_HISTORY_LEN), and to bridge single-cycle NICT scrape gaps (see
    NICT_CARRY_FORWARD_MAX_SECONDS); any failure (missing file, first run,
    schema drift) degrades to no prior info rather than breaking the pipeline."""
    try:
        with open(DATA_JSON_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        return {
            s["id"]: {
                "es_index": s.get("es_index"),
                "trend_history": s.get("trend_history") or [],
                "nict": s.get("nict"),
                "nict_as_of_ts": s.get("nict_as_of_ts"),
            }
            for s in prev.get("stations", []) if "id" in s
        }
    except Exception:  # noqa: BLE001 - degrade gracefully
        return {}


def read_prev_tropo():
    """Best-effort read of last cycle's data['tropo'] dict, used by fetch_tropo.run()
    to decide whether GFS has likely updated yet (see MIN_REFETCH_SECONDS there).
    Any failure (missing file, first run, schema drift) degrades to None, which
    fetch_tropo.run() treats as "always fetch"."""
    try:
        with open(DATA_JSON_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        return prev.get("tropo")
    except Exception:  # noqa: BLE001 - degrade gracefully
        return None


def summarize_status(psk_result, nict_result, noaa, kp_forecast, tropo_result):
    """Plain-language, non-technical summary of whether each upstream data
    source came back OK this cycle - NOT a raw error log. The point isn't for
    the user to "fix" anything (these are external services outside their
    control) but so a glance at the dashboard explains an odd-looking number
    ("ああ、今NICTが取れてないからだ") without needing to ask/investigate, and
    so a screenshot of this panel tells us immediately which upstream source
    to look at. level is one of "ok"/"warn"/"error" (mirrors the .pill
    ok/watch/vhigh CSS classes already used elsewhere in the UI)."""
    items = []

    psk_statuses = [b.get("status") for b in psk_result.values()]
    if psk_statuses and all(s == "ok" for s in psk_statuses):
        items.append({"name": "実伝播データ(PSKReporter)", "level": "ok", "note": "正常"})
    elif any(s == "ok" for s in psk_statuses):
        items.append({"name": "実伝播データ(PSKReporter)", "level": "warn", "note": "一部帯域のみ取得成功"})
    else:
        items.append({"name": "実伝播データ(PSKReporter)", "level": "error",
                       "note": "取得失敗(統計ベースラインのみで継続中)"})

    if nict_result.get("status") == "ok":
        items.append({"name": "NICT実測(電離層観測)", "level": "ok", "note": "正常"})
    else:
        items.append({"name": "NICT実測(電離層観測)", "level": "error",
                       "note": "ページ取得失敗(各局は直近の実測値を最大45分引き継ぎ中)"})

    if noaa.get("status") == "ok":
        items.append({"name": "宇宙天気(NOAA Kp実測)", "level": "ok", "note": "正常"})
    else:
        items.append({"name": "宇宙天気(NOAA Kp実測)", "level": "error", "note": "取得失敗"})

    if kp_forecast.get("status") == "ok":
        items.append({"name": "宇宙天気(NOAA Kp予報)", "level": "ok", "note": "正常"})
    else:
        items.append({"name": "宇宙天気(NOAA Kp予報)", "level": "warn", "note": "取得失敗(予報なしで継続中)"})

    ts = tropo_result.get("status")
    if ts == "ok":
        note = "正常(前回値を使用中、GFS更新待ち)" if tropo_result.get("reused") else "正常(再取得済み)"
        items.append({"name": "ダクト予報(GFS)", "level": "ok", "note": note})
    elif ts == "partial":
        items.append({"name": "ダクト予報(GFS)", "level": "warn", "note": "一部地点のみ取得成功"})
    else:
        items.append({"name": "ダクト予報(GFS)", "level": "error", "note": "取得失敗"})

    return items


def run():
    now_jst = jst_now()
    now_utc_iso = datetime.utcnow().isoformat()
    now_ts = time.time()

    prev_stations = read_prev_stations()
    prev_tropo = read_prev_tropo()

    noaa = fetch_noaa.run()
    kp = noaa.get("kp") if noaa.get("status") == "ok" else None

    # NOAA's own 3-day Kp forecast (not just the live reading above). Used to let
    # aurora-sensitive stations react to an incoming geomagnetic disturbance
    # before Kp has actually risen, instead of only catching up afterwards - see
    # kp_forecast.py's docstring... actually see fetch_noaa.run_forecast().
    kp_forecast = fetch_noaa.run_forecast()
    kp_forecast_peak = kp_forecast.get("peak") if kp_forecast.get("status") == "ok" else None
    # Whichever is higher - the live reading or a forecast disturbance arriving
    # within the next FORECAST_HORIZON_HOURS - drives the aurora-sensitivity
    # boost in climatology_index()/kp_modifier(). Falls back to plain `kp` when
    # no forecast is available.
    kp_for_modifier = kp
    if kp_forecast_peak is not None:
        kp_for_modifier = max(kp or 0.0, kp_forecast_peak)

    psk_result, psk_debug = fetch_pskreporter.run()
    nict_result = fetch_nict.run()

    # Tropospheric ducting (UHF/VHF) index grid - a completely separate
    # NOAA-GFS-derived signal from the Es (sporadic-E) machinery above. Throttled
    # internally (fetch_tropo.MIN_REFETCH_SECONDS) since GFS only updates ~every 6h.
    try:
        tropo_result = fetch_tropo.run(prev_tropo=prev_tropo, now_ts=None)
    except Exception as exc:  # noqa: BLE001 - never let a tropo fetch failure kill the whole pipeline
        tropo_result = {"status": "error", "error": str(exc)}

    # EXPERIMENTAL shadow signal (see fetch_jetstream.py docstring) - logged to
    # history.csv/data.json for a future backtest, NOT used in es_index/heatmap.
    try:
        jet_result = fetch_jetstream.run(now_ts=now_ts)
    except Exception as exc:  # noqa: BLE001 - never let this experimental fetch kill the whole pipeline
        jet_result = {"status": "error", "error": str(exc)}

    # --- append this sample to history ---
    row = {
        "timestamp_utc": now_utc_iso, "kp": kp if kp is not None else "",
        "jet250_kmh": jet_result.get("jet250_kmh", "") if jet_result.get("status") == "ok" else "",
    }
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
        prev_station = prev_stations.get(s["id"]) or {}
        clima = climatology_index(now_jst, s, kp_for_modifier)
        # True only for an aurora-sensitive station where the FORECAST (not the
        # live Kp) is what's driving the boost - i.e. a disturbance is expected
        # but hasn't arrived yet. Purely informational for the UI badge.
        aurora_forecast_active = bool(
            s.get("aurora_sensitive") and kp_forecast_peak is not None and kp_forecast_peak > (kp or 0.0) + 0.5
        )
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
        nict_storm = None
        nict_dis = None
        nict_as_of_ts = None
        stale_minutes = None
        if nict_station:
            nict_status = nict_station["esp_status"]
            foes_mhz = nict_station["foes_mhz"]
            nict_storm = nict_station["storm"]
            nict_dis = nict_station["dis"]

        if nict_status in ("ok", "quiet"):
            # A genuinely fresh reading this cycle - reset the "as of" clock.
            nict_as_of_ts = now_ts
        else:
            # This cycle's scrape didn't yield a usable reading for this station
            # (e.g. a single-sounding parsing/data gap). Rather than throwing away
            # a still-relevant recent measurement, carry the last known-good "ok"
            # reading forward for a bounded grace window - keeping the ORIGINAL
            # measurement timestamp (not resetting it to now) so staleness keeps
            # accumulating across consecutive gap cycles and eventually expires.
            prev_nict = prev_station.get("nict") or {}
            prev_as_of = prev_station.get("nict_as_of_ts")
            if prev_nict.get("status") == "ok" and prev_nict.get("foes_mhz") is not None and prev_as_of is not None:
                age = now_ts - prev_as_of
                if age <= NICT_CARRY_FORWARD_MAX_SECONDS:
                    nict_status = "ok"
                    foes_mhz = prev_nict.get("foes_mhz")
                    nict_storm = prev_nict.get("storm")
                    nict_dis = prev_nict.get("dis")
                    nict_as_of_ts = prev_as_of
                    stale_minutes = max(1, round(age / 60))

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

        # A real NICT ionosonde reading is ground truth for what's happening right
        # now, and must not be capped by a low climatology baseline (e.g. a strong
        # early-morning Es event outside the 11:00/16:30 climatological peaks was
        # previously getting multiplied down to a near-zero score). See
        # nict_floor_from_foes() for the MUF-based reasoning behind the thresholds.
        if nict_status == "ok" and foes_mhz is not None:
            es_index = max(es_index, nict_floor_from_foes(foes_mhz))

        es_index = max(0.0, min(100.0, es_index))

        prev_index = prev_station.get("es_index")
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

        # Rolling window of the last few per-cycle trend steps (oldest first), so
        # the UI can show a short "momentum" arrow sequence (e.g. down/flat/up/up)
        # instead of just the single latest up/down/flat. A step is only appended
        # once it's actually known (not "unknown", e.g. the very first run ever).
        trend_history = list(prev_station.get("trend_history") or [])
        if trend != "unknown":
            trend_history.append(trend)
        trend_history = trend_history[-TREND_HISTORY_LEN:]

        stations_out.append({
            "id": s["id"], "name": s["name"], "loc": s["loc"], "lat": s["lat"], "lon": s["lon"],
            "es_index": round(es_index, 1),
            "trend": trend, "trend_delta": trend_delta, "trend_history": trend_history,
            "climatology_index": round(clima, 1),
            "aurora_forecast_active": aurora_forecast_active,
            "live_evidence": {"ft8_6m_spots_15min": count6, "ft8_10m_spots_15min": count10},
            "nict": {
                "status": nict_status,
                "foes_mhz": foes_mhz,
                "storm": nict_storm,
                "dis": nict_dis,
                "stale_minutes": stale_minutes,
            },
            "nict_as_of_ts": nict_as_of_ts,
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
    heatmap_grid = heatmap.compute(now_jst, kp_for_modifier, all_psk_points, nict_stations)

    system_status = summarize_status(psk_result, nict_result, noaa, kp_forecast, tropo_result)

    EXCLUDE_KEYS = {"region_counts_raw", "heatmap_points"}
    data = {
        "generated_at": now_utc_iso + "Z",
        "generated_at_jst": now_jst.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "kp": {"status": noaa.get("status"), "value": kp, "time_tag": noaa.get("time_tag")},
        "kp_forecast": kp_forecast,
        "kp_history": kp_history,
        "pskreporter": {k: {kk: vv for kk, vv in v.items() if kk not in EXCLUDE_KEYS} for k, v in psk_result.items()},
        "nict": {"status": nict_result.get("status"), "stations": nict_stations},
        "stations": stations_out,
        "heatmap": heatmap_grid,
        "tropo": tropo_result,
        "jet250": jet_result,
        "system_status": system_status,
        "history_rows": len(history_rows),
        "models_trained": sorted(models.keys()),
    }

    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    with open(DEBUG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"pskreporter_debug": psk_debug, "nict_findings": nict_result,
                    "tropo_status": {k: v for k, v in tropo_result.items() if k != "grid"}},
                   f, ensure_ascii=False, indent=2)

    print(json.dumps({"stations": [(s["id"], s["es_index"]) for s in stations_out], "kp": kp,
                       "history_rows": len(history_rows), "models_trained": list(models.keys())}, ensure_ascii=False))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run()
