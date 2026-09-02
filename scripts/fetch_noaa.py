"""Fetch the latest planetary Kp index, and NOAA's own 3-day Kp forecast, from
NOAA SWPC (reliable public JSON feeds, no API key needed)."""
import sys
from datetime import datetime, timezone

import requests

URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
# NOAA's own predicted Kp product: ~7 days of past observed/estimated values plus
# ~3 days of "predicted" values ahead, all at 3-hour resolution. This is a genuine
# forecast (not just the live reading above) - used to let aurora-sensitive
# stations (Wakkanai) react to an incoming geomagnetic disturbance BEFORE Kp has
# actually risen yet, instead of only after the fact.
FORECAST_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
HEADERS = {"User-Agent": "es-forecast-dashboard/1.0 (personal amateur radio project)"}

FORECAST_HORIZON_HOURS = 24  # how far ahead to look for the "kp_forecast_peak" used by the model


def run():
    try:
        resp = requests.get(URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return {"status": "error", "error": "empty response"}
        last = data[-1]
        return {
            "status": "ok",
            "time_tag": last.get("time_tag"),
            "kp": last.get("kp_index"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def run_forecast(now_ts=None):
    """Returns {"status": "ok", "series": [{"time": iso, "kp": float, "kind":
    "predicted"|"estimated"}, ...] (future entries only, within
    FORECAST_HORIZON_HOURS), "peak": float or None, "peak_time": iso or None}.
    "kind" keeps "estimated" (NOAA's own near-term nowcast-ish value, not a pure
    observation) separate from "predicted" (the actual forward forecast) so the
    UI can be honest about which is which; both are treated as forward-looking
    for the peak calculation since "observed" (the past) is excluded either way."""
    import time
    if now_ts is None:
        now_ts = time.time()
    try:
        resp = requests.get(FORECAST_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return {"status": "error", "error": "empty response"}

        now_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
        series = []
        for row in rows:
            kind = row.get("observed")
            if kind not in ("predicted", "estimated"):
                continue
            time_tag = row.get("time_tag")
            kp = row.get("kp")
            if time_tag is None or kp is None:
                continue
            try:
                t = datetime.fromisoformat(time_tag).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            hours_ahead = (t - now_dt).total_seconds() / 3600.0
            if hours_ahead < -1.0 or hours_ahead > FORECAST_HORIZON_HOURS:
                continue
            series.append({"time": time_tag + "Z", "kp": float(kp), "kind": kind})

        series.sort(key=lambda r: r["time"])
        if not series:
            return {"status": "ok", "series": [], "peak": None, "peak_time": None}

        peak_row = max(series, key=lambda r: r["kp"])
        return {
            "status": "ok",
            "series": series,
            "peak": peak_row["kp"],
            "peak_time": peak_row["time"],
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    import json
    json.dump({"current": run(), "forecast": run_forecast()}, sys.stdout, ensure_ascii=False, indent=2)
