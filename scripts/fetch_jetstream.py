"""EXPERIMENTAL / shadow data collection only - NOT used in es_index, the
heatmap, or any user-facing score yet.

Logs 250hPa (~10.4km, jet-stream altitude) wind speed near Japan into
history.csv/data.json every cycle, purely so a future session has enough
real historical data to check whether jet-stream position/strength actually
correlates with Es activity (accuracy-roadmap item 3, discussed 2026-09-02
with the user). At the time this was added, history.csv held barely 1 day of
rows - nowhere near enough for a meaningful backtest - so this module's only
job for now is to start accumulating that history quietly in the background.

Uses the same free Open-Meteo GFS forecast API as fetch_tropo.py (see that
module's docstring for the "blocked from the sandbox itself, confirmed
reachable from GitHub Actions" caveat - the same applies here), just at the
jet stream's characteristic altitude instead of the near-surface levels
tropo ducting looks at.
"""
from datetime import datetime, timezone

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FETCH_TIMEOUT_SECONDS = 20
MODEL = "gfs_seamless"

# A handful of points spanning Japan north-to-south (roughly matching the 4
# Es-monitoring stations), averaged into one representative "jet stream near
# Japan" value. A first-pass shadow dataset doesn't need a full spatial grid
# like the tropo index does - this can be made finer later if an initial
# correlation check looks promising.
POINTS = [
    (45.4, 141.7),   # near Wakkanai
    (35.7, 139.5),   # near Kokubunji
    (31.2, 130.6),   # near Yamagawa
    (26.7, 128.2),   # near Oogimi
]


def run(now_ts=None):
    import time
    if now_ts is None:
        now_ts = time.time()
    now_hour = datetime.fromtimestamp(now_ts, tz=timezone.utc).hour

    lats = ",".join(str(p[0]) for p in POINTS)
    lons = ",".join(str(p[1]) for p in POINTS)
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": "windspeed_250hPa,winddirection_250hPa",
        "forecast_days": 1,
        "models": MODEL,
        "timezone": "UTC",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=FETCH_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            return {"status": "error", "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}
        data = resp.json()
        entries = data if isinstance(data, list) else [data]

        speeds = []
        dirs = []
        for entry in entries:
            hourly = entry.get("hourly", {}) or {}
            ws = hourly.get("windspeed_250hPa") or []
            wd = hourly.get("winddirection_250hPa") or []
            if now_hour < len(ws) and ws[now_hour] is not None:
                speeds.append(ws[now_hour])
            if now_hour < len(wd) and wd[now_hour] is not None:
                dirs.append(wd[now_hour])

        if not speeds:
            return {"status": "error", "error": "no windspeed data in response"}

        avg_speed = sum(speeds) / len(speeds)
        avg_dir = sum(dirs) / len(dirs) if dirs else None
        return {
            "status": "ok",
            "jet250_kmh": round(avg_speed, 1),
            "jet250_dir_deg": round(avg_dir, 0) if avg_dir is not None else None,
            "points_used": len(speeds),
        }
    except Exception as exc:  # noqa: BLE001 - a shadow/experimental feature must never break the main pipeline
        return {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), ensure_ascii=False, indent=2))
