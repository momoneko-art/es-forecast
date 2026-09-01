"""Fetch the latest planetary Kp index from NOAA SWPC (reliable public JSON feed)."""
import sys

import requests

URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
HEADERS = {"User-Agent": "es-forecast-dashboard/1.0 (personal amateur radio project)"}


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


if __name__ == "__main__":
    import json
    json.dump(run(), sys.stdout, ensure_ascii=False, indent=2)
