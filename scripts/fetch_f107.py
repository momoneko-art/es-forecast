"""EXPERIMENTAL / shadow data collection only - NOT used in es_index yet,
same "log first, trust later" approach as fetch_jetstream.py.

F10.7cm solar radio flux is the standard long-run proxy for overall solar UV
output and therefore overall ionospheric ionization. It is a well-established
space-weather index, but its established predictive value is mostly for the
F-region (MUF/foF2, long-haul HF propagation) rather than sporadic-E
specifically, which is driven by a different, largely-independent mechanism
(E-region neutral wind shear - see the "wind shear theory" research noted in
DEVELOPMENT_LOG.md). It is logged here purely so a future session can check,
with real history, whether it adds anything to Es specifically - not assumed
in advance.

Same JSON-feed domain/pattern already used for the Kp index in fetch_noaa.py:
    https://services.swpc.noaa.gov/json/f107_cm_flux.json       (observed)
    https://services.swpc.noaa.gov/json/predicted_f107cm_flux.json (forecast)
"""
import time

import requests

OBSERVED_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
FETCH_TIMEOUT_SECONDS = 20


def run(now_ts=None, session=None):
    """Returns {"status": "ok"|"error", "error": str|None, "fetched_at": ts,
    "f107": float|None, "f107_time": str|None} - the single most recent
    observed F10.7 reading. (No forecast value is fetched for now; the
    observed daily figure is what's relevant for "today's flux level", and
    keeping this module small/simple matters more than completeness while
    it's still an unproven shadow signal.)"""
    if now_ts is None:
        now_ts = time.time()
    try:
        getter = session.get if session else requests.get
        resp = getter(OBSERVED_URL, timeout=FETCH_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            return {"status": "error", "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                    "fetched_at": now_ts, "f107": None, "f107_time": None}
        data = resp.json()
        if not isinstance(data, list) or not data:
            return {"status": "error", "error": "empty/unexpected response",
                    "fetched_at": now_ts, "f107": None, "f107_time": None}
        # Entries are chronological; the last one is the most recent observation.
        last = data[-1]
        flux = last.get("flux") if isinstance(last, dict) else None
        when = last.get("time_tag") if isinstance(last, dict) else None
        if flux is None:
            return {"status": "error", "error": "no flux field in latest entry",
                    "fetched_at": now_ts, "f107": None, "f107_time": None}
        return {"status": "ok", "error": None, "fetched_at": now_ts,
                "f107": float(flux), "f107_time": when}
    except Exception as exc:  # noqa: BLE001 - a shadow/experimental feature must never break the main pipeline
        return {"status": "error", "error": str(exc), "fetched_at": now_ts, "f107": None, "f107_time": None}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), ensure_ascii=False, indent=2))
