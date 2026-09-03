"""Best-effort secondary source for foEs: prop.kc2g.com aggregates real-time
digisonde (ionosonde) readings from the GIRO/DIDBase network plus NOAA and
SWS Australia, as a single free JSON endpoint (no API key):

    https://prop.kc2g.com/api/stations.json

Investigated 2026-09-03 after the user asked what other public data could
help Es/tropo forecasting. Checking the actual station list showed our 4
NICT sites (Wakkanai/Kokubunji/Yamagawa, and a station very close to Oogimi)
appear in it under GIRO station codes, with source:"giro" - almost certainly
the SAME underlying NICT measurements, just re-published as clean JSON
instead of the HTML page fetch_nict.py has to scrape. So this module is
NOT meant to be an independent vote alongside NICT; it has two narrower
jobs:

  1. A fallback for fetch_nict.py: if NICT's own page fails to scrape AND
     the existing carry-forward window (NICT_CARRY_FORWARD_MAX_SECONDS in
     build_index.py) has also expired, a matching kc2g reading is a better
     last resort than falling all the way back to the pure climatology
     baseline - it is still a real measurement, just relayed through a
     different door.
  2. Shadow-logging a few nearby non-Japan stations (Korea, Russian Far
     East) that might show Es activity shortly before/after it reaches our
     own 4 sites - purely recorded for now (see build_index.py's history
     row), NOT folded into es_index until there is enough history to check
     whether it actually adds predictive value (same "log first, trust
     later" approach already used for jet250_kmh).

Like fetch_tropo.py's Open-Meteo calls, prop.kc2g.com is blocked by the
organisation egress policy from every environment available during
development (cloud sandbox, the PC device-bridge shell) - confirmed
reachable only via the WebFetch tool's own separate network path, not from
requests.get() in this environment. GitHub Actions runners have general
internet access and fetch several other external hosts today without
trouble, so this is expected to work there, but - same as the ECMWF
addition in fetch_tropo.py - it has NOT been live-verified yet. Check
debug.json after the next real Actions cycle.
"""
import time

import requests

STATIONS_URL = "https://prop.kc2g.com/api/stations.json"
FETCH_TIMEOUT_SECONDS = 20

# Matched by nearest-distance (see match_station()), not by hardcoding a kc2g
# station code - the code strings are treated as stable identifiers by kc2g
# itself, but distance-matching means this module doesn't silently go blind
# if kc2g ever renumbers/re-labels a station.
MATCH_MAX_KM = 120.0

# A short, deliberately hand-picked list of nearby-but-distinct stations for
# the shadow-logging "early warning" signal (see module docstring, job 2).
# Hardcoded rather than auto-selected by a distance rule so this can't
# accidentally start including a far-away or unrelated station if the global
# network's roster changes; a code going missing just drops silently from
# the output (see run()) rather than erroring.
NEIGHBOR_CODES = {
    "JJ433": "済州島(韓国)",
    "IC437": "利川(韓国)",
    "KB548": "ハバロフスク(ロシア)",
}


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _great_circle_km(lat1, lon1, lat2, lon2):
    import math
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _fetch(session=None):
    getter = session.get if session else requests.get
    resp = getter(STATIONS_URL, timeout=FETCH_TIMEOUT_SECONDS)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"expected a JSON array, got {type(data).__name__}")
    return data


def match_station(entries, target_lat, target_lon, max_km=MATCH_MAX_KM):
    """Nearest kc2g entry to (target_lat, target_lon) with a foEs reading,
    within max_km. Returns (entry, distance_km) or (None, None)."""
    best, best_dist = None, None
    for e in entries:
        st = e.get("station") or {}
        lat, lon = _to_float(st.get("latitude")), _to_float(st.get("longitude"))
        foes = e.get("foes")
        if lat is None or lon is None or foes is None:
            continue
        dist = _great_circle_km(target_lat, target_lon, lat, lon)
        if dist <= max_km and (best_dist is None or dist < best_dist):
            best, best_dist = e, dist
    return best, best_dist


def run(stations, now_ts=None, session=None, fetch=_fetch):
    """stations: the shared STATIONS list from stations.py (needs id/lat/lon).
    Returns {"status": "ok"|"error", "error": str|None, "fetched_at": ts,
    "by_station_id": {station_id: {"foes_mhz","kc2g_code","kc2g_name",
    "distance_km","kc2g_time"} or None, ...}, "neighbors": [...]}."""
    if now_ts is None:
        now_ts = time.time()
    try:
        entries = fetch(session=session)
    except Exception as exc:  # noqa: BLE001 - purely a fallback/shadow source, never allowed to raise
        return {"status": "error", "error": str(exc), "fetched_at": now_ts,
                "by_station_id": {}, "neighbors": []}

    by_station_id = {}
    for s in stations:
        entry, dist = match_station(entries, s["lat"], s["lon"])
        if entry is None:
            by_station_id[s["id"]] = None
            continue
        st = entry.get("station") or {}
        by_station_id[s["id"]] = {
            "foes_mhz": entry.get("foes"),
            "kc2g_code": st.get("code"),
            "kc2g_name": st.get("name"),
            "distance_km": round(dist, 1),
            "kc2g_time": entry.get("time"),
        }

    by_code = {}
    for e in entries:
        st = e.get("station") or {}
        code = st.get("code")
        if code:
            by_code[code] = e

    neighbors = []
    for code, label in NEIGHBOR_CODES.items():
        e = by_code.get(code)
        if e is None or e.get("foes") is None:
            continue
        st = e.get("station") or {}
        neighbors.append({
            "code": code,
            "label": label,
            "name": st.get("name"),
            "foes_mhz": e.get("foes"),
            "kc2g_time": e.get("time"),
        })

    return {
        "status": "ok",
        "error": None,
        "fetched_at": now_ts,
        "by_station_id": by_station_id,
        "neighbors": neighbors,
    }


if __name__ == "__main__":
    import json
    import sys
    sys.path.insert(0, ".")
    from stations import STATIONS
    print(json.dumps(run(STATIONS), ensure_ascii=False, indent=2))
