"""Fetch recent FT8 reception reports from PSKReporter for Es-relevant bands (10m / 6m)."""
import sys
import xml.etree.ElementTree as ET

import requests

from stations import maidenhead_to_latlon, nearest_station_by_lat, in_japan_bbox, great_circle_km, STATIONS

TOP_CONTACTS_PER_BAND = 6

BASE = "https://retrieve.pskreporter.info/query"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Referer": "https://pskreporter.info/pskmap.html",
    "Accept": "*/*",
}

BANDS = {
    "10m": (28070000, 28078000),
    "6m": (50300000, 50320000),
}


def fetch_band(lo, hi, window_seconds=-900):
    params = {
        "frange": f"{lo}-{hi}",
        "mode": "FT8",
        "flowStartSeconds": str(window_seconds),
        "rronly": "1",
    }
    try:
        resp = requests.get(BASE, params=params, headers=HEADERS, timeout=25)
        if resp.status_code != 200:
            return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
        return {"ok": True, "text": resp.text, "status_code": resp.status_code}
    except Exception as exc:  # noqa: BLE001 - we want this pipeline step to degrade gracefully
        return {"ok": False, "error": str(exc)}


def parse_reports(xml_text):
    """Return a list of dicts with receiverLocator / senderCallsign / frequency, tolerant of schema drift."""
    reports = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return reports
    for el in root.iter():
        attrs = el.attrib
        if not attrs:
            continue
        lowered = {k.lower(): v for k, v in attrs.items()}
        if "receiverlocator" in lowered or "receivercall" in lowered:
            reports.append({
                "sender_callsign": lowered.get("sendercallsign"),
                "sender_locator": lowered.get("senderlocator"),
                "receiver_callsign": lowered.get("receivercallsign"),
                "receiver_locator": lowered.get("receiverlocator"),
                "frequency": lowered.get("frequency"),
                "sn_r": lowered.get("sn_r") or lowered.get("snr"),
                "flow_start_seconds": lowered.get("flowstartseconds"),
            })
    return reports


def summarize_band(band_name, lo, hi):
    raw = fetch_band(lo, hi)
    if not raw.get("ok"):
        return {"status": "error", "error": raw.get("error")}, None

    reports = parse_reports(raw["text"])
    region_counts = {s["id"]: 0 for s in STATIONS}
    counted = 0
    heatmap_points = []  # [lat, lon] for every receiver inside Japan, used for the nationwide heatmap
    pair_distances = []  # every report where BOTH ends resolve, distance in km
    contacts = []        # same, plus callsigns/freq, kept for the "notable long-distance" list
    for r in reports:
        latlon = maidenhead_to_latlon(r.get("receiver_locator"))
        if not latlon:
            continue
        lat, lon = latlon
        if in_japan_bbox(lat, lon):
            heatmap_points.append([round(lat, 3), round(lon, 3)])
        station_id = nearest_station_by_lat(lat, lon)
        if station_id:
            region_counts[station_id] += 1
            counted += 1

        # Pair distance: only meaningful when the receiver is somewhere in Japan
        # (matches the heatmap's scope) and the sender's own locator also resolves.
        if not in_japan_bbox(lat, lon):
            continue
        sender_latlon = maidenhead_to_latlon(r.get("sender_locator"))
        if not sender_latlon:
            continue
        s_lat, s_lon = sender_latlon
        dist_km = great_circle_km(lat, lon, s_lat, s_lon)
        pair_distances.append(dist_km)
        contacts.append({
            "sender_callsign": r.get("sender_callsign"),
            "receiver_callsign": r.get("receiver_callsign"),
            "distance_km": round(dist_km),
            "sn_r": r.get("sn_r"),
        })

    contacts.sort(key=lambda c: c["distance_km"], reverse=True)
    # De-duplicate by (sender, receiver) pair - the same two stations can appear
    # in several FT8 cycles inside one 15-minute window.
    seen_pairs = set()
    top_contacts = []
    for c in contacts:
        key = (c["sender_callsign"], c["receiver_callsign"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        top_contacts.append(c)
        if len(top_contacts) >= TOP_CONTACTS_PER_BAND:
            break

    pair_stats = {
        "matched_pairs": len(pair_distances),
        "avg_distance_km": round(sum(pair_distances) / len(pair_distances)) if pair_distances else None,
        "max_distance_km": round(max(pair_distances)) if pair_distances else None,
        "top_contacts": top_contacts,
    }

    summary = {
        "status": "ok",
        "total_reports": len(reports),
        "matched_to_station": counted,
        "region_counts": region_counts,
        "heatmap_points": heatmap_points,
        "pair_stats": pair_stats,
    }
    # keep a small raw sample for debugging schema drift, not the full payload
    debug_sample = raw["text"][:1500]
    return summary, debug_sample


def run():
    result = {}
    debug = {}
    for name, (lo, hi) in BANDS.items():
        summary, sample = summarize_band(name, lo, hi)
        result[name] = summary
        if sample is not None:
            debug[name] = sample
    return result, debug


if __name__ == "__main__":
    import json
    res, dbg = run()
    json.dump({"result": res, "debug": dbg}, sys.stdout, ensure_ascii=False, indent=2)
