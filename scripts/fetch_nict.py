"""Fetch real-time NICT ionosonde status (Storm / SID-WA / Es) for the 4 stations.

Source: NICT's own lightweight mobile status page, which lists, per station,
the Storm and SID/WA alert codes plus the current foEs (E-layer critical
frequency, MHz) reading - refreshed roughly every 15 minutes:
    https://wdc.nict.go.jp/Ionosphere/realtime/ISDJ/ionospheric-signal-i.html

Example page content (plain text once HTML tags are stripped):
    NICT:Ionospheric Status@16:15(JST)
    Wak: Qui. Qui. 5.2
    Kok: Qui. Qui. 3.9
    Yam: Qui. Qui. 4.5
    Oki: Qui. Qui. 4.9

Column meaning (per NICT's own explanation page):
    Sto. (Storm)     : Qui. / Atn+ / Atn- / Wrn+ / Wrn- / ?
    Dis. (SID/WA)     : Qui. / Atn. / Wrn. / ?
    Esp  (Es)         : Qui. (no event) OR a numeric foEs value in MHz
                        (values >= 8MHz are suffixed with '*' on the source page) OR ?

This is real, ground-truth ionosonde data (not a statistical guess), so when it's
available it should be treated as the strongest single signal in the pipeline.
"""
import re
import sys

import requests

URL = "https://wdc.nict.go.jp/Ionosphere/realtime/ISDJ/ionospheric-signal-i.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
}

# Page abbreviation -> our internal station id (see stations.py)
STATION_MAP = {
    "Wak": "wakkanai",
    "Kok": "kokubunji",
    "Yam": "yamagawa",
    "Oki": "oogimi",
}

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

# Matches e.g. "Wak: Qui. Qui. 5.2" / "Wak Qui Qui 8.3*" / "Wak: ? ? ?"
# Captures: station abbr, storm token, dis token, esp token
ROW_RE = re.compile(
    r"\b(Wak|Kok|Yam|Oki)\b\s*:?\s*"
    r"([A-Za-z?][A-Za-z.+\-]*)\s+"
    r"([A-Za-z?][A-Za-z.+\-]*)\s+"
    r"([0-9]+\.?[0-9]*\*?|Qui\.?|\?)"
)


def _parse_esp(token):
    """Return (foes_mhz: float|None, status: 'ok'|'quiet'|'unknown')."""
    if token is None:
        return None, "unknown"
    t = token.strip().rstrip(".")
    if t == "?":
        return None, "unknown"
    if t.lower().startswith("qui"):
        return None, "quiet"
    t = t.rstrip("*")
    try:
        return float(t), "ok"
    except ValueError:
        return None, "unknown"


def fetch_raw():
    try:
        resp = requests.get(URL, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
        return {"ok": True, "text": resp.text}
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        return {"ok": False, "error": str(exc)}


def parse(html_text):
    """Return dict: station_id -> {storm, dis, foes_mhz, esp_status}."""
    plain = WS_RE.sub(" ", TAG_RE.sub(" ", html_text))
    out = {}
    for m in ROW_RE.finditer(plain):
        abbr, storm, dis, esp = m.groups()
        station_id = STATION_MAP.get(abbr)
        if not station_id or station_id in out:
            continue
        foes_mhz, esp_status = _parse_esp(esp)
        out[station_id] = {
            "storm": storm.strip(),
            "dis": dis.strip(),
            "foes_mhz": foes_mhz,
            "esp_status": esp_status,
        }
    return out


def run():
    raw = fetch_raw()
    if not raw.get("ok"):
        return {"status": "error", "error": raw.get("error"), "stations": {}}

    stations = parse(raw["text"])
    if not stations:
        # Page reachable but format didn't match what we expect - degrade gracefully
        # and keep a sample for debugging instead of crashing the pipeline.
        return {"status": "unconfirmed", "stations": {}, "sample": raw["text"][:1000]}

    return {"status": "ok", "stations": stations}


if __name__ == "__main__":
    import json
    res = run()
    json.dump(res, sys.stdout, ensure_ascii=False, indent=2)
