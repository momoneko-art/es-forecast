"""Best-effort attempt to locate machine-readable Es data behind NICT's realtime chart pages.

NICT does not publish a documented API for this. The chart is rendered client-side, so this
script fetches the raw page source (which WebFetch-style markdown conversion cannot see) and
looks for any referenced .json/.csv/.txt data URLs it can follow. If nothing is found, it
reports status "unconfirmed" and the dashboard falls back to the climatology model plus a
direct link to the official page, rather than showing a fabricated number.
"""
import re
import sys

import requests

HEADERS = {"User-Agent": "es-forecast-dashboard/1.0 (personal amateur radio project)"}
PAGES = [
    "https://wdc.nict.go.jp/Ionosphere/realtime/latest-fxEs.html",
    "https://swc.nict.go.jp/trend/es.html",
]

DATA_URL_RE = re.compile(r"""["'](?P<url>[^"']+\.(?:json|csv|txt))["']""", re.IGNORECASE)


def try_fetch(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # noqa: BLE001
        return None


def run():
    findings = []
    for page in PAGES:
        html = try_fetch(page)
        if html is None:
            findings.append({"page": page, "status": "fetch_failed"})
            continue
        candidates = sorted(set(m.group("url") for m in DATA_URL_RE.finditer(html)))
        confirmed = []
        for c in candidates[:8]:
            if c.startswith("//"):
                c = "https:" + c
            elif c.startswith("/"):
                from urllib.parse import urljoin
                c = urljoin(page, c)
            elif not c.startswith("http"):
                from urllib.parse import urljoin
                c = urljoin(page, c)
            body = try_fetch(c)
            confirmed.append({"url": c, "fetched": body is not None, "sample": (body or "")[:300]})
        findings.append({"page": page, "status": "scanned", "candidate_data_urls": confirmed})
    status = "ok" if any(f.get("candidate_data_urls") for f in findings) else "unconfirmed"
    return {"status": status, "findings": findings}


if __name__ == "__main__":
    import json
    json.dump(run(), sys.stdout, ensure_ascii=False, indent=2)
