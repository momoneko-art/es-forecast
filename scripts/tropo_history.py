"""Save the tropo-duct index for hours that have already passed, so the
dashboard can later replay the past ~month ("過去を見る").

Added 2026-09-25 at the user's request: instead of re-fetching past weather
data (which would need the duct-index calculation to be re-implemented and
could not be tested from the development environment), simply keep the
numbers this pipeline already computes. Every run, each hour of
data['tropo']['grid'] that is already in the past (<= now, UTC) is written
into one small file per UTC day:

    tropo_hist/YYYY-MM-DD.json   {"date", "index_version", "grid": {...meta},
                                  "frames": {"YYYY-MM-DDTHH:00": [[row][col] ints]}}
    tropo_hist/index.json        {"format", "dates": [...], "keep_days": N}

Newer model runs overwrite older values for the same hour (a fresher model run
is the better estimate of what actually happened). Values are rounded to
integers (0-100) to keep each day file around 10-15 KB. Files older than
KEEP_DAYS are deleted.

Safety: this module is completely separate from data.json. build_index.py
calls save() only AFTER data.json/debug.json have been written, inside a
try/except, so any failure here can never break the normal forecast update.
A day file is only rewritten when its content actually changed, so it does not
cause extra commits.
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone

HIST_DIRNAME = "tropo_hist"
KEEP_DAYS = 40          # a little over a month
FORMAT_VERSION = 1
GRID_META_KEYS = ("lat_min", "lat_max", "lat_step", "lon_min", "lon_max", "lon_step", "rows", "cols")


def _parse_hour(iso):
    """Open-Meteo hourly strings are 'YYYY-MM-DDTHH:MM' in UTC (no offset)."""
    try:
        return datetime.strptime(iso[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _round_frame(values, ti):
    out = []
    for row in values:
        r = []
        for series in row:
            v = series[ti] if (series is not None and ti < len(series)) else None
            r.append(None if v is None else int(round(v)))
        out.append(r)
    return out


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_if_changed(path, obj):
    new_text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    try:
        with open(path, encoding="utf-8") as f:
            if f.read() == new_text:
                return False
    except OSError:
        pass
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new_text)
    os.replace(tmp, path)
    return True


def save(root, tropo_result, now_ts=None):
    """Returns a small summary dict (for logging). Never raises on bad input -
    returns {"skipped": reason} instead."""
    if not isinstance(tropo_result, dict) or tropo_result.get("status") != "ok":
        return {"skipped": "tropo status not ok"}
    grid = tropo_result.get("grid") or {}
    times, values = grid.get("times"), grid.get("values")
    if not times or not values:
        return {"skipped": "no grid"}
    meta = {k: grid.get(k) for k in GRID_META_KEYS}
    if any(v is None for v in meta.values()):
        return {"skipped": "incomplete grid meta"}

    now = datetime.fromtimestamp(now_ts if now_ts is not None else time.time(), tz=timezone.utc)
    hist_dir = os.path.join(root, HIST_DIRNAME)
    os.makedirs(hist_dir, exist_ok=True)

    # group past hours by UTC date
    by_day = {}
    for ti, iso in enumerate(times):
        t = _parse_hour(iso)
        if t is None or t > now:
            continue
        key = t.strftime("%Y-%m-%dT%H:00")
        by_day.setdefault(t.strftime("%Y-%m-%d"), {})[key] = _round_frame(values, ti)

    changed_days = []
    for day, frames in by_day.items():
        path = os.path.join(hist_dir, day + ".json")
        doc = _load_json(path)
        if (not isinstance(doc, dict) or doc.get("grid") != meta
                or doc.get("index_version") != tropo_result.get("index_version")
                or doc.get("format") != FORMAT_VERSION):
            # new day, or the grid/index definition changed: start the day fresh
            doc = {"format": FORMAT_VERSION, "date": day, "grid": meta,
                   "index_version": tropo_result.get("index_version"), "frames": {}}
        doc["frames"].update(frames)   # newer model run wins for the same hour
        if _write_if_changed(path, doc):
            changed_days.append(day)

    # retention + index
    cutoff = (now - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    dates = []
    for name in sorted(os.listdir(hist_dir)):
        if not (name.endswith(".json") and len(name) == 15 and name[4] == "-" and name[7] == "-"):
            continue
        day = name[:-5]
        if day < cutoff:
            try:
                os.remove(os.path.join(hist_dir, name))
            except OSError:
                pass
            continue
        dates.append(day)
    index = {"format": FORMAT_VERSION, "dates": dates, "keep_days": KEEP_DAYS}
    old_index = _load_json(os.path.join(hist_dir, "index.json")) or {}
    if old_index.get("dates") != dates or old_index.get("keep_days") != KEEP_DAYS or old_index.get("format") != FORMAT_VERSION:
        _write_if_changed(os.path.join(hist_dir, "index.json"), index)
    return {"changed_days": changed_days, "days_kept": len(dates)}
