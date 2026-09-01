"""Fetch NOAA GFS near-surface pressure-level profiles (temperature / relative
humidity / geopotential height) across a coarse grid over Japan, via the free
Open-Meteo forecast API (https://open-meteo.com/ - plain JSON, no API key, no
GRIB/eccodes dependency), and compute a tropospheric-ducting index time series
(now through ~7 days ahead, hourly) from the vertical gradient of modified
refractivity (M-units).

Physics (standard, ITU-R P.453 style):
  N = 77.6*(P/T) + 3.73e5*(e/T^2)      radio refractivity, N-units (T in Kelvin)
  M = N + 0.157*h                       modified refractivity, M-units (h in metres)
A "standard atmosphere" has dM/dh ~= +117 M-units/km (no duct). Ducting occurs
where dM/dh goes negative anywhere in the near-surface layers (equivalent to the
well-known dN/dh < -157 N-units/km trapping-layer rule used by hobbyist tropo
forecasters, e.g. https://tropo.otacanada.com/ , which uses the same GFS
temperature/humidity/geopotential-height -> M-gradient approach). This module
looks only at the strongest (most negative) gradient across the near-surface
layers - it is a simplified single-column index, not a ray-traced path
prediction, and is meant to answer "is a duct present here", not exact path loss.

Forecast range: requests FORECAST_DAYS (7) days of hourly GFS data per grid
point in a single Open-Meteo call (models=gfs_seamless, so pressure-level
fields are guaranteed out to the full range rather than whatever the
"best_match" blend happens to carry), then samples every STEP_HOURS (1, i.e.
every hour Open-Meteo returns) to build a time series per point - matching
both the 2026-09 user request for an "about a week ahead" outlook and their
follow-up request to match dxinfocentre.com's hour-by-hour display, rather
than the coarser 6h steps this module used at first. Skill beyond a few days
is inherently limited (this is a single-column heuristic on top of a
medium-range NWP model, not a verified ducting forecast product), which the
UI should make clear to the user rather than presenting far-out steps with
false confidence.

The map deliberately does NOT clip the duct-index fill to the Japan coastline
(unlike the Es heatmap) - ducting is at least as significant over water as
over land (in fact many real-world tropo openings are strongest along
coastlines and over the sea), so clipping to land would hide the most
relevant part of the picture. The coastline is drawn as a reference outline
on top of the full grid instead.

CONFIRMED WORKING (2026-09-01): debug.json's tropo_status showed status "ok"
with no errors after the first GitHub Actions run following deployment, so
Open-Meteo is reachable from GitHub Actions runners as expected (this could
not be tested directly from within the development session itself - both the
cloud sandbox and the device-bridge shell on the user's own PC sit behind an
organisation-controlled egress allowlist that blocks api.open-meteo.com).
"""
import math

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Near-surface layers most relevant to VHF/UHF tropo ducting. The real trapping
# layers behind tropo openings (marine subsidence inversions, nocturnal radiation
# inversions) are typically only a few hundred metres thick - with only the
# coarse 1000/925/850/700hPa set (roughly 100m/750m/1500m/3000m, ~650-1500m
# gaps) a thin duct gets averaged away into a much gentler mean gradient across
# each big gap and essentially never registers as a real (dM/dh<0) duct. Using
# every near-surface level Open-Meteo offers (roughly 100/320/540/760/990/1460m)
# keeps each gap to ~200-500m, closer to the real layer thickness, so a genuine
# duct is far more likely to show up as an actual negative gradient somewhere
# in the profile instead of being smoothed out. 700hPa stays in as an upper
# reference point for an elevated duct sitting higher than the near-surface set.
LEVELS_HPA = [1000, 975, 950, 925, 900, 850, 700]

GRID_LAT_MIN, GRID_LAT_MAX, GRID_LAT_STEP = 24.0, 46.0, 2.0
GRID_LON_MIN, GRID_LON_MAX, GRID_LON_STEP = 123.0, 149.0, 2.0

FORECAST_DAYS = 7         # "about a week ahead" per user request; GFS itself supports up to 16
STEP_HOURS = 1            # hourly, per user request to match dxinfocentre.com's hour-by-hour display
MODEL = "gfs_seamless"    # pin to GFS explicitly so pressure-level fields are populated for the full range

BATCH_SIZE = 12           # a week of hourly data at 7 pressure levels per point is a much bigger payload
                          # than the old single-level "current conditions only" call, so batches are
                          # smaller than before (reduced further when LEVELS_HPA grew from 4 to 7 levels)
FETCH_TIMEOUT_SECONDS = 60
# GFS runs every 6h (00/06/12/18 UTC) and the user confirmed dxinfocentre.com itself
# only refreshes on that same 6h cadence, so re-fetching every 15min build cycle (or
# even every 50min, this module's first throttle value) was needlessly wasteful -
# almost every one of those fetches would have pulled back the same underlying model
# run. 5.5h leaves a margin under the full 6h so a slightly late Open-Meteo publish
# doesn't push us to miss an entire cycle.
MIN_REFETCH_SECONDS = int(5.5 * 60 * 60)

# Bumped whenever a change to the duct-index FORMULA (not just the grid/JSON
# schema) would make old cached values wrong even though they still look
# structurally valid (same keys, still has "times", etc) - e.g. LEVELS_HPA
# changing what altitudes are sampled. The reuse check below requires an exact
# match, so a version bump forces an immediate re-fetch on the very next run
# instead of silently reusing pre-fix numbers for up to MIN_REFETCH_SECONDS.
# v2: added 975/950/925/900/850hPa (was only 1000/925/850/700hPa) so thin
# near-surface ducts stop getting averaged away to ~0.
INDEX_VERSION = 2


def _frange(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def grid_points():
    lats = _frange(GRID_LAT_MIN, GRID_LAT_MAX, GRID_LAT_STEP)
    lons = _frange(GRID_LON_MIN, GRID_LON_MAX, GRID_LON_STEP)
    return lats, lons


def saturation_vapor_pressure_hpa(temp_c):
    """Bolton (1980) approximation of saturation vapor pressure, hPa."""
    return 6.1121 * math.exp((18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c)))


def refractivity_n(pressure_hpa, temp_c, rh_pct):
    """ITU-R P.453 radio refractivity N, in N-units."""
    t_k = temp_c + 273.15
    e = max(0.0, min(100.0, rh_pct)) / 100.0 * saturation_vapor_pressure_hpa(temp_c)
    return 77.6 * (pressure_hpa / t_k) + 3.73e5 * (e / (t_k ** 2))


def duct_index_from_profile(levels):
    """levels: iterable of dicts with pressure/height_m/temp_c/rh (any order, any
    level subset - missing/None entries are skipped). Returns (index_0_100,
    strongest_dM/dh_in_M-units_per_km or None if fewer than 2 usable levels)."""
    pts = []
    for lv in levels:
        if lv.get("height_m") is None or lv.get("temp_c") is None or lv.get("rh") is None:
            continue
        n = refractivity_n(lv["pressure"], lv["temp_c"], lv["rh"])
        m = n + 0.157 * lv["height_m"]
        pts.append((lv["height_m"], m))
    if len(pts) < 2:
        return 0.0, None

    pts.sort(key=lambda p: p[0])
    min_grad = None
    for i in range(len(pts) - 1):
        h0, m0 = pts[i]
        h1, m1 = pts[i + 1]
        if h1 - h0 < 1:
            continue
        grad_per_km = (m1 - m0) / (h1 - h0) * 1000.0
        if min_grad is None or grad_per_km < min_grad:
            min_grad = grad_per_km
    if min_grad is None:
        return 0.0, None

    # +117 M/km = standard atmosphere (no duct). Below that is increasingly
    # anomalous; below 0 is a true trapping layer (duct). Thresholds are a
    # simplified engineering scale, not a precise path-loss prediction.
    if min_grad >= 117:
        idx = 0.0
    elif min_grad >= 40:
        idx = 20.0 * (117 - min_grad) / (117 - 40)
    elif min_grad >= 0:
        idx = 20.0 + 20.0 * (40 - min_grad) / 40.0
    else:
        idx = 40.0 + min(60.0, -min_grad * 1.2)
    return round(max(0.0, min(100.0, idx)), 1), round(min_grad, 1)


def _hourly_params():
    vars_ = []
    for lv in LEVELS_HPA:
        vars_ += [f"temperature_{lv}hPa", f"relative_humidity_{lv}hPa", f"geopotential_height_{lv}hPa"]
    return ",".join(vars_)


def _fetch_batch(points, session=None):
    lats = ",".join(str(p[0]) for p in points)
    lons = ",".join(str(p[1]) for p in points)
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": _hourly_params(),
        "forecast_days": FORECAST_DAYS,
        "models": MODEL,
        "timezone": "UTC",
    }
    getter = session.get if session else requests.get
    resp = getter(OPEN_METEO_URL, params=params, timeout=FETCH_TIMEOUT_SECONDS)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    entries = data if isinstance(data, list) else [data]
    if len(entries) != len(points):
        raise RuntimeError(f"expected {len(points)} entries, got {len(entries)}")
    return entries


def _entry_to_series(entry, step_hours=STEP_HOURS):
    """Returns (index_series, times_iso) for one point - one duct index per
    sampled hour, plus the matching ISO8601 UTC timestamps. Missing/short
    arrays degrade to whatever length is actually available rather than
    raising, since Open-Meteo occasionally trims a field a few hours short."""
    hourly = entry.get("hourly", {}) or {}
    times = hourly.get("time") or []
    idxs = list(range(0, len(times), step_hours))

    level_arrays = {}
    for lv in LEVELS_HPA:
        level_arrays[lv] = (
            hourly.get(f"temperature_{lv}hPa") or [],
            hourly.get(f"relative_humidity_{lv}hPa") or [],
            hourly.get(f"geopotential_height_{lv}hPa") or [],
        )

    series = []
    for i in idxs:
        levels = []
        for lv in LEVELS_HPA:
            t_arr, rh_arr, h_arr = level_arrays[lv]
            levels.append({
                "pressure": lv,
                "temp_c": t_arr[i] if i < len(t_arr) else None,
                "rh": rh_arr[i] if i < len(rh_arr) else None,
                "height_m": h_arr[i] if i < len(h_arr) else None,
            })
        idx, _grad = duct_index_from_profile(levels)
        series.append(idx)

    times_out = [times[i] for i in idxs]
    return series, times_out


def fetch_grid(session=None, fetch_batch=_fetch_batch, step_hours=STEP_HOURS):
    """Returns (values_by_point dict[(lat,lon)] -> list[index] one per sampled
    hour, times list[str] ISO8601 UTC shared across all points, errors
    list[str]). times is taken from the first point that returned data since
    every point is fetched with the same forecast_days/model/timezone and
    should therefore share an identical timeline."""
    lats, lons = grid_points()
    points = [(la, lo) for la in lats for lo in lons]

    values = {}
    times_out = None
    errors = []
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        try:
            entries = fetch_batch(batch, session=session)
            for (lat, lon), entry in zip(batch, entries):
                series, times = _entry_to_series(entry, step_hours=step_hours)
                values[(lat, lon)] = series
                if times_out is None and times:
                    times_out = times
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, one bad batch shouldn't kill the rest
            errors.append(f"batch@{batch[0]}: {exc}")
    return values, times_out, errors


def run(prev_tropo=None, now_ts=None):
    """prev_tropo: the previous cycle's data['tropo'] dict (or None), used to skip
    re-fetching when GFS almost certainly hasn't updated yet. now_ts: unix
    timestamp (defaults to time.time()) - overridable for tests."""
    import time
    if now_ts is None:
        now_ts = time.time()

    # A prev_tropo written by an older version of this module (e.g. the
    # single-current-value shape from before the weekly time series was
    # added, or a 6h-step grid from before hourly steps) is never reused,
    # however fresh - otherwise a stale writer (a PC agent process that
    # hasn't been restarted since a deploy) can keep the whole site stuck on
    # an old schema/resolution until its own throttle window happens to
    # expire. "times" is only present in the current schema's grid.
    prev_grid = (prev_tropo or {}).get("grid") or {}
    prev_is_current_schema = bool(prev_grid.get("times"))
    prev_is_current_formula = prev_tropo and prev_tropo.get("index_version") == INDEX_VERSION

    if (prev_tropo and prev_is_current_schema and prev_is_current_formula
            and prev_tropo.get("status") == "ok" and prev_tropo.get("fetched_at")):
        age = now_ts - prev_tropo["fetched_at"]
        if age < MIN_REFETCH_SECONDS:
            out = dict(prev_tropo)
            out["reused"] = True
            return out

    lats, lons = grid_points()
    values, times_out, errors = fetch_grid()

    if not values or not times_out:
        return {
            "status": "error",
            "error": "; ".join(errors[:3]) if errors else "no data",
            "fetched_at": now_ts,
            "index_version": INDEX_VERSION,
        }

    n_steps = len(times_out)
    grid_values = []
    for la in lats:
        row = []
        for lo in lons:
            series = values.get((la, lo))
            if not series or len(series) != n_steps:
                series = [0.0] * n_steps
            row.append(series)
        grid_values.append(row)

    return {
        "status": "ok" if not errors else "partial",
        "errors": errors[:5],
        "fetched_at": now_ts,
        "index_version": INDEX_VERSION,
        "grid": {
            "lat_min": GRID_LAT_MIN, "lat_max": GRID_LAT_MAX, "lat_step": GRID_LAT_STEP,
            "lon_min": GRID_LON_MIN, "lon_max": GRID_LON_MAX, "lon_step": GRID_LON_STEP,
            "rows": len(lats), "cols": len(lons),
            "times": times_out,
            "values": grid_values,
        },
    }


if __name__ == "__main__":
    import json
    import sys
    result = run()
    grid = result.pop("grid", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if grid:
        print(f"grid: {grid['rows']}x{grid['cols']}x{len(grid['times'])} steps, "
              f"sample series: {grid['values'][0][0][:4]}...", file=sys.stderr)
