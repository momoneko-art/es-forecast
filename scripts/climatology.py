"""Seasonal / diurnal Es climatology model — the fallback baseline when no live evidence exists."""
import math
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))


def gauss(x, mu, sigma):
    d = (x - mu) / sigma
    return math.exp(-0.5 * d * d)


def jst_now():
    return datetime.now(timezone.utc).astimezone(JST)


def day_of_year(dt):
    return dt.timetuple().tm_yday


def seasonal_factor(doy, station):
    peak = gauss(doy, 197, 38)
    winter = max(gauss(doy, 0, 26), gauss(doy, 365, 26))
    v = station["floor"] + (station["baseline"] - station["floor"]) * peak + station["winter_bump"] * winter
    return min(1.0, v)


def diurnal_factor(hour_decimal):
    v = 0.10 + 0.55 * gauss(hour_decimal, 11, 2.1) + 0.62 * gauss(hour_decimal, 16.5, 2.4)
    return min(1.0, v)


def kp_modifier(station, kp):
    weight = float(station.get("aurora_sensitive") or 0.0)  # True/False from real stations, 0..1 from virtual_station_at
    if weight <= 0.0 or kp is None:
        return 1.0
    k = max(0.0, min(1.0, (kp - 3) / 6))
    return 1.0 + 0.18 * k * weight


def _interp1d(x, xs, ys):
    """Simple linear interpolation with flat extrapolation (xs must be sorted ascending)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            t = (x - xs[i]) / span if span else 0.0
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def virtual_station_at(lat, stations):
    """Build a synthetic station-like dict for an arbitrary latitude by linearly
    interpolating baseline/floor/winter_bump across the real 4 stations (sorted by
    latitude, flat extrapolation beyond Wakkanai/Oogimi). Used to extend the
    climatology baseline into a full grid for the nationwide heatmap - longitude
    is not modeled, a reasonable v1 since the 4 real anchors already span Japan
    east-west. Aurora sensitivity fades in smoothly approaching Wakkanai's
    latitude instead of the real stations' hard True/False."""
    pts = sorted(stations, key=lambda s: s["lat"])
    lats = [s["lat"] for s in pts]
    baseline = _interp1d(lat, lats, [s["baseline"] for s in pts])
    floor = _interp1d(lat, lats, [s["floor"] for s in pts])
    winter_bump = _interp1d(lat, lats, [s["winter_bump"] for s in pts])
    aurora_lats = [s["lat"] for s in pts if s.get("aurora_sensitive")]
    if aurora_lats:
        ref = min(aurora_lats)
        aurora = max(0.0, min(1.0, (lat - (ref - 6)) / 6))
    else:
        aurora = 0.0
    return {"baseline": baseline, "floor": floor, "winter_bump": winter_bump, "aurora_sensitive": aurora}


def climatology_index(dt, station, kp):
    doy = day_of_year(dt)
    hour_decimal = dt.hour + dt.minute / 60
    idx = 100 * seasonal_factor(doy, station) * diurnal_factor(hour_decimal) * kp_modifier(station, kp)
    return max(0.0, min(100.0, idx))
