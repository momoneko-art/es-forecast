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


def nict_floor_from_foes(foes_mhz):
    """Live-measurement floor for es_index, independent of time-of-day climatology.

    foEs (the sporadic-E layer's critical frequency, MHz), measured directly by an
    ionosonde, is ground truth for what is happening RIGHT NOW - unlike the
    climatology model above, which only encodes "how likely is Es at this hour of
    day / time of year" and can therefore badly under-react to a real event that
    lands outside its usual peak hours (e.g. diurnal_factor is tiny at 06:45, long
    before the 11:00/16:30 climatological peaks, so a real early-morning Es event
    was getting multiplied down to almost nothing before this floor existed).

    The floor is grounded in the classic single-hop Es MUF rule of thumb used by
    HF/VHF DXers: MUF(oblique) is roughly 3x foEs for typical single-hop
    distances (~1000-2200km). So foEs ~9MHz implies a maximum usable frequency
    near 27MHz - i.e. right in the 11m/CB band this app is built for - and the
    thresholds below are chosen so the floor crosses into the "watch"(30) /
    "high"(50) dashboard tiers right around the foEs values that imply an 11m
    opening is becoming plausible, not an arbitrary curve. This is still a rough
    heuristic (the real oblique multiplication factor varies ~2.5-4x with hop
    distance/geometry), not a verified MUF prediction."""
    if foes_mhz is None:
        return 0.0
    f = foes_mhz
    if f < 4.0:
        return 0.0
    if f < 6.0:
        return 20.0 * (f - 4.0) / 2.0
    if f < 8.0:
        return 20.0 + 15.0 * (f - 6.0) / 2.0
    if f < 10.0:
        return 35.0 + 20.0 * (f - 8.0) / 2.0
    if f < 14.0:
        return 55.0 + 25.0 * (f - 10.0) / 4.0
    return min(95.0, 80.0 + 15.0 * min(1.0, (f - 14.0) / 6.0))


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
