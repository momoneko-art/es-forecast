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
    if not station.get("aurora_sensitive") or kp is None:
        return 1.0
    k = max(0.0, min(1.0, (kp - 3) / 6))
    return 1.0 + 0.18 * k


def climatology_index(dt, station, kp):
    doy = day_of_year(dt)
    hour_decimal = dt.hour + dt.minute / 60
    idx = 100 * seasonal_factor(doy, station) * diurnal_factor(hour_decimal) * kp_modifier(station, kp)
    return max(0.0, min(100.0, idx))
