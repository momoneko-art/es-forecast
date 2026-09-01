"""Build a nationwide Es-activity grid by spatially interpolating the climatology
model (by latitude) with a Gaussian-kernel density of live PSKReporter evidence
and NICT foEs anchors. Pure Python (no numpy dependency) so it degrades along
with the rest of the pipeline rather than needing an extra optional import.
"""
import math

from stations import STATIONS
from climatology import day_of_year, diurnal_factor, seasonal_factor, kp_modifier, virtual_station_at

GRID_LAT_MIN, GRID_LAT_MAX, GRID_LAT_STEP = 24.0, 46.0, 0.5
GRID_LON_MIN, GRID_LON_MAX, GRID_LON_STEP = 122.5, 149.5, 0.5

PSK_KERNEL_SIGMA_KM = 100.0   # how far a single PSKReporter receiver's evidence spreads
PSK_KERNEL_SCALE = 2.0        # weighted-point-equivalent that saturates the boost to 1.0
NICT_KERNEL_SIGMA_KM = 160.0  # how far a single NICT ionosonde reading's evidence spreads
KERNEL_CUTOFF_SIGMAS = 3.0    # skip points further than this many sigmas (negligible weight)


def _frange(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def _km_dist(lat1, lon1, lat2, lon2):
    lat_mean_rad = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111.32 * math.cos(lat_mean_rad)
    dy = (lat2 - lat1) * 110.57
    return math.hypot(dx, dy)


def compute(now_jst, kp, psk_points, nict_stations):
    """psk_points: list of [lat, lon]. nict_stations: dict station_id -> fetch_nict station dict."""
    lats = _frange(GRID_LAT_MIN, GRID_LAT_MAX, GRID_LAT_STEP)
    lons = _frange(GRID_LON_MIN, GRID_LON_MAX, GRID_LON_STEP)

    doy = day_of_year(now_jst)
    hour_decimal = now_jst.hour + now_jst.minute / 60
    diurnal = diurnal_factor(hour_decimal)

    nict_anchors = []
    for s in STATIONS:
        st = nict_stations.get(s["id"])
        if st and st.get("esp_status") == "ok" and st.get("foes_mhz") is not None:
            boost = max(0.0, min(1.0, (st["foes_mhz"] - 2.0) / 8.0))
            if boost > 0:
                nict_anchors.append((s["lat"], s["lon"], boost))

    # Pre-filter PSK points to only those that could plausibly influence this grid's
    # bbox at all (cheap safety net; in practice fetch_pskreporter already restricts
    # to the Japan bbox).
    psk_pts = [(p[0], p[1]) for p in psk_points]

    values = []
    for lat in lats:
        vstation = virtual_station_at(lat, STATIONS)
        clima = 100 * seasonal_factor(doy, vstation) * diurnal * kp_modifier(vstation, kp)
        clima = max(0.0, min(100.0, clima))

        row = []
        for lon in lons:
            dens = 0.0
            cutoff = PSK_KERNEL_SIGMA_KM * KERNEL_CUTOFF_SIGMAS
            for (plat, plon) in psk_pts:
                if abs(plat - lat) > 3.0 or abs(plon - lon) > 3.0:
                    continue  # cheap pre-check before the real distance calc
                d = _km_dist(lat, lon, plat, plon)
                if d < cutoff:
                    dens += math.exp(-0.5 * (d / PSK_KERNEL_SIGMA_KM) ** 2)
            psk_boost = min(1.0, dens / PSK_KERNEL_SCALE)

            nict_boost = 0.0
            for (slat, slon, boost) in nict_anchors:
                d = _km_dist(lat, lon, slat, slon)
                w = boost * math.exp(-0.5 * (d / NICT_KERNEL_SIGMA_KM) ** 2)
                if w > nict_boost:
                    nict_boost = w

            combined = 1 - (1 - psk_boost) * (1 - nict_boost)
            v = max(0.0, min(100.0, clima * (1 + 0.6 * combined)))
            row.append(round(v, 1))
        values.append(row)

    return {
        "lat_min": GRID_LAT_MIN, "lat_max": GRID_LAT_MAX, "lat_step": GRID_LAT_STEP,
        "lon_min": GRID_LON_MIN, "lon_max": GRID_LON_MAX, "lon_step": GRID_LON_STEP,
        "rows": len(lats), "cols": len(lons),
        "values": values,
    }
