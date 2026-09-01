"""Shared station definitions and Maidenhead grid helpers for the Es forecast pipeline."""
import math

# The 4 NICT ionosonde sites referenced by the original EDFS blog post.
STATIONS = [
    {"id": "wakkanai",  "name": "稚内",   "loc": "北海道",   "lat": 45.4, "lon": 141.7,
     "baseline": 0.55, "floor": 0.10, "winter_bump": 0.20, "aurora_sensitive": True},
    {"id": "kokubunji", "name": "国分寺", "loc": "東京都",   "lat": 35.7, "lon": 139.5,
     "baseline": 0.80, "floor": 0.09, "winter_bump": 0.05, "aurora_sensitive": False},
    {"id": "yamagawa",  "name": "山川",   "loc": "鹿児島県", "lat": 31.2, "lon": 130.6,
     "baseline": 0.94, "floor": 0.11, "winter_bump": 0.02, "aurora_sensitive": False},
    {"id": "oogimi",    "name": "大宜味", "loc": "沖縄県",   "lat": 26.7, "lon": 128.2,
     "baseline": 1.00, "floor": 0.14, "winter_bump": 0.00, "aurora_sensitive": False},
]


def maidenhead_to_latlon(locator):
    """Decode a 4 or 6 character Maidenhead grid locator to (lat, lon). Returns None on failure."""
    if not locator or len(locator) < 4:
        return None
    loc = locator.strip().upper()
    try:
        lon = (ord(loc[0]) - ord('A')) * 20 - 180
        lat = (ord(loc[1]) - ord('A')) * 10 - 90
        lon += int(loc[2]) * 2
        lat += int(loc[3]) * 1
        if len(loc) >= 6 and loc[4].isalpha() and loc[5].isalpha():
            lon += (ord(loc[4].lower()) - ord('a')) * (2 / 24)
            lat += (ord(loc[5].lower()) - ord('a')) * (1 / 24)
        else:
            lon += 1
            lat += 0.5
        return (lat, lon)
    except (ValueError, IndexError):
        return None


JAPAN_BBOX = {"lat_min": 24.0, "lat_max": 46.0, "lon_min": 122.5, "lon_max": 149.5}


def in_japan_bbox(lat, lon):
    """Loose bounding-box check used for the nationwide heatmap (broader than the
    4-station assignment below, so it also captures reports away from the anchors)."""
    if lat is None or lon is None:
        return False
    return (JAPAN_BBOX["lat_min"] <= lat <= JAPAN_BBOX["lat_max"] and
            JAPAN_BBOX["lon_min"] <= lon <= JAPAN_BBOX["lon_max"])


def nearest_station_by_lat(lat, lon):
    """Assign a receiver to the closest of the 4 stations, restricted to roughly the Japan longitude band."""
    if lon is None or not (122 <= lon <= 148):
        return None
    best = min(STATIONS, key=lambda s: abs(s["lat"] - lat))
    if abs(best["lat"] - lat) > 6:
        return None
    return best["id"]


def great_circle_km(lat1, lon1, lat2, lon2):
    """Standard haversine great-circle distance in km. Shared by the pair-distance
    tracking in fetch_pskreporter.py (the nationwide heatmap keeps its own copy in
    heatmap.py to avoid touching working code)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
