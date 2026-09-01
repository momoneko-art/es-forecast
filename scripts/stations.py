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


def nearest_station_by_lat(lat, lon):
    """Assign a receiver to the closest of the 4 stations, restricted to roughly the Japan longitude band."""
    if lon is None or not (122 <= lon <= 148):
        return None
    best = min(STATIONS, key=lambda s: abs(s["lat"] - lat))
    if abs(best["lat"] - lat) > 6:
        return None
    return best["id"]
