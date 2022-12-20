from math import cos, asin, sqrt, pi

from app.services.scraper.models.position import Position as PositionOld
from app.models.position import Position as PositionNew


# returns distance in km on the globe
def distance(pos1, pos2):
    if isinstance(pos1, PositionOld) or isinstance(pos1, PositionNew):
        lat1 = float(pos1.lat)
        lon1 = float(pos1.lon)
    else:
        lat1 = float(pos1[0])
        lon1 = float(pos1[1])
    if isinstance(pos2, PositionOld) or isinstance(pos2, PositionNew):
        lat2 = float(pos2.lat)
        lon2 = float(pos2.lon)
    else:
        lat2 = float(pos2[0])
        lon2 = float(pos2[1])
    p = pi / 180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))


def is_in_poland(lat, lon):
    return (49 < float(lat) < 55) and (14 < float(lon) < 24)
