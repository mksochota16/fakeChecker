class Position:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def __str__(self):
        return str(F"[{self.lat}, {self.lon}]")

    def to_dict(self):
        return {
            "lat": self.lat,
            "lon": self.lon
        }
