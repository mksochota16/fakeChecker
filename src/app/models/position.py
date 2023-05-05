from pydantic import BaseModel
from services.scraper.models.position import Position as OldPosition

class Position(BaseModel):
    lat: float
    lon: float

    def __str__(self):
        return str(F"[{self.lat}, {self.lon}]")

    def to_old_model(self) -> OldPosition:
        return OldPosition(self.lat, self.lon)

    def make_approximation(self, precision: int = 1):
        return Position(
            lat = round(self.lat, precision),
            lon = round(self.lon, precision))