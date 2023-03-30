from pydantic import BaseModel
from app.services.scraper.models.position import Position as OldPosition

class Position(BaseModel):
    lat: float
    lon: float

    def __str__(self):
        return str(F"[{self.lat}, {self.lon}]")

    def to_old_model(self) -> OldPosition:
        return OldPosition(self.lat, self.lon)