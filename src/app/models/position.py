from pydantic import BaseModel


class Position(BaseModel):
    lat: float
    lon: float

    def __str__(self):
        return str(F"[{self.lat}, {self.lon}]")