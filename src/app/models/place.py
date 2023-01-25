import uuid
from typing import Optional, Any

from pydantic import BaseModel

from app.models.base_mongo_model import MongoDBModel
from app.models.position import Position
from app.models.types_cluster import CLUSTER_TYPES


class PlaceBase(BaseModel):
    name: str
    url: str
    address: str
    number_of_reviews: int
    localization: Optional[Position]
    rating: float
    type_of_object: str
    cluster: Optional[CLUSTER_TYPES]
    identifier: Optional[str]

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.identifier = f"{self.name} {self.address}"


class Place(PlaceBase):
    pass

class PlaceInDB(PlaceBase, MongoDBModel):
    pass