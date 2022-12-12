from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from bson.objectid import ObjectId

from app.models.position import Position
from app.models.types_cluster import CLUSTER_TYPES


class ReviewBase(BaseModel):
    review_id: str
    place_name: str
    rating: int
    content: str
    reviewer_url: str
    reviewer_id: str
    place_url: str
    localization: Position
    photos_urls: Optional[List[str]]
    type_of_object: str
    response_content: Optional[str]
    date: datetime
    is_private: Optional[bool]
    cluster: Optional[CLUSTER_TYPES]
    is_real: Optional[bool]

class ReviewOldBase(ReviewBase):
    full_flag: bool
    wrong_address: Optional[bool]
    wrong_url_flag: Optional[bool]

class ReviewOldInDB(ReviewOldBase):
    _id: ObjectId

class ReviewNewBase(ReviewBase):
    place_address: str
    place_number_of_reviews: int
    is_local_guide: bool
    number_of_opinions: int
    profile_photo_url: str
    reviewer_name: str

class ReviewNewInDB(ReviewNewBase):
    _id: ObjectId
