from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.models.base_mongo_model import MongoDBModel, MongoObjectId
from app.models.position import Position
from app.models.types_cluster import CLUSTER_TYPES


class ReviewBase(BaseModel):
    review_id: str
    rating: int
    content: str
    reviewer_url: str
    reviewer_id: str
    photos_urls: Optional[List[str]]
    response_content: Optional[str]
    date: datetime
    is_private: Optional[bool]
    is_real: Optional[bool]

class ReviewOldBase(ReviewBase):
    place_name: str
    place_url: str
    localization: Position
    type_of_object: str
    cluster: Optional[CLUSTER_TYPES]

    full_flag: bool
    wrong_address: Optional[bool]
    wrong_url_flag: Optional[bool]

    sentiment_rating: Optional[float]
    test_prediction: Optional[bool]

class ReviewOld(ReviewOldBase):
    pass

class ReviewOldInDB(ReviewOldBase, MongoDBModel):
    pass

class ReviewNewBase(ReviewBase):
    is_local_guide: bool
    number_of_reviews: int
    profile_photo_url: str
    reviewer_name: str
    place_id: MongoObjectId

class ReviewNew(ReviewNewBase):
    pass

class ReviewNewInDB(ReviewNewBase, MongoDBModel):
    pass


class ReviewPartialBase(BaseModel):
    review_id: str
    reviewer_id: str
    place_name: str
    place_address: str
    localization: Position
    rating: int
    date: datetime
    content: Optional[str]
    response_content: Optional[str]
    photos_urls: Optional[List[str]]

class ReviewPartial(ReviewPartialBase):
    pass

class ReviewPartialInDB(ReviewPartialBase, MongoDBModel):
    pass