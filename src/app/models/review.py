from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from models.base_mongo_model import MongoDBModel, MongoObjectId
from models.position import Position
from models.types_cluster import CLUSTER_TYPES
from services.analysis.geolocation import is_in_poland


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
    localization: Optional[Position]
    type_of_object: Optional[str]
    cluster: Optional[CLUSTER_TYPES]

    full_flag: Optional[bool]
    wrong_address: Optional[bool]
    wrong_url_flag: Optional[bool]

    sentiment_rating: Optional[float]
    test_prediction: Optional[bool]

    account_id: Optional[MongoObjectId]

class ReviewOld(ReviewOldBase):
    pass

class ReviewOldInDB(ReviewOldBase, MongoDBModel):
    pass

    def to_dict(self):
        localization = self.localization.dict() if self.localization else None
        return {
            "_id": self.id,
            "review_id": self.review_id,
            "place_name": self.place_name,
            "rating": self.rating,
            "content": self.content,
            "reviewer_url": self.reviewer_url,
            "reviewer_id": self.reviewer_id,
            "place_url": self.place_url,
            "localization": localization,
            "photos_urls": self.photos_urls,
            "type_of_object": self.type_of_object,
            "response_content": self.response_content,
            "date": self.date,
            "is_real": self.is_real,
            "account_id": self.account_id,
            "cluster": self.cluster

        }

class ReviewNewBase(ReviewBase):
    is_local_guide: bool
    number_of_reviews: int
    profile_photo_url: Optional[str]
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
    localization: Optional[Position]
    rating: int
    date: datetime
    content: Optional[str]
    response_content: Optional[str]
    photos_urls: Optional[List[str]]
    new_scrape: Optional[bool] = False

class ReviewPartial(ReviewPartialBase):
    pass

class ReviewPartialInDB(ReviewPartialBase, MongoDBModel):
    scraped_fully: Optional[bool] = False
    pass

class ReviewInGMR_PLDB(MongoDBModel):
    review_id: str
    rating: int
    content: str
    reviewer_url: str
    reviewer_id: str
    photos_urls: Optional[List[str]]
    response_content: Optional[str]
    date: datetime
    is_real: Optional[bool]
    place_name: str
    place_url: str
    localization: Optional[Position]
    type_of_object: Optional[str]
    cluster: Optional[CLUSTER_TYPES]
    content_not_full: bool = False
    content_translated: bool = False
    not_in_poland: bool = False
    localization_missing: bool = False

    account_id: Optional[MongoObjectId]

    def to_dict(self):
        localization = self.localization.dict() if self.localization else None
        return {
            "_id": self.id,
            "review_id": self.review_id,
            "place_name": self.place_name,
            "rating": self.rating,
            "content": self.content,
            "reviewer_url": self.reviewer_url,
            "reviewer_id": self.reviewer_id,
            "place_url": self.place_url,
            "localization": localization,
            "photos_urls": self.photos_urls,
            "type_of_object": self.type_of_object,
            "response_content": self.response_content,
            "date": self.date,
            "is_real": self.is_real,
            "account_id": self.account_id,
            "cluster": self.cluster,
            "content_not_full": self.content_not_full,
            "content_translated": self.content_translated,
            "not_in_poland": self.not_in_poland,
            "localization_missing": self.localization_missing

        }

    @classmethod
    def from_old_model(cls, review_old: ReviewOldInDB):
        return cls(
            review_id=review_old.review_id,
            rating=review_old.rating,
            content=review_old.content,
            reviewer_url=review_old.reviewer_url,
            reviewer_id=review_old.reviewer_id,
            photos_urls=review_old.photos_urls,
            response_content=review_old.response_content,
            date=review_old.date,
            is_real=review_old.is_real,
            place_name=review_old.place_name,
            place_url=review_old.place_url,
            localization=review_old.localization,
            type_of_object=review_old.type_of_object,
            cluster=review_old.cluster,
            account_id=review_old.account_id,
            _id=review_old.id,
            content_not_full="   Więcej" in review_old.content,
            content_translated="(Przetłumaczone przez Google)" in review_old.content,
            not_in_poland=review_old.localization is not None and (not is_in_poland(review_old.localization.lat, review_old.localization.lon)),
            localization_missing=review_old.localization is None

        )

class ReviewInAnonymisedGMR_PLDB(MongoDBModel):
    rating: int
    content: str
    photos_urls: Optional[List[str]]
    response_content: Optional[str]
    date: datetime
    is_real: Optional[bool]

    approximate_localization: Optional[Position]
    type_of_object: Optional[str]
    cluster: Optional[CLUSTER_TYPES]

    content_not_full: bool = False
    content_translated: bool = False
    not_in_poland: bool = False
    localization_missing: bool = False

    account_id: MongoObjectId

    def to_dict(self):
        approximate_localization = self.approximate_localization.dict() if self.approximate_localization else None
        return {
            "_id": self.id,
            "rating": self.rating,
            "content": self.content,
            "photos_urls": self.photos_urls,
            "response_content": self.response_content,
            "date": self.date,
            "is_real": self.is_real,
            "approximate_localization": approximate_localization,
            "type_of_object": self.type_of_object,
            "cluster": self.cluster,
            "account_id": self.account_id,
            "content_not_full": self.content_not_full,
            "content_translated": self.content_translated,
            "not_in_poland": self.not_in_poland,
            "localization_missing": self.localization_missing

        }