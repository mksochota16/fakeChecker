from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from bson.objectid import ObjectId

from app.config import NLP, ENGLISH_TRANSLATION_CLUSTER_DICT
from app.models.account import AccountOldInDB
from app.models.base_mongo_model import MongoDBModel, MongoObjectId
from app.models.place import PlaceInDB
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

    def parse_to_prediction_list(self) -> list:
        pass

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

    def parse_to_prediction_list(self, account: AccountOldInDB, exclude_localization = True) -> list:
        number_of_reviews = account.number_of_reviews
        reviewer_name = account.name
        is_local_guide = account.local_guide_level is not None
        if exclude_localization:
            localization = None
        else:
            localization = self.localization

        return _parse_review_data_to_prediction_list(number_of_reviews,
                                                     reviewer_name,
                                                     self.cluster,
                                                     is_local_guide,
                                                     localization,
                                                     self)

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

    def parse_to_prediction_list(self, place_in_db: PlaceInDB, exclude_localization = True) -> list:
        if exclude_localization:
            localization = None
        else:
            localization = place_in_db.localization
        return _parse_review_data_to_prediction_list(self.number_of_reviews,
                                          self.reviewer_name,
                                          place_in_db.cluster,
                                          self.is_local_guide,
                                          localization,
                                          self)

class ReviewNew(ReviewNewBase):
    pass

class ReviewNewInDB(ReviewNewBase, MongoDBModel):
    pass

def _parse_review_data_to_prediction_list(number_of_reviews: int,
                                          reviewer_name: str,
                                          cluster_name: str,
                                          is_local_guide: bool,
                                          localization: Optional[Position],
                                          review: ReviewBase) -> list:
    if number_of_reviews is None:
        number_of_reviews = 0
    review_data = [number_of_reviews, is_local_guide]

    name_score = NLP.analyze_name_of_account(reviewer_name)
    review_data.append(name_score)

    review_data.append(review.rating)
    clusters_list = list(ENGLISH_TRANSLATION_CLUSTER_DICT.values())
    review_data.append(clusters_list.index(cluster_name))

    review_data.append(len(review.content))
    if review.response_content is not None:
        review_data.append(len(review.response_content))
    else:
        review_data.append(0)
    review_data.append(NLP.sentiment_analyzer.analyze(review.content))

    review_data.append(NLP.get_capslock_score(review.content))
    review_data.append(NLP.get_interpunction_score(review.content))

    # review_data.append(NLP.get_emotional_interpunction_score(review.content))
    # review_data.append(NLP.get_consecutive_emotional_interpunction_score(review.content))
    # review_data.append(NLP.get_emojis_score(review.content))

    if localization is not None:
        review_data.append(localization.lat)
        review_data.append(localization.lon)

    return review_data
