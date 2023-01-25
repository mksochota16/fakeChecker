from typing import List

from pydantic import BaseModel

from app.models.base_mongo_model import MongoObjectId
from app.models.review import ReviewNewInDB


class AccountResponse(BaseModel):
    type="account_response"
    url: str
    is_fake: bool

class AccountIsPrivateResponse(BaseModel):
    type = "account_is_private_response"
    message: str = "Account is private, therefore we can't check it"

class AccountIsPrivateException(Exception):
    pass
class PlaceResponse(BaseModel):
    type = "place_response"
    number_of_reviews_scanned: int
    number_of_fake_reviews: int
    fake_percentage: float
    fake_reviews: List[ReviewNewInDB]

class NoReviewsFoundResponse(BaseModel):
    type = "no_reviews_found_response"
    message: str = "No reviews found on the specified place"
class FailedToCollectDataResponse(BaseModel):
    type = "failed_to_collect_data_response"
    message: str = "Failed to collect data from the given URL, if you are sure that the URL is correct, please call renew-markers endpoint first"
class BackgroundTaskRunningResponse(BaseModel):
    type = "background_task_running_response"
    message: str = "Background task is running, please wait the given wait-time and call check-results endpoint with the given task_id"
    task_id: MongoObjectId
    estimated_wait_time: int



