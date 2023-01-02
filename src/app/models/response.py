from typing import List

from pydantic import BaseModel

from app.models.review import ReviewNewInDB


class AccountResponse(BaseModel):
    is_fake: bool

class AccountIsPrivateResponse(BaseModel):
    message: str = "Account is private, therefore we can't check it"

class AccountIsPrivateException(Exception):
    pass
class PlaceResponse(BaseModel):
    disclaimer: str = "Keep in mind that our model has about 95% accuracy, therefore if 100 real reviews were checked, 5 of them could be marked as fake by mistake."
    number_of_reviews_scanned: int
    number_of_fake_reviews: int
    fake_percentage: float
    fake_reviews: List[ReviewNewInDB]

class NoReviewsFoundResponse(BaseModel):
    message: str = "No reviews found on the specified place"
class FailedToCollectDataResponse(BaseModel):
    message: str = "Failed to collect data from the given URL, if you are sure that the URL is correct, pls call renew-markers endpoint first"



