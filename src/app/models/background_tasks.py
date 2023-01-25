from enum import Enum
from typing import Union

from pydantic import BaseModel

from app.models.base_mongo_model import MongoObjectId, MongoDBModel
from app.models.response import BackgroundTaskRunningResponse, AccountResponse, AccountIsPrivateResponse, PlaceResponse, FailedToCollectDataResponse, NoReviewsFoundResponse

FAKE_CHECKER_ACCOUNT_RESPONSES = Union[AccountResponse, AccountIsPrivateResponse, FailedToCollectDataResponse, BackgroundTaskRunningResponse]
FAKE_CHECKER_PLACE_RESPONSES = Union[PlaceResponse, NoReviewsFoundResponse, FailedToCollectDataResponse, BackgroundTaskRunningResponse]
class BackgroundTaskTypes(str, Enum):
    CHECK_ACCOUNT = "check_account"
    CHECK_PLACE = "check_place"
    RENEW_MARKERS = "renew_markers"
    GET_MORE_DATA = "get_more_data"
    RUNNING = "running"

class BackgroundTask(BaseModel):
    task_id: MongoObjectId
    type: BackgroundTaskTypes
    fake_checker_response: Union[BaseModel, dict]


class BackgroundTaskInDB(BackgroundTask, MongoDBModel):
    pass

class BackgroundTaskAccount(BackgroundTask):
    type: BackgroundTaskTypes = BackgroundTaskTypes.CHECK_ACCOUNT
    fake_checker_response: FAKE_CHECKER_ACCOUNT_RESPONSES

class BackgroundTaskPlace(BackgroundTask):
    type: BackgroundTaskTypes = BackgroundTaskTypes.CHECK_PLACE
    fake_checker_response: FAKE_CHECKER_PLACE_RESPONSES

class BackgroundTaskRenewMarkers(BackgroundTask):
    type: BackgroundTaskTypes = BackgroundTaskTypes.RENEW_MARKERS
    fake_checker_response: dict = {"message": "Markers renewed"}

class BackgroundTaskGetMoreData(BackgroundTask):
    type: BackgroundTaskTypes = BackgroundTaskTypes.GET_MORE_DATA
    fake_checker_response: dict

class BackgroundTaskRunning(BackgroundTask):
    type: BackgroundTaskTypes = BackgroundTaskTypes.RUNNING
    future_type: BackgroundTaskTypes
    fake_checker_response: dict = {"message": "Background task is running"}


