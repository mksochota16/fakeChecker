import traceback
from typing import List, Tuple

import uvicorn
from bson import ObjectId
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from config import ADMIN_API_KEY
from dao.dao_accounts_new import DAOAccountsNew
from dao.dao_background_tasks import DAOBackgroundTasks
from dao.dao_places import DAOPlaces
from dao.dao_reviews_new import DAOReviewsNew
from dao.dao_reviews_partial import DAOReviewsPartial
from models.account import AccountNewInDB
from models.background_tasks import BackgroundTaskPlace, BackgroundTaskRunning, BackgroundTaskTypes, \
    BackgroundTaskAccount, BackgroundTaskRenewMarkers, BackgroundTaskGetMoreData
from models.base_mongo_model import MongoObjectId
from models.place import Place, PlaceInDB
from models.response import PlaceResponse, NoReviewsFoundResponse, FailedToCollectDataResponse, AccountResponse, \
    AccountIsPrivateException, AccountIsPrivateResponse, BackgroundTaskRunningResponse
from models.review import ReviewNewInDB, ReviewPartialInDB
from services.predictions.prediction_tools import predict_reviews_from_place, predict_account
from services.scraper.tools.usage import ScraperUsage

app = FastAPI()

@app.get("/check-place/",
         response_model=BackgroundTaskRunningResponse)
def place_scraper(url: str, background_tasks: BackgroundTasks,
                          max_scroll_time: int = 10):
    dao_background_tasks = DAOBackgroundTasks()
    mongo_object_id = MongoObjectId()
    background_running = BackgroundTaskRunning(
        task_id=mongo_object_id,
        future_type=BackgroundTaskTypes.CHECK_PLACE)
    dao_background_tasks.insert_one(background_running)
    background_tasks.add_task(_place_scraper, url=url, mongo_object_id=mongo_object_id, max_scroll_time=max_scroll_time)
    return BackgroundTaskRunningResponse(
        task_id=mongo_object_id,
        estimated_wait_time=max_scroll_time*4
    )

def _place_scraper(url: str, mongo_object_id: MongoObjectId, max_scroll_time: int = 10, new_scrape = False):
    usage = ScraperUsage(headless=True)
    dao_background_tasks = DAOBackgroundTasks()
    try:
        scraper_result: Tuple[MongoObjectId, int] = usage.collect_data_from_place(url=url,
                                                                                  max_scroll_seconds=max_scroll_time,
                                                                                  new_scrape=new_scrape)
        scrapped_place_id: MongoObjectId = scraper_result[0]
        misread_reviews: int = scraper_result[1]
    except Exception as e:
        traceback.print_exc()
        usage.driver.quit()
        background_task_result = BackgroundTaskPlace(
            task_id = mongo_object_id,
            fake_checker_response = FailedToCollectDataResponse()
        )
        dao_background_tasks.replace_one(background_task_result)
        return

    if misread_reviews > 5:
        print("Too many misread reviews")
    usage.driver.quit()
    predict_reviews_from_place(scrapped_place_id)
    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()
    all_reviews_from_current_place: List[ReviewNewInDB] = dao_reviews_new.find_many_by_query(
        {'place_id': scrapped_place_id})
    fake_reviews_from_current_scrape: List[ReviewNewInDB] = dao_reviews_new.find_many_by_query(
        {'place_id': scrapped_place_id, 'is_real': False})
    if len(all_reviews_from_current_place) > 0:
        percentage_of_fake_reviews = len(fake_reviews_from_current_scrape) / len(all_reviews_from_current_place) * 100
        place_response =  PlaceResponse(
            number_of_reviews_scanned=len(all_reviews_from_current_place),
            number_of_fake_reviews=len(fake_reviews_from_current_scrape),
            fake_percentage=percentage_of_fake_reviews,
            fake_reviews=fake_reviews_from_current_scrape
        )
        background_task_result = BackgroundTaskPlace(
            task_id=mongo_object_id,
            fake_checker_response=place_response
        )
        dao_background_tasks.replace_one(background_task_result)
        return
    else:
        background_task_result = BackgroundTaskPlace(
            task_id=mongo_object_id,
            fake_checker_response=NoReviewsFoundResponse()
        )
        dao_background_tasks.replace_one(background_task_result)
        return

@app.get("/check-account/",
         response_model=BackgroundTaskRunningResponse)
def account_scraper(url: str, background_tasks: BackgroundTasks, max_scroll_time: int = 10):
    dao_background_tasks = DAOBackgroundTasks()
    mongo_object_id = MongoObjectId()
    background_running = BackgroundTaskRunning(
        task_id=mongo_object_id,
        future_type=BackgroundTaskTypes.CHECK_ACCOUNT)
    dao_background_tasks.insert_one(background_running)
    background_tasks.add_task(_account_scraper, url=url, mongo_object_id=mongo_object_id, max_scroll_time=max_scroll_time)
    return BackgroundTaskRunningResponse(
        task_id=mongo_object_id,
        estimated_wait_time=max_scroll_time * 4
    )

def _account_scraper(url: str, mongo_object_id: MongoObjectId, max_scroll_time: int = 10):
    usage = ScraperUsage(headless=True)
    dao_background_tasks = DAOBackgroundTasks()
    try:
        scraper_result: Tuple[MongoObjectId, int] = usage.collect_data_from_person(url=url, max_scroll_seconds=max_scroll_time)
        scrapped_account_id: MongoObjectId = scraper_result[0]
        misread_reviews: int = scraper_result[1]
    except Exception as e:
        traceback.print_exc()
        usage.driver.quit()
        background_task_result = BackgroundTaskAccount(
            task_id=mongo_object_id,
            fake_checker_response=FailedToCollectDataResponse()
        )
        dao_background_tasks.replace_one(background_task_result)
        return
    usage.driver.quit()
    if misread_reviews > 5:
        print("Too many misread reviews")
    try:
        prediction = predict_account(account_id=scrapped_account_id)
    except AccountIsPrivateException:
        background_task_result = BackgroundTaskAccount(
            task_id=mongo_object_id,
            fake_checker_response=AccountIsPrivateResponse()
        )
        dao_background_tasks.replace_one(background_task_result)
        return

    response =  AccountResponse(url= url, is_fake=prediction)
    background_task_result = BackgroundTaskAccount(
        task_id=mongo_object_id,
        fake_checker_response=response
    )
    dao_background_tasks.replace_one(background_task_result)
    return

@app.get("/renew-markers/")
def renew_html_markers(background_tasks: BackgroundTasks):
    dao_background_tasks = DAOBackgroundTasks()
    mongo_object_id = MongoObjectId()
    background_running = BackgroundTaskRunning(
        task_id=mongo_object_id,
        future_type=BackgroundTaskTypes.RENEW_MARKERS)
    dao_background_tasks.insert_one(background_running)
    background_tasks.add_task(_renew_html_markers, mongo_object_id=mongo_object_id)
    return BackgroundTaskRunningResponse(
        task_id=mongo_object_id,
        estimated_wait_time=90
    )

def _renew_html_markers(mongo_object_id: MongoObjectId):
    usage = ScraperUsage(headless=True)
    dao_background_tasks = DAOBackgroundTasks()
    usage.discover_new_markers()
    usage.driver.quit()
    background_task_result = BackgroundTaskRenewMarkers(
        task_id=mongo_object_id
    )
    dao_background_tasks.replace_one(background_task_result)
    return

@app.get("/get-more-data/")
def get_more_data(admin_api_key: str, background_tasks: BackgroundTasks,  ignore_exceptions: bool = False, collection_limit: int = 5):
    if admin_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="API key is invalid")
    dao_background_tasks = DAOBackgroundTasks()
    mongo_object_id = MongoObjectId()
    background_running = BackgroundTaskRunning(
        task_id=mongo_object_id,
        future_type=BackgroundTaskTypes.RENEW_MARKERS)
    dao_background_tasks.insert_one(background_running)
    background_tasks.add_task(_get_more_data, mongo_object_id=mongo_object_id, ignore_exceptions=ignore_exceptions, collection_limit=collection_limit)
    return BackgroundTaskRunningResponse(
        task_id=mongo_object_id,
        estimated_wait_time=collection_limit*20*4
    )

def _get_more_data(mongo_object_id: MongoObjectId, ignore_exceptions: bool, collection_limit: int = 5):

    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()
    dao_accounts_new: DAOAccountsNew = DAOAccountsNew()
    dao_background_tasks = DAOBackgroundTasks()
    all_reviews: List[ReviewNewInDB] = dao_reviews_new.find_many_by_query({})
    usage = ScraperUsage(headless=True)
    scrapped_accounts = 0
    for review in all_reviews:
        try:
            res = dao_accounts_new.find_one_by_query({'reviewer_id': review.reviewer_id})
            if res is not None:
                continue
        except:
            pass

        try:
            usage.collect_data_from_person(reviewer_id=review.reviewer_id, max_scroll_seconds=20, url="")
        except Exception as e:
            if ignore_exceptions:
                traceback.print_exc()
                continue
            else:
                traceback.print_exc()
                usage.driver.quit()
                background_task_result = BackgroundTaskGetMoreData(
                    task_id=mongo_object_id,
                    fake_checker_response={
                        "error": str(e),
                        "Collected accounts count": scrapped_accounts
                    }
                )
                dao_background_tasks.replace_one(background_task_result)
                return

        scrapped_accounts+=1
        if scrapped_accounts >= collection_limit:
            break

    usage.driver.quit()
    background_task_result = BackgroundTaskGetMoreData(
        task_id=mongo_object_id,
        fake_checker_response={"Collected accounts count": scrapped_accounts}
    )
    dao_background_tasks.replace_one(background_task_result)
    return

@app.get("/check-results/")
def check_results(results_id: str):
    dao_background_tasks = DAOBackgroundTasks()
    result: dict = dao_background_tasks.find_one_by_query_return_raw({'task_id': ObjectId(results_id)})
    if result is not None:
        return result
    else:
        raise HTTPException(status_code=404, detail="Task not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)