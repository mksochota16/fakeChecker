import traceback
from typing import List, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException

from app.dao.dao_reviews_new import DAOReviewsNew
from app.models.base_mongo_model import MongoObjectId
from app.models.response import PlaceResponse, NoReviewsFoundResponse, FailedToCollectDataResponse, AccountResponse, \
    AccountIsPrivateException, AccountIsPrivateResponse
from app.models.review import ReviewNewInDB
from app.services.predictions.prediction_tools import predict_reviews_from_place, predict_account
from app.services.scraper.tools.usage import ScraperUsage

app = FastAPI()

@app.get("/check-place/",
         response_model=PlaceResponse,
         responses={
             400: {"model": FailedToCollectDataResponse},
             422: {"model": NoReviewsFoundResponse}})
def place_scraper_testing(url: str, max_scroll_time: int = 10):
    usage = ScraperUsage(headless=True)
    try:
        scraper_result: Tuple[MongoObjectId, int] = usage.collect_data_from_place(url=url, max_scroll_seconds=max_scroll_time)
        scrapped_place_id: MongoObjectId = scraper_result[0]
        misread_reviews: int = scraper_result[1]
    except Exception as e:
        traceback.print_exc()
        usage.driver.quit()
        raise HTTPException(status_code=400, detail="Failed data collection")
        # return FailedToCollectDataResponse()

    if misread_reviews > 5:
        print("Too many misread reviews")
    usage.driver.quit()
    predict_reviews_from_place(scrapped_place_id)
    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()
    all_reviews_from_current_place: List[ReviewNewInDB] = dao_reviews_new.find_many_by_query({'place_id': scrapped_place_id})
    fake_reviews_from_current_scrape: List[ReviewNewInDB] = dao_reviews_new.find_many_by_query({'place_id': scrapped_place_id, 'is_real': False})
    if len(all_reviews_from_current_place) > 0:
        percentage_of_fake_reviews = len(fake_reviews_from_current_scrape) / len(all_reviews_from_current_place) * 100
        return PlaceResponse(
            number_of_reviews_scanned=len(all_reviews_from_current_place),
            number_of_fake_reviews=len(fake_reviews_from_current_scrape),
            fake_percentage=percentage_of_fake_reviews,
            fake_reviews = fake_reviews_from_current_scrape
        )
    else:
        raise HTTPException(status_code=422, detail="No reviews found")
        # return NoReviewsFoundResponse()
@app.get("/check-account/",
         response_model=AccountResponse,
         responses={
             400: {"model": FailedToCollectDataResponse},
             422: {"model": AccountIsPrivateResponse}})
def account_scraper(url: str, max_scroll_time: int = 10):
    usage = ScraperUsage(headless=True)
    try:
        scraper_result: Tuple[MongoObjectId, int] = usage.collect_data_from_person(url=url, max_scroll_seconds=max_scroll_time)
        scrapped_account_id: MongoObjectId = scraper_result[0]
        misread_reviews: int = scraper_result[1]
    except Exception as e:
        traceback.print_exc()
        usage.driver.quit()
        raise HTTPException(status_code=400, detail="Failed data collection")
        # return FailedToCollectDataResponse()
    usage.driver.quit()
    if misread_reviews > 5:
        print("Too many misread reviews")
    try:
        prediction = predict_account(account_id=scrapped_account_id)
    except AccountIsPrivateException:
        raise HTTPException(status_code=422, detail="Account is private")
        # return AccountIsPrivateResponse()
    return AccountResponse(is_fake=prediction)

@app.get("/renew-markers/")
def renew_html_markers():
    usage = ScraperUsage(headless=True)
    usage.discover_new_markers()
    usage.driver.quit()
    return {"message": "Markers renewed"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)