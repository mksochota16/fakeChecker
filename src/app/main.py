from collections import Counter
from typing import List

import uvicorn
from fastapi import FastAPI
from matplotlib import pyplot as plt

from app.config import NLP
from app.dao.dao_reviews_old import DAOReviewsOld
from app.models.base_mongo_model import MongoObjectId
from app.models.review import ReviewOldInDB
from app.services.analysis.nlp_analysis import StyloMetrixResults
from app.services.predictions.prediction_tools import predict_reviews_from_place, get_and_prepare_accounts_data, \
    get_and_prepare_reviews_data, get_prepared_reviews_data_from_file, get_train_and_test_datasets, \
    build_model_return_predictions, calculate_basic_metrics, calculate_metrics, k_fold_validation, \
    predict_all_reviews_from_new_scrape
from app.services.scraper.tools.usage import ScraperUsage

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

def get_vote_statistics():
    results_dict ={
    "-7 ": 112,
    "-6 ": 7,
    "-5 ": 39,
    "-4 ": 3,
    "-3 ": 19,
    "-2 ": 3,
    "-1 ": 3,
    "0 ": 1,
    "1 ": 1,
    "2 ": 0,
    "3 ": 3,
    "4 ": 2,
    "5 ": 18,
    "6 ": 1,
    "7 ": 108
    }
    plt.bar(results_dict.keys(), results_dict.values(), 0.9)
    plt.ylabel('Number of accounts')
    plt.yscale('log')
    plt.xlabel('Voting results')
    plt.show()

def scraper_testing():
    usage = ScraperUsage(headless=True)
    # usage.discover_new_markers()
    scrapped_place_id: MongoObjectId = usage.collect_data_from_place(
        url="https://www.google.pl/maps/place/Salon+meblowy+Black+Red+White+-+meble+Warszawa/@52.1915787,20.9598395,14z/data=!4m5!3m4!1s0x471934adc20fc469:0xeffca447261b4baa!8m2!3d52.2030371!4d20.9362185")
    usage.driver.quit()
    predict_reviews_from_place(scrapped_place_id)

def prediction_testing():
    #get_and_prepare_accounts_data(save_to_file=False)
    # get_and_prepare_reviews_data(save_to_file=True, exclude_localization=True)
    data = get_prepared_reviews_data_from_file(file_name = 'app/data/formatted_reviews_data_w_sentiment_caps_inter.csv', exclude_localization=True) # get_prepared_accounts_data_from_file(ignore_empty_accounts=True) # get_and_prepare_accounts_data(save_to_file=True)
    # for i in range(20):
    #     prepared_data = get_train_and_test_datasets(3/5, data, resolve_backpack_problem=True)
    #     predicts = build_model_return_predictions(prepared_data[0], prepared_data[1], prepared_data[2])
    #     TP, TN, FP, FN, ALL = calculate_basic_metrics(predicts, prepared_data[3])
    #     calculate_metrics(TP, TN, FP, FN, ALL)
    k_fold_validation(10, data, resolve_backpack_problem=True)
    predict_all_reviews_from_new_scrape()
    # print("FINISHED")

def sentiment_testing():
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
    # all_reviews: List[ReviewOldInDB] = dao_reviews_old.find_all()
    # counter = 0
    # print("#"*20)
    # for review in all_reviews:
    #     sentiment_rating = SENTIMENT_ANALYZER.analyze(review.content)
    #     dao_reviews_old.update_one({"_id": review.id}, {"$set":{'sentiment_rating': sentiment_rating}})
    #     counter += 1
    #     if counter >= len(all_reviews)/20:
    #         counter = 0
    #         print("#", end="")

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real':False, "rating": 5})
    real_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True, "rating": 5})
    fake_sentiment: List[float] = [review.sentiment_rating for review in fake_reviews if (review.sentiment_rating > 0 or review.sentiment_rating < -0.25)]
    real_sentiment: List[float] = [review.sentiment_rating for review in real_reviews if (review.sentiment_rating > 0 or review.sentiment_rating < -0.25)]

    plt.hist(fake_sentiment, bins=20, range=[-1,1])
    plt.title(f'Sentiment of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Sentiment value')
    plt.show()

    plt.hist(real_sentiment, bins=20, range=[-1, 1])
    plt.title(f'Sentiment of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Sentiment value')
    plt.show()

def nlp_testing():
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False, "rating": 5})
    real_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True, "rating": 5})
    fake_sentiment: List[float] = [review.sentiment_rating for review in fake_reviews]
    real_sentiment: List[float] = [review.sentiment_rating for review in real_reviews]


    plt.hist(fake_sentiment, bins=20, range=[-1, 1])
    plt.title(f'Sentiment of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Sentiment value')
    plt.show()

    plt.hist(real_sentiment, bins=20, range=[-1, 1])
    plt.title(f'Sentiment of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Sentiment value')
    plt.show()

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real':False, "rating": 5})
    real_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True, "rating": 5})
    fake_complexity: List[int] = [len(review.content) for review in fake_reviews]
    real_complexity: List[int] = [len(review.content) for review in real_reviews]

    plt.hist(fake_complexity, bins=20, range=[0, 100])
    plt.title(f'Complexity of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Complexity value')
    plt.show()

    plt.hist(real_complexity, bins=20, range=[0, 100])
    plt.title(f'Complexity of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Complexity value')
    plt.show()

    fake_complexity: List[float] = [NLP.get_capslock_score(review.content) for review in fake_reviews]
    real_complexity: List[float] = [NLP.get_capslock_score(review.content) for review in real_reviews]

    plt.hist(fake_complexity, bins=20, range=[0, 150])
    plt.title(f'CAPSLOCK of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('CAPSLOCK value')
    plt.show()

    plt.hist(real_complexity, bins=20, range=[0, 150])
    plt.title(f'CAPSLOCK of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('CAPSLOCK value')
    plt.show()

    fake_complexity: List[float] = [NLP.get_interpunction_score(review.content) for review in fake_reviews]
    real_complexity: List[float] = [NLP.get_interpunction_score(review.content) for review in real_reviews]

    plt.hist(fake_complexity, bins=20, range=[0, 150])
    plt.title(f'Interpunction of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Interpunction value')
    plt.show()

    plt.hist(real_complexity, bins=20, range=[0, 150])
    plt.title(f'Interpunction of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Interpunction value')
    plt.show()

    fake_complexity: List[float] = [NLP.get_emotional_interpunction_score(review.content) for review in fake_reviews]
    real_complexity: List[float] = [NLP.get_emotional_interpunction_score(review.content) for review in real_reviews]

    plt.hist(fake_complexity, bins=20, range=[0, 150])
    plt.title(f'Emotional interpunction of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Emotional interpunction value')
    plt.show()

    plt.hist(real_complexity, bins=20, range=[0, 150])
    plt.title(f'Emotional interpunction of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Emotional interpunction value')
    plt.show()

    fake_complexity: List[float] = [NLP.get_consecutive_emotional_interpunction_score(review.content) for review in fake_reviews]
    real_complexity: List[float] = [NLP.get_consecutive_emotional_interpunction_score(review.content) for review in real_reviews]

    plt.hist(fake_complexity, bins=20, range=[0, 150])
    plt.title(f'Consecutive emotional interpunction of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Consecutive emotional interpunction value')
    plt.show()

    plt.hist(real_complexity, bins=20, range=[0, 150])
    plt.title(f'Consecutive emotional interpunction of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Consecutive emotional interpunction value')
    plt.show()

    fake_complexity: List[float] = [NLP.get_emojis_score(review.content) for review in fake_reviews]
    real_complexity: List[float] = [NLP.get_emojis_score(review.content) for review in real_reviews]

    plt.hist(fake_complexity, bins=20, range=[0, 150])
    plt.title(f'Emojis score of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Emojis score value')
    plt.show()

    plt.hist(real_complexity, bins=20, range=[0, 150])
    plt.title(f'Emojis score of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Emojis score value')
    plt.show()

def test_stylo_metrix():
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False, "rating": 5})
    real_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True, "rating": 5})

    fake_stylo: List[dict] = [NLP.get_stylo_metrix_metrics(review.content).dict() for review in fake_reviews]
    real_stylo: List[dict] = [NLP.get_stylo_metrix_metrics(review.content).dict() for review in real_reviews]

    stylo_keys = StyloMetrixResults.get_list_of_metrics()
    for key in stylo_keys:
        fake_results = [stylo[key] for stylo in fake_stylo]
        real_results = [stylo[key] for stylo in real_stylo]
        test_stylo_metrix_attribute(fake_results, real_results, key)


def test_stylo_metrix_attribute(fake_results: List, real_results: List, key: str):

    plt.hist(fake_results, bins=20)
    plt.title(f'{key} score of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('{key} score value')
    plt.savefig(f'app/results/{key}fake.png')

    plt.hist(real_results, bins=20)
    plt.title(f'{key} score of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('{key} score value')
    plt.savefig(f'app/results/{key}real.png')



def analyze_fake_texts():
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True})
    fake_review_combined: str = (" ".join([review.content for review in fake_reviews])).lower().replace("!","").replace(".","").replace(",","").replace("?","").replace("(","").replace(")","").replace("-","").replace(":","").replace(";","").replace("  "," ")

    words = fake_review_combined.split()
    counted_words = Counter(words)
    counted_words = dict(sorted(counted_words.items(), key=lambda item: item[1]))
    for word in counted_words:
        print(f"{word}={counted_words[word]}")

if __name__ == "__main__":
    # prediction_testing()
    #nlp_testing()
    test_stylo_metrix()
    #uvicorn.run(app, host="0.0.0.0", port=8000)