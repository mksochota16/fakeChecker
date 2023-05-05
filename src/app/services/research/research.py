from datetime import datetime, timedelta
from statistics import mean

import matplotlib.dates as mdates
from collections import Counter
from typing import List, Optional
import time

import numpy as np
import pandas
import folium
import re
import uvicorn
from fastapi import FastAPI
from fastapi.openapi.models import Response
from matplotlib import pyplot as plt
from tabulate import tabulate

from config import NLP, MONGO_CLIENT
from dao.dao_accounts_gmr_pl import DAOAccountsGMR_PL
from dao.dao_accounts_gmr_pl_ano import DAOAccountsGMR_PL_Ano
from dao.dao_accounts_new import DAOAccountsNew
from dao.dao_accounts_old import DAOAccountsOld
from dao.dao_places import DAOPlaces
from dao.dao_reviews_gmr_pl import DAOReviewsGMR_PL
from dao.dao_reviews_gmr_pl_ano import DAOReviewsGMR_PL_Ano
from dao.dao_reviews_partial import DAOReviewsPartial
from models.place import PlaceInDB
from services.scraper.models.position import Position
from dao.dao_reviews_new import DAOReviewsNew
from dao.dao_reviews_old import DAOReviewsOld
from models.account import AccountOldInDB, AccountNewInDB, AccountOld, AccountInAnonymisedGMR_PLInDB, AccountInGMR_PLInDB
from models.base_mongo_model import MongoObjectId
from models.response import PlaceResponse, NoReviewsFoundResponse, FailedToCollectDataResponse, AccountResponse, \
    AccountIsPrivateException, AccountIsPrivateResponse
from models.review import ReviewOldInDB, ReviewNewInDB, ReviewOldBase, ReviewPartialInDB, ReviewInAnonymisedGMR_PLInDB, \
    ReviewInGMR_PLInDB
from services.analysis import geolocation
from services.analysis.AnalysisTools import calculateNMeans, get_distances_to_centroids
from services.analysis.geolocation import is_in_poland
from services.analysis.nlp_analysis import StyloMetrixResults
from services.database.database import Database
from services.predictions.prediction_tools import predict_reviews_from_place, get_and_prepare_accounts_data, \
    get_and_prepare_reviews_data, get_prepared_reviews_data_from_file, get_train_and_test_datasets, \
    build_model_return_predictions, calculate_basic_metrics, calculate_metrics, k_fold_validation, \
    predict_all_reviews_from_new_scrape_one_model, predict_account
from services.scraper.tools.usage import ScraperUsage


def review_new_to_review_old(review_new: ReviewNewInDB) -> ReviewOldBase:
    dao_places: DAOPlaces = DAOPlaces()
    place: PlaceInDB = dao_places.find_by_id(review_new.place_id)
    return ReviewOldBase(
        review_id=review_new.review_id,
        rating=review_new.rating,
        content=review_new.content,
        reviewer_url=review_new.reviewer_url,
        reviewer_id=review_new.reviewer_id,
        photos_urls=review_new.photos_urls,
        response_content=review_new.response_content,
        date=review_new.date,
        is_private=review_new.is_private,
        is_real=review_new.is_real,
        place_name=place.name,
        place_url=place.url,
        localization=place.localization,
        type_of_object=place.type_of_object,
        cluster=place.cluster,
        full_flag=True,
        wrong_address=False,
        wrong_url_flag=False,
        sentiment_rating=None,
        test_prediction=None)


def get_vote_statistics():
    results_dict = {
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
    fake = {
        "-7 ": 112,
        "-6 ": 7,
        "-5 ": 39,
        "-4 ": 3,
        "-3 ": 19,
        "-2 ": 3,
        "-1 ": 3,
        "0 ": 1,
    }
    # neutral = {
    #
    # }
    real = {
        "1 ": 1,
        "2 ": 0,
        "3 ": 3,
        "4 ": 2,
        "5 ": 18,
        "6 ": 1,
        "7 ": 108
    }
    plt.bar(fake.keys(), fake.values(), 0.9, color='r', label="fake")
    # plt.bar(neutral.keys(), neutral.values(), 0.9, color='b', label="neutral")
    plt.bar(real.keys(), real.values(), 0.9, color='g', label="genuine")
    plt.ylabel('Number of accounts')
    plt.yscale('log')
    plt.xlabel('Voting results')
    plt.legend()
    plt.show()


def prediction_testing():
    # get_and_prepare_accounts_data(save_to_file=False)
    # get_and_prepare_reviews_data(save_to_file=True, exclude_localization=True, file_name = 'app/data/formatted_reviews_data_w_less_nlp.csv')
    data = get_prepared_reviews_data_from_file(file_name='app/data/formatted_reviews_data_w_sentiment_caps_inter.csv',
                                               exclude_localization=True)  # get_prepared_accounts_data_from_file(ignore_empty_accounts=True) # get_and_prepare_accounts_data(save_to_file=True)
    # for i in range(20):
    #     prepared_data = get_train_and_test_datasets(3/5, data, resolve_backpack_problem=True)
    #     predicts = build_model_return_predictions(prepared_data[0], prepared_data[1], prepared_data[2])
    #     TP, TN, FP, FN, ALL = calculate_basic_metrics(predicts, prepared_data[3])
    #     calculate_metrics(TP, TN, FP, FN, ALL)
    k_fold_validation(10, data, resolve_backpack_problem=True)
    # predict_all_reviews_from_new_scrape()
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

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False, "rating": 5})
    real_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True, "rating": 5})
    fake_sentiment: List[float] = [review.sentiment_rating for review in fake_reviews if
                                   (review.sentiment_rating > 0 or review.sentiment_rating < -0.25)]
    real_sentiment: List[float] = [review.sentiment_rating for review in real_reviews if
                                   (review.sentiment_rating > 0 or review.sentiment_rating < -0.25)]

    plt.hist(fake_sentiment, bins=20, range=[-1, 1])
    # plt.title(f'Sentiment of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Sentiment value')
    plt.show()

    plt.hist(real_sentiment, bins=20, range=[-1, 1])
    # plt.title(f'Sentiment of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Sentiment value')
    plt.show()


def nlp_testing():
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False, "rating": 5})
    real_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True, "rating": 5})
    fake_reviews: List[ReviewOldInDB] = [review for review in fake_reviews if len(review.content) > 0]
    real_reviews: List[ReviewOldInDB] = [review for review in real_reviews if len(review.content) > 0]

    fake_sentiment: List[float] = [review.sentiment_rating for review in fake_reviews]
    real_sentiment: List[float] = [review.sentiment_rating for review in real_reviews]

    plt.hist(fake_sentiment, bins=20, edgecolor='black', range=[-1, 1])
    plt.title(f'Sentiment of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Sentiment value')
    plt.show()

    plt.hist(real_sentiment, bins=20, edgecolor='black', range=[-1, 1])
    plt.title(f'Sentiment of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Sentiment value')
    plt.show()

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False, "rating": 5})
    real_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True, "rating": 5})
    fake_complexity: List[int] = [len(review.content) for review in fake_reviews]
    real_complexity: List[int] = [len(review.content) for review in real_reviews]

    plt.hist(fake_complexity, bins=20, edgecolor='black', range=[0, 100])
    plt.title(f'Length of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Length in characters')
    plt.show()

    plt.hist(real_complexity, bins=20, edgecolor='black', range=[0, 100])
    plt.title(f'Length of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Length in characters')
    plt.show()

    fake_complexity: List[float] = [NLP.get_capslock_score(review.content) for review in fake_reviews]
    real_complexity: List[float] = [NLP.get_capslock_score(review.content) for review in real_reviews]

    plt.hist(fake_complexity, bins=20, edgecolor='black', range=[0, 150])
    plt.title(f'Capitalized letters of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Capitalized letters score')
    plt.show()

    plt.hist(real_complexity, bins=20, edgecolor='black', range=[0, 150])
    plt.title(f'Capitalized letters of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Capitalized letters score')
    plt.show()

    fake_complexity: List[float] = [NLP.get_interpunction_score(review.content) for review in fake_reviews]
    real_complexity: List[float] = [NLP.get_interpunction_score(review.content) for review in real_reviews]

    plt.hist(fake_complexity, bins=20, edgecolor='black', range=[0, 150])
    plt.title(f'Interpunction of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Interpunction score')
    plt.show()

    plt.hist(real_complexity, bins=20, edgecolor='black', range=[0, 150])
    plt.title(f'Interpunction of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel('Interpunction score')
    plt.show()

    # fake_complexity: List[float] = [NLP.get_emotional_interpunction_score(review.content) for review in fake_reviews]
    # real_complexity: List[float] = [NLP.get_emotional_interpunction_score(review.content) for review in real_reviews]
    #
    # plt.hist(fake_complexity, bins=20, range=[0, 150])
    # plt.title(f'Emotional interpunction of fake reviews')
    # plt.ylabel('Number of reviews')
    # plt.xlabel('Emotional interpunction value')
    # plt.show()
    #
    # plt.hist(real_complexity, bins=20, range=[0, 150])
    # plt.title(f'Emotional interpunction of real reviews')
    # plt.ylabel('Number of reviews')
    # plt.xlabel('Emotional interpunction value')
    # plt.show()
    #
    # fake_complexity: List[float] = [NLP.get_consecutive_emotional_interpunction_score(review.content) for review in fake_reviews]
    # real_complexity: List[float] = [NLP.get_consecutive_emotional_interpunction_score(review.content) for review in real_reviews]
    #
    # plt.hist(fake_complexity, bins=20, range=[0, 150])
    # plt.title(f'Consecutive emotional interpunction of fake reviews')
    # plt.ylabel('Number of reviews')
    # plt.xlabel('Consecutive emotional interpunction value')
    # plt.show()
    #
    # plt.hist(real_complexity, bins=20, range=[0, 150])
    # plt.title(f'Consecutive emotional interpunction of real reviews')
    # plt.ylabel('Number of reviews')
    # plt.xlabel('Consecutive emotional interpunction value')
    # plt.show()
    #
    # fake_complexity: List[float] = [NLP.get_emojis_score(review.content) for review in fake_reviews]
    # real_complexity: List[float] = [NLP.get_emojis_score(review.content) for review in real_reviews]
    #
    # plt.hist(fake_complexity, bins=20, range=[0, 150])
    # plt.title(f'Emojis score of fake reviews')
    # plt.ylabel('Number of reviews')
    # plt.xlabel('Emojis score value')
    # plt.show()
    #
    # plt.hist(real_complexity, bins=20, range=[0, 150])
    # plt.title(f'Emojis score of real reviews')
    # plt.ylabel('Number of reviews')
    # plt.xlabel('Emojis score value')
    # plt.show()


def test_stylo_metrix():
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False, "rating": 5})
    real_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True, "rating": 5})

    print("Calculating metrics")
    start = time.time()
    fake_stylo: List[dict] = [NLP.get_stylo_metrix_metrics(review.content).dict() for review in fake_reviews]
    real_stylo: List[dict] = [NLP.get_stylo_metrix_metrics(review.content).dict() for review in real_reviews]
    end = time.time()
    print(end - start)
    print("Metrics calculated")

    stylo_keys = StyloMetrixResults.get_list_of_metrics()
    for key in stylo_keys:
        fake_results = [stylo[key] for stylo in fake_stylo]
        real_results = [stylo[key] for stylo in real_stylo]
        test_stylo_metrix_attribute(fake_results, real_results, key)


def test_stylo_metrix_attribute(fake_results: List, real_results: List, key: str):
    plt.hist(fake_results, bins=20)
    plt.title(f'{key} score of fake reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel(f'{key} score value')
    plt.savefig(f'app/data/results/{key}fake.png')
    plt.show()

    plt.hist(real_results, bins=20)
    plt.title(f'{key} score of real reviews')
    plt.ylabel('Number of reviews')
    plt.xlabel(f'{key} score value')
    plt.savefig(f'app/data/results/{key}real.png')
    plt.show()


def analyze_fake_texts():
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True})
    fake_review_combined: str = (" ".join([review.content for review in fake_reviews])).lower().replace("!",
                                                                                                        "").replace(".",
                                                                                                                    "").replace(
        ",", "").replace("?", "").replace("(", "").replace(")", "").replace("-", "").replace(":", "").replace(";",
                                                                                                              "").replace(
        "  ", " ")

    words = fake_review_combined.split()
    counted_words = Counter(words)
    counted_words = dict(sorted(counted_words.items(), key=lambda item: item[1]))
    for word in counted_words:
        print(f"{word}={counted_words[word]}")


def synchronize_new_db_scrape():
    dao_reviews_partial: DAOReviewsPartial = DAOReviewsPartial()
    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()

    reviews_partial: List[ReviewPartialInDB] = dao_reviews_partial.find_many_by_query({'scraped_fully': True})
    for review_partial in reviews_partial:
        try:
            dao_reviews_new.update_one({'review_id': review_partial.review_id}, {'$set': {'new_scrape': True}})
        except:
            print(review_partial.review_id)


def get_cluster_statistics():
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()
    dao_places: DAOPlaces = DAOPlaces()
    database: Database = Database()

    data = []
    cluster_names = sorted(database.get_cluster_names())
    for cluster_name in cluster_names:
        count_PL = 0
        real_count_PL = 0
        fake_count_PL = 0
        count = 0
        real_count = 0
        fake_count = 0

        new_places_in_cluster: List[PlaceInDB] = dao_places.find_many_by_query({'cluster': cluster_name})
        for place in new_places_in_cluster:
            if place.localization is None:
                in_poland = True
            else:
                in_poland = is_in_poland(place.localization.lat, place.localization.lon)
            reviews_real_in_cluster: List[ReviewNewInDB] = dao_reviews_new.find_many_by_query(
                {'new_scrape': True, 'place_id': place.id})
            if in_poland:
                count_PL += len(reviews_real_in_cluster)
                real_count_PL += len(reviews_real_in_cluster)
            count += 1
            real_count += len(reviews_real_in_cluster)

        reviews_fake_in_cluster: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query(
            {'cluster': cluster_name, 'is_real': False})
        for fake_review in reviews_fake_in_cluster:
            in_poland = is_in_poland(fake_review.localization.lat, fake_review.localization.lon)
            if in_poland:
                count_PL += 1
                fake_count_PL += 1
            count += 1
            fake_count += 1
        data.append(
            [f"[{count}] {cluster_name}", fake_count_PL, real_count_PL, count_PL, fake_count, real_count, count])

    headers = ["Cluster", 'fake_count_PL', 'real_count_PL', 'count_PL', 'fake_count', 'real_count',
               'count']
    print(tabulate(data, headers))

    df = pandas.DataFrame(data, columns=headers)
    plt.figure(figsize=(9, 9))
    plt.ylabel('Number of reviews')
    plt.xlabel('CLUSTERS')
    # plt.yscale('log')
    x_axis = np.arange(len(df['Cluster']))
    # plt.ylabel('fake_count_PL')
    plt.bar(x_axis - 0.2, df['fake_count_PL'], width=0.4, label='fake_count_PL')
    plt.bar(x_axis + 0.2, df['real_count_PL'], width=0.4, label='real_count_PL')
    plt.xticks(x_axis, df['Cluster'], rotation=90)
    plt.tight_layout()
    plt.yscale('log')
    plt.legend()
    plt.savefig('polskie_dane.png')
    plt.show()

    plt.figure(figsize=(9, 9))
    plt.ylabel('Number of reviews')
    plt.xlabel('CLUSTERS')
    x_axis = np.arange(len(df['Cluster']))
    # plt.ylabel('fake_count_PL')
    # plt.yscale('log')
    plt.bar(x_axis - 0.2, df['fake_count'], width=0.4, label='Fake')
    plt.bar(x_axis + 0.2, df['real_count'], width=0.4, label='Genuine')
    plt.xticks(x_axis, df['Cluster'], rotation=90)
    plt.tight_layout()
    plt.yscale('log')
    plt.legend()
    plt.savefig('wszystkie_dane.png')
    plt.show()


def plot_dates():
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
    reviews_fake: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False})
    dates_fake: List[datetime] = [review.date for review in reviews_fake]

    reviews_real: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True})
    dates_real: List[datetime] = [review.date for review in reviews_real]

    # plot it
    fig, ax = plt.subplots(1, 1)
    ax.hist(dates_fake, bins=20, color='r', label='fake reviews')
    ax.hist(dates_real, bins=30, color='g', alpha=0.5, label='real reviews')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.ylabel('Number of reviews posted')
    plt.xlabel('Date')
    plt.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.yscale('log')
    plt.show()


def calculate_centroids(reviewer_id):
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
    reviews: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(reviewer_id)
    pos_list = [review.localization for review in reviews]

    kmeans_algorithm = calculateNMeans(pos_list)
    centrum_radius = []
    distances = []
    for n in range(len(kmeans_algorithm[0])):
        cluster = kmeans_algorithm[0][n]
        # calculate distances to centroids
        max_distance = 0
        for index in cluster:
            temp_position = pos_list[index]
            distance = geolocation.distance(temp_position, kmeans_algorithm[1][n])
            if distance > max_distance:
                max_distance = distance
            distances.append(distance)
        centrum_radius.append([kmeans_algorithm[1][n], max_distance])
    print(mean(distances))
    print(centrum_radius)


def test_local_guide_levels():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    real_accounts: List[AccountOldInDB] = dao_accounts_old.find_many_by_query(
        {'fake_service': 'real', 'number_of_reviews': {'$gt': 0}})
    fake_accounts: List[AccountOldInDB] = dao_accounts_old.find_many_by_query(
        {'fake_service': {'$ne': 'real'}, 'number_of_reviews': {'$gt': 0}})

    real_levels: List[int] = [account.local_guide_level if account.local_guide_level is not None else 0 for account in
                              real_accounts]
    fake_levels: List[int] = [account.local_guide_level if account.local_guide_level is not None else 0 for account in
                              fake_accounts]

    bins = max([max(real_levels), max(fake_levels)]) + 1
    plt.bar(*np.unique(fake_levels, return_counts=True), edgecolor='black')
    # plt.hist(real_levels, bins=bins, edgecolor='black',range=(0,bins), align = "left")
    plt.title(f'Local guide levels of fake accounts')
    plt.ylabel('Number of accounts')
    plt.xlabel('Local guide level')
    plt.xlim([-0.5, bins - 0.5])
    plt.xticks(np.arange(0, 10, step=1))
    plt.show()

    plt.bar(*np.unique(real_levels, return_counts=True), edgecolor='black')
    # plt.hist(fake_levels, bins=bins, edgecolor='black',range=(0,bins), align = "left")
    plt.title(f'Local guide levels of real accounts')
    plt.ylabel('Number of accounts')
    plt.xlabel('Local guide level')
    plt.xlim([-0.5, bins - 0.5])
    plt.xticks(np.arange(0, 10, step=1))
    plt.show()

    print("done")


def test_review_number():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()

    dao_accounts_new: DAOAccountsNew = DAOAccountsNew()

    real_accounts: List[AccountNewInDB] = dao_accounts_new.find_many_by_query(
        {'new_scrape': True, 'number_of_reviews': {'$gt': 0}})
    fake_accounts: List[AccountOldInDB] = dao_accounts_old.find_many_by_query(
        {'fake_service': {'$ne': 'real'}, 'number_of_reviews': {'$gt': 0}})

    real_levels: List[int] = [account.number_of_reviews for account in real_accounts]
    fake_levels: List[int] = [account.number_of_reviews for account in fake_accounts]

    max_reviews = max([max(real_levels), max(fake_levels)])
    size = max(real_levels) / 40
    bins_fake = int(max(fake_levels) / size)
    # plt.bar(*np.unique(fake_levels, return_counts=True), edgecolor='black')
    plt.hist(real_levels, bins=40, edgecolor='black')
    # plt.title(f'Number of reviews of real accounts')
    plt.ylabel('Number of accounts')
    plt.xlabel('Number of reviews')
    plt.xlim([0, max_reviews])
    plt.show()

    # plt.bar(*np.unique(real_levels, return_counts=True), edgecolor='black')
    plt.hist(fake_levels, bins=bins_fake, edgecolor='black')
    # plt.title(f'Number of reviews of fake accounts')
    plt.ylabel('Number of accounts')
    plt.xlabel('Number of reviews')
    plt.xlim([0, max_reviews])
    plt.show()
    print("done")

    real_levels: List[int] = [lvl if lvl < 200 else 200 for lvl in real_levels]
    fake_levels: List[int] = [lvl if lvl < 200 else 200 for lvl in fake_levels if lvl < 200]

    max_reviews = max([max(real_levels), max(fake_levels)])
    size = max(real_levels) / 40
    bins_fake = int(max(fake_levels) / size) + 2
    # plt.bar(*np.unique(fake_levels, return_counts=True), edgecolor='black')
    plt.hist(real_levels, bins=40, range=(0, max_reviews), edgecolor='black')
    # plt.title(f'Number of reviews of real accounts')
    plt.ylabel('Number of accounts')
    plt.xlabel('Number of reviews')
    plt.xlim([0, max_reviews])
    plt.show()

    # plt.bar(*np.unique(real_levels, return_counts=True), edgecolor='black')
    plt.hist(fake_levels, bins=40, range=(0, max_reviews), edgecolor='black')
    # plt.title(f'Number of reviews of fake accounts')
    plt.ylabel('Number of accounts')
    plt.xlabel('Number of reviews')
    plt.xlim([0, max_reviews])
    plt.show()
    print("done")


def get_number_of_local_guide():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False})
    real_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': True})

    fake_guide = 0
    fake_no_guide = 0
    for fake in fake_reviews:
        account: AccountOldInDB = dao_accounts_old.find_one_by_query({'reviewer_id': fake.reviewer_id})
        if account.local_guide_level is None:
            fake_no_guide += 1
        else:
            fake_guide += 1

    real_guide = 0
    real_no_guide = 0
    for real in real_reviews:
        account: AccountOldInDB = dao_accounts_old.find_one_by_query({'reviewer_id': real.reviewer_id})
        if account.local_guide_level is None:
            real_no_guide += 1
        else:
            real_guide += 1

    print(f'Fake reviews: {fake_guide} are local guides, {fake_no_guide} are not')
    print(f'Real reviews: {real_guide} are local guides, {real_no_guide} are not')


def get_ratings_histograms():
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False})
    fake_ratings = [review.rating for review in fake_reviews]
    dao_accounts_new: DAOAccountsNew = DAOAccountsNew()
    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()

    real_accounts: List[AccountNewInDB] = dao_accounts_new.find_many_by_query(
        {'new_scrape': True, 'number_of_reviews': {'$gt': 0}})
    real_ratings = []
    for account in real_accounts:
        real_reviews: List[ReviewNewInDB] = dao_reviews_new.find_many_by_query({'reviewer_id': account.reviewer_id})
        ratings = [review.rating for review in real_reviews]
        real_ratings.extend(ratings)

    plt.bar(*np.unique(fake_ratings, return_counts=True), edgecolor='black', width=1.0)
    # plt.hist(fake_ratings, bins=5, range=[1,5],  edgecolor='black', align='mid')
    # plt.title(f"Histogram of fake reviews' ratings")
    plt.ylabel('Number of ratings')
    plt.xlabel('Rating in number of stars')
    plt.show()

    plt.bar(*np.unique(real_ratings, return_counts=True), edgecolor='black', width=1.0)
    # plt.hist(real_ratings, bins=5, range=[1, 5], edgecolor='black', align='mid')
    # plt.title(f"Histogram of real reviews' ratings")
    plt.ylabel('Number of ratings')
    plt.xlabel('Rating in number of stars')
    plt.show()
    print("done")


def check_if_accounts_really_are_private_or_deleted():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    all_accounts: List[AccountOldInDB] = dao_accounts_old.find_all()
    usage = ScraperUsage(headless=True)
    for account in all_accounts:
        result = usage.check_if_account_is_deleted(None, reviewer_id=account.reviewer_id)
        if result == "private":
            private = True
            deleted = False
        elif result == "deleted":
            private = False
            deleted = True
        else:
            private = False
            deleted = False

        dao_accounts_old.update_one({'reviewer_id': account.reviewer_id},
                                    {'$set': {'is_private': private, 'is_probably_banned': deleted}})

    usage.driver.quit()


def rename_is_prob_is_deleted():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    all_accounts: List[AccountOldInDB] = dao_accounts_old.find_all()
    for account in all_accounts:
        dao_accounts_old.update_one({'reviewer_id': account.reviewer_id},
                                    {'$rename': {'is_probably_banned': 'is_deleted'}})


def get_distances_to_centroids_collective_distribution():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    dao_accounts_new: DAOAccountsNew = DAOAccountsNew()
    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()

    fake_accounts: List[AccountOldInDB] = dao_accounts_old.find_many_by_query({'fake_service': {'$ne': 'real'}})
    real_accounts: List[AccountNewInDB] = dao_accounts_new.find_many_by_query({'new_scrape': True})

    fake_distances: List[float] = []
    for fake_account in fake_accounts:
        if fake_account.number_of_reviews is not None:
            distances = get_distances_to_centroids_old(fake_account.reviewer_id, dao_reviews_old)
            fake_distances.extend(distances)

    real_distances: List[float] = []
    for real_account in real_accounts:
        if real_account.number_of_reviews is not None:
            distances = get_distances_to_centroids_new_accounts(real_account.reviewer_id)
            real_distances.extend(distances)

    plt.hist(fake_distances, bins=20, edgecolor='black')
    # plt.title(f'Geographic distribution of fake accounts\' reviews')
    plt.ylabel('Number of locations')
    plt.xlabel('Distance to the nearest centroid [km]')
    plt.show()

    # plt.bar(*np.unique(real_levels, return_counts=True), edgecolor='black')
    plt.hist(real_distances, bins=20, edgecolor='black')
    # plt.title(f'Geographic distribution of real accounts\' reviews')
    plt.ylabel('Number of locations')
    plt.xlabel('Distance to the nearest centroid [km]')
    plt.show()


def get_distances_to_centroids_old(account_id: str, dao_reviews_old: DAOReviewsOld) -> List[float]:
    reviews_of_account: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(account_id)
    pos_list: List[Position] = [review.localization for review in reviews_of_account if ((
                                                                                                     review.localization is not None) and review.localization.lon != 0.0 and review.localization.lat != 0.0)]
    if len(pos_list) < 2:
        return []
    kmeans_algorithm = calculateNMeans(pos_list)
    distances_to_centroids: List[float] = get_distances_to_centroids(kmeans_algorithm, pos_list, if_round=True)
    return distances_to_centroids


def get_distances_to_centroids_new_accounts(account_id: str) -> List[float]:
    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()
    dao_places: DAOPlaces = DAOPlaces()
    reviews_of_account: List[ReviewNewInDB] = dao_reviews_new.find_many_by_query({'reviewer_id': account_id})
    pos_list: List[Position] = []
    for review in reviews_of_account:
        place: PlaceInDB = dao_places.find_by_id(review.place_id)
        if place is not None and place.localization is not None:
            pos_list.append(place.localization.to_old_model())
    if len(pos_list) < 2:
        return []
    kmeans_algorithm = calculateNMeans(pos_list)
    distances_to_centroids: List[float] = get_distances_to_centroids(kmeans_algorithm, pos_list, if_round=True)
    return distances_to_centroids


def get_histograms_of_geographic_distributions():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    dao_accounts_new: DAOAccountsNew = DAOAccountsNew()

    fake_accounts: List[AccountOldInDB] = dao_accounts_old.find_many_by_query({'fake_service': {'$ne': 'real'}})
    real_accounts: List[AccountNewInDB] = dao_accounts_new.find_many_by_query({'new_scrape': True})

    fake_distances: List[float] = []
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
    for fake_account in fake_accounts:
        if fake_account.number_of_reviews is not None:
            distances = get_distances_to_centroids_old(fake_account.reviewer_id, dao_reviews_old)
            fake_distances.extend(distances)

    real_distances: List[float] = []
    for real_account in real_accounts:
        if real_account.number_of_reviews is not None:
            distances = get_distances_to_centroids_new_accounts(real_account.reviewer_id)
            real_distances.extend(distances)

    fake_distances = [distance if distance < 400 else 400 for distance in fake_distances]
    real_distances = [distance if distance < 400 else 400 for distance in real_distances if distance < 400]

    plt.hist(fake_distances, bins=20, edgecolor='black')
    plt.title(f'Geographic distribution of fake accounts\' reviews')
    plt.ylabel('Number of locations')
    plt.xlabel('Distance to the nearest centroid [km]')
    plt.show()

    plt.hist(real_distances, bins=20, edgecolor='black')
    plt.title(f'Geographic distribution of real accounts\' reviews')
    plt.ylabel('Number of locations')
    plt.xlabel('Distance to the nearest centroid [km]')
    plt.show()


def get_histograms_of_name_scores():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    dao_accounts_new: DAOAccountsNew = DAOAccountsNew()

    fake_accounts: List[AccountOldInDB] = dao_accounts_old.find_many_by_query({'fake_service': {'$ne': 'real'}})
    real_accounts: List[AccountNewInDB] = dao_accounts_new.find_many_by_query({'new_scrape': True})

    fake_name_scores: List[float] = [NLP.analyze_name_of_account(account.name) for account in fake_accounts]
    real_name_scores: List[float] = [NLP.analyze_name_of_account(account.name) for account in real_accounts]
    print(mean(fake_name_scores))
    print(mean(real_name_scores))
    fake_name_scores = [score if score < 1200000 else 1200000 for score in fake_name_scores]
    real_name_scores = [score if score < 1200000 else 1200000 for score in real_name_scores]

    # plt.hist(fake_name_scores, bins=20, edgecolor='black', weights=np.ones(len(fake_name_scores))/len(fake_name_scores))
    # #plt.bar(real.keys(), real.values(), 0.9, color='g', label="genuine")
    # # plt.title(f'Histogram of fake accounts\' name scores')
    # plt.ylabel('Number of accounts')
    # plt.xlabel('Name score')
    # plt.show()
    #
    # plt.hist(real_name_scores, bins=20, edgecolor='black', weights=np.ones(len(real_name_scores))/len(real_name_scores))
    # # plt.title(f'Histogram of real accounts\' name scores')
    # plt.ylabel('Number of accounts')
    # plt.xlabel('Name score')
    # plt.show()

    fake_name_scores = [score for score in fake_name_scores if score < 400000]
    real_name_scores = [score for score in real_name_scores if score < 400000]

    print(mean(fake_name_scores))
    print(mean(real_name_scores))
    plt.hist(fake_name_scores, bins=20, edgecolor='black',
             weights=np.ones(len(fake_name_scores)) / len(fake_name_scores))
    # plt.title(f'Histogram of fake accounts\' name scores (zoomed)')
    plt.ylabel('Account share')
    plt.xlabel('Name score')
    plt.show()

    plt.hist(real_name_scores, bins=20, edgecolor='black',
             weights=np.ones(len(real_name_scores)) / len(real_name_scores))
    # plt.title(f'Histogram of real accounts\' name scores (zoomed)')
    plt.ylabel('Account share')
    plt.xlabel('Name score')
    plt.show()


def get_map_distributions_of_accounts_real():
    dao_accounts_new: DAOAccountsNew = DAOAccountsNew()
    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()
    dao_places: DAOPlaces = DAOPlaces()
    accounts: List[AccountNewInDB] = dao_accounts_new.find_many_by_query({'new_scrape': True})
    m = folium.Map(location=[52, 19], zoom_start=7)
    for account in accounts:
        account_reviews = dao_reviews_new.find_many_by_query({'reviewer_id': account.reviewer_id})
        for review in account_reviews:
            place: PlaceInDB = dao_places.find_by_id(review.place_id)
            if place is not None and place.localization is not None:
                pos1 = place.localization.to_old_model()
            else:
                continue
            if is_in_poland(pos1.lat, pos1.lon):
                folium.Circle(radius=2000, location=(pos1.lat, pos1.lon), color='red').add_to(m)

    print("======= FINISHED =======")
    m.save(f'D:/OneDrive - Politechnika Warszawska/Studia/Szkoła Orłów/Artykuł/obrazki_nowe/collected_real.html')


def get_map_distributions_of_accounts_fake():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    accounts: List[AccountOldInDB] = dao_accounts_old.find_many_by_query({'fake_service': {'$ne': 'real'}})
    m = folium.Map(location=[52, 19], zoom_start=7)
    for account in accounts:
        account_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'reviewer_id': account.reviewer_id})
        for review in account_reviews:
            pos1 = review.localization
            if is_in_poland(pos1.lat, pos1.lon):
                folium.Circle(radius=2000, location=(pos1.lat, pos1.lon), color='red').add_to(m)

    print("======= FINISHED =======")
    m.save(f'D:/OneDrive - Politechnika Warszawska/Studia/Szkoła Orłów/Artykuł/obrazki_nowe/collected_fake.html')


def create_GMR_PL_dataset():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    dao_places: DAOPlaces = DAOPlaces()
    dao_accounts_new: DAOAccountsNew = DAOAccountsNew()
    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()

    accounts_fake: List[AccountOldInDB] = dao_accounts_old.find_many_by_query({'fake_service': {'$ne': 'real'}})
    accounts_fake_transformed: List[AccountOldInDB] = []
    for account in accounts_fake:
        account_transformed: AccountOldInDB = AccountOldInDB(
            name=account.name,
            reviewer_id=account.reviewer_id,
            local_guide_level=account.local_guide_level,
            number_of_reviews=account.number_of_reviews,
            is_private=account.is_private,
            reviewer_url=account.reviewer_url,
            fake_service=account.fake_service,
            is_deleted=account.is_deleted,
            _id=account.id
        )
        accounts_fake_transformed.append(account_transformed)

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False})
    fake_reviews_transformed: List[ReviewOldInDB] = []
    for fake_review in fake_reviews:
        account: AccountOldInDB = dao_accounts_old.find_one_by_query({'reviewer_id': fake_review.reviewer_id})
        review_transformed: ReviewOldInDB = ReviewOldInDB(
            place_name=fake_review.place_name,
            place_url=fake_review.place_url,
            localization=fake_review.localization,
            type_of_object=fake_review.type_of_object,
            cluster=fake_review.cluster,

            review_id=fake_review.review_id,
            rating=fake_review.rating,
            content=fake_review.content,
            reviewer_url=fake_review.reviewer_url,
            reviewer_id=fake_review.reviewer_id,
            photos_urls=fake_review.photos_urls,
            response_content=fake_review.response_content,
            date=fake_review.date,
            is_real=False,
            _id=fake_review.id,
            account_id=account.id
        )
        fake_reviews_transformed.append(review_transformed)

    accounts_real: List[AccountNewInDB] = dao_accounts_new.find_many_by_query({'new_scrape': True})
    accounts_real_transformed: List[AccountOldInDB] = [account.to_old_model("real") for account in accounts_real]

    real_reviews_transformed: List[ReviewOldInDB] = []
    for real_account in accounts_real:
        reviews_of_account: List[ReviewNewInDB] = dao_reviews_new.find_many_by_query(
            {'reviewer_id': real_account.reviewer_id, 'new_scrape': True})
        for review in reviews_of_account:
            connected_place: PlaceInDB = dao_places.find_by_id(review.place_id)
            review_transformed: ReviewOldInDB = ReviewOldInDB(
                place_name=connected_place.name,
                place_url=connected_place.url,
                localization=connected_place.localization,
                type_of_object=connected_place.type_of_object,
                cluster=connected_place.cluster,

                review_id=review.review_id,
                rating=review.rating,
                content=review.content,
                reviewer_url=review.reviewer_url,
                reviewer_id=review.reviewer_id,
                photos_urls=review.photos_urls,
                response_content=review.response_content,
                date=review.date,
                is_real=True,
                _id=review.id,
                account_id=real_account.id
            )
            real_reviews_transformed.append(review_transformed)

    client = MONGO_CLIENT
    db = client['gmr_pl_full']
    accounts = db["accounts"]
    accounts_real_transformed.extend(accounts_fake_transformed)
    accounts_in_gmr_pl_db: List[AccountInGMR_PLInDB] = [AccountInGMR_PLInDB.from_old_model(account) for account in
                                                        accounts_real_transformed]
    accounts.insert_many([account.to_dict() for account in accounts_in_gmr_pl_db])
    reviews = db["reviews"]
    real_reviews_transformed.extend(fake_reviews_transformed)
    reviews_in_gmr_pl_db: List[ReviewInGMR_PLInDB] = [ReviewInGMR_PLInDB.from_old_model(review) for review in
                                                      real_reviews_transformed]
    reviews.insert_many([review.to_dict() for review in reviews_in_gmr_pl_db])


def create_GMR_PL_Anonymised_dataset():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    dao_places: DAOPlaces = DAOPlaces()
    dao_accounts_new: DAOAccountsNew = DAOAccountsNew()
    dao_reviews_new: DAOReviewsNew = DAOReviewsNew()

    accounts_fake: List[AccountOldInDB] = dao_accounts_old.find_many_by_query({'fake_service': {'$ne': 'real'}})
    accounts_fake_transformed: List[AccountInAnonymisedGMR_PLInDB] = []
    for account in accounts_fake:
        account_transformed: AccountInAnonymisedGMR_PLInDB = AccountInAnonymisedGMR_PLInDB(
            name_score=NLP.analyze_name_of_account(account.name),
            local_guide_level=account.local_guide_level,
            number_of_reviews=account.number_of_reviews,
            is_real=False,
            is_private=account.is_private,
            is_deleted=account.is_deleted,
            _id=account.id
        )
        accounts_fake_transformed.append(account_transformed)

    fake_reviews: List[ReviewOldInDB] = dao_reviews_old.find_many_by_query({'is_real': False})
    fake_reviews_transformed: List[ReviewInAnonymisedGMR_PLInDB] = []
    for fake_review in fake_reviews:
        account: AccountOldInDB = dao_accounts_old.find_one_by_query({'reviewer_id': fake_review.reviewer_id})
        approximate_localization: Optional[
            Position] = fake_review.localization.make_approximation() if fake_review.localization is not None else None
        review_transformed: ReviewInAnonymisedGMR_PLInDB = ReviewInAnonymisedGMR_PLInDB(
            rating=fake_review.rating,
            content=fake_review.content,
            photos_urls=fake_review.photos_urls,
            response_content=fake_review.response_content,
            date=fake_review.date,
            is_real=False,

            approximate_localization=approximate_localization,
            type_of_object=fake_review.type_of_object,
            cluster=fake_review.cluster,

            _id=fake_review.id,
            account_id=account.id,

            content_not_full="   Więcej" in fake_review.content,
            content_translated="(Przetłumaczone przez Google)" in fake_review.content,
            not_in_poland=fake_review.localization is not None and (
                not is_in_poland(fake_review.localization.lat, fake_review.localization.lon)),
            localization_missing=fake_review.localization is None
        )
        fake_reviews_transformed.append(review_transformed)

    accounts_real: List[AccountNewInDB] = dao_accounts_new.find_many_by_query({'new_scrape': True})
    accounts_real_transformed: List[AccountInAnonymisedGMR_PLInDB] = []
    for account in accounts_real:
        account_transformed: AccountInAnonymisedGMR_PLInDB = AccountInAnonymisedGMR_PLInDB(
            name_score=NLP.analyze_name_of_account(account.name),
            local_guide_level=account.local_guide_level,
            number_of_reviews=account.number_of_reviews,
            is_real=True,
            is_private=account.is_private,
            is_deleted=account.is_deleted,
            _id=account.id
        )
        accounts_real_transformed.append(account_transformed)

    real_reviews_transformed: List[ReviewInAnonymisedGMR_PLInDB] = []
    for real_account in accounts_real:
        reviews_of_account: List[ReviewNewInDB] = dao_reviews_new.find_many_by_query(
            {'reviewer_id': real_account.reviewer_id, 'new_scrape': True})
        for review in reviews_of_account:
            connected_place: PlaceInDB = dao_places.find_by_id(review.place_id)
            approximate_localization: Optional[
                Position] = connected_place.localization.make_approximation() if connected_place.localization is not None else None
            review_transformed: ReviewInAnonymisedGMR_PLInDB = ReviewInAnonymisedGMR_PLInDB(
                rating=review.rating,
                content=review.content,
                photos_urls=review.photos_urls,
                response_content=review.response_content,
                date=review.date,
                is_real=True,

                approximate_localization=approximate_localization,
                type_of_object=connected_place.type_of_object,
                cluster=connected_place.cluster,

                _id=review.id,
                account_id=real_account.id,

                content_not_full="   Więcej" in review.content,
                content_translated="(Przetłumaczone przez Google)" in review.content,
                not_in_poland=connected_place.localization is not None and (
                    not is_in_poland(connected_place.localization.lat, connected_place.localization.lon)),
                localization_missing=connected_place.localization is None
            )
            real_reviews_transformed.append(review_transformed)

    client = MONGO_CLIENT
    db = client['gmr_pl_anonymous']
    accounts = db["accounts"]
    accounts_real_transformed.extend(accounts_fake_transformed)
    accounts_dicts = [account.to_dict() for account in accounts_real_transformed]
    accounts.insert_many(accounts_dicts)
    reviews = db["reviews"]
    real_reviews_transformed.extend(fake_reviews_transformed)
    reviews_dicts = [review.to_dict() for review in real_reviews_transformed]
    reviews.insert_many(reviews_dicts)

def check_reviews_for_anonymity():
    dao_accounts_gmr_pl: DAOAccountsGMR_PL = DAOAccountsGMR_PL()
    dao_reviews_gmr_pl: DAOReviewsGMR_PL = DAOReviewsGMR_PL()
    dao_reviews_gmr_pl_ano: DAOReviewsGMR_PL_Ano = DAOReviewsGMR_PL_Ano()

    ano_reviews: List[ReviewInAnonymisedGMR_PLInDB] = dao_reviews_gmr_pl_ano.find_all()
    counter = 0
    skip_flag = True
    for ano_review in ano_reviews:
        full_account: AccountInGMR_PLInDB = dao_accounts_gmr_pl.find_by_id(ano_review.account_id)
        full_review: ReviewInGMR_PLInDB = dao_reviews_gmr_pl.find_by_id(ano_review.id)
        counter += 1
        print(counter)
        if counter =< 17979:
            continue

        new_content = ano_review.content
        new_response_content = ano_review.response_content
        supposedly_name = full_account.name.split(" ")[0]
        new_response_content = censor_text(supposedly_name, new_response_content)
        try:
            supposedly_surname = full_account.name.split(" ")[1]
            new_response_content = censor_text(supposedly_surname, new_response_content)
        except IndexError:
            pass

        if full_review.place_name:
            split_place_name = full_review.place_name.split(" ")
            for word in split_place_name:
                new_content = censor_text(word, new_content)
                new_response_content = censor_text(word, new_response_content)

        if new_content != ano_review.content or new_response_content != ano_review.response_content:
            dao_reviews_gmr_pl_ano.update_one({'_id':ano_review.id}, {"$set":{'content': new_content, 'response_content': new_response_content, "censor_text": True}})
            print("Updated")



def censor_text(search_text, text_to_censor) -> Optional[str]:
    text_og = text_to_censor
    search_text = search_text.lower()

    search_text = search_text.replace("(","")
    search_text = search_text.replace(")","")
    search_text = search_text.replace("*","")
    skip_check = False
    if text_to_censor is None:
        return None
    if len(search_text) > 3:
        search_text = search_text[:-2]
    elif len(search_text) == 3:
        search_text = search_text[:-1]
    else:
        skip_check = True

    to_censor = []
    if not skip_check:
        search = re.findall(fr"\b{search_text}[^ ]{{0,3}}(?:,|\s)", text_to_censor.lower())
        if search is not None:
            for result in search:
                #start = result[0]
                to_censor.append(result)
                true_len = len(result)
                #censored_text = text_to_censor[result[0]:result[1]]
                if "," in result:
                    true_len -= 1
                if " " in result:
                    true_len -= 1
                if "." in result:
                    true_len -= 1
                #new_content = text_to_censor[:start] + f"{'*' * (finish - start)}" + text_to_censor[finish:]

                compiled = re.compile(re.escape(result[:true_len]), re.IGNORECASE)
                res = compiled.sub('*'*true_len, text_to_censor)
                text_to_censor = str(res)

    new_content = text_to_censor

    if new_content == text_og:
        return text_og

    print(f"To censor: {to_censor}")
    print("Proposed change:")
    print(text_og)
    print("=====================================")
    print(new_content)
    print("=====================================")
    print("y/n:")
    answer = input()
    if answer == "y":
        return new_content
    else:
        return text_og


if __name__ == '__main__':
    check_reviews_for_anonymity()
