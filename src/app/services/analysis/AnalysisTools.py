import random
import statistics
from statistics import mean, median, variance
from typing import List, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils import distance_metric, type_metric
from tabulate import tabulate

from config import ENGLISH_TRANSLATION_CLUSTER_DICT, NLP
from dao.dao_reviews_old import DAOReviewsOld
from models.account import AccountBase, AccountOldInDB
from models.place import PlaceInDB
from models.review import ReviewOldInDB, ReviewPartialInDB, ReviewNewInDB, ReviewBase
from services.analysis import geolocation
from services.analysis.nlp_analysis import StyloMetrixResults
from services.predictions.prediction_constants import AttributesModes
from services.scraper.models.position import Position

import pandas


def performKMeansForAnalysis(url, position_list):
    print(f"Number of locations: {len(position_list)}")
    print("Found centroids:")
    kmeans_algorithm = calculateNMeans(position_list)
    distances_to_centroids = get_distances_to_centroids(kmeans_algorithm, position_list)
    print(kmeans_algorithm[1])
    print(f"Average distant co centroid = {round(np.mean(distances_to_centroids), 2)}km")
    plt.hist(distances_to_centroids, bins=20, range=[0, 200])
    plt.title(f'Localizations of Reviewer ID: {url.split("/")[5]}')
    plt.ylabel('Number of locations')
    plt.xlabel('Distance in km to the nearest centroid')
    plt.show()


def perform_kmeans_centroid_analysis_for_account(account_id, database):
    pos_list = database.get_list_of_positions_by_id(account_id)
    print(f"Number of locations: {len(pos_list)}")
    print("Found centroids:")
    kmeans_algorithm = calculateNMeans(pos_list)
    distances_to_centroids = get_distances_to_centroids(kmeans_algorithm, pos_list)
    print(kmeans_algorithm[1])
    print(f"Average distant co centroid = {round(np.mean(distances_to_centroids), 2)}km")
    plt.style.use({'figure.facecolor': 'white'})
    plt.hist(distances_to_centroids, bins=20, range=[0, 200])
    plt.title(f'Localizations of Reviewer ID: {account_id}')
    plt.ylabel('Number of locations')
    plt.xlabel('Distance in km to the nearest centroid')
    plt.show()


def perform_all_prepared_analyses(account_id, database):
    print("============== CENTROID ANALYSIS ==============")
    perform_kmeans_centroid_analysis_for_account(account_id, database)
    print("============== RATINGS ANALYSIS ==============")
    show_ratings_histograms_for_account(account_id, database)


def get_ratings_distribution_metrics(account_id: str, reviews_of_account: Optional[List[Union[ReviewOldInDB, ReviewPartialInDB]]] = None):
    if reviews_of_account is None:
        dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
        reviews_of_account: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(account_id)
    ratings_list = [review.rating for review in reviews_of_account]
    list_mean = mean(ratings_list)
    list_median = median(ratings_list)
    try:
        list_variance = variance(ratings_list)
    except statistics.StatisticsError:
        list_variance = 0

    return list_mean, list_median, list_variance


def get_geolocation_distribution_metrics(account_id, reviews_of_account: Optional[List[Union[ReviewOldInDB, ReviewPartialInDB]]] = None):
    if reviews_of_account is None:
        dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
        reviews_of_account: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(account_id)
    pos_list: List[Position] = [review.localization for review in reviews_of_account if ((review.localization is not None) and review.localization.lon != 0.0 and review.localization.lat != 0.0)]

    kmeans_algorithm = calculateNMeans(pos_list)
    distances_to_centroids = get_distances_to_centroids(kmeans_algorithm, pos_list, if_round=False)
    list_mean = mean(distances_to_centroids)
    list_median = median(distances_to_centroids)
    try:
        list_variance = variance(distances_to_centroids)
    except statistics.StatisticsError:
        list_variance = 0
    return list_mean, list_median, list_variance


def get_percentage_of_photographed_reviews(account_id,
                                           reviews_of_account: Optional[List[Union[ReviewOldInDB, ReviewPartialInDB]]] = None) -> float:
    if reviews_of_account is None:
        dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
        reviews_of_account: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(account_id)
    photos_urls_list: List[Optional[List[str]]] = [review.photos_urls for review in reviews_of_account]

    none_count = photos_urls_list.count(None)
    return none_count / len(photos_urls_list)


def get_percentage_of_responded_reviews(account_id,
                                        reviews_of_account: Optional[List[Union[ReviewOldInDB, ReviewPartialInDB]]] = None) -> float:
    if reviews_of_account is None:
        dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
        reviews_of_account: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(account_id)
    responses_list: List[Optional[str]] = [review.response_content for review in reviews_of_account]

    none_count = responses_list.count(None)
    return none_count / len(responses_list)


def show_ratings_histograms_for_account(account_id, database):
    ratings_list = database.get_list_of_ratings_by_id(account_id)
    plt.style.use({'figure.facecolor': 'white'})
    plt.hist(ratings_list, bins=5, range=[1, 5], align='mid')
    plt.title(f'Ratings of Reviewer ID: {account_id}')
    plt.ylabel('Number of ratings')
    plt.xlabel('Rating in stars')
    plt.show()


def get_type_of_objects_counts_for_account(account_id, reviews_of_account: Optional[List[Union[ReviewOldInDB, ReviewPartialInDB]]] = None):
    if reviews_of_account is None:
        dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
        reviews_of_account: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(account_id)
    counter_dict = {}
    cluster_names = ENGLISH_TRANSLATION_CLUSTER_DICT.values()
    for cluster_name in cluster_names:
        counter_dict[cluster_name] = 0
    for review in reviews_of_account:
        if review.cluster in counter_dict:
            counter_dict[review.cluster] += 1
        else:
            print(f"Error, cluster '{review.cluster}' not known")
    return counter_dict


def get_distances_to_centroids(kmeans_algorithm_result, position_list, if_round=True):
    distances_to_centroids = []
    for n in range(len(kmeans_algorithm_result[0])):
        cluster = kmeans_algorithm_result[0][n]
        # calculate distances to centroids
        for index in cluster:
            temp_position = position_list[index]
            distance = geolocation.distance(temp_position, kmeans_algorithm_result[1][n])
            if if_round and distance > 400:
                distance = 400
            distances_to_centroids.append(distance)
    return distances_to_centroids


def calculateAverageDistance(position_list):
    avg_distance = []
    for pos1 in position_list:
        for pos2 in position_list:
            distance = geolocation.distance(pos1, pos2)
            avg_distance.append(distance)
    return np.mean(avg_distance)


def calculateNMeans(position_list):
    n = int(len(position_list) / 7)
    if n > 20:
        n = 20
    # custom_function = lambda pos1, pos2: GeoLocation.distance(pos1, pos2)
    metric = distance_metric(type_metric.USER_DEFINED, func=lambda pos1, pos2: geolocation.distance(pos1, pos2))

    # create K-Means algorithm with specific distance metric
    start_centers = []
    for m in range(n + 1):
        start_centers.append([random.uniform(49, 54), random.uniform(14, 23)])  # roughly coordinates of Poland
    sample = []
    for position in position_list:
        sample.append([position.lat, position.lon])

    if len(sample) == 0:
        return [[], []]
    kmeans_instance = kmeans(sample, start_centers, metric=metric, tolerance=0.000001)

    # run cluster analysis and obtain results
    kmeans_instance.process()
    return [kmeans_instance.get_clusters(), kmeans_instance.get_centers()]


def get_cluster_statistics(database):
    data = []
    cluster_names = database.get_cluster_names()
    for cluster_name in cluster_names:
        reviews_in_cluster = database.find_by_cluster(cluster_name)
        count_PL = 0
        real_count_PL = 0
        fake_count_PL = 0
        count = 0
        real_count = 0
        fake_count = 0
        for review in reviews_in_cluster:
            loc = review['localization']
            lat = loc['lat']
            lon = loc['lon']
            in_poland = geolocation.is_in_poland(lat, lon)
            if in_poland:
                count_PL += 1
            count += 1
            try:
                if review['is_real']:
                    if in_poland:
                        real_count_PL += 1
                    real_count += 1
                else:
                    if in_poland:
                        fake_count_PL += 1
                    fake_count += 1
            except:
                print(f"ERROR IN REVIEW_ID = {review['review_id']}")
        data.append([cluster_name, fake_count_PL, real_count_PL, count_PL, fake_count, real_count, count])
    headers = ["Cluster", 'fake_count_PL', 'real_count_PL', 'count_PL', 'fake_count', 'real_count',
               'count']
    print(tabulate(data, headers))

    df = pandas.DataFrame(data, columns=headers)
    plt.figure(figsize=(9, 9))
    plt.xlabel('CLUSTERS')
    x_axis = np.arange(len(df['Cluster']))
    # plt.ylabel('fake_count_PL')
    plt.bar(x_axis - 0.2, df['fake_count_PL'], width=0.4, label='fake_count_PL')
    plt.bar(x_axis + 0.2, df['real_count_PL'], width=0.4, label='real_count_PL')
    plt.xticks(x_axis, df['Cluster'], rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.savefig('polskie_dane.png')
    plt.show()

    plt.figure(figsize=(9, 9))
    plt.xlabel('CLUSTERS')
    x_axis = np.arange(len(df['Cluster']))
    # plt.ylabel('fake_count_PL')
    plt.bar(x_axis - 0.2, df['fake_count'], width=0.4, label='fake_count')
    plt.bar(x_axis + 0.2, df['real_count'], width=0.4, label='real_count')
    plt.xticks(x_axis, df['Cluster'], rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.savefig('wszystkie_dane.png')
    plt.show()


def is_type_already_known(type_of_object, cluster_dict):
    for key in cluster_dict:
        if type_of_object in cluster_dict[key]:
            return True
    return False

def parse_account_to_prediction_list(account: AccountBase, reviews_of_account: List[Union[ReviewOldInDB, ReviewPartialInDB]], with_scraped_reviews=False, bare_data=False) -> list:
    if not bare_data:
        return _parse_account_data_to_prediction_list(account=account, reviews_of_account=reviews_of_account,
                                                      with_scraped_reviews=with_scraped_reviews)
    else:
        return _parse_account_data_to_prediction_list_bare(account=account, reviews_of_account=reviews_of_account,
                                                      with_scraped_reviews=with_scraped_reviews)

def _parse_account_data_to_prediction_list(account: AccountBase,
                                           reviews_of_account: List[Union[ReviewOldInDB, ReviewPartialInDB]],
                                           with_scraped_reviews=False) -> list:
    account_data = []

    name_score = NLP.analyze_name_of_account(account.name)
    account_data.append(name_score)

    local_guide_level = account.local_guide_level
    if local_guide_level is None:
        local_guide_level = 0
    account_data.append(local_guide_level)

    number_of_reviews = account.number_of_reviews
    if number_of_reviews is None:
        number_of_reviews = 0
    account_data.append(number_of_reviews)

    is_private = account.is_private
    is_probably_deleted = False
    if is_private and number_of_reviews > 0:
        is_probably_deleted = True
    # account_data.append(is_private)
     #account_data.append(is_probably_deleted)

    if len(reviews_of_account) == 0:
        account_data.append(0)
        account_data.append(0)
        account_data.append(0)
    else:
        ratings_metrics = get_ratings_distribution_metrics(account.reviewer_id, reviews_of_account)
        ratings_mean = ratings_metrics[0]
        ratings_median = ratings_metrics[1]
        ratings_variance = ratings_metrics[2]
        account_data.append(ratings_mean)
        account_data.append(ratings_median)
        account_data.append(ratings_variance)

    if len(reviews_of_account) == 0:
        account_data.append(0)
        account_data.append(0)
        account_data.append(0)
    else:
        geolocation_metrics = get_geolocation_distribution_metrics(account.reviewer_id, reviews_of_account)
        geolocation_mean = geolocation_metrics[0]
        geolocation_median = geolocation_metrics[1]
        geolocation_variance = geolocation_metrics[2]
        account_data.append(geolocation_mean)
        account_data.append(geolocation_median)
        account_data.append(geolocation_variance)

    if len(reviews_of_account) == 0:
        account_data.append(0)
    else:
        photo_reviews_percentage = get_percentage_of_photographed_reviews(account.reviewer_id, reviews_of_account)
        account_data.append(photo_reviews_percentage)

    if len(reviews_of_account) == 0:
        account_data.append(0)
    else:
        responded_reviews_percentage = get_percentage_of_responded_reviews(account.reviewer_id, reviews_of_account)
        account_data.append(responded_reviews_percentage)

    if len(reviews_of_account) == 0:
        account_data.append(0)
        account_data.append(0)
        account_data.append(0)

        account_data.append(0)
        account_data.append(0)
        account_data.append(0)

        account_data.append(0)
        account_data.append(0)
        account_data.append(0)

    else:
        sentiment = []
        capslock = []
        interpunction = []
        for review in reviews_of_account:
            sentiment.append(NLP.sentiment_analyzer.analyze(review.content))
            capslock.append(NLP.get_capslock_score(review.content))
            interpunction.append(NLP.get_interpunction_score(review.content))

        account_data.append(mean(sentiment))
        account_data.append(median(sentiment))
        try:
            account_data.append(variance(sentiment))
        except statistics.StatisticsError:
            account_data.append(0)

        account_data.append(mean(capslock))
        account_data.append(median(capslock))
        try:
            account_data.append(variance(capslock))
        except statistics.StatisticsError:
            account_data.append(0)

        account_data.append(mean(interpunction))
        account_data.append(median(interpunction))
        try:
            account_data.append(variance(interpunction))
        except statistics.StatisticsError:
            account_data.append(0)

    if with_scraped_reviews:
        cluster_names = ENGLISH_TRANSLATION_CLUSTER_DICT.values()
        cluster_counter = get_type_of_objects_counts_for_account(account.reviewer_id, reviews_of_account)
        for cluster_name in cluster_names:
            account_data.append(cluster_counter[cluster_name])

    return account_data

def _parse_account_data_to_prediction_list_bare(account: AccountBase,
                                           reviews_of_account: List[Union[ReviewOldInDB, ReviewPartialInDB]],
                                           with_scraped_reviews=False) -> list:
    account_data = []

    local_guide_level = account.local_guide_level
    if local_guide_level is None:
        local_guide_level = 0
    account_data.append(local_guide_level)

    number_of_reviews = account.number_of_reviews
    if number_of_reviews is None:
        number_of_reviews = 0
    account_data.append(number_of_reviews)

    is_private = account.is_private
    is_probably_deleted = False
    if is_private and number_of_reviews > 0:
        is_probably_deleted = True
    # account_data.append(is_private)
    # account_data.append(is_probably_deleted)

    if len(reviews_of_account) == 0:
        account_data.append(0)
        account_data.append(0)
        account_data.append(0)
    else:
        ratings_metrics = get_ratings_distribution_metrics(account.reviewer_id, reviews_of_account)
        ratings_mean = ratings_metrics[0]
        ratings_median = ratings_metrics[1]
        ratings_variance = ratings_metrics[2]
        account_data.append(ratings_mean)
        account_data.append(ratings_median)
        account_data.append(ratings_variance)

    if len(reviews_of_account) == 0:
        account_data.append(0)
    else:
        photo_reviews_percentage = get_percentage_of_photographed_reviews(account.reviewer_id, reviews_of_account)
        account_data.append(photo_reviews_percentage)

    if len(reviews_of_account) == 0:
        account_data.append(0)
    else:
        responded_reviews_percentage = get_percentage_of_responded_reviews(account.reviewer_id, reviews_of_account)
        account_data.append(responded_reviews_percentage)

    return account_data

def parse_old_review_to_prediction_list(review: ReviewOldInDB, account: AccountOldInDB, prediction_mode: AttributesModes, exclude_localization = True) -> list:
    number_of_reviews = account.number_of_reviews
    reviewer_name = account.name
    is_local_guide = account.local_guide_level is not None
    if exclude_localization:
        localization = None
    else:
        localization = review.localization

    return _parse_review_data_to_prediction_list(number_of_reviews,
                                                 reviewer_name,
                                                 review.cluster,
                                                 is_local_guide,
                                                 prediction_mode,
                                                 localization,
                                                 review)

def parse_new_review_to_prediction_list(review: ReviewNewInDB, place_in_db: PlaceInDB, prediction_mode: AttributesModes, exclude_localization = True) -> list:
    if exclude_localization:
        localization = None
    else:
        localization = place_in_db.localization
    return _parse_review_data_to_prediction_list(review.number_of_reviews,
                                      review.reviewer_name,
                                      place_in_db.cluster,
                                      review.is_local_guide,
                                      prediction_mode,
                                      localization,
                                      review)

def _parse_review_data_to_prediction_list(number_of_reviews: int,
                                          reviewer_name: str,
                                          cluster_name: str,
                                          is_local_guide: bool,
                                          prediction_mode: AttributesModes,
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

    if prediction_mode == AttributesModes.SENTIMENT:
        return review_data

    review_data.append(NLP.get_capslock_score(review.content))
    review_data.append(NLP.get_interpunction_score(review.content))

    if prediction_mode == AttributesModes.SENTIMENT_CAPS_INTER:
        return review_data

    if prediction_mode == AttributesModes.SIMPLE_NLP:
        review_data.append(NLP.get_emotional_interpunction_score(review.content))
        review_data.append(NLP.get_consecutive_emotional_interpunction_score(review.content))
        review_data.append(NLP.get_emojis_score(review.content))
        return review_data


    stylo_metrix_analyzed: StyloMetrixResults = NLP.get_stylo_metrix_metrics(review.content)
    review_data.append(stylo_metrix_analyzed.G_V)
    review_data.append(stylo_metrix_analyzed.G_N)

    review_data.append(stylo_metrix_analyzed.IN_V_INFL)

    review_data.append(stylo_metrix_analyzed.L_TCCT1)

    review_data.append(stylo_metrix_analyzed.PS_M_VALa)
    review_data.append(stylo_metrix_analyzed.PS_M_AROa)
    review_data.append(stylo_metrix_analyzed.PS_M_DOMa)

    review_data.append(stylo_metrix_analyzed.PS_M_AROb)

    if prediction_mode == AttributesModes.LESS_NLP:
        return review_data

    review_data.append(stylo_metrix_analyzed.G_ADV)
    review_data.append(stylo_metrix_analyzed.PS_M_VALb)
    review_data.append(stylo_metrix_analyzed.L_TCCT5)
    review_data.append(stylo_metrix_analyzed.IN_V_1S)

    if prediction_mode == AttributesModes.ALL_NLP:
        return review_data

    if localization is not None:
        review_data.append(localization.lat)
        review_data.append(localization.lon)

    return review_data
