import random
import statistics
from statistics import mean, median, variance
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils import distance_metric, type_metric
from tabulate import tabulate

from app.config import ENGLISH_TRANSLATION_CLUSTER_DICT
from app.dao.dao_reviews_old import DAOReviewsOld
from app.models.review import ReviewOldInDB
from app.services.analysis import geolocation
from app.services.scraper.models.position import Position

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


def get_ratings_distribution_metrics(account_id: str, reviews_of_account: Optional[List[ReviewOldInDB]] = None):
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


def get_geolocation_distribution_metrics(account_id, reviews_of_account: Optional[List[ReviewOldInDB]] = None):
    if reviews_of_account is None:
        dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
        reviews_of_account: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(account_id)
    pos_list: List[Position] = [review.localization for review in reviews_of_account]

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
                                           reviews_of_account: Optional[List[ReviewOldInDB]] = None) -> float:
    if reviews_of_account is None:
        dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
        reviews_of_account: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(account_id)
    photos_urls_list: List[Optional[List[str]]] = [review.photos_urls for review in reviews_of_account]

    none_count = photos_urls_list.count(None)
    return none_count / len(photos_urls_list)


def get_percentage_of_responded_reviews(account_id,
                                        reviews_of_account: Optional[List[ReviewOldInDB]] = None) -> float:
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


def get_type_of_objects_counts_for_account(account_id, reviews_of_account: Optional[List[ReviewOldInDB]] = None):
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
            if if_round and distance > 199:
                distance = 199
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
