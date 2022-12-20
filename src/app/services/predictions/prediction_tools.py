# coding=ISO-8859-2
import random
from math import floor
from typing import List

from sklearn.utils import shuffle

from sklearn import tree
import numpy as np
import csv
import pickle

from app.config import NLP, ENGLISH_TRANSLATION_CLUSTER_DICT
from app.dao.dao_accounts_old import DAOAccountsOld
from app.dao.dao_places import DAOPlaces
from app.dao.dao_reviews_new import DAOReviewsNew

from app.dao.dao_reviews_old import DAOReviewsOld
from app.models.account import AccountOldInDB
from app.models.base_mongo_model import MongoObjectId
from app.models.place import Place, PlaceInDB
from app.models.review import ReviewOldInDB, ReviewNew, ReviewNewInDB
from app.services.analysis.AnalysisTools import get_ratings_distribution_metrics, get_geolocation_distribution_metrics, \
    get_type_of_objects_counts_for_account, get_percentage_of_photographed_reviews, get_percentage_of_responded_reviews


def get_and_prepare_accounts_data(save_to_file=False):
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    accounts: List[AccountOldInDB] = dao_accounts_old.find_all()
    prepared_data = []
    classes = []
    progress = 0
    amount = len(accounts)
    print("Progress: ")
    print("###################")
    for account in accounts:
        account_data = []
        dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
        reviews_of_account: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(account.reviewer_id)

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
        account_data.append(is_private)
        account_data.append(is_probably_deleted)

        if number_of_reviews == 0:
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

        if number_of_reviews == 0:
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

        cluster_names = ENGLISH_TRANSLATION_CLUSTER_DICT.values()
        cluster_counter = get_type_of_objects_counts_for_account(account.reviewer_id, reviews_of_account)
        for cluster_name in cluster_names:
            account_data.append(cluster_counter[cluster_name])

        if number_of_reviews == 0:
            account_data.append(0)
        else:
            photo_reviews_percentage = get_percentage_of_photographed_reviews(account.reviewer_id, reviews_of_account)
            account_data.append(photo_reviews_percentage)

        if number_of_reviews == 0:
            account_data.append(0)
        else:
            responded_reviews_percentage = get_percentage_of_responded_reviews(account.reviewer_id, reviews_of_account)
            account_data.append(responded_reviews_percentage)

        prepared_data.append(account_data)
        if account.fake_service != "real":
            fake = True
        else:
            fake = False

        classes.append(fake)
        progress += 1
        if progress >= 0.05 * amount:
            progress = 0
            print(f"#", end="")

    print("\n")
    if save_to_file:
        with open('app/data/formatted_accounts_data.csv', mode='w') as f:
            employee_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for sample, _class in zip(prepared_data, classes):
                sample_copy = sample.copy()
                sample_copy.append(_class)
                employee_writer.writerow(sample_copy)
    return prepared_data, classes


def get_and_prepare_reviews_data(save_to_file=False, exclude_localization = True):
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
    reviews: List[ReviewOldInDB] = dao_reviews_old.find_all()
    prepared_data = []
    classes = []
    progress = 0
    amount = len(reviews)
    clusters_list  = list(ENGLISH_TRANSLATION_CLUSTER_DICT.values())
    print("Progress: ")
    print("###################")
    for review in reviews:
        reviewer_id = review.reviewer_id

        dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
        account: AccountOldInDB = dao_accounts_old.find_one_by_query({"reviewer_id": reviewer_id})

        review_data = review.parse_to_prediction_list(account,exclude_localization)
        prepared_data.append(review_data)
        classes.append(not review.is_real)

        progress += 1
        if progress >= 0.05 * amount:
            progress = 0
            print(f"#", end="")

    print("\n")
    if save_to_file:
        if exclude_localization:
            file_name = 'app/data/formatted_reviews_data_wo_loc.csv'
        else:
            file_name = 'app/data/formatted_reviews_data.csv'
        with open(file_name, mode='w') as f:
            employee_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for sample, _class in zip(prepared_data, classes):
                sample_copy = sample.copy()
                sample_copy.append(_class)
                employee_writer.writerow(sample_copy)

    return prepared_data, classes


def get_prepared_accounts_data_from_file(ignore_empty_accounts=False):
    file_path = 'app/data/formatted_accounts_data.csv'
    samples, classes = get_prepared_data_from_file(file_path, ignore_empty_accounts=ignore_empty_accounts)
    return samples, classes

def get_prepared_reviews_data_from_file(file_name: str = None, exclude_localization = True):
    if file_name is None:
        if exclude_localization:
            file_path = 'app/data/formatted_reviews_data_wo_loc.csv'
        else:
            file_path = 'app/data/formatted_reviews_data.csv'
    else:
        file_path = file_name

    samples, classes = get_prepared_data_from_file(file_path)
    return samples, classes

def get_prepared_data_from_file(file_path, ignore_empty_accounts=False):
    samples = []
    classes = []
    with open(file_path) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            if len(row) <= 0:
                continue

            formatted_row = format_row_from_csv(row)
            if ignore_empty_accounts and formatted_row[2] == 0:
                continue
            samples.append(formatted_row[:-1])
            classes.append(formatted_row[-1])

    print("Loaded data from file")
    return samples, classes


def format_row_from_csv(row):
    new_row = []
    for val in row:
        # int
        if val.isdigit():
            new_row.append(int(val))
            continue
        if val == "True":
            new_row.append(True)
            continue
        if val == "False":
            new_row.append(False)
            continue
        try:
            new_row.append(float(val))
            continue
        except ValueError:
            raise Exception(f"Type of {val}, {type(val)} not known")
    return new_row


def get_train_and_test_datasets(frac, whole_dataset=None, resolve_backpack_problem=False):
    samples, classes = get_and_reshuffle_data(whole_dataset)
    train_samples = []
    train_classes = []
    test_samples = []
    test_classes = []
    if resolve_backpack_problem:
        return resolve_backpack_packing_single(frac, samples, classes)
    for sample, _class in zip(samples, classes):
        rand = random.uniform(0, 1)
        if rand > frac:
            test_samples.append(sample)
            test_classes.append(_class)
        else:
            train_samples.append(sample)
            train_classes.append(_class)

    return train_samples, train_classes, test_samples, test_classes

def resolve_backpack_packing_single(frac, samples, classes):
    combined, max_amount = separate_data_by_accounts(
        classes, frac, samples)
    train_samples = []
    train_classes = []
    test_samples = []
    test_classes = []

    for reviewer_id in combined:
        rand = random.uniform(0, 1)
        current_amount = len(train_samples)
        if rand < frac and current_amount + len(combined[reviewer_id]) < max_amount:
            for sample_class in combined[reviewer_id]:
                train_samples.append(sample_class[0:-2])
                train_classes.append(sample_class[-1])
        else:
            for sample_class in combined[reviewer_id]:
                test_samples.append(sample_class[0:-2])
                test_classes.append(sample_class[-1])

    return train_samples, train_classes, test_samples, test_classes


def resolve_backpack_packing_k_fold(k, samples, classes):
    combined, max_amount = separate_data_by_accounts(classes, (1/k), samples)
    dict_samples = {}
    dict_classes = {}
    for fold in range(k):
        dict_samples[fold] = []
        dict_classes[fold] = []

    for reviewer_id in combined:
        fold = find_available_fold(k, max_amount, combined, dict_samples, reviewer_id)
        for sample_class in combined[reviewer_id]:
            dict_samples[fold].append(sample_class[0:-1])
            dict_classes[fold].append(sample_class[-1])

    return dict_samples, dict_classes

def find_available_fold(k, max_amount, combined, dict_samples, reviewer_id):
    available_folds = []
    for fold in range(k):
        if len(dict_samples[fold]) + len(combined[reviewer_id]) < max_amount:
            available_folds.append(fold)
    if available_folds:
        return random.choice(available_folds)
    else:
        return floor(random.uniform(0, k))


def separate_data_by_accounts(classes, frac, samples):
    combined = {}
    for sample, _class in zip(samples, classes):
        if sample[-1] not in combined:
            combined[sample[-1]] = []
        sample_copy = sample.copy()
        sample_copy.append(_class)
        combined[sample[-1]].append(sample_copy)
    max_amount = frac * len(samples)
    keys = list(combined)
    random.shuffle(keys)
    combined = {k: combined[k] for k in keys}
    return combined, max_amount


def build_model_return_predictions(train_samples, train_classes, test_samples):
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(train_samples, train_classes)
    return decision_tree.predict(test_samples), decision_tree


def calculate_basic_metrics(predictions, test_classes):
    # some magic to convert boolean array to int array
    diffs = np.subtract(predictions * 1, test_classes * 1)
    sums = np.add(predictions * 1, test_classes * 1)

    FP = np.count_nonzero(diffs == -1)
    FN = np.count_nonzero(diffs == 1)
    TP = np.count_nonzero(sums == 2)
    TN = np.count_nonzero(sums == 0)
    ALL = TP + TN + FP + FN
    return TP, TN, FP, FN, ALL


def calculate_metrics(TP, TN, FP, FN, ALL):
    prevalence = (TP + FP) / ALL
    accuracy = (TP + TN) / ALL
    F1 = (2 * TP) / (2 * TP + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(
        f"Accuracy = {round(accuracy, 3)} | Prevalence = {round(prevalence, 3)} |\nF1 = {round(F1, 3)} | Precision = {round(precision, 3)} | Recall = {round(recall, 3)}")


def k_fold_validation(k, whole_dataset, resolve_backpack_problem=False):
    if not resolve_backpack_problem:
        dict_samples, dict_classes = separate_data_into_k_folds(k, whole_dataset)
    else:
        dict_samples, dict_classes = separate_data_into_k_folds_resolve_backpack(k, whole_dataset)
    TP, TN, FP, FN, ALL = 0, 0, 0, 0, 0
    best_model = None
    best_precission = 0
    for key in dict_samples:
        test_samples = dict_samples[key]
        test_classes = dict_classes[key]
        train_samples = []
        train_classes = []
        for key_inner in dict_samples:
            if key_inner != key:
                train_samples += dict_samples[key_inner]
                train_classes += dict_classes[key_inner]
        predictions, trained_model = build_model_return_predictions(train_samples, train_classes, test_samples)
        TP_k, TN_k, FP_k, FN_k, ALL_k = calculate_basic_metrics(predictions, test_classes)
        try:
            fold_precision = TP_k / (TP_k + FP_k)
        except ZeroDivisionError:
            fold_precision = 0
        if fold_precision > best_precission:
            best_model = trained_model

        TP += TP_k
        TN += TN_k
        FP += FP_k
        FN += FN_k
        ALL += ALL_k
    print(f"Average metrics of {k}-fold validation:")
    pickle.dump(best_model, open("app/pickled_prediction_models/model", 'wb'))
    print("dumped")
    calculate_metrics(TP, TN, FP, FN, ALL)


def separate_data_into_k_folds(k, whole_dataset):
    samples, classes = get_and_reshuffle_data(whole_dataset)
    dict_samples = {}
    dict_classes = {}
    fold = 0
    for sample, _class in zip(samples, classes):
        if fold not in dict_classes:
            dict_samples[fold] = []
            dict_classes[fold] = []
        dict_samples[fold].append(sample)
        dict_classes[fold].append(_class)
        fold = (fold + 1) % k

    return dict_samples, dict_classes


def separate_data_into_k_folds_resolve_backpack(k, whole_dataset):
    samples, classes = get_and_reshuffle_data(whole_dataset)
    return resolve_backpack_packing_k_fold(k, samples, classes)


def get_and_reshuffle_data(whole_dataset):
    if whole_dataset is None:
        samples, classes = get_and_prepare_accounts_data()
    else:
        samples, classes = whole_dataset
    samples, classes = shuffle(samples, classes, random_state=42)
    return samples, classes


def knapSack(W, wt, val, n):
    dp = [0 for i in range(W + 1)]  # Making the dp array

    for i in range(1, n + 1):  # taking first i elements
        for w in range(W, 0, -1):  # starting from back,so that we also have data of
            # previous computation when taking i-1 items
            if wt[i - 1] <= w:
                # finding the maximum value
                dp[w] = max(dp[w], dp[w - wt[i - 1]] + val[i - 1])

    return dp[W]  # returning the maximum value of knapsack

def predict_reviews_from_place(place_id: MongoObjectId):
    dao_places: DAOPlaces = DAOPlaces()
    dao_reviews: DAOReviewsNew = DAOReviewsNew()

    place: PlaceInDB = dao_places.find_by_id(place_id)
    reviews_on_place: List[ReviewNewInDB] = dao_reviews.find_reviews_of_place(place_id)
    model = pickle.load(open("app/pickled_prediction_models/model", 'rb'))
    for review in reviews_on_place:
        prediction = model.predict([review.parse_to_prediction_list(place)])
        dao_reviews.update_one({"_id": review.id}, {"$set": {"is_real": not prediction[0]}})
    print("Reviews predictions updated")

def predict_all_reviews_from_new_scrape():
    dao_places: DAOPlaces = DAOPlaces()
    dao_reviews: DAOReviewsNew = DAOReviewsNew()

    reviews: List[ReviewNewInDB] = dao_reviews.find_all()
    model = pickle.load(open("app/pickled_prediction_models/model", 'rb'))
    for review in reviews:
        place: PlaceInDB = dao_places.find_by_id(review.place_id)
        prediction = model.predict([review.parse_to_prediction_list(place)])
        dao_reviews.update_one({"_id": review.id}, {"$set": {"is_real": not prediction[0]}})
    print("Reviews predictions updated")




if __name__ == '__main__':
    #get_and_prepare_accounts_data(save_to_file=False)
    get_and_prepare_reviews_data(save_to_file=True, exclude_localization=True)
    data = get_prepared_reviews_data_from_file(exclude_localization=True) # get_prepared_accounts_data_from_file(ignore_empty_accounts=True) # get_and_prepare_accounts_data(save_to_file=True)
    # for i in range(20):
    #     prepared_data = get_train_and_test_datasets(3/5, data, resolve_backpack_problem=True)
    #     predicts = build_model_return_predictions(prepared_data[0], prepared_data[1], prepared_data[2])
    #     TP, TN, FP, FN, ALL = calculate_basic_metrics(predicts, prepared_data[3])
    #     calculate_metrics(TP, TN, FP, FN, ALL)
    k_fold_validation(10, data, resolve_backpack_problem=True)
    predict_all_reviews_from_new_scrape()
    # print("FINISHED")
