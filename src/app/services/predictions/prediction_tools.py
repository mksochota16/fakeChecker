# coding=ISO-8859-2
import random
from enum import Enum
from math import floor
from typing import List, Optional

from sklearn.utils import shuffle
from sklearn import tree

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import numpy as np
import csv
import pickle

from config import NLP, ENGLISH_TRANSLATION_CLUSTER_DICT
from dao.dao_accounts_new import DAOAccountsNew
from dao.dao_accounts_old import DAOAccountsOld
from dao.dao_places import DAOPlaces
from dao.dao_reviews_new import DAOReviewsNew

from dao.dao_reviews_old import DAOReviewsOld
from dao.dao_reviews_partial import DAOReviewsPartial
from models.account import AccountOldInDB, AccountNewInDB
from models.base_mongo_model import MongoObjectId
from models.place import PlaceInDB
from models.response import AccountIsPrivateException
from models.review import ReviewOldInDB, ReviewNewInDB, ReviewPartialInDB
from services.analysis.AnalysisTools import get_ratings_distribution_metrics, get_geolocation_distribution_metrics, \
    get_type_of_objects_counts_for_account, get_percentage_of_photographed_reviews, get_percentage_of_responded_reviews, \
    parse_account_to_prediction_list, parse_old_review_to_prediction_list, parse_new_review_to_prediction_list
from services.predictions.prediction_constants import AttributesModes

global current_classifier

class AvailablePredictionModels(Enum):
    # DECISION_TREE = DecisionTreeClassifier()
    RANDOM_FOREST = RandomForestClassifier(n_estimators=100)# RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1) precision: 0.5082382762991128 f1: 0.6515028432168968
    # NEURAL_NETWORK = MLPClassifier(max_iter=1000) #MLPClassifier(alpha=1, max_iter=1000) precision: 0.2256020278833967 f1: 0.3371212121212121
    # K_NEIGHBORS = KNeighborsClassifier() # KNeighborsClassifier(3) precision: 0.9607097591888466 f1: 0.9171203871748337
    # SUPPORT_VECTOR_MACHINE = SVC() # SVC(gamma=2, C=1) precision: 0.0012674271229404308 f1: 0.002531645569620253
    # not enough RAM #GAUSSIAN_PROCESS = GaussianProcessClassifier(1.0 * RBF(1.0))
    # ADABOOST = AdaBoostClassifier(learning_rate=0.55)
    # NAIVE_BAYES = GaussianNB()
    # QUADRATIC_DISCRIMINANT_ANALYSIS = QuadraticDiscriminantAnalysis()


def get_and_prepare_accounts_data(save_to_file=False, bare_data=False):
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    accounts: List[AccountOldInDB] = dao_accounts_old.find_all()
    prepared_data = []
    classes = []
    progress = 0
    amount = len(accounts)
    print("Progress: ")
    print("###################")
    for account in accounts:
        dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
        reviews_of_account: List[ReviewOldInDB] = dao_reviews_old.find_reviews_of_account(account.reviewer_id)

        account_data: List = parse_account_to_prediction_list(account=account, reviews_of_account=reviews_of_account, bare_data=bare_data)
        if account.number_of_reviews == 0:
            continue
        prepared_data.append(account_data)
        if account.fake_service != "real":
            classes.append(True)
        else:
            classes.append(False)

        progress += 1
        if progress >= 0.05 * amount:
            progress = 0
            print(f"#", end="")

    print("\n")
    if save_to_file:
        if not bare_data:
            file_path = 'data/formatted_accounts_no_count_data.csv'
        else:
            file_path = 'data/formatted_accounts_data_bare.csv'
        with open(file_path, mode='w') as f:
            employee_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for sample, _class in zip(prepared_data, classes):
                sample_copy = sample.copy()
                sample_copy.append(_class)
                employee_writer.writerow(sample_copy)
    return prepared_data, classes


def get_and_prepare_reviews_data(attribute_mode: AttributesModes,save_to_file=False, exclude_localization=True,
                                 file_name=None,
                                 ):
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()
    reviews: List[ReviewOldInDB] = dao_reviews_old.find_all()
    prepared_data = []
    classes = []
    progress = 0
    amount = len(reviews)
    print("Progress: ")
    print("###################")
    for review in reviews:
        reviewer_id = review.reviewer_id

        dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
        account: AccountOldInDB = dao_accounts_old.find_one_by_query({"reviewer_id": reviewer_id})

        review_data = parse_old_review_to_prediction_list(review, account, attribute_mode, exclude_localization)
        review_data.append(reviewer_id)
        prepared_data.append(review_data)
        classes.append(not review.is_real)

        progress += 1
        if progress >= 0.05 * amount:
            progress = 0
            print(f"#", end="")


    print("\n")
    if save_to_file:
        if not exclude_localization:
            file_name = 'data/formatted_reviews_data.csv'
        if file_name is None:
            file_name = attribute_mode.value
        with open(file_name, mode='w') as f:
            employee_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for sample, _class in zip(prepared_data, classes):
                sample_copy = sample.copy()
                sample_copy.append(_class)
                employee_writer.writerow(sample_copy)

    return prepared_data, classes


def get_prepared_accounts_data_from_file(ignore_empty_accounts=False, bare_data = False):
    if not bare_data:
        file_path = 'data/formatted_accounts_no_count_data.csv' # data/formatted_accounts_data.csv'
    else:
        file_path = 'data/formatted_accounts_data_bare.csv'
    samples, classes = get_prepared_data_from_file(file_path, ignore_empty_accounts=ignore_empty_accounts)
    return samples, classes


def get_prepared_reviews_data_from_file(attribute_mode:AttributesModes, file_name: str = None):
    if file_name is not None:
        file_path = file_name
    else:
        file_path = attribute_mode.value

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
            if ignore_empty_accounts and formatted_row[2] == 0: #FIXME for normal data it should be 2
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


def get_train_and_test_datasets(frac, whole_dataset=None, resolve_backpack_problem=True, equal_datasets=False, cut_reviewer_id=True):
    samples, classes = get_and_reshuffle_data(whole_dataset)
    train_samples = []
    train_classes = []
    test_samples = []
    test_classes = []
    if resolve_backpack_problem:
        if equal_datasets:
            return resolve_backpack_packing_single_equal_datasets(frac, samples, classes)
        else:
            return resolve_backpack_packing_single(frac, samples, classes, cut_reviewer_id)
    for sample, _class in zip(samples, classes):
        rand = random.uniform(0, 1)
        if rand > frac:
            test_samples.append(sample)
            test_classes.append(_class)
        else:
            train_samples.append(sample)
            train_classes.append(_class)

    return train_samples, train_classes, test_samples, test_classes


def resolve_backpack_packing_single(frac, samples, classes, cut_reviewer_id=True):
    combined, max_amount = separate_data_by_accounts(
        classes, frac, samples)
    train_samples = []
    train_classes = []
    test_samples = []
    test_classes = []

    split_data_by_frac_controlled(train_samples, train_classes, test_samples, test_classes, frac, combined,
                                  max_amount, cut_reviewer_id)

    return train_samples, train_classes, test_samples, test_classes


def split_data_by_frac_controlled(train_samples, train_classes, test_samples, test_classes, frac, dict_to_separate, max_amount, cut_reviewer_id=True):
    for reviewer_id in dict_to_separate:
        rand = random.uniform(0, 1)
        current_amount = len(train_samples)
        if rand < frac and current_amount + len(dict_to_separate[reviewer_id]) < max_amount:
            for sample_class in dict_to_separate[reviewer_id]:
                if cut_reviewer_id:
                    train_samples.append(sample_class[0:-2])
                    train_classes.append(sample_class[-1])
                else:
                    train_samples.append(sample_class[0:-1])
                    train_classes.append(sample_class[-1])
        else:
            for sample_class in dict_to_separate[reviewer_id]:
                if cut_reviewer_id:
                    test_samples.append(sample_class[0:-2])
                    test_classes.append(sample_class[-1])
                else:
                    test_samples.append(sample_class[0:-1])
                    test_classes.append(sample_class[-1])



def resolve_backpack_packing_single_equal_datasets(frac, samples, classes):
    real, fake, max_amount = separate_data_by_accounts_and_genuineness(classes, frac, samples)
    train_samples = []
    train_classes = []
    test_samples = []
    test_classes = []

    split_data_by_frac_controlled(train_samples, train_classes, test_samples, test_classes, frac, real,
                                  max_amount/2)

    split_data_by_frac_controlled(train_samples, train_classes, test_samples, test_classes, frac, fake,
                                  max_amount / 2)

    return train_samples, train_classes, test_samples, test_classes


def resolve_backpack_packing_k_fold(k, samples, classes):
    combined, max_amount = separate_data_by_accounts(classes, (1 / k), samples)
    dict_samples = {}
    dict_classes = {}
    for fold in range(k):
        dict_samples[fold] = []
        dict_classes[fold] = []

    for reviewer_id in combined:
        fold = find_available_fold(k, max_amount, combined, dict_samples, reviewer_id)
        for sample_class in combined[reviewer_id]:
            dict_samples[fold].append(sample_class[0:-2])
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

def separate_data_by_accounts_and_genuineness(classes, frac, samples):
    fake = {}
    real = {}
    for sample, _class in zip(samples, classes):
        if _class: # real
            if sample[-1] not in real:
                real[sample[-1]] = []
            sample_copy = sample.copy()
            sample_copy.append(_class)
            real[sample[-1]].append(sample_copy)
        else: # fake
            if sample[-1] not in fake:
                fake[sample[-1]] = []
            sample_copy = sample.copy()
            sample_copy.append(_class)
            fake[sample[-1]].append(sample_copy)
    max_amount = frac * len(samples)

    keys_real = list(real)
    random.shuffle(keys_real)
    real = {k: real[k] for k in keys_real}

    keys_fake = list(fake)
    random.shuffle(keys_fake)
    fake = {k: fake[k] for k in keys_fake}

    return real, fake, max_amount


def build_model_return_predictions(train_samples, train_classes, test_samples):
    global current_classifier
    if current_classifier is None:
        decision_tree = RandomForestClassifier(n_estimators=100)# tree.DecisionTreeClassifier()
        decision_tree = decision_tree.fit(train_samples, train_classes)
        return decision_tree.predict(test_samples), decision_tree
    else:
        current_classifier = current_classifier.fit(train_samples, train_classes)
        return current_classifier.predict(test_samples), current_classifier


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

def perform_vote_of_models(lists_predictions: List[List[float]], list_of_f1: List[float]):
    denominator = sum(list_of_f1)
    dict_of_votes = {}
    for model_predictions in lists_predictions:
        for index, prediction in enumerate(model_predictions):
            if index not in dict_of_votes:
                dict_of_votes[index] = 0
            dict_of_votes[index] += prediction
    return [int((x / denominator) >= 0) for x in list(dict_of_votes.values())]

def calculate_basic_metrics_from_vote(lists_predictions: List[List[float]], test_classes, list_of_f1: List[float]):
    list_of_vote_results = perform_vote_of_models(lists_predictions, list_of_f1)
    TP, TN, FP, FN, ALL = calculate_basic_metrics(list_of_vote_results, test_classes)
    calculate_metrics(TP, TN, FP, FN, ALL)


def calculate_metrics(TP, TN, FP, FN, ALL):
    prevalence = (TP + FP) / ALL
    accuracy = (TP + TN) / ALL
    try:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    except ZeroDivisionError:
        F1 = 0
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
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
    # pickle.dump(best_model, open(f"../../pickled_prediction_models/{what_to_predict}/model", 'wb'))
    # print("dumped")
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
    samples, classes = shuffle(samples, classes)
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


def predict_reviews_from_place(place_id: MongoObjectId, model_path: str = "pickled_prediction_models/reviews/RANDOM_FOREST_BEST_SENT_CAPS_INTER",attribute_mode: AttributesModes = AttributesModes.BEST):
    dao_places: DAOPlaces = DAOPlaces()
    dao_reviews: DAOReviewsNew = DAOReviewsNew()

    place: PlaceInDB = dao_places.find_by_id(place_id)
    reviews_on_place: List[ReviewNewInDB] = dao_reviews.find_reviews_of_place(place_id)
    model = pickle.load(open(model_path, 'rb'))
    for review in reviews_on_place:
        prediction = model.predict([parse_new_review_to_prediction_list(review, place, attribute_mode)])
        dao_reviews.update_one({"_id": review.id}, {"$set": {"is_real": not prediction[0]}})
    print("Reviews predictions updated")

def predict_account(account_id: MongoObjectId, model_path: str = "pickled_prediction_models/accounts/RANDOM_FOREST_BEST", with_scraped_reviews: bool = False):
    dao_accounts_new: DAOAccountsNew = DAOAccountsNew()
    dao_reviews_partial: DAOReviewsPartial = DAOReviewsPartial()

    account: AccountNewInDB = dao_accounts_new.find_by_id(account_id)
    if account.is_private or (account.is_deleted is not None and account.is_deleted):
        raise AccountIsPrivateException()
    reviews_of_account: List[ReviewPartialInDB] = dao_reviews_partial.find_reviews_of_account(account.reviewer_id)
    model = pickle.load(open(model_path, 'rb'))
    if with_scraped_reviews:
        raise NotImplementedError
    else:
        account_info_list = [parse_account_to_prediction_list(account=account, reviews_of_account=reviews_of_account, with_scraped_reviews=False)]
        if None in account_info_list:
            return
        try:
            prediction = model.predict(account_info_list)
        except ValueError:
            return
        dao_accounts_new.update_one({"_id": account.id}, {"$set": {"is_real": not prediction[0]}})
    print("Account prediction updated")
    return prediction[0]


def predict_all_reviews_from_new_scrape_one_model(attribute_mode:AttributesModes):
    dao_places: DAOPlaces = DAOPlaces()
    dao_reviews: DAOReviewsNew = DAOReviewsNew()

    reviews: List[ReviewNewInDB] = dao_reviews.find_all()
    model = pickle.load(open("../../pickled_prediction_models/reviews/model", 'rb'))
    for review in reviews:
        place: PlaceInDB = dao_places.find_by_id(review.place_id)
        prediction = model.predict(parse_new_review_to_prediction_list(review, place, attribute_mode))
        dao_reviews.update_one({"_id": review.id}, {"$set": {"is_real": not prediction[0]}})
    print("Reviews new predictions updated")

def predict_all_old_reviews():
    dao_accounts_old: DAOAccountsOld = DAOAccountsOld()
    dao_reviews_old: DAOReviewsOld = DAOReviewsOld()

    reviews: List[ReviewOldInDB] = dao_reviews_old.find_all()
    model = pickle.load(open("../../pickled_prediction_models/reviews/model", 'rb'))
    for review in reviews:
        account: AccountOldInDB = dao_accounts_old.find_one_by_query({"reviewer_id": review.reviewer_id})
        prediction = model.predict([review.parse_to_prediction_list(account)])
        dao_reviews_old.update_one({"_id": review.id}, {"$set": {"test_prediction": not prediction[0]}})
    print("Reviews old predictions updated")

def predict_all_reviews_from_new_scrape_all_models():
    dao_places: DAOPlaces = DAOPlaces()
    dao_reviews: DAOReviewsNew = DAOReviewsNew()

    reviews: List[ReviewNewInDB] = dao_reviews.find_all()
    model = pickle.load(open("../../pickled_prediction_models/reviews/model", 'rb'))
    for review in reviews:
        place: PlaceInDB = dao_places.find_by_id(review.place_id)
        prediction = model.predict([review.parse_to_prediction_list(place)])
        dao_reviews.update_one({"_id": review.id}, {"$set": {"is_real": not prediction[0]}})
    print("Reviews predictions updated")

def fit_and_get_model(train_samples: List, train_classes: List, test_samples: List, test_classes: List,
                      selected_model: AvailablePredictionModels):
    model = selected_model.value
    model.fit(train_samples, train_classes)
    predictions = model.predict(test_samples)
    TP, TN, FP, FN, ALL = calculate_basic_metrics(predictions, test_classes)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TN + FN == 0:
        f_precision = 0
    else:
        f_precision = TN / (TN + FN)

    if TP + FP + FN == 0:
        f1 = 0
    else:
        f1 = (2 * TP) / (2 * TP + FP + FN)

    if TN + FN + FP == 0:
        f_f1 = 0
    else:
        f_f1 = (2 * TN) / (2 * TN + FN + FP)

    try:
        recall = TP / (TP + FN)
        f_recall = TN / (TN + FP)
    except ZeroDivisionError:
        recall = 0
        f_recall = 0

    try:
        f_recall = TN / (TN + FP)
    except ZeroDivisionError:
        f_recall = 0

    return model, precision, f1, recall, f_precision, f_f1, f_recall

def combine_samples_and_classes(samples: List, classes: List):
    combined = []
    for sample, _class in zip(samples, classes):
        sample_copy = sample.copy()
        combined.append(sample_copy.append(_class))
    return combined

def fit_and_tests_all_models(all_data, what_to_predict: str, frac=0.8):
    train_samples, train_classes, test_samples, test_classes = get_train_and_test_datasets(whole_dataset=all_data, frac=frac, resolve_backpack_problem=True, cut_reviewer_id=False)
    test_samples = cut_reviewer_id(test_samples)
    train_data = (train_samples, train_classes)
    trained_models = []
    print("#"*100)
    for available_model in AvailablePredictionModels:
        f1 = 0
        for i in range(100):
            print("#", end="")
            temp_train_samples, temp_train_classes, temp_test_samples, temp_test_classes = get_train_and_test_datasets(
                whole_dataset=train_data, frac=0.8, resolve_backpack_problem=True)
            temp_model, temp_precision, temp_f1, temp_recall, temp_f_precision, temp_f_f1, temp_f_recall= fit_and_get_model(temp_train_samples, temp_train_classes, temp_test_samples, temp_test_classes, available_model)
            if temp_f1 > f1:
                f1 = temp_f1
                model = temp_model
                precision = temp_precision
                recall = temp_recall
                f_precision = temp_f_precision
                f_f1 = temp_f_f1
                f_recall = temp_f_recall
        print("")
        pickle.dump(model, open(f"pickled_prediction_models/{what_to_predict}/{available_model.name}", 'wb'))
        with open(f"pickled_prediction_models/{what_to_predict}/{available_model.name}.metrics", 'w') as metrics_file:
            metrics_file.write(f'{precision} {f1}')
        print(f"Model {available_model.name} \n"
              f"FAKE: precision: {precision} f1: {f1} recall: {recall} \n"
              f"REAL: precision: {f_precision} f1: {f_f1} recall: {f_recall} \n")
        trained_models.append([model, precision, f1])

    print("All models trained")
    return trained_models, test_samples, test_classes

def train_best_from_every_available_models(data, what_to_predict: str, frac=0.7, bare_data=False, resolve_backpack_problem=True):
    if not bare_data:
        file_path_add = ''
    else:
        file_path_add = '/bare_data'
    for available_model in AvailablePredictionModels:
        f1 = 0
        for i in range(100):
            print("#", end="")
            temp_train_samples, temp_train_classes, temp_test_samples, temp_test_classes = get_train_and_test_datasets(
                whole_dataset=data, frac=frac, resolve_backpack_problem=resolve_backpack_problem)
            temp_model, temp_precision, temp_f1, temp_recall, temp_f_precision, temp_f_f1, temp_f_recall = fit_and_get_model(temp_train_samples, temp_train_classes,
                                                                                 temp_test_samples, temp_test_classes,
                                                                                 available_model)
            if temp_f1 > f1:
                f1 = temp_f1
                model = temp_model
                precision = temp_precision
                recall = temp_recall
                f_precision = temp_f_precision
                f_f1 = temp_f_f1
                f_recall = temp_f_recall
        print("")

        pickle.dump(model, open(f"pickled_prediction_models/{what_to_predict}{file_path_add}/{available_model.name}", 'wb'))
        with open(f"pickled_prediction_models/{what_to_predict}{file_path_add}/{available_model.name}.metrics",
                  'w') as metrics_file:
            metrics_file.write(f'{precision} {f1}')
        print(f"Model {available_model.name} \n"
              f"FAKE: precision: {precision} f1: {f1} recall: {recall} \n"
              f"REAL: precision: {f_precision} f1: {f_f1} recall: {f_recall} \n")


def load_all_trained_models(what_to_predict: str):
    trained_models = []
    for available_model in AvailablePredictionModels:
        model = pickle.load(open(f"pickled_prediction_models/{what_to_predict}/{available_model.name}", 'rb'))
        with open(f"pickled_prediction_models/{what_to_predict}/{available_model.name}.metrics", 'r') as metrics_file:
            precision, f1 = metrics_file.read().split()
            precision = float(precision)
            f1 = float(f1)
        trained_models.append([model, precision, f1])
    return trained_models

def test_vote_of_trained_models(test_samples: List, test_classes: List):
    trained_models = load_all_trained_models()
    predictions = []
    list_of_f1 = []
    for model, precision, f1 in trained_models:
        model_predictions_bool: List[bool] = model.predict(test_samples)
        model_predictions_float = [f1 if item else -f1 for item in model_predictions_bool]
        predictions.append(model_predictions_float)
        list_of_f1.append(f1)

    calculate_basic_metrics_from_vote(predictions, test_classes, list_of_f1)

def predict_by_vote_of_models(sample_list_to_test: List[List]) -> List[int]:
    trained_models = load_all_trained_models()
    predictions = []
    list_of_f1 = []
    for model, precision, f1 in trained_models:
        model_predictions_bool: List[bool] = model.predict(sample_list_to_test)
        model_predictions_float = [f1 if item else -f1 for item in model_predictions_bool]
        predictions.append(model_predictions_float)
        list_of_f1.append(f1)

    list_of_vote_results = perform_vote_of_models(predictions, list_of_f1)
    return list_of_vote_results

def update_predictions_of_reviews_from_new_scrape(attribute_mode:AttributesModes):
    dao_places: DAOPlaces = DAOPlaces()
    dao_reviews: DAOReviewsNew = DAOReviewsNew()

    reviews: List[ReviewNewInDB] = dao_reviews.find_all()
    for review in reviews:
        place: PlaceInDB = dao_places.find_by_id(review.place_id)
        review_to_predict = parse_new_review_to_prediction_list(review, place, attribute_mode)
        prediction = predict_by_vote_of_models(review_to_predict)[0] == 1
        dao_reviews.update_one({"_id": review.id}, {"$set": {"is_real": not prediction}})
    print("Reviews predictions updated")

def prepare_data_for_all_modes():
    for attribute_mode in AttributesModes:
        get_and_prepare_reviews_data(attribute_mode=attribute_mode, save_to_file=True, exclude_localization=True)

def cut_reviewer_id(samples: List[List]) -> List[List]:
    return [sample[:-1] for sample in samples]



if __name__ == '__main__':
    # prepare_data_for_all_modes()
    # get_and_prepare_accounts_data(save_to_file=True, bare_data=True)
    # data = get_prepared_accounts_data_from_file(ignore_empty_accounts=True, bare_data=False)
    # get_and_prepare_reviews_data(attribute_mode=AttributesModes.LESS_NLP, save_to_file=True, exclude_localization=True)
    data = get_prepared_reviews_data_from_file(attribute_mode=AttributesModes.BEST)  # get_prepared_accounts_data_from_file(ignore_empty_accounts=True) # get_and_prepare_accounts_data(save_to_file=True)
        # for i in range(20):
        #     prepared_data = get_train_and_test_datasets(3/5, data, resolve_backpack_problem=True)
        #     predicts = build_model_return_predictions(prepared_data[0], prepared_data[1], prepared_data[2])
        #     TP, TN, FP, FN, ALL = calculate_basic_metrics(predicts, prepared_data[3])
        #     calculate_metrics(TP, TN, FP, FN, ALL)
    # for i in range(10):
    # global current_classifier
    # for classifier in AvailablePredictionModels:
    #     current_classifier = classifier.value
    #     print(f"Classifier: {classifier.name}")
    #     k_fold_validation(10, data, resolve_backpack_problem=True)

    # predict_all_reviews_from_new_scrape_one_model()
    # predict_all_old_reviews()
        # print("FINISHED")
    # trained_models, test_samples, test_classes = fit_and_tests_all_models(data, frac=0.8)
    # test_vote_of_trained_models(test_samples, test_classes)
    # update_predictions_of_reviews_from_new_scrape(attribute_mode=AttributesModes.SENTIMENT_CAPS_INTER)
    train_best_from_every_available_models(data, "reviews", frac = 0.8, bare_data=False, resolve_backpack_problem=True)