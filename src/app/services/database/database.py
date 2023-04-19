from pymongo import MongoClient

from services.scraper.models.position import Position


class Database:
    def __init__(self, original_database: bool = True):
        self.client = MongoClient('localhost', 27017)
        if original_database:
            self.db = self.client.scraperDatabase
        else:
            self.db = self.client.fakeChecker
            self.reviews_partial = self.db.reviews_partial
        self.accounts_collection = self.db.accounts
        self.reviews_collection = self.db.reviews
        self.old_reviews_collection = self.db.reviews_old

    def save_account(self, account):
        return self.accounts_collection.insert_one(account.to_dict())

    def save_review(self, review):
        return self.reviews_collection.insert_one(review.to_dict())

    def is_account_already_in(self, account_id):
        check = self.accounts_collection.find_one({"reviewer_id": account_id})
        return check is not None

    def is_review_already_in(self, review_id):
        check = self.reviews_collection.find_one({"review_id": review_id})
        return check is not None

    def update_name_and_type_of_review_by_id(self, review_id, name, type_of_object):
        self.reviews_collection.find_one_and_update({"review_id": review_id},
                                                    {"$set": {"place_name": name, "type_of_object": type_of_object}})

    def update_place_url_by_id(self, review_id, place_url):
        self.reviews_collection.find_one_and_update({"review_id": review_id}, {"$set": {"place_url": place_url}})

    def update_full_flag_by_id(self, review_id, full_flag):
        self.reviews_collection.find_one_and_update({"review_id": review_id}, {"$set": {"full_flag": full_flag}})

    def update_weird_flag_by_id(self, review_id, weird_flag):
        self.reviews_collection.find_one_and_update({"review_id": review_id},
                                                    {"$set": {"weird_flag": weird_flag, "full_flag": False}})

    def update_private_flag_by_id(self, reviewer_id, is_private):
        self.reviews_collection.find_one_and_update({"reviewer_id": reviewer_id},
                                                    {"$set": {"is_private": is_private}})

    def update_account_private_flag_by_id(self, reviewer_id, is_private):
        self.accounts_collection.find_one_and_update({"reviewer_id": reviewer_id},
                                                     {"$set": {"is_private": is_private}})

    def get_all_reviews(self):
        return self.reviews_collection.find({})

    def get_all_full_reviews(self):
        return self.reviews_collection.find({"full_flag": True})

    def get_all_not_full_reviews(self):
        return self.reviews_collection.find({"$or": [{"full_flag": False}, {"full_flag": {"$exists": False}}]})

    def get_all_accounts(self):
        return self.accounts_collection.find({})

    def get_account_by_id(self, reviewer_id):
        return self.accounts_collection.find_one({"reviewer_id": reviewer_id})

    def get_all_public_accounts(self):
        return self.accounts_collection.find({'is_private': False})

    def get_reviews_of_account(self, reviewer_id):
        return self.reviews_collection.find({"reviewer_id": reviewer_id})

    def get_fake_accounts(self):
        return self.accounts_collection.find({"fake_service": {'$ne': "real"}})

    def get_fake_accounts_id_list(self):
        fake_accounts = self.get_fake_accounts()
        ids = []
        for account in fake_accounts:
            ids.append(account['reviewer_id'])
        return ids

    def get_real_accounts(self):
        return self.accounts_collection.find({"fake_service": "real"})

    def get_fake_reviews(self):
        fake_accounts = self.get_fake_accounts()
        fake_reviews = []
        for account in fake_accounts:
            reviews = self.get_reviews_of_account(account['reviewer_id'])
            fake_reviews.append(reviews)
        return fake_reviews

    def get_real_reviews(self):
        real_accounts = self.get_real_accounts()
        real_reviews = []
        for account in real_accounts:
            reviews = self.get_reviews_of_account(account['reviewer_id'])
            real_reviews.append(reviews)
        return real_reviews

    def update_place_type_cluster(self, type_of_object, cluster):
        self.reviews_collection.update_many({"type_of_object": type_of_object},
                                            {"$set": {"cluster": cluster}})

    def update_wrong_address_flag_by_id(self, review_id, wrong_address_flag):
        self.reviews_collection.find_one_and_update({"review_id": review_id},
                                                    {"$set": {"wrong_address": wrong_address_flag}})

    def correct_similar_wrong_address_reviews(self, base_review_id, wrong_address_flag):
        base_review = self.get_review_by_id(base_review_id)
        place_url = base_review['place_url']
        place_name = base_review['place_name']
        type_of_object = base_review['type_of_object']
        try:
            cluster = base_review['cluster']
        except:
            cluster = None
        localization = base_review['localization']
        self.reviews_collection.update_many({"localization": localization},
                                            {"$set": {"wrong_address": wrong_address_flag, "place_url": place_url,
                                                      "place_name": place_name, "type_of_object": type_of_object,
                                                      "cluster": cluster}})

    def update_similar_wrong_address_reviews(self, localization, place_data, wrong_address_flag):
        place_url = place_data[1]
        place_name = place_data[0]
        type_of_object = place_data[2]
        cluster = place_data[3]
        new_localization = place_data[4]
        self.reviews_collection.update_many({"localization": localization},
                                            {"$set": {"wrong_address": wrong_address_flag, "place_url": place_url,
                                                      "place_name": place_name, "type_of_object": type_of_object,
                                                      "cluster": cluster, "localization": localization,
                                                      "auto_update": True, "full_flag": True}})

    def update_wrong_url_flag_by_id(self, review_id, wrong_url_flag):
        self.reviews_collection.find_one_and_update({"review_id": review_id},
                                                    {"$set": {"wrong_url_flag": wrong_url_flag}})

    def get_list_of_positions_by_id(self, reviewer_id):
        reviews = self.get_reviews_of_account(reviewer_id)
        pos_list = []
        for review in reviews:
            loc = review['localization']
            lat = loc['lat']
            lon = loc['lon']
            pos_list.append(Position(lat, lon))
        return pos_list

    def get_list_of_ratings_by_id(self, reviewer_id):
        reviews = self.get_reviews_of_account(reviewer_id)
        rating_list = []
        for review in reviews:
            rating_list.append(review['rating'])
        return rating_list

    def get_review_by_id(self, review_id):
        return self.reviews_collection.find_one({"review_id": review_id})

    def get_list_of_reviews_by_review_id(self, review_ids):
        list_of_reviews = []
        for review_id in review_ids:
            list_of_reviews.append(self.reviews_collection.find_one({"review_id": review_id}))
        return list_of_reviews

    def get_keys_in_clusters_dict(self):
        number_of_clusters = self.get_max_cluster_number()
        cluster_dict = {}
        for number in range(number_of_clusters):
            cluster_dict[number] = self.reviews_collection.find({"cluster": number})
        cluster_dict2 = {}
        for key in cluster_dict:
            temp_list = []
            for review in cluster_dict.get(key):
                temp_list.append(review)
            cluster_dict2[key] = temp_list
        return cluster_dict2

    def get_max_cluster_number(self):
        max_number = -2
        for review in self.get_all_full_reviews():
            try:
                cluster_number = review['cluster']
            except:
                continue
            if cluster_number > max_number:
                max_number = cluster_number
        return max_number + 1

    def find_by_cluster(self, cluster):
        return self.reviews_collection.find({"cluster": cluster})

    def update_cluster_names(self, cluster_dict):
        for key in cluster_dict:
            type_of_object_list = cluster_dict[key]
            for type_of_object in type_of_object_list:
                self.reviews_collection.update_many({"type_of_object": type_of_object},
                                                    {"$set": {"cluster": key}})

    def update_cluster_names_to_english(self):
        english_dict ={
            "BUDOWNICTWO": "Constructions",
            "DOSTAWCY I PRODUCENCI": "Suppliers",
            "EDUKACJA": "Education",
            "GASTRONOMIA": "Gastronomy",
            "MIEJSCA I INSTYTUCJE PUBLICZNE i PAŃSTWOWE": "Institutions",
            "MOTORYZACJA": "Automotive",
            "NOCLEGI": "Lodging",
            "PRAWO I UBEZPIECZENIA": "Legal",
            "PRZYRODA": "Nature",
            "SERIWSY I NAPRAWY": "Repairs",
            "SKLEPY": "Shops",
            "TRANSPORT": "Transport",
            "TURYSTYKA, ROZRYWKA I SPORT": "Leisure",
            "USŁUGI": "Other services",
            "USŁUGI I PLACÓWKI MEDYCZNE": "Medical",
            "ZABYTKI I BUDYNKI SAKRALNE": "Sacral and Monuments",
            "INNE": "Other"
        }
        for key in english_dict:
            self.reviews_collection.update_many({"cluster": key},{"$set": {"cluster": english_dict[key]}})

    def update_review_cluster_by_id(self, review_id, cluster):
        self.reviews_collection.find_one_and_update({"review_id": review_id},
                                                    {"$set": {"cluster": cluster}})

    def get_reviews_without_clusters(self):
        return self.reviews_collection.find({"$or": [{"cluster": None}, {"cluster": {"$exists": False}}]})

    def update_false_real_reviews(self):
        real_reviews = self.get_real_reviews()
        for review in real_reviews:
            for rev in review:
                self.reviews_collection.find_one_and_update({"review_id": rev['review_id']},
                                                            {"$set": {"is_real": True}})
        fake_reviews = self.get_fake_reviews()
        for review in fake_reviews:
            for rev in review:
                self.reviews_collection.find_one_and_update({"review_id": rev['review_id']},
                                                            {"$set": {"is_real": False}})

    def percentage_of_photographed_reviews(self, account_id):
        reviews_count = self.get_reviews_of_account(account_id).count()
        photographed_count = self.reviews_collection.find(
            {"account_id": account_id, "photos_url": {"$ne": "null"}}).count()
        if reviews_count == 0:
            return 0
        return photographed_count / reviews_count

    def percentage_of_responded_reviews(self, account_id):
        reviews_count = self.get_reviews_of_account(account_id).count()
        photographed_count = self.reviews_collection.find(
            {"account_id": account_id, "response_content": {"$ne": "null"}}).count()
        if reviews_count == 0:
            return 0
        return photographed_count / reviews_count

    def get_cluster_names(self):
        cluster_names = list(self.reviews_collection.distinct('cluster'))
        cluster_names = sorted(cluster_names)
        return cluster_names

    def get_reviews_in_clusters(self):
        cluster_names = self.reviews_collection.distinct('cluster')
        reviews_in_clusters = {}
        for cluster_name in cluster_names:
            reviews_in_clusters[cluster_name] = self.find_by_cluster(cluster_name)

        return reviews_in_clusters

    def get_old_review_of_id(self, review_id):
        return self.old_reviews_collection.find_one({"review_id": review_id})

    def get_wrong_address_reviews(self):
        return self.reviews_collection.find({"wrong_address": True})

    def get_wrong_url_reviews(self):
        return self.reviews_collection.find({"wrong_url_flag": True})

    def add_vote_result_to_account(self, reviewer_id, vote_result):
        self.accounts_collection.find_one_and_update({"reviewer_id": reviewer_id},
                                                     {"$set": {"is_real_vote_result": vote_result}})

    def add_probably_banned_value(self):
        accounts = self.get_all_accounts()
        for account in accounts:
            is_banned = (account['is_private']) and (account['number_of_reviews'] is not None)
            self.accounts_collection.find_one_and_update({"reviewer_id": account['reviewer_id']},
                                                         {"$set": {"is_probably_banned": is_banned}})

    def add_vote_results(self):
        accounts = self.get_all_accounts()
        accounts_id = []
        for account in accounts:
            try:
                vote_result = account['is_real_vote_result'] > 0
                algo_result = account['fake_service'] != 'real'
                if vote_result == algo_result:
                    accounts_id.append(account['reviewer_id'])
            except KeyError:
                continue

    def find_oldest_and_newest(self):
        reviews = self.get_all_reviews()
        dates = []
        for review in reviews:
            dates.append(review['date'])

        print(min(dates))
        print(max(dates))


if __name__ == '__main__':
    db = Database(False)
    reviews_partial = db.reviews_partial.find()
    for review in reviews_partial:
        db.reviews_partial.update_one({'review_id':review['review_id']}, {'$set': {'rating': int(review['rating'])}})

