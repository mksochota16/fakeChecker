from gensim.models import KeyedVectors
from numpy import unique
from numpy import where
from sklearn.cluster import *
import numpy as np
from sklearn.mixture import GaussianMixture
import fasttext

from app.models.types_cluster import CLUSTER_TYPES
from app.services.scraper.tools import io_files_handler


def make_clusters_of_data(data, info_dict, number_of_clusters, algorithm=None):
    X = data
    # define the model
    if algorithm == 'KMeans':
        model = KMeans(n_clusters=number_of_clusters)
    elif algorithm == 'GaussianMixture':
        model = GaussianMixture(n_components=number_of_clusters)
    elif algorithm == 'SpectralClustering':  #
        model = SpectralClustering(n_clusters=number_of_clusters)
    elif algorithm == 'OPTICS':  #
        model = OPTICS(eps=np.Inf, min_samples=10)
    elif algorithm == 'MeanShift':  #
        model = MeanShift()
    elif algorithm == 'Birch':
        model = Birch(threshold=0.1, n_clusters=number_of_clusters)
    elif algorithm == 'AgglomerativeClustering':  #
        model = AgglomerativeClustering(n_clusters=number_of_clusters)
    elif algorithm == 'AffinityPropagation':
        model = AffinityPropagation(damping=0.6)
    else:
        model = KMeans(n_clusters=number_of_clusters)

    if algorithm in ['KMeans', 'GaussianMixture', 'AffinityPropagation', 'Birch'] or algorithm is None:
        model.fit(X)
        yhat = model.predict(X)
    else:
        yhat = model.fit_predict(X)
    # GaussianMixture(n_components=number_of_clusters) - good
    # SpectralClustering(n_clusters=number_of_clusters) - no good results
    # OPTICS(eps=np.Inf, min_samples=10) - no good results
    # MeanShift() - no good results
    # Birch(threshold=0.1, n_clusters=number_of_clusters) - good results
    # AgglomerativeClustering(n_clusters=number_of_clusters) - good results
    # AffinityPropagation(damping=0.6) - no good results
    # KMeans(n_clusters=number_of_clusters) - good results
    # ==== for Affinity, KMeans, Birch, GaussianMixture ===
    # model.fit(X)
    # yhat = model.predict(X)
    # ==== for Agglomerative, MeanShift, OPTICS, SpectralClustering ===
    # yhat = model.fit_predict(X)
    clusters = unique(yhat)
    vector_cluster_dict = {}
    cluster_number = 0
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        x = X[row_ix, :]
        x = x[0, :, :]
        for vector in x:
            key = find_key_in_dict(info_dict, vector)
            vector_cluster_dict[key] = cluster_number
        cluster_number += 1
    return vector_cluster_dict

    # for vector in data:
    #     cluster_distance = []
    #     for cluster in cluster_centers:
    #         dis = distance.euclidean(vector, cluster)
    #         cluster_distance.append(float(dis))
    #     key = find_key_in_dict(info_dict, vector)
    #     vector_cluster_dict[key] = cluster_distance.index(min(cluster_distance))
    # return [vector_cluster_dict, cluster_centers]
    # create scatter plot for samples from each cluster


def find_key_in_dict(info_dict, vector):
    for key in info_dict:
        if np.array_equal(info_dict[key][0], vector):
            return key


class Sth2Vec:
    def __init__(self, model_provider='gensim', english_translation_dict=None):
        if english_translation_dict is None:
            english_translation_dict = {}
        self.model_provider = model_provider
        if self.model_provider == 'gensim':
            self.word2vec = KeyedVectors.load(
                "D:/Dev/Word2Vec/word2vec_800_3/word2vec_800_3_polish.bin")  # ("word2vec/word2vec_100_3_polish.bin")
        elif self.model_provider == 'fasttext':
            self.fasttext_model = fasttext.load_model("D:/Dev/Word2Vec/cc.pl.300.bin/cc.pl.300.bin")
        self.english_translation_dict = english_translation_dict
        print("Initialized Sth2Vec Module")

    def get_similarity_between_words(self, word1, word2):
        if self.model_provider == 'gensim':
            word1 = word1.lower()
            word2 = word2.lower()
            try:
                return self.word2vec.similarity(word1, word2)
            except:
                return -1
        elif self.model_provider == 'fasttext':
            raise Exception('Not supported action')

    def get_similarity_between_sentence_and_word(self, sentence, word2):
        if self.model_provider == 'gensim':
            sentence = sentence.lower().split(' ')
            word2 = word2.lower()
            scores = []
            for word in sentence:
                try:
                    scores.append(self.word2vec.similarity(word, word2))
                except:
                    scores.append(-1)
            return max(scores)
        elif self.model_provider == 'fasttext':
            raise Exception('Not supported action')

    def return_the_most_similar_word(self, sentence, list_of_words):
        if self.model_provider == 'gensim':
            scores = []
            for keyword in list_of_words:
                scores.append(self.get_similarity_between_sentence_and_word(sentence, keyword))
            index = scores.index(max(scores))
            return list_of_words[index]
        elif self.model_provider == 'fasttext':
            raise Exception('Not supported action')

    def get_vector_of_word(self, word):
        if self.model_provider == 'gensim':
            return self.word2vec.wv[word]
        elif self.model_provider == 'fasttext':
            return self.fasttext_model.get_word_vector(word)

    def get_vector_of_sentence(self, sentence):
        sentence = sentence.lower().replace(' z ', ' ').replace(' ze ', ' ')
        if self.model_provider == 'gensim':
            values = []
            try:
                sentence = sentence.split(' ')
            except:
                return np.zeros(800)
            for word in sentence:
                try:
                    values.append(self.word2vec.wv[word])  # , len(word)])
                except:
                    values.append(np.zeros(800))
            # vector_weighted_sum = sum(x[0] * x[1] for x in values)
            # weight_sum = sum(x[1] for x in values)
            vector_mean = sum(values) / len(values)
            # if weight_sum == 0:
            #     return np.zeros(100)
            # result = vector_weighted_sum / weight_sum
            result = vector_mean
            return result
        elif self.model_provider == 'fasttext':
            try:
                return self.fasttext_model.get_sentence_vector(sentence)
            except:
                return np.zeros(300)

    def find_closest_cluster(self, type_of_object, cluster_dict, mode):
        if mode == 'centers':
            centers_dict = {}
            for key in cluster_dict:
                if self.model_provider == 'gensim':
                    vector_sum = np.zeros(800)
                elif self.model_provider == 'fasttext':
                    vector_sum = np.zeros(300)
                else:
                    raise Exception('Not supported action')

                for object_type in cluster_dict[key]:
                    vector_sum += self.get_vector_of_sentence(object_type)
                centers_dict[key] = vector_sum / len(cluster_dict[key])
            dist_dict = {}
            for key in centers_dict:
                dist_dict[key] = np.linalg.norm(
                    self.get_vector_of_sentence(type_of_object) - centers_dict[key])
            return min(dist_dict, key=lambda k: dist_dict[k])
        elif mode == 'dist_mean':
            dist_dict = {}
            for key in cluster_dict:
                dist_sum = 0
                new_vector = self.get_vector_of_sentence(type_of_object)
                for object_type in cluster_dict[key]:
                    known_vector = self.get_vector_of_sentence(object_type)
                    dist_sum += np.linalg.norm(new_vector - known_vector)
                dist_dict[key] = dist_sum / len(cluster_dict[key])
            return min(dist_dict, key=lambda k: dist_dict[k])
        elif mode == 'min_dist':
            dist_dict = {}
            for key in cluster_dict:
                dist_list = []
                new_vector = self.get_vector_of_sentence(type_of_object)
                for object_type in cluster_dict[key]:
                    known_vector = self.get_vector_of_sentence(object_type)
                    dist_list.append(np.linalg.norm(new_vector - known_vector))
                dist_dict[key] = min(dist_list)
            return min(dist_dict, key=lambda k: dist_dict[k])

    def classify_type_of_object(self, type_of_object, cluster_dict=None, mode='min_dist') -> CLUSTER_TYPES:
        if cluster_dict is None:
            cluster_dict = io_files_handler.get_clusters()
        # first check if we already know this type
        inne = False
        for key in cluster_dict:
            if type_of_object in cluster_dict[key]:
                return CLUSTER_TYPES(self.english_translation_dict[key])
        # find closest cluster

        polish_result: str = self.find_closest_cluster(type_of_object, cluster_dict, mode)
        return CLUSTER_TYPES(self.english_translation_dict[polish_result])
