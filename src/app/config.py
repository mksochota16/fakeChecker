import os
from dotenv import load_dotenv
from pymongo import MongoClient

from app.services.analysis.nlp_analysis import NLPanalysis
from app.services.analysis.sth2vec import Sth2Vec

import warnings

warnings.filterwarnings("ignore")

load_dotenv()
POSITIONSTACK_API_KEY = os.getenv("POSITIONSTACK_GEOCODE_API_KEY")
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_GEOCODE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_GEOCODE_API_KEY")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_PORT = int(os.getenv("MONGODB_PORT"))
MONGODB_OLD_DB_NAME= os.getenv("MONGODB_OLD_DB_NAME")
MONGODB_NEW_DB_NAME= os.getenv("MONGODB_NEW_DB_NAME")

MONGO_CLIENT = MongoClient(MONGODB_URI, MONGODB_PORT)

ENGLISH_TRANSLATION_CLUSTER_DICT ={
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

STH2VEC: Sth2Vec = Sth2Vec(english_translation_dict=ENGLISH_TRANSLATION_CLUSTER_DICT)
NLP: NLPanalysis = NLPanalysis()

if __name__ == '__main__':
    print(STH2VEC.get_vector_of_sentence('Moja piękna dziewczyna jest super'))