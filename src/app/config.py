import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
POSITIONSTACK_API_KEY = os.getenv("POSITIONSTACK_GEOCODE_API_KEY")


MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_PORT = int(os.getenv("MONGODB_PORT"))
MONGODB_OLD_DB_NAME= os.getenv("MONGODB_OLD_DB_NAME")
MONGODB_NEW_DB_NAME= os.getenv("MONGODB_NEW_DB_NAME")

MONGO_CLIENT = MongoClient(MONGODB_URI, MONGODB_PORT)