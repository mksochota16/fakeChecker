from pymongo.database import Database as MongoDB
from pymongo.collection import Collection as MongoCollection
from pymongo import MongoClient

class BaseDao:
    client: MongoClient
    db: MongoDB
    collection_name: MongoCollection

    def __init__(self, client: MongoClient, db: MongoDB, collection_name: str):
        self.client = client
        self.db = db
        self.collection_name = collection_name