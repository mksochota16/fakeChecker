from typing import Type, List

from pydantic import BaseModel
from pymongo.database import Database as MongoDB
from pymongo.cursor import Cursor as MongoCursor
from pymongo.collection import Collection as MongoCollection
from pymongo import MongoClient


class BaseDao:
    client: MongoClient
    db: MongoDB
    collection: MongoCollection

    base_model: Type[BaseModel]
    model_in_db: Type[BaseModel]

    def __init__(self, client: MongoClient, db_name: str, collection_name: str, base_model: Type[BaseModel],
                 model_in_db: Type[BaseModel]):
        self.client = client
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.base_model = base_model
        self.model_in_db = model_in_db

    def find_all(self) -> List[BaseModel]:
        result: MongoCursor = self.collection.find({})
        return [self.model_in_db(**doc) for doc in list(result)]
