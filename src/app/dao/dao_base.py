from typing import Type, List

from pydantic import BaseModel
from pymongo.database import Database as MongoDB
from pymongo.cursor import Cursor as MongoCursor
from pymongo.collection import Collection as MongoCollection
from pymongo import MongoClient
from bson.objectid import ObjectId


class DAOBase:
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

    def find_by_id(self, id: ObjectId) -> BaseModel:
        result: dict = self.collection.find_one({"_id": id})
        return self.model_in_db(**result)

    def find_one_by_query(self, query: dict) -> BaseModel:
        result: dict = self.collection.find_one(query)
        return self.model_in_db(**result)

    def find_one_by_query_return_raw(self, query: dict) -> dict:
        result: dict = self.collection.find_one(query)
        return result

    def find_many_by_query(self, query: dict) -> List[BaseModel]:
        result: MongoCursor = self.collection.find(query)
        return [self.model_in_db(**doc) for doc in list(result)]

    def insert_one(self, obj: BaseModel| dict) -> ObjectId:
        if isinstance(obj, BaseModel):
            obj = obj.dict()
        result: ObjectId = self.collection.insert_one(obj).inserted_id
        return result

    def insert_many(self, obj_list: List[BaseModel]) -> List[ObjectId]:
        dict_list: List[dict] = [obj.dict() for obj in obj_list]
        result: List[ObjectId] = self.collection.insert_many(dict_list).inserted_ids
        return result

    def replace_one(self, field_name: str, value: any, obj: BaseModel|dict) -> bool:
        if isinstance(obj, BaseModel):
            obj = obj.dict()
        return self.collection.replace_one({field_name:value}, obj).acknowledged

    def update_one(self, query: dict, values: dict) -> int:
        return self.collection.update_one(query, values).matched_count