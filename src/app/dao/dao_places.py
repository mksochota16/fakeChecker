import pymongo
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from dao.dao_base import DAOBase
from config import MONGO_CLIENT, MONGODB_NEW_DB_NAME
from models.place import Place, PlaceInDB


class DAOPlaces(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_NEW_DB_NAME,
                         'places',
                         Place,
                         PlaceInDB)

    def insert_one(self, place: Place) -> ObjectId:
        self.collection.create_index([('identifier', 1)], unique=True)
        try:
            return super().insert_one(place)
        except DuplicateKeyError:
            super().replace_one('identifier', place.identifier, place)
            place_in_db: PlaceInDB = super().find_one_by_query({'identifier': place.identifier})
            return place_in_db.id
