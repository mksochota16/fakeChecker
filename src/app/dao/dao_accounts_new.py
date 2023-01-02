from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.dao.dao_base import DAOBase
from app.config import MONGO_CLIENT, MONGODB_NEW_DB_NAME
from app.models.account import AccountNew, AccountNewInDB


class DAOAccountsNew(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_NEW_DB_NAME,
                         'accounts',
                         AccountNew,
                         AccountNewInDB)


    def insert_one(self, account: AccountNew) -> ObjectId:
        self.collection.create_index([('reviewer_id', 1)], unique=True)
        try:
            return super().insert_one(account)
        except DuplicateKeyError:
            super().replace_one('reviewer_id', account.reviewer_id, account)
            account_in_db: AccountNewInDB = super().find_one_by_query({'reviewer_id': account.reviewer_id})
            return ObjectId(account_in_db.id)