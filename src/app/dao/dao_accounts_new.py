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
