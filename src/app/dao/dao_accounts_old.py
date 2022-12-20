from app.dao.dao_base import DAOBase
from app.config import MONGO_CLIENT, MONGODB_OLD_DB_NAME
from app.models.account import AccountOld, AccountOldInDB


class DAOAccountsOld(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_OLD_DB_NAME,
                         'accounts',
                         AccountOld,
                         AccountOldInDB)

