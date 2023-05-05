from dao.dao_base import DAOBase
from config import MONGO_CLIENT, MONGODB_GMR_PL_FULL
from models.account import AccountInGMR_PLInDB, AccountInGMR_PL


class DAOAccountsGMR_PL(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_GMR_PL_FULL,
                         'accounts',
                         AccountInGMR_PL,
                         AccountInGMR_PLInDB)

