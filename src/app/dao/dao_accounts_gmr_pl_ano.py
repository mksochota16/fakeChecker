from dao.dao_base import DAOBase
from config import MONGO_CLIENT, MONGODB_GMR_PL_FULL_ANONYMOUS
from models.account import AccountInAnonymisedGMR_PL, AccountInAnonymisedGMR_PLInDB


class DAOAccountsGMR_PL_Ano(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_GMR_PL_FULL_ANONYMOUS,
                         'accounts',
                         AccountInAnonymisedGMR_PL,
                         AccountInAnonymisedGMR_PLInDB)

