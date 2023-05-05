from dao.dao_base import DAOBase
from config import MONGO_CLIENT, MONGODB_GMR_PL_FULL_ANONYMOUS
from models.review import ReviewInAnonymisedGMR_PL, ReviewInAnonymisedGMR_PLInDB


class DAOReviewsGMR_PL_Ano(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_GMR_PL_FULL_ANONYMOUS,
                         'reviews',
                         ReviewInAnonymisedGMR_PL,
                         ReviewInAnonymisedGMR_PLInDB)

