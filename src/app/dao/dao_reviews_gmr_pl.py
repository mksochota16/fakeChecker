from dao.dao_base import DAOBase
from config import MONGO_CLIENT, MONGODB_GMR_PL_FULL
from models.review import ReviewInGMR_PL, ReviewInGMR_PLInDB


class DAOReviewsGMR_PL(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_GMR_PL_FULL,
                         'reviews',
                         ReviewInGMR_PL,
                         ReviewInGMR_PLInDB)

