from app.dao.baseDao import BaseDao
from app.config import MONGO_CLIENT, MONGODB_OLD_DB_NAME
from app.models.review import ReviewOldBase, ReviewOldInDB


class ReviewsOldDao(BaseDao):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_OLD_DB_NAME,
                         'reviews',
                         ReviewOldBase,
                         ReviewOldInDB)

