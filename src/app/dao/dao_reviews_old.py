from app.dao.dao_base import DAOBase
from app.config import MONGO_CLIENT, MONGODB_OLD_DB_NAME
from app.models.review import ReviewOld, ReviewOldInDB


class DAOReviewsOld(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_OLD_DB_NAME,
                         'reviews',
                         ReviewOld,
                         ReviewOldInDB)

    def find_reviews_of_account(self, reviewer_id: str) -> list[ReviewOldInDB]:
        return super().find_many_by_query({'reviewer_id': reviewer_id})
