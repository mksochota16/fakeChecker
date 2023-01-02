from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.dao.dao_base import DAOBase
from app.config import MONGO_CLIENT, MONGODB_NEW_DB_NAME
from app.models.review import ReviewPartial, ReviewPartialInDB


class DAOReviewsPartial(DAOBase):
    def __init__(self):
        super().__init__(MONGO_CLIENT,
                         MONGODB_NEW_DB_NAME,
                         'reviews_partial',
                         ReviewPartial,
                         ReviewPartialInDB)

    def insert_one(self, review: ReviewPartial) -> ObjectId:
        self.collection.create_index([('review_id', 1)], unique=True)
        try:
            return super().insert_one(review)
        except DuplicateKeyError:
            super().replace_one('review_id', review.review_id, review)
            review_in_db: ReviewPartialInDB = super().find_one_by_query({'review_id': review.review_id})
            return ObjectId(review_in_db.id)


    def find_reviews_of_account(self, account_id: str) -> list[ReviewPartialInDB]:
        return super().find_many_by_query({'reviewer_id': account_id})

