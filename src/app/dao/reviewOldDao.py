from app.dao.baseDao import BaseDao


class ReviewOldDao(BaseDao):
    def __init__(self, client, db):
        super().__init__(client, db, 'reviews_old')