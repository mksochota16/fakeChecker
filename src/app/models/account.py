from typing import Optional

from pydantic import BaseModel

from app.models.base_mongo_model import MongoDBModel, MongoObjectId


class AccountBase(BaseModel):
    name: str
    reviewer_id: str
    local_guide_level: Optional[int]
    number_of_reviews: Optional[int]
    is_private: Optional[bool]
    reviewer_url: str
    is_deleted: Optional[bool] = False

class AccountOldBase(AccountBase):
    fake_service: str
    is_checked: Optional[bool]
    fake_service_old: Optional[str]
    is_real_vote_result: Optional[int]
    is_probably_banned: Optional[bool]


class AccountOld(AccountOldBase):
    pass


class AccountOldInDB(AccountOldBase, MongoDBModel):
    pass

    def to_dict(self):
        return {
            "_id": self.id,
            "name": self.name,
            "reviewer_id": self.reviewer_id,
            "local_guide_level": self.local_guide_level,
            "number_of_reviews": self.number_of_reviews,
            "is_private": self.is_private,
            "reviewer_url": self.reviewer_url,
            "fake_service": self.fake_service,
            "is_deleted": self.is_deleted
        }


class AccountNewBase(AccountBase):
    name: Optional[str]
    is_real: Optional[bool]
    #is_deleted: Optional[bool] = False
    new_scrape: Optional[bool] = False

class AccountNew(AccountNewBase):
    pass


class AccountNewInDB(AccountNewBase, MongoDBModel):
    pass

    def to_old_model(self, fake_service: str) -> AccountOldInDB:
        return AccountOldInDB(
            name = self.name,
            reviewer_id = self.reviewer_id,
            local_guide_level = self.local_guide_level,
            number_of_reviews = self.number_of_reviews,
            is_private = self.is_private,
            reviewer_url = self.reviewer_url,
            fake_service = fake_service,
            is_deleted = self.is_deleted,
            _id = self.id
        )


class AccountInGMR_PLDB(MongoDBModel):
    name: str
    reviewer_id: str
    local_guide_level: Optional[int]
    number_of_reviews: Optional[int]
    is_private: Optional[bool]
    reviewer_url: str
    fake_service: str
    is_deleted: Optional[bool] = False

    def to_dict(self):
        return {
            "_id": self.id,
            "name": self.name,
            "reviewer_id": self.reviewer_id,
            "local_guide_level": self.local_guide_level,
            "number_of_reviews": self.number_of_reviews,
            "is_private": self.is_private,
            "reviewer_url": self.reviewer_url,
            "fake_service": self.fake_service,
            "is_deleted": self.is_deleted
        }

    @classmethod
    def from_old_model(cls, old_model: AccountOldInDB):
        return cls(
            name = old_model.name,
            reviewer_id = old_model.reviewer_id,
            local_guide_level = old_model.local_guide_level,
            number_of_reviews = old_model.number_of_reviews,
            is_private = old_model.is_private,
            reviewer_url = old_model.reviewer_url,
            fake_service = old_model.fake_service,
            is_deleted = old_model.is_deleted,
            _id = old_model.id
        )
class AccountInAnonymisedGMR_PLDB(MongoDBModel):
    name_score: int
    local_guide_level: Optional[int]
    number_of_reviews: Optional[int]
    is_real: bool
    is_private: Optional[bool]
    is_deleted: Optional[bool] = False

    def to_dict(self):
        return {
            "_id": self.id,
            "name_score": self.name_score,
            "local_guide_level": self.local_guide_level,
            "number_of_reviews": self.number_of_reviews,
            "is_real": self.is_real,
            "is_private": self.is_private,
            "is_deleted": self.is_deleted
        }

