from typing import Optional

from pydantic import BaseModel

from app.models.base_mongo_model import MongoDBModel

class AccountBase(BaseModel):
    name: str
    reviewer_id: str
    local_guide_level: Optional[int]
    number_of_reviews: Optional[int]
    is_private: Optional[bool]
    reviewer_url: str

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


class AccountNewBase(AccountBase):
    is_real: Optional[bool]
    pass

class AccountNew(AccountNewBase):
    pass


class AccountNewInDB(AccountNewBase, MongoDBModel):
    pass



