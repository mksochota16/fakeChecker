import pydantic

from pydantic import BaseModel, Field
from bson.objectid import ObjectId
from bson.errors import InvalidId


class MongoObjectId(ObjectId):

  @classmethod
  def __get_validators__(cls):
    yield cls.validate

  @classmethod
  def validate(cls, v):
    try:
      ObjectId(str(v))
    except InvalidId:
      raise ValueError("Expected %s but got %s" %
                       (ObjectId.__name__, type(v).__name__))
    return ObjectId(str(v))

  @classmethod
  def __modify_schema__(cls, field_schema):
    field_schema.update(type="string")


pydantic.json.ENCODERS_BY_TYPE[ObjectId] = str
pydantic.json.ENCODERS_BY_TYPE[MongoObjectId] = str


class MongoDBModel(BaseModel):
  id: MongoObjectId = Field(..., alias='_id')
