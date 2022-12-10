from enum import Enum

# Copied from Orchestra
class MethodsEnum(str, Enum):
  GET = 'GET'
  POST = 'POST'
  DELETE = 'DELETE'
  PATCH = 'PATCH'
  PUT = 'PUT'
