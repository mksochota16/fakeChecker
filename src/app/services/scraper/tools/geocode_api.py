from enum import Enum

from requests import (Response, Request, PreparedRequest, Session)

from app.config import POSITIONSTACK_API_KEY, GEOAPIFY_API_KEY
from app.models.http_methods import MethodsEnum
from app.services.scraper.models.position import Position as PositionOld
from app.models.position import Position as PositionNew

class AvailableGeocodeAPI(str, Enum):
    POSITIONSTACK = "positionstack"
    GEOAPIFY = "geoapify"

def forward_geocode(address: str, limit: int = 1, new_model = False, which_api: AvailableGeocodeAPI = AvailableGeocodeAPI.GEOAPIFY) -> PositionOld | PositionNew:
    if len(address.split(", ")) < 3:
        address = address + ", Polska"
    if which_api == AvailableGeocodeAPI.POSITIONSTACK:
        return forward_geocode_positionstack(address, limit, new_model)
    elif which_api == AvailableGeocodeAPI.GEOAPIFY:
        return forward_geocode_geoapify(address, limit, new_model)



def forward_geocode_positionstack(address: str, limit: int = 1, new_model = False, nested_level = 0) -> PositionOld | PositionNew:
    request: Request = Request(
        MethodsEnum.GET.value,
        "http://api.positionstack.com/v1/forward",
        params={"access_key": POSITIONSTACK_API_KEY,
                "query": address.encode("utf-8"),
                "limit": limit})
    prep_request: PreparedRequest = request.prepare()
    session: Session = Session()
    response: Response = session.send(prep_request)
    response.raise_for_status()
    result = response.json()
    if len(result["data"]) == 0 or not result["data"][0]:
        if nested_level == 0:
            if len(address.split(", ")[0].split(" ")) > 2:
                new_address = " ".join(address.split(" ")[1:])
                return forward_geocode_positionstack(new_address, limit, new_model, nested_level + 1)
            else:
                return forward_geocode_positionstack(address, limit, new_model, nested_level + 1)
        elif nested_level == 1:
            new_address = " ".join(address.split(", ")[1:])
            return forward_geocode_positionstack(new_address, limit, new_model, nested_level + 1)
        else:
            return PositionNew(lat=0.0, lon=0.0)
    result = result["data"][0]
    if new_model:
        return PositionNew(lat=result["latitude"], lon=result["longitude"])
    else:
        return PositionOld(result["latitude"], result["longitude"])


def forward_geocode_geoapify(address: str, limit: int = 1, new_model = False) -> PositionOld | PositionNew:
    request: Request = Request(
        MethodsEnum.GET.value,
        "https://api.geoapify.com/v1/geocode/search",
        params={"apiKey": GEOAPIFY_API_KEY,
                "text": address.encode("utf-8"),
                "limit": limit})
    prep_request: PreparedRequest = request.prepare()
    session: Session = Session()
    response: Response = session.send(prep_request)
    response.raise_for_status()
    result = response.json()
    result = result["features"][0]['geometry']['coordinates']
    if result:
        if new_model:
            return PositionNew(lat=result[1], lon=result[0])
        else:
            return PositionOld(result[1], result[0])
    else:
        if new_model:
            return PositionNew(lat=0.0, lon=0.0)
        else:
            return PositionOld(0.0, 0.0)