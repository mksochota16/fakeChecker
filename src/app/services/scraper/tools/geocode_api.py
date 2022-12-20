from requests import (Response, Request, PreparedRequest, Session)

from app.config import POSITIONSTACK_API_KEY
from app.models.http_methods import MethodsEnum
from app.services.scraper.models.position import Position as PositionOld
from app.models.position import Position as PositionNew


def forward_geocode(address: str, limit: int = 1, new_model = False) -> PositionOld | PositionNew:
    request: Request = Request(
        MethodsEnum.GET.value,
        "http://api.positionstack.com/v1/forward",
        params={"access_key": POSITIONSTACK_API_KEY,
                "query": address,
                "limit": limit})
    prep_request: PreparedRequest = request.prepare()
    session: Session = Session()
    response: Response = session.send(prep_request)
    response.raise_for_status()
    result = response.json()
    result = result["data"][0]
    if new_model:
        return PositionNew(lat=result["latitude"], lon=result["longitude"])
    else:
        return PositionOld(result["latitude"], result["longitude"])
