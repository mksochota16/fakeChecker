from requests import (Response, Request, PreparedRequest, Session)

from app.config import POSITIONSTACK_API_KEY
from app.models.http_methods import MethodsEnum
from app.services.scraper.models.position import Position


def forward_geocode(address: str, limit: int = 1) -> Position:
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
    return Position(result["latitude"], result["longitude"])
