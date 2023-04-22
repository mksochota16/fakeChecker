from services.scraper.models.position import Position


class Review:
    def __init__(self, review_id=None, place_name=None, rating=None, content=None, reviewer_url=None, place_url=None, localization=None,
                 photos_urls=None, type_of_object=None, date=None, response_content=None, dictionary=None):
        if dictionary is None:
            self.review_id = review_id #
            self.place_name = place_name  #
            self.rating = rating  #
            self.content = content  #
            self.reviewer_url = reviewer_url  #
            self.reviewer_id = reviewer_url.split('/')[5]  #
            self.place_url = place_url #
            self.localization = localization #
            self.photos_urls = photos_urls #
            self.type_of_object = type_of_object #
            self.response_content = response_content #
            self.date = date #
        else:
            self.review_id = dictionary["review_id"]
            self.place_name = dictionary["place_name"]
            self.rating = dictionary["rating"]
            self.content = dictionary["content"]
            self.reviewer_url = dictionary["reviewer_url"]
            self.reviewer_id = dictionary["reviewer_id"]
            self.place_url = dictionary["place_url"]
            pos = dictionary["localization"]
            self.localization = Position(pos["lat"], pos["lon"])
            self.photos_urls = dictionary["photos_urls"]
            self.type_of_object = dictionary["type_of_object"]
            self.response_content = dictionary["response_content"]
            self.date = dictionary["date"]

    def __str__(self):
        return f"{self.place_name}; {self.rating}; {self.content}; {self.reviewer_url}; {self.place_url}; {self.localization}; \n"

    def to_dict(self):
        return {
            "review_id":self.review_id,
            "place_name": self.place_name,
            "rating": self.rating,
            "content": self.content,
            "reviewer_url": self.reviewer_url,
            "reviewer_id": self.reviewer_id,
            "place_url": self.place_url,
            "localization": self.localization.to_dict(),
            "photos_urls": self.photos_urls,
            "type_of_object":self.type_of_object,
            "response_content":self.response_content,
            "date":self.date
        }
