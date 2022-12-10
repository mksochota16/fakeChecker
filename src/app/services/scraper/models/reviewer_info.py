class ReviewerInfo:
    def __init__(self, name=None, local_guide_level=None, number_of_reviews=None, reviewer_url=None, fake_service=None, reviewer_id=None, is_checked=False, is_private=False, dictionary=None):
        if dictionary is None:
            self.name = name
            if reviewer_id is None:
                self.reviewer_id = reviewer_url.split("/")[5]
            else:
                self.reviewer_id = reviewer_id
            self.local_guide_level = local_guide_level
            self.number_of_reviews = number_of_reviews
            self.reviewer_url = reviewer_url
            self.fake_service = fake_service
            self.is_checked = is_checked
            self.is_private = is_private
        else:
            self.name = dictionary["name"]
            self.reviewer_id = dictionary["reviewer_id"]
            self.local_guide_level = dictionary["local_guide_level"]
            self.number_of_reviews = dictionary["number_of_reviews"]
            self.reviewer_url = dictionary["reviewer_url"]
            self.fake_service = dictionary["fake_service"]
            self.is_checked = dictionary["is_checked"]
            self.is_private = dictionary["is_private"]

    def __str__(self):
        return str(self.name + " | level: " + str(self.local_guide_level) + " | number of reviews: " + str(
            self.number_of_reviews))

    def to_dict(self):
        return {
            "name": self.name,
            "reviewer_id": self.reviewer_id,
            "local_guide_level": self.local_guide_level,
            "number_of_reviews": self.number_of_reviews,
            "reviewer_url": self.reviewer_url,
            "fake_service": self.fake_service,
            "is_checked": self.is_checked,
            "is_private": self.is_private
        }
