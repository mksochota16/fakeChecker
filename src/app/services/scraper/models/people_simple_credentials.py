class PeopleSimpleCredentials:
    def __init__(self, name, reviewer_url):
        self.name = name
        self.reviewer_url = reviewer_url
        self.reviewer_id = reviewer_url.split("/")[5]

    def __str__(self):
        return f"{self.name} {self.reviewer_url}"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.reviewer_id == other.reviewer_id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)
