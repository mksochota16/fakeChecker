class AccountDetails:
    def __init__(self, account_url):
        self.url = account_url
        self.id = account_url.split("/")[5]
        self.position_list = []
        self.ratings_list = []

    def addPosition(self, position):
        self.position_list.append(position)

    def addRating(self, rating):
        self.ratings_list.append(int(rating))

    def __eq__(self, other):
        if isinstance(other, AccountDetails):
            return self.id == other.id
        else:
            return False
