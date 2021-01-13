class tweet_obj:
    def __init__(self, full_text, date, id, username):
        self.full_text = full_text
        self.date = date
        self.id = id
        self.username = username


    def __str__(self):
        return 'TWEET: \n Text: {} \n Date: {} \n ID: {} \n username: {}'\
            .format(self.full_text, self.date, self.id, self.username)