class tweet_obj:
    def __init__(self, tweet_id, fire_ID, text, date, author_id, entities):
        self.full_text = text
        self.date = date
        self.tweet_id = tweet_id
        self.author_id = author_id
        self.fire_ID = fire_ID
        self.entities = entities

    def __str__(self):
        return 'TWEET: \n Text: {} \n Date: {} \n ID: {}'.format(self.full_text, self.date, self.id)