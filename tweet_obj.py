class tweet_obj:
    def __init__(self, tweet_id, fire_ID, text, date, dtime, author_id, author_name, author_username, entities, retweet_count, sentiment=None, magnitude=None):
        self.full_text = text
        self.date = date
        self.dtime = dtime
        self.tweet_id = tweet_id
        self.author_id = author_id
        self.author_name = author_name
        self.author_username = author_username
        self.fire_ID = fire_ID
        self.entities = entities
        self.retweet_count = retweet_count
        self.sentiment = sentiment
        self.magnitude = magnitude

    def __str__(self):
        return 'TWEET: \n Text: {} \n Date: {} \n usename: {} \n RT: {} \n Sentiment {} Magnitude: {}'\
            .format(self.full_text, self.date, self.author_username, self.retweet_count, self.sentiment, self.magnitude)