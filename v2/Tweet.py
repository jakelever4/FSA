class Tweet:
    def __init__(self, full_text, date, location, entities, sentiment):
        self.full_text = full_text
        self.date = date
        self.location = location
        self.entities = entities
        self.sentiment = sentiment

    def __str__(self):
        return 'TWEET: \n Text: {} \n Date: {} \n Location: {} \n Sentiment: {}'\
            .format(self.full_text, self.date, self.location, self.sentiment) #[print(entity) for entity in self.entities],