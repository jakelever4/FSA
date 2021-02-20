class entity:
    def __init__(self, tweet_ID, hashtags, urls):
        self.tweet_ID = tweet_ID
        self.hashtags = hashtags
        self.urls = urls


def save_entities(tweet_ID, entity_dic):
    try:
        hashtags = entity_dic['hashtags']
        urls = entity_dic['urls']
        return entity(tweet_ID, hashtags, urls)
    except KeyError:
        return None