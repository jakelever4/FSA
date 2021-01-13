import tweepy
import twitter
import google_analyse_entities
import google_analyse_sentiment
from datetime import datetime

from v2 import Tweet

query = '#CaliforniaFires'
filters = ['like wildfire', 'LIKE WILDFIRE']
tweet_geocode = None #'36.73,119.78,700mi'
from_date = '2020-11-07'
to_date = '2020-11-09'


def filter_words(full_text):
    if any(word not in full_text for word in filters):
        return True
    else:
        return False


def get_full_text(result):
    try:
        full_text = result['retweeted_status']['full_text']
    except KeyError:
        pass

    try:
        full_text = result['full_text']
    except KeyError:
        full_text = result['text']

    return full_text


def get_datetime(result):
    date_str = result['created_at']
    date_time = datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')
    return date_time


def get_location(result):
    if result['geo'] is not None:
        return result['geo']
    elif result['coordinates'] is not None:
        return result['coordinates']
    elif result['place'] is not None:
        return result['place']
    elif result['user']['location'] is not None:
        return result['user']['location']
    else:
        return None


def get_dates(tweets):
    dates = []
    for tweet in tweets:
        date = tweet.date.date()
        if date not in dates:
            dates.append(date)

    return dates


def get_tweets_for_date(date, tweets):
    tweets_for_day = []
    for tweet in tweets:
        tweet_date = tweet.date.date()
        if tweet_date == date:
            tweets_for_day.append(tweet)

    return tweets_for_day


def get_avg_sentiment_score(tweets_for_date):
    total_sentiment = 0
    num_tweets = len(tweets_for_date)
    for tweet in tweets_for_date:
        if sentiment is not None:
            total_sentiment += 10 * tweet.sentiment.score * tweet.sentiment.magnitude
        else:
            total_sentiment += 0

    avg_sentiment_score = total_sentiment / num_tweets
    return avg_sentiment_score


def convert_date(string):
    new_date = string.replace('-', '') + '0000'
    return new_date


api = twitter.connect_to_twitter()
query_result = tweepy.Cursor(api.search_full_archive, environment_name='dev', query=query, fromDate=convert_date(from_date), maxResults=10).items(10)
search_results = [ status._json for status in query_result]
# search_results = [status._json for status in tweepy.Cursor(api.search, q=query, count=10, tweet_mode='extended').items(100)]

tweets = []
for result in search_results:
    ind = search_results.index(result)
    print('{} of {}'.format(ind, len(search_results)))
    full_text = get_full_text(result)
    if filter_words(full_text):
        date = get_datetime(result)
        location_info = get_location(result)

        try:
            entities = google_analyse_entities.analyze_entities(full_text)
        except:
            entities = None

        try:
            sentiment = google_analyse_sentiment.analyze(full_text)
        except:
            sentiment = None

        tweet = Tweet.Tweet(full_text, date, location_info, entities, sentiment)
        print(tweet)
        tweets.append(tweet)


# sort tweets by ascending time
tweets.sort(key=lambda tweet: tweet.date, reverse=False)
for tweet in tweets:
    print(tweet.date)


dates = get_dates(tweets)
print(dates)


for date in dates:
    tweets_for_date = get_tweets_for_date(date, tweets)
    sentiment = get_avg_sentiment_score(tweets_for_date)
    print(date)
    print(sentiment)





