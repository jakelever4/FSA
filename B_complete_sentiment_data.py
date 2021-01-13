import snscrape.modules.twitter as sntwitter
import tweet_obj
import google_analyse_sentiment
import csv
from datetime import datetime, timedelta
import random
import numpy as np

filters = 'like wildfire'
days_lag_pre = 2
days_lag_post = 5


def get_queries_from_location(state, state_short, location_list):
    state_list = location_list[0].split()
    location_list[0] = state_list[0]

    q3 = ''
    for location in location_list:
        q3 += location + ' OR '
    # print(q3)

    state_1 = '#' + state.replace(' ', '') + 'Wildfires'
    state_2 = '#' + state.replace(' ', '') + 'Fires'
    state_short_1 = '#' + state_short + 'Fires'
    state_short_2 = '#' + state_short + 'Wildfires'

    q1 = '(' + q3 + state + ' OR ' + state_short + ') AND ( Wildfire OR Wildfires OR fires OR fire OR burning OR burn OR Landscape burn OR wildland burn )'
    q2 = state_1  + ' OR ' + state_2 + ' OR ' + state_short_1 + ' OR ' + state_short_2
    q = q1 + ' OR ' + q2
    print(q)

    return q


def check_sentiment_column(row):
    if row[13] == '':
        print('no sentiment found for fire_id: {}.'.format(row[0]))
        return True
    return False


def get_start_end_dates(row):
    try:
        from_date = str(datetime.strptime(row[5], '%Y-%m-%d').date() - timedelta(days=days_lag_pre))
        to_date = str(datetime.strptime(row[6], '%Y-%m-%d').date() + timedelta(days=days_lag_post))
        # print('from date {} to date {}'.format(from_date, to_date))
        # print('duration {}'.format(row[7]))
        return from_date, to_date
    except KeyError:
        print('Cannot generate start and end dates of fire.')


def get_location_list(row):
    location_list = row[10].split(',')
    state = row[11]
    state_short = row[12]
    return location_list, state, state_short


def get_tweets(start_date, end_date, query, filters):
    # Get tweets for particular day and query
    query_string = query + ' -' + filters + ' ' + 'since:' + start_date + ' until:' + end_date + ' -filter:replies'
    results = enumerate(sntwitter.TwitterSearchScraper(query_string).get_items())

    # save results in tweet object list
    tweets = []
    for tweet_tuple in results:
        tweet = tweet_tuple[1]

        tweet = tweet_obj.tweet_obj(tweet.content, tweet.date, tweet.id, tweet.username)
        tweets.append(tweet)

    print("NUMBER OF TWEETS FOR FIRE: {}".format(len(tweets)))
    return tweets


def group_tweet_texts(tweets):
    tweet_text = ''

    for tweet in tweets:
        text = tweet.full_text
        if text[-1] != '.':
            text += '. '

        tweet_text += text

    return tweet_text


def get_sentiment_for_text(text):
    sentiment = google_analyse_sentiment.analyze(text)
    return sentiment


with open("datasets/AUS_Ignitions_2016_I.csv", 'r') as dataset_incomplete:

    reader = csv.reader(dataset_incomplete, delimiter=',')

    for row in reader:
        if check_sentiment_column(row) and int(row[0]) > -1 :
            start_date, end_date = get_start_end_dates(row)
            location_list, state, state_short = get_location_list(row)
            query = get_queries_from_location(state, state_short, location_list)

            tweets = get_tweets(start_date,end_date, query, filters)
            grouped_text = group_tweet_texts(tweets)

            if grouped_text != '':
                sentiment = get_sentiment_for_text(grouped_text)
                score = sentiment.score
                magnitude = sentiment.magnitude
                num_tweets = len(tweets)
            else:
                score = 0
                magnitude = 0
                num_tweets = 0

            row[13] = score
            row.append(magnitude)
            row.append(num_tweets)

            with open('NA_ignitions_2016.csv', 'a') as dataset:
                writer = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(row)

            print('row ID {} saved'.format(row[0]))

    print('dataset sentiment analysed successfully')