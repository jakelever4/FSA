import requests
import json
from datetime import datetime
import rfc3339
import time
from dateutil.parser import parse
import entity_obj
import tweet_obj
import re
import tweepy

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAM2zMgEAAAAAjIFbBetAWCuAzaEL%2B5jSMyofgKE%3DwCGeSfjOYu91nXq0LJiygBheEegg7mU5dhecl2jD2IJIPiwbQI'
search_url = "https://api.twitter.com/2/tweets/search/all"

api_key = 'RiiQO55ccrGJhuECOVi2pyjXf'
api_key_secret = 'RsVmLMKBNTgpGfMw7Br6ViQ317ej9fbSoEE8RFEcOl1V6aS7c2'


query = 'Dixie Fire'
filters = 'like wildfire'


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers, query_params):
    response = requests.request("GET", url, headers=headers, params=query_params)
    # print(response.status_code)
    status_code = response.status_code

    if status_code == 429:
        while status_code == 429:
            print('429 Twitter exception: too many requests. Waiting 10 secs and retrying')
            time.sleep(10)
            response = requests.request("GET", url, headers=headers, params=query_params)
            status_code = response.status_code
            print(status_code)

    elif status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def get_rfc33339_date(date):
    return rfc3339.rfc3339(date)


def full_archive_search(query, start_date, end_date, next_token):
    try:
        start_date = get_rfc33339_date(datetime.strptime(start_date, '%Y-%m-%d'))
        end_date = get_rfc33339_date(datetime.strptime(end_date, '%Y-%m-%d'))
    except:
        print('could not convert dates for archive search')
        return None, None

    headers = create_headers(bearer_token)
    query_params = {'query': query, 'start_time': start_date, 'end_time': end_date, 'tweet.fields': 'author_id,created_at,entities,geo,id,text,public_metrics,referenced_tweets,reply_settings,withheld', 'user.fields': 'description,id,name,username', 'next_token':next_token, 'max_results': 500, 'expansions': 'author_id'}
    json_response = connect_to_endpoint(search_url, headers, query_params)


    next_token = None
    try:
        next_token = json_response['meta']['next_token']
        print('next token found, more tweets to collect.')

    except KeyError:
        print('no next_token found')



    # print(json.dumps(json_response, indent=4, sort_keys=True))
    return json_response, next_token


def get_user_from_id(author_id, users):
    for user in users:
        user_id = user['id']
        if user_id == author_id:
            return user['name'], user['username']
    print('Username Not Found')
    return 'NA', 'NA'


def get_tweets(query, start_date, end_date, fire_id):
    results, next_token = full_archive_search(query, start_date, end_date, next_token=None)
    user_expansions = []
    all_results = []

    try:
        all_results += results['data']
        user_expansions += results['includes']['users']
    except KeyError:
        print('no tweets found for fire')
        return []

    while next_token is not None:
        new_results, next_token = full_archive_search(query, start_date,end_date, next_token=next_token)
        all_results += new_results['data']
        user_expansions += new_results['includes']['users']
        print('{} tweets found so far'.format(len(all_results)))

    tweets = []
    no_user_found = 0
    num_tweets = len(all_results)
    for tweet in all_results:
        if 'referenced_tweet' in tweet:
            continue

        dtime = parse(tweet['created_at'])
        date = dtime.date()
        author_name, author_username = get_user_from_id(tweet['author_id'], user_expansions)
        if author_name == 'NA' and author_username == 'NA':
            no_user_found += 1

        try:
            entities = entity_obj.save_entities(tweet['id'], tweet['entities'])
        except KeyError:
            entities = None
        rt_count = tweet['public_metrics']['retweet_count'] + tweet['public_metrics']['quote_count']
        tweet_text_no_url = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet['text'])

        tweet_object = tweet_obj.tweet_obj(tweet['id'], fire_id, tweet_text_no_url, date, dtime,
                                           tweet['author_id'], author_name, author_username, entities, rt_count)
        tweets.append(tweet_object)

    print('FOR {} RESULTS, COULDNT MATCH USERS FOR {}'.format(num_tweets, no_user_found))
    return tweets


#
# # Tweepy streaming listener for streaming tweets. TODO test
# class StreamListener(tweepy.StreamListener):
#     def on_status(self, status):
#         print(status.id_str)
#
#     def on_error(self, status_code):
#         print("Encountered streaming error with code: {}".format(status_code))
#         return






# tweets = get_tweets(query, '2021-07-13', '2021-08-11', fire_id =None)
#
# for tweet in tweets:
#     print(tweet)