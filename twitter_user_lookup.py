import requests
import pandas as pd

# You can enter up to 100 comma-separated values.
usernames = "usernames=BCGovFireInfo"
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAM2zMgEAAAAAjIFbBetAWCuAzaEL%2B5jSMyofgKE%3DwCGeSfjOYu91nXq0LJiygBheEegg7mU5dhecl2jD2IJIPiwbQI'


def create_url(user_id):
    return "https://api.twitter.com/2/users/{}/tweets".format(user_id)


def create_lookup_url(usernames):
    user_fields = "user.fields=description,entities,id,location,name,username"
    url = "https://api.twitter.com/2/users/by?{}&{}".format(usernames, user_fields)
    return url


def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserTweetsPython"
    return r


def lookup_bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserLookupPython"
    return r


def connect_to_lookup_endpoint(url):
    response = requests.request("GET", url, auth=lookup_bearer_oauth,)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def connect_to_endpoint(user_id, next_token):
    url = create_url(user_id)
    params = {'tweet.fields': 'author_id,created_at,entities,geo,id,text,public_metrics,referenced_tweets,reply_settings,source,withheld',
              'max_results':100,
              'pagination_token':next_token
              }

    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    # print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    json_response = response.json()
    next_token = None
    try:
        next_token = json_response['meta']['next_token']
    except KeyError:
        print('no next_token found')

    return json_response, next_token


def lookup_usernames(usernames):
    url = create_lookup_url(usernames)
    response = connect_to_lookup_endpoint(url)
    return response['data']


def lookup_user_id(user_id):
    results, next_token = connect_to_endpoint(user_id, next_token=None)
    geo_info = []
    try:
        all_results = results['data']
    except KeyError:
        print('no tweets found for fire')
        return []

    while next_token is not None:
        new_results, next_token = connect_to_endpoint(user_id, next_token=next_token)
        try:
            all_results += new_results['data']
        except KeyError:
            continue

    tweets = []

    for result in all_results:
        ctime = result['created_at']
        tweet_id = result['id']
        if 'referenced_tweets' in result:
            is_rt = True
        else:
            is_rt = False
        likes = result['public_metrics']['like_count']
        quotes = result['public_metrics']['quote_count']
        replies = result['public_metrics']['reply_count']
        rts = result['public_metrics']['retweet_count']
        text = result['text']

        row = [ctime, tweet_id, is_rt, likes, quotes, replies, rts, text]
        tweets.append(row)

    cols = ['created_at', 'tweet_id', 'is_RT', 'like_count', 'quote_count', 'reply_count', 'RT_count', 'text']
    df = pd.DataFrame(tweets, columns=cols)
    print(df)
    return df


users = lookup_usernames(usernames)

for user in users:
    user_id = user['id']
    username = user['username']

    df = lookup_user_id(user_id)
    df.to_csv('tweets/{}_tweets.csv'.format(username))
    print('Tweets for {} saved'.format(username))