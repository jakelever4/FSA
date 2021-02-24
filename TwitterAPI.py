import requests
import json
from datetime import datetime
import rfc3339
import time

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAM2zMgEAAAAAjIFbBetAWCuAzaEL%2B5jSMyofgKE%3DwCGeSfjOYu91nXq0LJiygBheEegg7mU5dhecl2jD2IJIPiwbQI'
search_url = "https://api.twitter.com/2/tweets/search/all"

api_key = 'lVDQ566wOv7ch377C6msQcORY'
api_key_secret = 'LYRIaLhNVBrdwuk738s7e3dxKucg62ODTmSUjDVWiEFq4v3QGn'


query = 'California wildfires'
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
    query_params = {'query': query, 'start_time': start_date, 'end_time': end_date, 'tweet.fields': 'author_id,context_annotations,created_at,entities,geo,id,text', 'user.fields': 'description', 'next_token':next_token, 'max_results': 500}
    json_response = connect_to_endpoint(search_url, headers, query_params)


    next_token = None
    try:
        next_token = json_response['meta']['next_token']
        print('next token found, more tweets to collect.')

    except KeyError:
        print('no next_token found')



    # print(json.dumps(json_response, indent=4, sort_keys=True))
    return json_response, next_token


# print(full_archive_search(query, '2019-09-10', '2019-09-25', filters))