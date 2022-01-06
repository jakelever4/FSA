import requests
import json
import visualise_stream
from dateutil import parser


rules = [
    # {"value": "(wildfire california) OR bushfire", "tag": "california wildfires"},
    # {"value": "(Australia bushfire) OR wildfire", "tag": "australia bushfires"},
    # {"value": "(Europe bushfire) OR wildfire", "tag": "Europe bushfires"},
    {"value": "rishi sunak", "tag": "Sunak"},
    {'value': 'boris johnson', 'tag': 'Johnson'},
    {'value': 'dominic raab', 'tag': 'Raab'}
    # {"value": "from:BCGovFireInfo", "tag": "BC Gov fire info"}
]
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAM2zMgEAAAAAjIFbBetAWCuAzaEL%2B5jSMyofgKE%3DwCGeSfjOYu91nXq0LJiygBheEegg7mU5dhecl2jD2IJIPiwbQI'


def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FilteredStreamPython"
    return r


def get_users_tweets(user_id):
    response = requests.get()


def get_rules():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", auth=bearer_oauth
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    # print(json.dumps(response.json()))
    return response.json()


def delete_all_rules(rules):
    if rules is None or "data" not in rules:
        return None

    ids = list(map(lambda rule: rule["id"], rules["data"]))
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot delete rules (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    print(json.dumps(response.json()))


def set_rules(rules):
    payload = {"add": rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception(
            "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
        )

    print(json.dumps(response.json()))

    return response.json()


def get_stream(set):
    params = {'tweet.fields': 'author_id,created_at,entities,geo,id,text,public_metrics,referenced_tweets,reply_settings,source,withheld',
              'user.fields': 'description,entities,id,location,name,username',
              'place.fields': 'contained_within,country,country_code,full_name,geo,id,name,place_type',
              'media.fields': 'type,url,public_metrics',
              'expansions': 'author_id,attachments.media_keys,entities.mentions.username,geo.place_id',
              }
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream", auth=bearer_oauth, stream=True, params=params
    )
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    for response_line in response.iter_lines():
        if response_line:
            json_response = json.loads(response_line)
            # print(json.dumps(json_response, indent=4, sort_keys=True))
            text = process_response(json_response)
            # return text


def process_response(response):
     

    visualise_stream.update(new_tweet=tweet)
    # return tweet


def start_stream(rules):
    old_rules = get_rules()
    delete_all_rules(old_rules)
    set = set_rules(rules)
    data = get_stream(set)
    # return data


start_stream(rules)
