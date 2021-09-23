import tweet_obj
import google_analyse_sentiment
import csv
from datetime import datetime, timedelta
import time
import numpy as np
import TwitterAPI
import DarkSkyAPI
import matplotlib.pyplot as plt
import SQLite
import entity_obj
import json
from dateutil.parser import parse
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

wpop_density = np.loadtxt('gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11_2015_15_min_asc/gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev11_2015_15_min.asc', skiprows=6)

filters = ['like wildfire', 'feel the burn']
days_lag_pre = 3
days_lag_post = 3


def get_queries_from_location(state, country, location_list):
    state_list = location_list[0].split()
    for value in state_list:
        # value = value.strip()
        if value != 'county' and value != 'County' and value != 'HI' and value != 'OR' and value != 'no.'  \
                and value != 'No'  and value != 'No.' and value != 'no':
            location_list.append(value)

    i=0

    while i < len(location_list):
        loc = location_list[i]
        if loc.strip() == 'WA' or loc.strip() == 'Australia':
            location_list.remove(loc)
        else:
            i += 1

    # for loc in location_list:
    #     if loc.strip() == 'USA' or loc.strip() =='Canada' or loc.strip() == 'WA' or loc.strip() == 'Australia' ' Australia':
    #         location_list.remove(loc)

    location_list.append(state)

    hashtags = ''
    for location in location_list:
        loc = location.strip().replace(' ', '')
        loc_hashtag1 = '#' + loc + 'Wildfires'
        loc_hashtag2 = '#' + loc + 'Fires'

        hashtags += ' OR ' + loc_hashtag1 + ' OR ' + loc_hashtag2

    fire_keywords = ') ( Wildfire OR Wildfires OR "Landscape burn" OR "wildland burn"  OR bushfire )'

    # only relevant fro aus data obviously
    location_list = location_list[:-1]
    if state.strip() == 'Western Australia':
        state = '"Western Australia"'
    elif state.strip() == 'Northern Territory':
        state = '"Northern Territory"'
    elif state.strip() == 'South Australia':
        state = '"South Australia"'
    elif state.strip() == 'New South Wales':
        state = '"New South Wales"'

    location_list.append(state)

    location_keywords = '( '
    for location in location_list:
        loc = location.strip()
        if len(loc) > 2:
            if len(location_keywords) > 3:
                location_keywords += ' OR ' + loc
            else:
                location_keywords += loc

    query = location_keywords + fire_keywords + hashtags + ' -is:retweet -is:quote'
    print(query)

    return query


def check_sentiment_column(row):
    if len(row) == 15:
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
    location_list = row[12].split(',')
    state = row[13]
    state_short = row[14]

    if state_short == 'HI' or 'OR':
        state_short = 'USA'
    for loc in location_list:
        if loc == ' HI' or loc == ' OR':
            location_list.remove(loc)

    return location_list, state, state_short


def get_lat_lon(row):
    lat = row[1]
    lon = row[2]
    return [lat,lon]


def lookup_density(coords):
    lat_15_min = float(coords[0]) * 4
    lon_15_min = float(coords[1]) * 4

    lat_trans = round(360 - lat_15_min)
    lon_trans = round(720 + lon_15_min)

    # print([lat_trans,lon_trans])

    density = wpop_density[lat_trans][lon_trans]
    return density


def get_user_from_id(author_id, users):
    for user in users:
        user_id = user['id']
        if user_id == author_id:
            return user['name'], user['username']
    print('Username Not Found')
    return 'NA', 'NA'


def get_tweets(start_date, end_date, query, fire_ID):
    # Get tweets for particular query
    results, next_token = TwitterAPI.full_archive_search(query, start_date, end_date, next_token=None)
    user_expansions = []
    try:
        all_results = results['data']
        user_expansions += results['includes']['users']
    except KeyError:
        print('no tweets found for fire')
        return []


    # Check the next token to see if there is another page of results to get
    while next_token is not None:
        new_results, next_token = TwitterAPI.full_archive_search(query,start_date,end_date, next_token=next_token)
        all_results += new_results['data']
        user_expansions += new_results['includes']['users']

    # save results in tweet object list
    tweets = []
    no_user_found = 0
    num_res = len(all_results)
    for tweet in all_results:
        if 'referenced_tweets' in tweet:
            # ignore retweets
            # print('referenced tweet found. Type: {}'.format(tweet['referenced_tweets'][0]['type']))
            # print(tweet['text'])
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

        tweet_o = tweet_obj.tweet_obj(tweet['id'], fire_ID, tweet_text_no_url, date, dtime, tweet['author_id'],
                                      author_name, author_username, entities, rt_count)
        tweets.append(tweet_o)

    print('for {} results, couldnt match users for {}'.format(num_res, no_user_found))
    return tweets


def remove_special_chars(text):
    tweet_text = text.strip()
    tweet_text = re.sub(r"[^a-zA-Z0-9]+", ' ', tweet_text)
    sentence = tweet_text.replace('…', ' ').replace('...', ' ').replace('.', ',').replace('!', ' ') \
        .replace('?', ',').replace('\n', ' ').replace('|', ',').replace('"', "'")

    try:
        if sentence[0].isdigit():
            sentence = "A " + sentence
        if sentence[-1] == ',':
            sentence = sentence[:-1] + '. \n'
            # sentence[-1] = '. '
        elif sentence[-1] != '.':
            sentence += '. \n'
    except IndexError:
        # print(sentence)
        sentence = 'None. \n'

    return sentence


def group_tweet_texts(tweets):
    text = ''
    for tweet in tweets:
        sentence = remove_special_chars(tweet.full_text)
        text += sentence

    return text


def get_sentiment_for_text(text):
    sentiment = google_analyse_sentiment.analyze(text)
    return sentiment


def remove_filters(tweets, filters):
    filtered_tweets = []
    for tweet in tweets:
        append = True
        for filter in filters:
            if filter in tweet.full_text:
                append = False

        if append:
            filtered_tweets.append(tweet)

    return filtered_tweets


def convert_list_to_string(lst):
    st = ' '.join(str(x) for x in lst)
    return st


def save_tweets_to_db(tweets, fire_row_id, conn):
    for tweet in tweets:
        tweet_row = (int(tweet.tweet_id), fire_row_id, tweet.full_text, tweet.date, tweet.dtime, int(tweet.author_id),
                     tweet.author_name, tweet.author_username, tweet.retweet_count, tweet.sentiment, tweet.magnitude)

        tweet_row_id = SQLite.create_tweet(conn,tweet_row)

        if tweet.entities is not None:
            save_entities_to_db(tweet.entities, int(tweet.tweet_id))
    print('Tweets for fire_ID {} successfully saved to databse.'.format(fire_row_id))
    return None


def save_entities_to_db(entities, tweet_id):
    try:
        hashtags = json.dumps(entities.hashtags[0])
        urls = json.dumps(entities.urls[0])
        entity_row = (tweet_id, hashtags, urls)
        SQLite.create_entity(conn, entity_row)
    except:
        raise Exception('Couldnt save entities to db for tweet id {}'.format(tweet_id))


def save_fire_to_db(fire, conn):
    fire_row = (int(fire[0]),float(fire[1]),float(fire[2]),float(fire[3]),float(fire[4]),fire[5],fire[6],float(fire[7]),float(fire[8]),float(fire[9]),fire[10],fire[11],
                fire[12],fire[13],fire[14],fire[15],float(fire[16]),float(fire[17]),float(fire[18]))

    fire_row_ID = SQLite.create_fire(conn, fire_row)
    print('fire_ID: {} successfully saved to database.'.format(fire[0]))
    return fire_row_ID


def check_database_for_tweet(tweet, conn):
    id = tweet.tweet_id
    q = """SELECT * FROM tweets WHERE tweet_ID = {};""".format(id)
    result = SQLite.execute_query(q, conn, table='tweets')
    if len(result.index) >= 1:
        print('Tweet found in database! Using value')
        old_sentiment = result['sentiment'][0]
        old_magnitude = result['magnitude'][0]
        if tweet.sentiment is None and tweet.magnitude is None:
            tweet.sentiment = old_sentiment
            tweet.magnitude = old_magnitude
        return True, tweet
    else:
        return False, tweet


def check_database_for_fire(fire_ID):
    q = """SELECT * FROM fires WHERE fire_ID = {};""".format(fire_ID)
    result = SQLite.execute_query(q, conn, table='fires')
    if len(result.index) == 1:
        print('fire {} found in db already, skipping'.format(fire_ID))
        return True
    elif len(result.index) == 0:
        print('fire {} not found in db, analysing'.format(fire_ID))
        return False
    else:
        raise Exception


with open("datasets/aus_data_wcats.csv", 'r') as dataset_incomplete:

    reader = csv.reader(dataset_incomplete, delimiter=',')
    next(reader, None)  # skip the headers

    # WRITE HEADER
    with open('datasets/AUS_Ignitions.csv', 'w') as dataset:
        writer = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['fire_ID',
                         'latitude',
                         'longitude',
                         'size',
                         'perimeter',
                         'start_date',
                         'end_date',
                         'duration',
                         'speed',
                         'expansion',
                         'direction',
                         'landcover',
                         'location',
                         'state',
                         'state_short',
                         'pop_density',
                         # 'cloud_cover',
                         # 'humidity',
                         # 'precip_intensity',
                         # 'precip_prob',
                         # 'pressure',
                         # 'max_temp',
                         # 'uv_index',
                         # 'wind_bearing',
                         # 'wind_speed',
                         'sentiment',
                         'magnitude',
                         'num_tweets',
                        ])

    database = 'australia.db'
    # create a database connection
    conn = SQLite.create_connection(database)

    with conn:
        SQLite.create_tables(conn)
        for row in reader:
            # IF ROW IS NOT ANALYSED YET THEN ANALYSE
            # TODO: CHNAGE ROW ID HERE IF PARTIALLY ANALYSED
            fire_in_db = check_database_for_fire(row[0])
            # note highest US fire id is 123394
            if not fire_in_db and int(row[0]) >= 884578: # 865654:

                # GET START, END DATES, LOCATION WORDS AND GENERATE QUERY
                start_date, end_date = get_start_end_dates(row)
                location_list, state, state_short = get_location_list(row)
                query = get_queries_from_location(state, state_short, location_list)

                # FOR AUS DATA: REMOVE OLD SENTMENT, MAGNITUDE AND NUM_TWEETS VALES, AS WE ARE ABOUT TO RECALCULATE THEM.REMOVE FOR US  DATA
                row = row[:-3]

                # ADD POPULATION DENSITY TO ROW
                try:
                    lat, lon = get_lat_lon(row)
                    pop_density = lookup_density([lat, lon])
                    row.append(pop_density)
                except:
                    print('couldnt get pop density data for fire ID: {}. setting to 0'.format(row[0]))
                    row.append(0)

                # GET TWEETS FOR FIRE
                while True:
                    # tweets = get_tweets(start_date, end_date, query, row[0])
                    try:
                        tweets = get_tweets(start_date, end_date, query, row[0])
                        tweets = remove_filters(tweets, filters)
                        num_tweets = len(tweets)
                        print('{} tweets found.'.format(num_tweets))
                    except:
                        print('Error collecting tweets. Retrying')
                        continue
                    break

                if tweets != []:

                    # CHECK TO SEE WHICH TWEETS ARE ALREADY IN THE DATABASE
                    tweets_to_analyse = []
                    analysed_tweets = []
                    for tweet in tweets:
                        in_db, tweet = check_database_for_tweet(tweet, conn)
                        if not in_db:
                            tweets_to_analyse.append(tweet)
                        else:
                            analysed_tweets.append(tweet)

                    # unique_days, grouped_daily_tweets = split_tweets_into_daily(tweets_to_analyse)
                    num_analysed = len(analysed_tweets)
                    num_to_analyse = len(tweets_to_analyse)
                    print('{} tweets pulled from database, analysing {} extra tweets'.format(num_analysed,num_to_analyse))

                    grouped_text = group_tweet_texts(tweets_to_analyse)

                    # ANALYSE SENTIMENT OF GROUPED TWEETS
                    sentiment = get_sentiment_for_text(grouped_text)

                    if len(sentiment.sentences) != len(tweets_to_analyse):
                        print('sentences and tweets dont match. difference of {}'.format(abs(len(sentiment.sentences) - len(tweets_to_analyse))))
                        # raise Exception

                        non_matched_tweets = []
                        for tweet in tweets_to_analyse:
                            reduced_text = remove_special_chars(tweet.full_text).replace('.', '').replace(' \n', '').strip()
                            for sentence in sentiment.sentences:
                                sentence_text = sentence.text.replace('.', '').replace(' \n', '').strip()
                                if reduced_text == sentence_text:
                                    score = sentence.score
                                    magnitude = sentence.magnitude
                                    tweet.sentiment = score
                                    tweet.magnitude = magnitude

                                    # print(tweet)
                                    break
                            if tweet.sentiment is None:
                                print('cannot match Tweet: {}'.format(reduced_text))
                                non_matched_tweets.append(tweet)
                                # tweet.sentiment = 0
                                # tweet.magnitude = 0

                        print('total tweets not matched: {} trying to re-analyse'.format(len(non_matched_tweets)))
                        non_matched_grouped_text = group_tweet_texts(non_matched_tweets)
                        sentiment_2 = get_sentiment_for_text(non_matched_grouped_text)
                        non_matched_num = 0
                        for tweet in non_matched_tweets:
                            red_txt = remove_special_chars(tweet.full_text).replace(' \n', '').strip()
                            for sentence in sentiment_2.sentences:
                                sentence_text = sentence.text.replace('.', '').replace(' \n', '').strip()
                                fuzzy_ratio = fuzz.partial_ratio(red_txt,sentence_text)
                                print('{} AND {}. FUZZY RATIO: {}'.format(red_txt,sentence_text, fuzzy_ratio))
                                print(fuzzy_ratio)
                                if fuzzy_ratio > 90:
                                    print('{} \n {}. \n FUZZY RATIO: {} MATCHED'.format(red_txt,sentence_text, fuzzy_ratio))
                                    score = sentence.score
                                    magnitude = sentence.magnitude
                                    tweet.sentiment = score
                                    tweet.magnitude = magnitude
                                    # print(tweet)
                                    tweets_to_analyse.append(tweet)
                                    break

                            if tweet.sentiment is None:
                                print('still cannot match Tweet: {}'.format(reduced_text))
                                non_matched_num += 1
                                tweet.sentiment = 0
                                tweet.magnitude = 0

                        print('total non matched: {}'.format(non_matched_num))

                    else:
                        for sentence in sentiment.sentences:
                            sentence_index = sentence.index
                            tweet = tweets_to_analyse[sentence_index]
                            score = sentence.score
                            magnitude = sentence.magnitude
                            tweet.sentiment = score
                            tweet.magnitude = magnitude
                            # print(tweet)

                    analysed_tweets = analysed_tweets + tweets_to_analyse
                    analysed_tweets.sort(key=lambda x: x.dtime, reverse=False)
                    for tweet in analysed_tweets:
                        if tweet.sentiment is None and tweet.magnitude is None:
                            print('Non analysed tweet: {}'.format(tweet.full_text))
                            tweet.sentiment = 0
                            tweet.magnitude = 0

                    avg_sentiment = 0
                    magnitude = 0
                    for tweet in analysed_tweets:
                        avg_sentiment += tweet.sentiment
                        magnitude += tweet.magnitude
                    avg_sentiment = avg_sentiment / len(analysed_tweets)

                    print('SENTIMENT SCORE FOR FIRE {}. SCORE: {}, MAGNITUDE: {}'.format(row[0], avg_sentiment, magnitude))
                    row.append(avg_sentiment)
                    row.append(magnitude)
                    row.append(len(analysed_tweets))

                    #SAVE TWEETS, FIRE IN DATABASE
                    fire_row_id = save_fire_to_db(row, conn)
                    save_tweets_to_db(tweets, row[0], conn)

                    # SAVE FIRE DATA TO CSV
                    # with open('datasets/V4_Ignitions_2016.csv', 'a') as dataset:
                    #     writer = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    #     writer.writerow(row)
                else:
                    row.append(0)
                    row.append(0)
                    row.append(0)
                    fire_row_id = save_fire_to_db(row, conn)

            print('row ID {} saved'.format(row[0]))

        print('dataset sentiment analysed successfully')


# weather_vector = DarkSkyAPI.get_weather_vector(start_date, end_date, lat, lon)
# for x in weather_vector:
#     print(x)

# fig, (ax1, ax2) = plt.subplots(2)
# fig.suptitle('fire ID: {} over burn period ({} to {})'.format(row[0], start_date, end_date))
# ax1.set_title('SUM(Sentiment)')
# ax2.set_title('SUM(Magnitude)')
# ax2.plot(unique_days, magnitude_vector, color='g')
#
# ax1.plot(unique_days, sentiment_vector, color='b')
# ax1.bar(unique_days, positive_sentiment_vector, color='g')
# ax1.bar(unique_days, negative_sentiment_vector, color='r')
#
# plt.show()
#
# def get_unique_days(list_of_days):
#     s = set(list_of_days)
#     lst = list(s)
#     lst.sort()
#     return lst

# def split_tweets_into_daily(tweets):
#     days = []
#     for tweet in tweets:
#         datetime = parse(tweet.date)
#         day = datetime.date().strftime("%Y/%m/%d")
#         # day = datetime.strptime(tweet.date, '%Y-%m-%d')
#         days.append(day)
#
#     unique = get_unique_days(days)
#     binned_tweets = []
#
#     for day in unique:
#         days_tweets = []
#
#         for tweet in tweets:
#             tweet_day = datetime.strptime(tweet.date, '%Y-%m-%d')
#             if tweet_day == day:
#                 days_tweets.append(tweet.full_text)
#
#         binned_tweets.append(days_tweets)
#
#     return unique, binned_tweets

#     sentiment_vector = []
#     positive_sentiment_vector = []
#     negative_sentiment_vector = []
#     magnitude_vector = []
#     # For each days tweets
#     for day in grouped_text:
#         # analyse tweets for day, add metrics to vector
#         sentiment = get_sentiment_for_text(day)
#
#         score = sentiment.score
#         magnitude = sentiment.magnitude
#         sentiment_vector.append(score)
#         magnitude_vector.append(magnitude)
#         positive_sentiment_vector.append(sentiment.overall_positive_sentiment)
#         negative_sentiment_vector.append(sentiment.overall_negative_sentiment)