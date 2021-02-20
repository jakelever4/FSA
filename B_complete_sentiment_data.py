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

wpop_density = np.loadtxt('gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11_2015_15_min_asc/gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev11_2015_15_min.asc', skiprows=6)

filters = ['like wildfire', 'feel the burn']
days_lag_pre = 1
days_lag_post = 1


def get_queries_from_location(state, country, location_list):
    state_list = location_list[0].split()
    for value in state_list:
        # value = value.strip()
        if value != 'county' and value != 'County' and value != 'HI' and value != 'OR':
            location_list.append(value)

    filt = ''
    for value in filters:
        filt += ' -' + value

    print(filt)

    location_list.append(state)
    hashtags = ''
    for location in location_list:
        loc = location.strip().replace(' ', '')
        loc_hashtag1 = '#' + loc + 'Wildfires'
        loc_hashtag2 = '#' + loc + 'Fires'

        hashtags += ' OR ' + loc_hashtag1 + ' OR ' + loc_hashtag2

    fire_keywords = ') ( Wildfire OR Wildfires OR Landscape burn OR "wildland burn" )'

    location_keywords = '( '
    for location in location_list:
        loc = location.strip()
        if len(loc) > 2:
            if len(location_keywords) > 3:
                location_keywords += ' OR ' + loc
            else:
                location_keywords += loc

    query = location_keywords + fire_keywords + hashtags
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


def get_tweets(start_date, end_date, query, fire_ID):
    # Get tweets for particular query
    results, next_token = TwitterAPI.full_archive_search(query, start_date, end_date, next_token=None)
    all_results = [results]

    # Check the next token to see if there is another page of results to get
    while next_token is not None:
        new_results, next_token = TwitterAPI.full_archive_search(query,start_date,end_date, next_token=next_token)
        all_results.append(new_results)

    # save results in tweet object list
    tweets = []
    for result_list in all_results:
        for tweet in result_list['data']:
            try:
                date = tweet['created_at'][:10]
                try:
                    entities = entity_obj.save_entities(tweet['id'], tweet['entities'])
                except KeyError:
                    entities = None

                tweet_o = tweet_obj.tweet_obj(tweet['id'], fire_ID, tweet['text'], date, tweet['author_id'], entities)
                tweets.append(tweet_o)
            except KeyError:
                print('Cannot find fields for tweet {}.'.format(tweet['id']))

    return tweets


def group_tweet_texts(binned_daily_tweets):
    daily_text = []
    for daily_tweets in binned_daily_tweets:
        days_tweet_text = ''

        for text in daily_tweets:
            if text[-1] != '.':
                text += '. '

            days_tweet_text += text

        daily_text.append(days_tweet_text)

    return daily_text


def get_sentiment_for_text(text):
    sentiment = google_analyse_sentiment.analyze(text)
    return sentiment


def get_unique_days(list_of_days):
    s = set(list_of_days)
    lst = list(s)
    lst.sort()
    return lst


def split_tweets_into_daily(tweets):
    days = []
    for tweet in tweets:
        day = datetime.strptime(tweet.date, '%Y-%m-%d')
        days.append(day)

    unique = get_unique_days(days)
    binned_tweets = []

    for day in unique:
        days_tweets = []

        for tweet in tweets:
            tweet_day = datetime.strptime(tweet.date, '%Y-%m-%d')
            if tweet_day == day:
                days_tweets.append(tweet.full_text)

        binned_tweets.append(days_tweets)

    return unique, binned_tweets


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
        tweet_row = (int(tweet.tweet_id), fire_row_id, tweet.full_text, tweet.date, int(tweet.author_id))

        tweet_row_id = SQLite.create_tweet(conn,tweet_row)

        if tweet.entities is not None:
            save_entities_to_db(tweet.entities, tweet_row_id)
    print('Tweets for fire_ID {} successfully saved to databse.'.format(fire_row_id))
    return None


def save_entities_to_db(entities, tweet_row_id):
    try:
        hashtags = json.dumps(entities.hashtags[0])
        urls = json.dumps(entities.urls[0])
        entity_row = (tweet_row_id, hashtags, urls)
        SQLite.create_entity(conn, entity_row)
    except:
        raise Exception('Couldnt save entities to db for tweet id {}'.format(tweet_row_id))


def save_fire_to_db(fire, conn):
    sent_as_str = convert_list_to_string(fire[16])
    ov_pos_sent = convert_list_to_string(fire[18])
    ov_neg_sent = convert_list_to_string([fire[19]])
    mag_as_str = convert_list_to_string(fire[20])
    num_tw_as_str = convert_list_to_string(fire[22])

    fire_row = (int(fire[0]),float(fire[1]),float(fire[2]),float(fire[3]),float(fire[4]),fire[5],fire[6],float(fire[7]),float(fire[8]),float(fire[9]),fire[10],fire[11],
                fire[12],fire[13],fire[14],fire[15],sent_as_str,float(fire[17]), ov_pos_sent, ov_neg_sent,mag_as_str,float(fire[21]),num_tw_as_str, fire[23])

    fire_row_ID = SQLite.create_fire(conn, fire_row)
    print('fire_ID: {} successfully saved to database.'.format(fire[0]))
    return fire_row_ID


with open("datasets/V4_Ignitions_2016_I.csv", 'r') as dataset_incomplete:

    reader = csv.reader(dataset_incomplete, delimiter=',')

    # WRITE HEADER
    with open('datasets/V4_Ignitions_2016.csv', 'w') as dataset:
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
                         'overall_sentiment',
                         'overall_positive_sentiment',
                         'overall_negative_sentiment',
                         'magnitude',
                         'overall_magnitude',
                         'num_tweets',
                         'total_tweets'
                        ])

    database = 'test.db'
    # create a database connection
    conn = SQLite.create_connection(database)

    with conn:
        SQLite.create_tables(conn)
        for row in reader:
            # IF ROW IS NOT ANALYSED YET THEN ANALYSE
            if check_sentiment_column(row) and int(row[0]) >= 0: # 866562:

                # GET START, END DATES, LOCATION WORDS AND GENERATE QUERY
                start_date, end_date = get_start_end_dates(row)
                location_list, state, state_short = get_location_list(row)
                query = get_queries_from_location(state, state_short, location_list)

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
                    # tweets = remove_filters(tweets, filters)
                    # num_tweets = len(tweets)

                    try:
                        tweets = get_tweets(start_date, end_date, query, row[0])
                        tweets = remove_filters(tweets, filters)
                        num_tweets = len(tweets)
                    except:
                        print('Error collecting tweets')
                        continue
                    break

                # GROUP TWEETS FOR ANALYSIS
                unique_days, grouped_daily_tweets = split_tweets_into_daily(tweets)

                num_tweets_vector = []
                for day_of_tweets in grouped_daily_tweets:
                    num_tweets_vector.append(len(day_of_tweets))
                    if len(day_of_tweets) < 5:
                        print('found less than 5 tweets for a day')

                grouped_text = group_tweet_texts(grouped_daily_tweets)

                if len(grouped_text) != 0:
                    # ANALYSE SENTIMENT OF GROUPED TWEETS
                    sentiment_vector = []
                    positive_sentiment_vector = []
                    negative_sentiment_vector = []
                    magnitude_vector = []
                    # For each days tweets
                    for day in grouped_text:
                        # analyse tweets for day, add metrics to vector
                        sentiment = get_sentiment_for_text(day)

                        score = sentiment.score
                        magnitude = sentiment.magnitude
                        sentiment_vector.append(score)
                        magnitude_vector.append(magnitude)
                        positive_sentiment_vector.append(sentiment.overall_positive_sentiment)
                        negative_sentiment_vector.append(sentiment.overall_negative_sentiment)

                    # weather_vector = DarkSkyAPI.get_weather_vector(start_date, end_date, lat, lon)
                    # for x in weather_vector:
                    #     print(x)

                    fig, (ax1, ax2) = plt.subplots(2)
                    fig.suptitle('fire ID: {} over burn period ({} to {})'.format(row[0], start_date, end_date))
                    ax1.set_title('SUM(Sentiment)')
                    ax2.set_title('SUM(Magnitude)')
                    ax2.plot(unique_days, magnitude_vector, color='g')

                    ax1.plot(unique_days, sentiment_vector, color='b')
                    ax1.bar(unique_days, positive_sentiment_vector, color='g')
                    ax1.bar(unique_days, negative_sentiment_vector, color='r')

                    plt.show()

                    # Sum social metrics
                    total_sentiment = sum(sentiment_vector)
                    total_magnitude = sum(magnitude_vector)
                    total_tweets = sum(num_tweets_vector)
                    print('TOTAL TWEETS FOR FIRE: {}'.format(total_tweets))

                    # add variables to row
                    row.append(sentiment_vector)
                    row.append(total_sentiment)
                    row.append(positive_sentiment_vector)
                    row.append(negative_sentiment_vector)
                    row.append(magnitude_vector)
                    row.append(total_magnitude)
                    row.append(num_tweets_vector)
                    row.append(total_tweets)

                    print(row)

                    #SAVE TWEETS, FIRE IN DATABASE
                    fire_row_id = save_fire_to_db(row, conn)
                    save_tweets_to_db(tweets, fire_row_id, conn)

                    # SAVE FIRE DATA TO CSV
                    with open('datasets/V4_Ignitions_2016.csv', 'a') as dataset:
                        writer = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(row)

                    print('row ID {} saved'.format(row[0]))

        print('dataset sentiment analysed successfully')