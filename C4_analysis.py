import pandas as pd
import SQLite
import json
import statistics
import numpy as np
from datetime import datetime


# load in AUS dataset
pd.set_option("display.max_rows", None, "display.max_columns", None)
db_file = 'database.db'
db_file_aus = 'australia.db'
conn = SQLite.create_connection(db_file)
conn_aus = SQLite.create_connection(db_file_aus)
crs = {'init': 'epsg:4326'}

# fires_df = SQLite.select_all_fires(conn)
fires_df_aus = SQLite.select_all_fires(conn_aus)
fires_df_aus['sentiment'] = fires_df_aus['sentiment'].apply(lambda x: json.loads(x))
fires_df_aus['magnitude'] = fires_df_aus['magnitude'].apply(lambda x: json.loads(x))

# remove fires with no tweets
indexNames = fires_df_aus[ fires_df_aus['num_tweets'] == 0].index
# Delete these row indexes from dataFrame
fires_df_aus.drop(indexNames, inplace=True)

fires_df_aus['s_mean'] = fires_df_aus['sentiment'].apply(lambda x: statistics.mean(x))
fires_df_aus['s_var'] = fires_df_aus['sentiment'].apply(lambda x: np.var(x))

fires_df_aus['m_mean'] = fires_df_aus['magnitude'].apply(lambda x: statistics.mean(x))
fires_df_aus['m_var'] = fires_df_aus['magnitude'].apply(lambda x: np.var(x))

fires_df_aus['direction_cat'] = fires_df_aus['direction'].astype('category').cat.codes
fires_df_aus['landcover_cat'] = fires_df_aus['landcover'].astype('category').cat.codes
fires_df_aus['state_cat'] = fires_df_aus['state'].astype('category').cat.codes

fires_df_aus['s_duration'] = fires_df_aus['duration']
fires_df_aus['overall_sentiment'] = fires_df_aus['sentiment'].apply(lambda x: sum(x))
fires_df_aus['overall_magnitude'] = fires_df_aus['magnitude'].apply(lambda x: sum(x))
fires_df_aus['total_tweets'] = fires_df_aus['num_tweets']


# Load in US Dataset
fires_df = pd.read_csv('fires_df.csv')

fires_df['sentiment'] = fires_df['sentiment'].apply(lambda x: json.loads(x))
fires_df['overall_positive_sentiment'] = fires_df['overall_positive_sentiment'].apply(lambda x: json.loads(x))
fires_df['overall_negative_sentiment'] = fires_df['overall_negative_sentiment'].apply(lambda x: json.loads(x))
fires_df['magnitude'] = fires_df['magnitude'].apply(lambda x: json.loads(x))
fires_df['num_tweets'] = fires_df['num_tweets'].apply(lambda x: json.loads(x))
fires_df['avg_sentiment'] = fires_df['avg_sentiment'].apply(lambda x: json.loads(x))

fires_df['start_doy'] = fires_df['start_date'].apply(lambda  x: datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday)
fires_df['end_doy'] = fires_df['end_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday)

# avg_sentiment = []
avg_magnitude = []
for ind, row in fires_df.iterrows():
    # s = [x / y for x,y in zip(row['sentiment'],row['num_tweets'])]
    m = [x / y for x,y in zip(row['magnitude'],row['num_tweets'])]
    # avg_sentiment.append(s)
    avg_magnitude.append(m)

# fires_df['s'] = avg_sentiment
fires_df['avg_magnitude'] = avg_magnitude

fires_df['s_mean'] = fires_df['sentiment'].apply(lambda x: statistics.mean(x))
fires_df['s_var'] = fires_df['sentiment'].apply(lambda x: np.var(x))

fires_df['m_mean'] = fires_df['magnitude'].apply(lambda x: statistics.mean(x))
fires_df['m_var'] = fires_df['magnitude'].apply(lambda x: np.var(x))

fires_df.drop(columns=['duration'])