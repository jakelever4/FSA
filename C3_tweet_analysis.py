import SQLite
from dateutil.parser import parse


db_file = 'US_V3.db'
conn = SQLite.create_connection(db_file)
tweets_df = SQLite.execute_query("""SELECT * FROM tweets;""")

tweets_df['start_doy'] = tweets_df['start_date'].apply(lambda  x: parse(x))



