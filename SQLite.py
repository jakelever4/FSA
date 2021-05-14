import sqlite3
from sqlite3 import Error
import pandas as pd

fire_columns = ['fire_ID', 'latitude', 'longitude', 'size', 'perimeter', 'start_date', 'end_date', 'duration',
                'speed', 'expansion', 'direction', 'landcover', 'location', 'state', 'state_short', 'pop_density',
                'sentiment', 'magnitude', 'num_tweets']

tweet_columns = ['tweet_ID', 'fire_ID', 'full_text', 'date', 'dtime', 'author_ID', 'author_name', 'author_username',
                 'rt_count', 'sentiment', 'magnitude']


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def create_fire(conn, fire):
    sql = ''' INSERT INTO fires(fire_ID,latitude,longitude,size,perimeter,start_date,end_date,duration,speed,expansion,
    direction,landcover,location,state,state_short,pop_density,sentiment,magnitude,num_tweets)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, fire)
    conn.commit()
    return cur.lastrowid


def update_fire(conn, fire):
    sql = ''' UPDATE fires
              SET priority = ? ,
                  begin_date = ? ,
                  end_date = ?
              WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, fire)
    conn.commit()


def create_tweet(conn, tweet):
    """
    Create a new task
    :param conn:
    :param task:
    :return:
    """

    sql = ''' INSERT INTO tweets(tweet_ID,fire_ID,full_text,date,dtime,author_ID,author_name,author_username,rt_count,sentiment,magnitude)
              VALUES(?,?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, tweet)
    conn.commit()
    return cur.lastrowid


def create_entity(conn, entity):
    """
        Create a new task
        :param conn:
        :param task:
        :return:
        """

    sql = ''' INSERT INTO entities(tweet_ID,hashtags,urls)
                  VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, entity)
    conn.commit()
    return cur.lastrowid


def create_tables(conn):
    sql_create_fires_table = """ CREATE TABLE IF NOT EXISTS fires (
                                        fire_ID integer,
                                        latitude real,
                                        longitude real,
                                        size real,
                                        perimeter real,
                                        start_date text,
                                        end_date text,
                                        duration real,
                                        speed real,
                                        expansion real,
                                        direction text,
                                        landcover text,
                                        location text,
                                        state text,
                                        state_short text,
                                        pop_density real,
                                        sentiment real,
                                        magnitude real,
                                        num_tweets int
                                    ); """

    sql_create_tweets_table = """CREATE TABLE IF NOT EXISTS tweets (
                                    tweet_ID integer,
                                    fire_ID integer,
                                    full_text text,
                                    date text,
                                    dtime text,
                                    author_ID int,
                                    author_name text,
                                    author_username text,
                                    rt_count int,
                                    sentiment real,
                                    magnitude real,
                                    FOREIGN KEY(fire_ID) REFERENCES fires(fire_ID)
                                );"""

    sql_create_entities_table = """CREATE TABLE IF NOT EXISTS entities (
                                        entity_ID PRIMARY KEY,
                                        tweet_ID integer,
                                        hashtags text,
                                        urls text,
                                        FOREIGN KEY(tweet_ID) REFERENCES tweets(tweet_ID)
                                    );"""

    # create tables
    if conn is not None:
        # create projects table
        create_table(conn, sql_create_fires_table)

        # create tasks table
        create_table(conn, sql_create_tweets_table)
        create_table(conn, sql_create_entities_table)
    else:
        print("Error! cannot create the database connection.")


def select_all_fires(conn):
    """
    Query rows in the fires table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM fires")
    rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=fire_columns)
    return df


def select_all_tweets(conn):
    """
    Query rows in the tweets table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tweets")
    rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=tweet_columns)
    df = convert_tweet_rows(df)

    return df


def execute_query(query, conn, table=None, cols=None):
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    if cols is None:
        if table == 'fires':
            cols = fire_columns
        elif table == 'tweets':
            cols = tweet_columns
        else:
            cols = None

    df = pd.DataFrame(rows, columns=cols)

    return df


def convert_tweet_rows(df):
    df['tweet_ID'] = df['tweet_ID'].astype(int)
    df['fire_ID'] = df['fire_ID'].astype(int)
    df['author_ID'] = df['author_ID'].astype(int)

    return df


def string_to_float_array(series):
    column = []
    for str in series:
        if str[0] == '[':
            str = str[1:]
        if str[-1] == ']':
            str = str[:-1]

        lst = str.split()
        lst_fl = [float(x.replace(',', '')) for x in lst]
        column.append(lst_fl)

    return column
