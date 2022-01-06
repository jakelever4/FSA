import nltk
import spacy
from spacy.lang.en import English
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from dateutil.parser import parse
import SQLite

pd.set_option("display.max_rows", 200, "display.max_columns", 200)
#
# nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])
db_file = 'australia.db'
conn = SQLite.create_connection(db_file)
tweets_test = SQLite.execute_query("""SELECT * FROM tweets;""", conn, table='tweets')
#
# tweets_df['dtime'] = tweets_df['dtime'].apply(lambda x: dateutil.parser.parse(x))


# tweets_test = pd.read_csv('tweets_df_test.csv')

spacy.load('en_core_web_sm')
parser = English()
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

black_words = ['tennessee', 'carolina', 'gatlinburg']

def tokenize(text):
    lda_tokens = []
    users = []
    try:
        tokens = parser(text)
    except:
        print('error')
        return [], []


    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            None
            # lda_tokens.append('URL')

        elif token.orth_.startswith('@'):
            # lda_tokens.append('SCREEN_NAME')
            users.append(token.lower_)
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens, users


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text):
    tokens, users = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [token for token in tokens if token not in black_words]
    tokens = [get_lemma(token) for token in tokens]
    return tokens, users


tweets_test['dtime'] = tweets_test['dtime'].apply(lambda x: parse(x))
tweets_test['doy'] = tweets_test['dtime'].apply(lambda x: x.to_pydatetime().timetuple().tm_yday)
tweets_by_day = tweets_test.groupby(by='doy')
print(tweets_by_day.groups)


DOYs = []
all_tokens = []
all_users = []

for doy, group in tweets_by_day:
    print('DOY: {}'.format(doy))
    tokens_for_day = []
    users_for_day = []
    for index, tweet in group.iterrows():
        if tweet['full_text'] is None:
            print('error')
        tokens, users = prepare_text_for_lda(tweet['full_text'])
        # print('tokens: {}'.format(tokens))
        # print('users: {}'.format(users))
        # print(tweet['full_text'])
        tokens_for_day += tokens
        users_for_day += users

    # val_cnt = pd.Series(tokens_for_day).value_counts()

    all_tokens += tokens_for_day
    all_users += users_for_day
    DOYs.append(doy)


vl_cnt = pd.Series(all_tokens).value_counts()

# for doy in DOYs:
print('doy: {}'.format(doy))
print(all_tokens)
print(all_users)
print(vl_cnt.head(100))


# tokens_by_day.index = DOYs
# users_by_day.index = DOYs


# text_data = []
# for line in tweets_test['dtime']:
#     tokens, users, hashtags = prepare_text_for_lda(line)
#     print(tokens)
#     text_data.append(tokens)