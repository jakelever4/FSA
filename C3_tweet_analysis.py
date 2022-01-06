import SQLite
from dateutil.parser import parse
import re
import string
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from textwrap import wrap
import matplotlib.pyplot as plt
from textblob import TextBlob
import dateutil
from collections import Counter
import numpy as np
import calmap
import calplot
import matplotlib.dates as mdates
import nltk
import spacy
from spacy.lang.en import English
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import google_analyse_sentiment
import tkinter



us_top_tokens = pd.read_csv('us_top_tokens_analysed.csv')
aus_top_tokens = pd.read_csv('aus_top_tokens_analysed.csv')

us_avg_s = us_top_tokens['sentiment'].mean()
aus_avg_s = aus_top_tokens['sentiment'].mean()

us_avg_m = us_top_tokens['magnitude'].mean()
aus_avg_m = aus_top_tokens['magnitude'].mean()

print('AUS AVG: S: {} , M: {}'.format(aus_avg_s, aus_avg_m))
print('US AVG: S: {} , M: {}'.format(us_avg_s, us_avg_m))



# # Analysis of top words
# aus_top_tokens = pd.read_csv('aus_top_tokens.csv')
# aus_top_tokens['token'] = aus_top_tokens['token'].apply(lambda x: x.strip())
# # us_top_tokens['token'] = us_top_tokens['token'].apply(lambda x: x.strip())
#
# word_sentiments = []
# word_magnitudes = []
# for index, row in aus_top_tokens.iterrows():
#     analysis = google_analyse_sentiment.analyze(row['token'])
#     word_sentiments.append(analysis.score)
#     word_magnitudes.append(analysis.magnitude)
#
# aus_top_tokens['sentiment'] = pd.Series(word_sentiments)
# aus_top_tokens['magnitude'] = pd.Series(word_magnitudes)
#
# aus_top_tokens.to_csv('aus_top_tokens_analysed.csv')


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

aus_top_tokens = pd.read_csv('aus_top_tokens_analysed.csv')
us_top_tokens = pd.read_csv('us_top_tokens_analysed.csv')


print('avg aus sentiment: {}'.format(aus_top_tokens['sentiment'].mean()))
print('avg us sentimemt: {}'.format(us_top_tokens['sentiment'].mean()))
print()
print('avg aus mag: {}'.format(aus_top_tokens['magnitude'].mean()))
print('avg us ag: {}'.format(us_top_tokens['magnitude'].mean()))


x = us_top_tokens['token'].iloc[0:40]
y = us_top_tokens['count'].iloc[0:40]
fig, ax = plt.subplots()
ax.bar(x, y)

# First, let's remove the top, right and left spines (figure borders)
# which really aren't necessary for a bar chart.
# Also, make the bottom spine gray instead of black.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')

# Third, add a horizontal grid (but keep the vertical grid hidden).
# Color the lines a light gray as well.
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

ax.set_xlabel('Word', color='#333333', fontsize=20)
ax.set_ylabel('Count', fontsize=20, color='#333333')
ax.set_title('Top 40 Tokens in US Dataset', pad=15, color='#333333',
             weight='bold', fontsize=20)
plt.xticks(rotation=45, ha='right')
fig.tight_layout()

print('saving / showing fig')
plt.savefig('graphs2/top_words_us.png')
print('saved fig')
plt.show()
# print(top_20_words)



# pd.set_option("display.max_rows", None, "display.max_columns", None)
#
# nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])
db_file = 'australia.db'
conn = SQLite.create_connection(db_file)
tweets_df = SQLite.execute_query("""SELECT * FROM tweets;""", conn, table='tweets')

tweets_df['dtime'] = tweets_df['dtime'].apply(lambda x: dateutil.parser.parse(x))


# first = tweets_df.head(50000)
#
# first.to_csv('tweets_df_test.csv')




# Dictionary of English Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                      "can't": "cannot","can't've": "cannot have",
                      "'cause": "because","could've": "could have","couldn't": "could not",
                      "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                      "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                      "hasn't": "has not","haven't": "have not","he'd": "he would",
                      "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                      "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                      "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                      "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                      "it'd": "it would","it'd've": "it would have","it'll": "it will",
                      "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                      "mayn't": "may not","might've": "might have","mightn't": "might not",
                      "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                      "mustn't've": "must not have", "needn't": "need not",
                      "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                      "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                      "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                      "she'll": "she will", "she'll've": "she will have","should've": "should have",
                      "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                      "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                      "there'd've": "there would have", "they'd": "they would",
                      "they'd've": "they would have","they'll": "they will",
                      "they'll've": "they will have", "they're": "they are","they've": "they have",
                      "to've": "to have","wasn't": "was not","we'd": "we would",
                      "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                      "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                      "what'll've": "what will have","what're": "what are", "what've": "what have",
                      "when've": "when have","where'd": "where did", "where've": "where have",
                      "who'll": "who will","who'll've": "who will have","who've": "who have",
                      "why've": "why have","will've": "will have","won't": "will not",
                      "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                      "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                      "y'all'd've": "you all would have","y'all're": "you all are",
                      "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                      "you'll": "you will","you'll've": "you will have", "you're": "you are",
                      "you've": "you have"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

# # Expanding Contractions
# tweets_df['full_text'] = tweets_df['full_text'].apply(lambda x:expand_contractions(x))
# # lower case all words
# tweets_df['full_text'] = tweets_df['full_text'].apply(lambda x: x.lower())
# # remove digits
# tweets_df['full_text']=tweets_df['full_text'].apply(lambda x: re.sub('\w*\d\w*','', x))
# # remove punctuation
# tweets_df['full_text']=tweets_df['full_text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
# # remove extra spaces
# tweets_df['full_text']=tweets_df['full_text'].apply(lambda x: re.sub(' +',' ',x))
# # lemmatize using nlp model
# # tweets_df['lemmatized']=tweets_df['full_text'].apply(lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (token.is_stop==False)]))






DAYS = ['Sun.', 'Mon.', 'Tues.', 'Wed.', 'Thurs.', 'Fri.', 'Sat.']
MONTHS = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'June', 'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.']


def date_heatmap(series, start=None, end=None, mean=False, ax=None, **kwargs):
    '''Plot a calendar heatmap given a datetime series.

    Arguments:
        series (pd.Series):
            A series of numeric values with a datetime index. Values occurring
            on the same day are combined by sum.
        start (Any):
            The first day to be considered in the plot. The value can be
            anything accepted by :func:`pandas.to_datetime`. The default is the
            earliest date in the data.
        end (Any):
            The last day to be considered in the plot. The value can be
            anything accepted by :func:`pandas.to_datetime`. The default is the
            latest date in the data.
        mean (bool):
            Combine values occurring on the same day by mean instead of sum.
        ax (matplotlib.Axes or None):
            The axes on which to draw the heatmap. The default is the current
            axes in the :module:`~matplotlib.pyplot` API.
        **kwargs:
            Forwarded to :meth:`~matplotlib.Axes.pcolormesh` for drawing the
            heatmap.

    Returns:
        matplotlib.collections.Axes:
            The axes on which the heatmap was drawn. This is set as the current
            axes in the `~matplotlib.pyplot` API.
    '''
    # Combine values occurring on the same day.
    dates = series.index.floor('D')
    group = series.groupby(dates)
    series = group.mean() if mean else group.sum()

    # Parse start/end, defaulting to the min/max of the index.
    start = pd.to_datetime(start or series.index.min())
    end = pd.to_datetime(end or series.index.max())

    # We use [start, end) as a half-open interval below.
    end += np.timedelta64(1, 'D')

    # Get the previous/following Sunday to start/end.
    # Pandas and numpy day-of-week conventions are Monday=0 and Sunday=6.
    start_sun = start - np.timedelta64((start.dayofweek + 1) % 7, 'D')
    end_sun = end + np.timedelta64(7 - end.dayofweek - 1, 'D')

    # Create the heatmap and track ticks.
    num_weeks = (end_sun - start_sun).days // 7
    heatmap = np.zeros((7, num_weeks))
    ticks = {}  # week number -> month name
    for week in range(num_weeks):
        for day in range(7):
            date = start_sun + np.timedelta64(7 * week + day, 'D')
            if date.day == 1:
                ticks[week] = MONTHS[date.month - 1]
            if date.dayofyear == 1:
                ticks[week] += f'\n{date.year}'
            if start <= date < end:
                heatmap[day, week] = series.get(date, 0)

    # Get the coordinates, offset by 0.5 to align the ticks.
    y = np.arange(8) - 0.5
    x = np.arange(num_weeks + 1) - 0.5

    # Plot the heatmap. Prefer pcolormesh over imshow so that the figure can be
    # vectorized when saved to a compatible format. We must invert the axis for
    # pcolormesh, but not for imshow, so that it reads top-bottom, left-right.
    ax = ax or plt.gca()
    mesh = ax.pcolormesh(x, y, heatmap, **kwargs)
    ax.invert_yaxis()

    # Set the ticks.
    ax.set_xticks(list(ticks.keys()))
    ax.set_xticklabels(list(ticks.values()))
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels(DAYS)

    # Set the current image and axes in the pyplot API.
    plt.sca(ax)
    plt.sci(mesh)

    plt.show()
    plt.savefig('A_newgraphs/heatmap.pdf')

    return ax

# sentiment_by_day = tweets_df.resample('d', on='dtime')['sentiment'].sum()
# print('creating resample')
# magnitude_by_day = tweets_df.resample('d', on='dtime')['magnitude'].sum()
# print('saving set')
# magnitude_by_day.to_csv('days_magnitude.csv')


#
sentiment_by_day = pd.read_csv('days_magnitude.csv')
senti_by_day_aus = pd.read_csv('days_magnitude_aus.csv')

# sentiment_by_day['dtime'] = sentiment_by_day['dtime'].apply(lambda x: dateutil.parser.parse(x))

# sent_2016_aus = senti_by_day_aus.loc[118:483, :]
# sent_2016 = sentiment_by_day.loc[126:491, :]
#
# dr_2016 = pd.date_range(sent_2016['dtime'].iloc[0], sent_2016['dtime'].iloc[-1])
#
# sent = pd.Series(sent_2016['magnitude'])
# sent_aus = pd.Series(sent_2016_aus['magnitude'])
# sent.index = dr_2016
# sent_aus.index = dr_2016
#
# logsent_aus = np.log10(abs(sent_aus))
# logsent_aus.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
# # idmax1 = logsent_aus.idxmin()
# # logsent_aus[idmax1] = logsent_aus.mean()
# # idmax2 = logsent_aus.idxmin()
# # logsent_aus[idmax2] = logsent_aus.mean()
#
#
# logsent = np.log10(abs(sent))
# logsent.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
#
#
# fig, ax1 = plt.subplots(1,1,figsize=(16,9), dpi= 80)
# ax1.plot(dr_2016, logsent, color='tab:red', label='US Magnitude')
#
# ax1.plot(dr_2016, logsent_aus, color='tab:blue', label='AUS Magnitude')
# ax1.legend()
# ax1.set_title('US & AUS Online Wildfire Social Magnitude for 2016', fontsize=24)
# ax1.set_xlabel('Date', fontsize=20)
# ax1.set_ylabel('Magnitude (Log 10)', fontsize=20)
# plt.grid()
# plt.xticks(rotation=45, ha='right')
# fig.autofmt_xdate()
# ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#
# plt.tight_layout()
# plt.savefig('graphs2/us_aus_magnitude_year_plot.png')
# plt.show()

# calplot.calplot(logsent_aus, cmap='afmhot_r')
# plt.show()
# # plt.savefig('heatmap_aus_data_log10.png')
# # calmap.yearplot(sent)
#

# # fig, ax = plt.subplots(figsize=(20,7))
# # ax = date_heatmap(sentiment_by_day, ax=ax)
# # plt.savefig('A_newgraphs/heatmap.png')
# # plt.show()
#
#
# text = " ".join(val for val in tweets_df.full_text)
# # text = text.replace('tennessee', '').replace('carolina', '')
# # text = text.replace('of', '').replace('in', '').replace('at', '').replace('to', '').replace('the', '').replace('for', '')\
# #     .replace('as', '').replace('a', '').replace('is', '').replace('on', '').replace('re', '').replace('nd', '')\
# #     .replace('et', '').replace('th', '').replace('-', '').replace('in', '')
# # text.replace('wildfires', '').replace('bushfires', '')





# get most popular users
# val_cnt = tweets_df['author_username'].value_counts()
# print("the top 20 users were: {}".format(val_cnt.head(20)))
# print("the top 20 users accounted for {} tweets".format(val_cnt.head(20).sum()))
# print('there are {} users total in the database'.format(len(val_cnt)))
#
# top_20 = val_cnt.sort_values(ascending=False)[0:20]
# fig = top_20.plot(kind='barh', figsize=(10,7)).get_figure()
# fig.savefig('A_newgraphs/top_20_users_aus.png')
# print('saved top 10 users')



# plot sentiment for year
# tweets_crono = tweets_df.sort_values(by='dtime')
# x = tweets_crono['dtime']
# y = tweets_crono['sentiment']
# fig, ax = plt.subplots()
# ax.plot_date(x, y, marker='', linestyle='-')
# ax.grid(b=True, which='minor', color='w', linewidth=0.75)
# print('saving / showing fig')
# fig.autofmt_xdate()
# plt.savefig('A_newgraphs/sentiment_year_us.png')
# print('saved fig')
# plt.show()



# # plot histogram of sneitment vlaues
# tweets_senti = tweets_df.sort_values(by='sentiment')
# x = tweets_senti['sentiment']
# plt.hist(x)
# plt.ylabel('Sentiment', fontsize=20)
# plt.xlabel('Data', fontsize=20)
# plt.suptitle('Histogram of AUS Tweet Sentiment Values', fontsize=18)
# plt.tight_layout()
#
# print('saving / showing fig')
# plt.savefig('graphs2/sentiment_histogram_aus.png')
# print('saved fig')
# plt.show()
#
#
# # plot histogram of magnitude values
# tweets_mag = tweets_df.sort_values(by='magnitude')
# x = tweets_mag['magnitude']
# plt.hist(x)
# plt.ylabel('Magntiude', fontsize=20)
# plt.xlabel('Data', fontsize=20)
# plt.suptitle('Histogram of AUS Tweet Magnitude Values', fontsize=18)
# plt.tight_layout()
#
# print('saving / showing fig')
# plt.savefig('graphs2/mag_histogram_aus.png')
# print('saved fig')
# plt.show()


# plt.figure()
# tweets_df['sentiment'].plot()
# hist.savefig('A_newgraphs/mag_hist_tweets_aus.png')

# print('saved histograms')


# print('splitting text')
# split_text = text.split()
# cnt = Counter(split_text)
# print('getting most common words')
# top_20_words = cnt.most_common(20)
# print('creating graph')
# x = [x[0] for x in top_20_words]
# y = [x[1] for x in top_20_words]
# fig, ax = plt.subplots()
# ax.bar(x, y)
#
# # First, let's remove the top, right and left spines (figure borders)
# # which really aren't necessary for a bar chart.
# # Also, make the bottom spine gray instead of black.
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_color('#DDDDDD')
#
# # Third, add a horizontal grid (but keep the vertical grid hidden).
# # Color the lines a light gray as well.
# ax.set_axisbelow(True)
# ax.yaxis.grid(True, color='#EEEEEE')
# ax.xaxis.grid(False)
#
# ax.set_xlabel('Word', labelpad=15, color='#333333')
# ax.set_ylabel('Count', labelpad=15, color='#333333')
# ax.set_title('Top 20 Words in US Dataset', pad=15, color='#333333',
#              weight='bold')
# plt.xticks(rotation=45, ha='right')
# fig.tight_layout()
#
# print('saving / showing fig')
# plt.savefig('A_newgraphs/top_words_us.png')
# print('saved fig')
# plt.show()
# print(top_20_words)





# word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
# plt.imshow(word_cloud, interpolation='bilinear')
# plt.axis("off")
# plt.show()
# plt.title('Word Cloud For US Tweet Data (including Keywords)')
#
# word_cloud.to_file("A_newgraphs/us_wordcloud_w_keywords.png")
# tweets_df['polarity'] = tweets_df['lemmatized'].apply(lambda x:TextBlob(x).sentiment.polarity)

# df_grouped=tweets_df[['fire_ID','lemmatized']].groupby(by='fire_ID').agg(lambda x:' '.join(x))
# df_grouped.head()
# cv=CountVectorizer(analyzer='word')
# data=cv.fit_transform(tweets_df['lemmatized'])
# df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names())
# df_dtm.index=tweets_df.index
# df_dtm.head(3)
#
# # Function for generating word clouds
# def generate_wordcloud(data,title):
#     wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
#     plt.figure(figsize=(10,8))
#     plt.imshow(wc, interpolation='bilinear')
#     plt.axis("off")
#     plt.title('\n'.join(wrap(title,60)),fontsize=13)
#     plt.show()
#
# # Transposing document term matrix
# df_dtm=df_dtm.transpose()
#
# # Plotting word cloud for each product
# for index,product in enumerate(df_dtm.columns):
#     generate_wordcloud(df_dtm[product].sort_values(ascending=False),product)