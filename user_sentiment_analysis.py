import pandas as pd
import google_analyse_sentiment
import google_analyse_entities

username = 'TwitterDev'

try:
    tweets_df = pd.read_csv('tweets/{}_tweets.csv'.format(username))
except FileNotFoundError:
    print('username not found in tweets folder')
    tweets_df = pd.DataFrame

merged_text = ''

for index, row in tweets_df.iterrows():
    text = row['text']
    merged_text += " "
    merged_text += text

analysis = google_analyse_sentiment.analyze(merged_text)
print(analysis)

entities = google_analyse_entities.analyze_entities(merged_text)
linked_entities = []
for entity in entities:
    if entity.salience != 0 and entity.metadata:
        print(entity)
