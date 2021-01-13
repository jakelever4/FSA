from google.cloud import language
from google.cloud.language import types
from google.cloud.language import enums

import Sentiment


def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        # print(
        #     "Sentence {} has a sentiment score of {}".format(index, sentence_sentiment)
        # )

    print(
        "Overall Sentiment: score of {} with magnitude of {}".format(score, magnitude)
    )
    return 0


def analyze(text_string):
    client = language.LanguageServiceClient.from_service_account_json("fire-sentiment-analysis-bf24604da498.json")

    document = types.Document(content=text_string, type=enums.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(document=document)

    # Print the results
    # Score is the overall emotional learning of the text
    # Magnitude indicates the overall strength of the emotion.
    print_result(annotations)

    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude
    sentences = []
    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = Sentiment.Sentence_Sentiment(index, sentence.sentiment.score, sentence.sentiment.magnitude)
        sentences.append(sentence_sentiment)

    sentimemt = Sentiment.Overall_Sentiment(score, magnitude, sentences)

    return sentimemt
