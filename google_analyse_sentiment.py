from google.cloud import language_v1
import Sentiment
from google.api_core import exceptions


def analyze(text_content):
    client = language_v1.LanguageServiceClient.from_service_account_json("fire-sentiment-analysis-bf24604da498.json")

    type_ = language_v1.Document.Type.PLAIN_TEXT
    document = {"content": text_content, "type_": type_}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    try:
        response = client.analyze_sentiment(request={'document': document, 'encoding_type': encoding_type})
    except exceptions.InvalidArgument:
        raise Exception(response)

    # Print the results
    # Score is the overall emotional learning of the text
    # Magnitude indicates the overall strength of the emotion.
    #print_result(response)

    score = response.document_sentiment.score
    magnitude = response.document_sentiment.magnitude
    sentences = []
    for index, sentence in enumerate(response.sentences):
        sentence_sentiment = Sentiment.Sentence_Sentiment(index, sentence.sentiment.score, sentence.sentiment.magnitude)
        sentences.append(sentence_sentiment)

    overall_sent = 0
    overall_mag = 0
    overall_positive_sentiment = 0
    overall_negative_sentiment = 0
    for sentence in sentences:
        overall_sent += sentence.score
        if sentence.score <= 0:
            overall_negative_sentiment += sentence.score
        else:
            overall_positive_sentiment += sentence.score
        overall_mag += sentence.magnitude

    sentimemt = Sentiment.Overall_Sentiment(overall_sent, overall_positive_sentiment, overall_negative_sentiment, overall_mag, sentences)
    print("Overall Sentiment: score of {} with magnitude of {}".format(overall_sent, overall_mag))

    return sentimemt


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
