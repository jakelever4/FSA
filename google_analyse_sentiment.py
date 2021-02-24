from google.cloud import language_v1
import Sentiment
from google.api_core import exceptions


def split_text_context(text_content, n_splits):
    score = 0
    pos_sentiment = 0
    neg_sentiment = 0
    magnitude = 0

    text_content = text_content.split('.')
    split_list = split(text_content, n_splits)
    for list in split_list:
        string = ''
        for item in list:
            string += item

        sentiment = analyze(string)
        score += sentiment.score
        pos_sentiment += sentiment.overall_positive_sentiment
        neg_sentiment += sentiment.overall_negative_sentiment
        magnitude += sentiment.magnitude

    sent = Sentiment.Overall_Sentiment(score, pos_sentiment, neg_sentiment, magnitude, None)
    return sent


def analyze(text_content):
    client = language_v1.LanguageServiceClient.from_service_account_json("fire-sentiment-analysis-bf24604da498.json")

    type_ = language_v1.Document.Type.PLAIN_TEXT
    document = {"content": text_content, "type_": type_}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    # response = client.analyze_sentiment(request={'document': document, 'encoding_type': encoding_type})
    try:
        # try to get a response
        response = client.analyze_sentiment(request={'document': document, 'encoding_type': encoding_type})
    except exceptions.InvalidArgument as e:
        # if there is an invalid exception
        print('google.api_core.exceptions.InvalidArgument: 400')
        print(e.message)
        # check if document size is big (140,000 = 1,000 x 140 char tweets for a day) split document into 10 slices
        if len(text_content) > 140000:
            print('DOCUMENT TOO BIG. PERFORMING SLICING')
            sent = split_text_context(text_content, 10)
        else:
            print('doc size normal, other invalid argument')
            sent = Sentiment.Overall_Sentiment(0, 0, 0, 0, None)

        return sent

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


def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


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
