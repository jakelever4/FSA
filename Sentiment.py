class Overall_Sentiment:
    def __init__(self, score, magnitude, sentences):
        self.score = score
        self.magnitude = magnitude
        self.sentences = sentences

    def __str__(self):
        return 'Overall Score: {} \n Overall Magnitude: {}'.format(self.score, self.magnitude)


class Sentence_Sentiment:
    def __init__(self, index, score, magnitude):
        self.index = index
        self.score = score
        self.magnitude = magnitude

    def __str__(self):
        return 'Sentence Index: {}. Score: {}. Magnitude: {}'.format(self.index, self.score, self.magnitude)