import google_analyse_sentiment
import matplotlib.pyplot as plt
import pandas as pd

colours = ['purple', 'pink', 'green', 'black', 'blue', 'yellow', 'red']

class Tweet:
    def __init__(self, author_id, created_at, entities, geo, tweet_id, retweet_count, reply_count, like_count,
                 quote_count, is_rt, text, rule, rule_id, sentiment = None, magnitude = None):
        self.author_id = author_id
        # self.author_username = twitter_user_lookup.lookup_user_id(self.author_id)
        self.created_at = created_at
        self.entities = entities
        self.geo = geo
        self.tweet_id = tweet_id
        self.retweet_count = retweet_count
        self.reply_count = reply_count
        self.like_count = like_count
        self.quote_count = quote_count
        self.is_rt = is_rt
        self.text = text
        self.rule = rule
        self.rule_id = rule_id
        self.sentiment = sentiment
        self.magnitude = magnitude


class Tweet_Rule:
    def __init__(self, rule_name, rule_id, tweets_for_rule, s_vals, m_vals):
        self.rule_name = rule_name
        self.rule_id = rule_id
        self.tweets_for_rule = tweets_for_rule
        self.s_vals = s_vals
        self.m_vals = m_vals

rules = []
binned_tweets = []

s_avg = []



# define and adjust figure
fig = plt.figure(figsize=(12,6), facecolor='#DEDEDE')
ax = plt.subplot(121)
ax1 = plt.subplot(122)
plt.style.use('seaborn-whitegrid')


def find_bin(tweet, binned_tweets):
    for bin in binned_tweets:
        bin_rule = bin.rule_name
        if bin_rule == tweet.rule:
            return bin
    print('cannot find matching rule')
    return None


def add_or_create_bin(tweet):
    if tweet.rule not in rules:
        new_bin = Tweet_Rule(tweet.rule, tweet.rule_id, [tweet], [tweet.sentiment], [tweet.magnitude])
        binned_tweets.append(new_bin)
        rules.append(tweet.rule)
    else:
        bin = find_bin(tweet, binned_tweets)
        if type(bin.tweets_for_rule) is not list:
            bin.tweets_for_rule = [bin.tweets_for_rule]
            bin.tweets_for_rule.append(tweet)
        else:
            bin.tweets_for_rule.append(tweet)

    return binned_tweets


def update(new_tweet):

    # run analysis and append to tweet object
    analysis = google_analyse_sentiment.analyze(new_tweet.text)
    new_tweet.sentiment = analysis.score
    new_tweet.magnitude = analysis.magnitude
    print('New Tweet: {} . Sentiment: {} . Magnitude: {} . Created at {} for rule: {}'.format(new_tweet.text,
                                                                                              new_tweet.sentiment,
                                                                                              new_tweet.magnitude,
                                                                                              new_tweet.created_at,
                                                                                              new_tweet.rule))
    # add the tweet to the appropriate bin according to rule
    bins = add_or_create_bin(new_tweet)

    # clear axis
    ax.cla()
    ax1.cla()

    for index, bin in enumerate(bins):
        # order tweets by created at time
        bin.tweets_for_rule.sort(key=lambda x: x.created_at, reverse=True)
        # init vis vectors for each bin
        s_vals = []
        m_vals = []
        ctimes = []
        s_av = []
        m_av = []
        for ind, tweet in enumerate(bin.tweets_for_rule):
            # add all tweet values to vectors for vis
            s_vals.append(tweet.sentiment)
            m_vals.append(tweet.magnitude)
            ctimes.append(tweet.created_at)


        data = {'s_vals': s_vals, 'm_vals': m_vals, 'ctimes': ctimes}
        df = pd.DataFrame(data)
        df['s_vals_rolling_average'] = df['s_vals'].rolling(5).mean()
        df['s_vals_rolling_average'] = df['s_vals_rolling_average'].fillna(0)
        # print(df)

            # TODO: add rolling average here as a curve
            # average_sentiment = sum(s_vals_bin) / len(s_vals_bin)
            # s_av_bin.append(average_sentiment)
            # average_magnitude = sum(m_vals_bin) / len(m_vals_bin)
            # m_av_bin.append(average_magnitude)
            #
            # prev_tweet_time = parser.parse(tweets_for_bin[ind - 1].created_at)
            # tweet_time = parser.parse(tweet.created_at)
            # if prev_tweet_time == tweet_time:
            #     merged_sentiment, merged_magnitude = merge_values(tweets_for_bin[ind - 1], tweet)
            #     s_av_bin[-1] = merged_sentiment
            #     m_av_bin[-1] = merged_magnitude

        # reorder values so that they are visualised in chronological order
        # sometimes the filtered stream delivers these in the wrong order
        # ctimes, s_vals = (list(t) for t in zip(*sorted(zip(ctimes,s_vals))))
        # ctimes, m_vals = (list(t) for t in zip(*sorted(zip(ctimes,m_vals))))

        bin_colour = colours[index]

        # set up axis
        ax.xaxis_date()
        ax.set_title("Sentiment")
        ax1.set_title("Magnitude")
        ax.set(xlabel='Time', ylabel='Sentiment')
        ax1.set(xlabel='Time', ylabel='Magnitude')
        ax.set_ylim([-1, 1])
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
        fig.suptitle('Real Time Sentiment Tracking')

        ax.scatter(ctimes, s_vals, color=bin_colour, marker='.', label=bin.rule_name)
        ax.plot(ctimes, df['s_vals_rolling_average'], color=bin_colour)
        ax.legend()

        ax1.scatter(ctimes, m_vals, color=bin_colour, label=bin.rule_name)
        # ax1.plot(ctimes, m_av_bin, color=bin_colour, label=bin.rule_name)
        ax1.legend()

    plt.pause(0.01)