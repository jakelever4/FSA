import tweepy


# Set twitter credentials
CONSUMER_KEY = 'nGWxOlRumgcq8vJgowFViH082'
CONSUMER_SECRET = 'Qh0thqFeFeStsKBCgDGJnwBsy8nuwRG4EYeNas2Dslp5nIwCOt'
ACCESS_TOKEN = '241460215-m3TjNEuV1NkvUxfMMNvNz3vTGxnXi5ZXs4PUuPGv'
ACCESS_TOKEN_SECRET = '6OZ6umC3HuaipM9ybL0rx2Asypigw0pXYUJMLfH7d20ux'


def connect_to_twitter():
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    # Create API object
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    try:
        api.verify_credentials()
        print("Twitter Authentication OK")
    except:
        print("Error during authentication")

    return api