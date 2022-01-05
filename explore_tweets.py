import tweepy
import pandas as pd
import ACCESSTOKENS
import numpy as np

def get_one_tweet_test():
    client = tweepy.Client(bearer_token=ACCESSTOKENS.tw_bearer_token)

    ids = ['1409935014725177344']
    tweets = client.get_tweets(ids=ids)
    for tweet in tweets.data:
        print(tweet.text)

def test_train_split():
    df = pd.read_csv("data/fdcl18_raw.csv")
    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= 0.7

    train = df[msk]
    test = df[~msk]
    train.to_csv(path_or_buf="data/train.csv", index=False)
    test.to_csv(path_or_buf="data/test.csv", index=False)


get_one_tweet_test()