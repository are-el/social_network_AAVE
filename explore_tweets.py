import tweepy
import pandas as pd
import ACCESSTOKENS
import numpy as np

client = tweepy.Client(bearer_token=ACCESSTOKENS.tw_bearer_token)

def get_tweet_list_from_ids(ids_list):
    tweets = client.get_tweets(ids=ids_list)
    return tweets.data


def get_tweet(id):
    tweet = client.get_tweet(id=id)
    if tweet is None:
        return np.nan
    else:
        return tweet.data


def test_train_split():
    df = pd.read_csv("data/fdcl18_raw.csv")
    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= 0.7

    train = df[msk]
    test = df[~msk]
    train.to_csv(path_or_buf="data/train.csv", index=False)
    test.to_csv(path_or_buf="data/test.csv", index=False)

def test_dev_split():
    df = pd.read_csv("data/test.csv")
    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= 0.5

    train = df[msk]
    test = df[~msk]
    train.to_csv(path_or_buf="data/dev.csv", index=False)
    test.to_csv(path_or_buf="data/test.csv", index=False)


