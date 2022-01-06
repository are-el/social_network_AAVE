import numpy as np
import pandas as pd
import csv
import codecs
import sys
import re
from more_itertools import sliced
import explore_tweets
import time

#LOL THIS IS A SCRIPT

###############################
#            GLOVE            #
###############################
def convert_to_binary(embedding_path):
    f = codecs.open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []
    with codecs.open(embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        count = 0
        for line in f:
            splitlines = line.split()
            try:
                vocab_write.write(splitlines[0].strip())
                vocab_write.write("\n")
                wv.append([float(val) for val in splitlines[1:]])
            except IndexError or ValueError:
                pass
        count += 1
    np.save(embedding_path + ".npy", np.array(wv))


def load_word_emb_binary(embedding_file_name_w_o_suffix):
    print("Loading binary word embedding from {0}.vocab and {0}.npy".format(embedding_file_name_w_o_suffix))
    with codecs.open(embedding_file_name_w_o_suffix + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    wv = np.load(embedding_file_name_w_o_suffix + '.npy', allow_pickle=True)
    word_embedding_map = {}

    for i, w in enumerate(index2word):
        try:
            word_embedding_map[w] = wv[i]
        except IndexError:
            pass
    print("GloVe model loading completed")
    return word_embedding_map

def get_glove(sentence, model):
    return np.array([model.get(val, np.zeros(100)) for val in sentence.split()], dtype=np.float64)

def test_glove():
    #convert_to_binary("GloVe/glove.840B.300d")
    model = load_word_emb_binary("GloVe/glove.840B.300d")
    print(get_glove("testing", model))


###############################
#             CSV             #
###############################

def populate_tw_text():
    # read dev set to dataframe
    pd.set_option('display.max_columns', None)
    df = pd.read_csv("data/dev.csv", usecols=['tweet_id','label'])
    new_df = pd.DataFrame(columns=['tweet_id', 'tweet_text', 'label'])
    CHUNK_SIZE = 100

    index_slices = sliced(range(len(df)), CHUNK_SIZE)

    #get tweet text via twitter API
    for index_slice in index_slices:
        chunk_df = df.iloc[index_slice]
        ids = chunk_df['tweet_id'].to_list()
        tweets = explore_tweets.get_tweet_list_from_ids(ids)
        tmp_dict = {}
        print(len(new_df))
        for tweet in tweets:
            tmp_dict['tweet_id'] = tweet.id
            tmp_dict['tweet_text'] = tweet.text
            #find row from chunk_df with id to retrieve lost label
            row = chunk_df.loc[chunk_df['tweet_id'] == tweet.id]
            tmp_dict['label'] = row['label']
            new_df = new_df.append(tmp_dict, ignore_index=True)

    print(new_df.head())
    #write dataframe to csv to save
    new_df.to_csv("data/dev_text.csv")



###############################
#           TW TEXT           #
###############################
# from https://gist.github.com/ppope/0ff9fa359fb850ecf74d061f3072633a

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()


# text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
# tokens = tokenize(text)
# print(tokens)

#NEXT STEP
#1. GONNA HAVE TO DO SOMETHING TO ACCOMODATE THE ABILITY TO TAKE SPACES IN GLOVE INPUT FUNCTION
#2. WHEN USING TOKENIZE, JUST REMOVE THE BRACKETED THINGS I GUESS
