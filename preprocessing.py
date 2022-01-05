import numpy as np
import pandas as pd
import csv
import codecs


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


#convert_to_binary("GloVe/glove.840B.300d")
model = load_word_emb_binary("GloVe/glove.840B.300d")
print(get_glove("testing", model))




