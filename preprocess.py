import string
import logging

import gensim
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


# load in data
path_to_data = 'result.txt'
all_tweets_df = pd.read_table(path_to_data, names = ['ID', 'class', 'tweet'])
print('Read in', len(all_tweets_df), 'tweets')


# tokenize tweets
tokenizer = TweetTokenizer()
all_tweets_df['tweet'] = all_tweets_df['tweet'].apply(tokenizer.tokenize)

# remove stopwords
stop = set(stopwords.words('english'))
all_tweets_df['tweet'] = all_tweets_df['tweet'].apply(lambda wordlist:
                                                      [word for word in wordlist if word not in stop])
# remove punctuation
punctuation = string.punctuation
all_tweets_df['tweet'] = all_tweets_df['tweet'].apply(lambda wordlist:
                                                      [word for word in wordlist if word not in punctuation])
# embed using Word2Vec
embedding_dimension = 300
print('Embedding tweets as', embedding_dimension, 'dimensional vectors')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin.gz', binary = True)

def build_vector(wordlist, embedding_dimension):
    vector = np.zeros(embedding_dimension)
    # vectorize tweet as average of word2vec vectors
    num_words = 0
    for word in wordlist:
        try:
            vector += model[word][:embedding_dimension]
            num_words += 1
        except KeyError:
            # ignore unrecognized words
            continue
    if num_words != 0:
        vector /= 1.0 * num_words
    return list(vector)

# build vectors
all_tweets_df['vector'] = all_tweets_df['tweet'].apply(lambda wordlist:
                                                       build_vector(wordlist, embedding_dimension))

# pickle data frame
all_tweets_df.to_pickle('pickled/all_tweets_df')
