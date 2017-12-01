import string
from functools import lru_cache

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

TOKENIZER = TweetTokenizer()
STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = string.punctuation


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        # return [t for t in stems if t not in PUNCTUATION]

def clean_tweet(tweet):
    '''Tokenize tweet and remove stopwords and punctuation.

    Parameters
    ----------
    tweet : str

    Returns
    -------
    list
    '''
    # tokenize
    wordlist = TOKENIZER.tokenize(tweet)
    # remove stopwords
    wordlist = [word for word in wordlist if word not in STOPWORDS]
    # remove punctuation
    wordlist = [word for word in wordlist if word not in PUNCTUATION]

    return(wordlist)

def lemmatize_tweet(wordlist):
    wordlist = [LEMMATIZER.lemmatize(word) for word in wordlist]
    return ''.join(wordlist)



def w2v_embed_tweet(wordlist, embedding_dimension):
    '''Embed wordlist using Google Word2Vec as average of all words in list.

    Parameters
    ----------
    wordlist : list of str
    embedding_dimension : int
        minimum 0, maxiumum 300

    Returns
    -------
    numpy array
    '''
    w2v_model = get_w2v_model()
    vector = np.zeros(embedding_dimension)
    # vectorize tweet as average of word2vec vectors
    num_words = 0
    for word in wordlist:
        try:
            vector += w2v_model[word][:embedding_dimension]
            num_words += 1
        except KeyError:
            # ignore unrecognized words
            continue

    if num_words != 0:
        vector /= 1.0 * num_words

    return vector

@lru_cache(maxsize=1)
def get_w2v_model():
    model = KeyedVectors.load('w2v_preloaded_model')
    return model