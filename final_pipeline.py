import pickle
import pandas as pd
import json
import random

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from nltk.classify import ClassifierI
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]



tfidf_tuple = ('tfidf', TfidfTransformer())
mnb_tuple = ('clf', MultinomialNB())
sgd_tuple = ('clf', SGDClassifier())

### Classifier Pipelines
# Multinomial Naive Bayes
tfidf_mnb_classifier = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', strip_accents='unicode', max_df=0.5, max_features=2000, ngram_range=(1, 2))), tfidf_tuple, mnb_tuple])

bow_mnb_classifier = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', strip_accents='unicode', max_df=0.5, max_features=5000, ngram_range=(1, 2))), mnb_tuple])

# Logistic regression
tfidf_logreg_classifier = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', strip_accents='unicode', max_df=0.5, max_features=None, ngram_range=(1, 2))), tfidf_tuple, ('clf', LogisticRegression(C=100))])

bow_logreg_classifier = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', strip_accents='unicode', max_df=0.5, max_features=5000, ngram_range=(1, 2))),('clf', LogisticRegression(C=1))])

# Stochastic gradient descent
tfidf_sgd_classifier = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', strip_accents='unicode', max_df=0.75, max_features=None, ngram_range=(1, 2))), tfidf_tuple, sgd_tuple])
bow_sgd_classifier = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', strip_accents='unicode', max_df=0.5, max_features=None, ngram_range=(1, 2))), sgd_tuple])

# Support Vector Machines
tfidf_svm_classifier = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', strip_accents='unicode', max_df=0.5, max_features=None, ngram_range=(1, 2))), tfidf_tuple, ('clf', SVC(C=10, gamma=0.01, kernel='linear'))])
bow_svm_classifier = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', strip_accents='unicode', max_df=0.5, max_features=None, ngram_range=(1, 1))), ('clf', SVC(C=10, gamma=0.01, kernel='rbf'))])

models_dict = {'bow__mnb_classifier': bow_mnb_classifier, 'tfidf__logreg_classifier': tfidf_logreg_classifier,'bow__logreg_classifier': bow_logreg_classifier, 'tfidf_sgd_classifier': tfidf_sgd_classifier, 'bow__sgd_classifier': bow_sgd_classifier,'tfidf__svm_classifier': tfidf_svm_classifier, 'bow__svm_classifier': bow_svm_classifier}

class VoteClassifier(ClassifierI):
    def __init__(self, classifiers):
        self._classifiers = classifiers
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.predict(features)[0]
            votes.append(v)
        mode = max(set(votes), key=votes.count)
        return mode
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.predict(features)[0]
            votes.append(v)
        mode = max(set(votes), key=votes.count)
        choice_votes = votes.count(mode)
        conf = choice_votes / len(votes)
        return conf

CONSUMER_KEY = 'tOpKHcNzNpu88PSZxAzCI87Ne'
CONSUMER_SECRET = 'EhSEN89oydJi058EkQP3iMjsVlYw6yLYZ2Uq2UAVSWS43wXju9'
ACCESS_TOKEN = '37501551-hUS1bgjvyBq9H1pplnXkQb1rBIqNfNzPsBHFtx8dw'
ACCESS_SECRET = '0MI19XnM6FXT8D8LWM70KbHYZDp5GefyZpYwD6hUUvtSD'

models = []
for name in models_dict.keys():
    model = pickle.load(open('pickled_classifiers/' + name + '.pickle', 'rb'))
    models.append(model)

VOTECLASSIFIER = VoteClassifier(models)

class TwitterListener(StreamListener):
    def on_data(self, data):
        try:
            json_data = json.loads(data)
            tweet = json_data['text']
            sentiment = 'positive' if VOTECLASSIFIER.classify([tweet]) else 'negative'
            confidence = VOTECLASSIFIER.confidence([tweet])
            # print(sentiment, confidence)

            with open('twitter-out.txt', 'a') as output:
                output.write(str(sentiment) + ',' + str(confidence))
                output.write('\n')
        except Exception as error:
            print(error)
            pass

        return True

    def on_error(self, status):
        print(status)
        pass

# authenticate
auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)


print('Classifying live Twitter data')

# clear output file
f = open('twitter-out.txt', 'w')
f.close()

# create streaming object
twitterStream = Stream(auth, TwitterListener())
twitterStream.filter(languages=['en'], track=['trump', 'donald trump'])
# twitterStream.filter(languages=['en'], track=['doug jones'])
# twitterStream.filter(languages=['en'], track=['roy moore'])
# twitterStream.filter(languages=['en'], track=['coco', 'pixar'])
