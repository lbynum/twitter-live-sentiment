import pandas as pd
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# from featurizers import clean_tweet, lemmatize_tweet
# from feature_transformer import FunctionToTransformer
from featurizers import LemmaTokenizer

# load data ####################################################################
path_to_data = 'result.txt'
print('Reading data...')
all_tweets_df = pd.read_table(path_to_data, names = ['ID', 'sentiment', 'tweet'])

# remove neutral tweets
all_tweets_df = all_tweets_df[all_tweets_df['sentiment'] != 'neutral']

# convert sentiment labels to binary (1 = positive, 0 = negative)
all_tweets_df['sentiment'] = (all_tweets_df['sentiment'] == 'positive').astype(int)

# split features and labels
X = all_tweets_df['tweet']
y = all_tweets_df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    shuffle=True)
print('Training data dimensions n,d = {}'.format(X_train.shape))
print('Test data dimensions n,d = {}'.format(X_test.shape))
# define pipelines #############################################################
################################################################################

# tfidf
tfidf_MNB = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(),
                             stop_words='english',
                             strip_accents='unicode',
                             max_df=0.5)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])
tfidf_SVM = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(),
                             stop_words='english',
                             strip_accents='unicode',
                             max_df=0.5)),
    ('tfidf', TfidfTransformer()),
    ('clf', svm.SVC())
])

# bow
bow_MNB = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(),
                             stop_words='english',
                             strip_accents='unicode',
                             max_df=0.5)),
    ('clf', MultinomialNB())
])


# ngram
twogram_MNB = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(),
                             stop_words='english',
                             strip_accents='unicode',
                             ngram_range=(1,2),
                             max_df=0.5)),
    ('clf', MultinomialNB())
])




# run ##########################################################################
print('Training TF-IDF Multinomial Naive Bayes...')
tfidf_MNB.fit(X_train, y_train)
print('Classifying TF-IDF Multinomial Naive Bayes...')
y_prediction = tfidf_MNB.predict(X_test)
print(y_prediction)
report = classification_report(y_test, y_prediction)
print(report)

print('Training BOW Multinomial Naive Bayes...')
bow_MNB.fit(X_train, y_train)
print('Classifying BOW Multinomial Naive Bayes...')
y_prediction = bow_MNB.predict(X_test)
report = classification_report(y_test, y_prediction)
print(report)

print('Training 2-gram Multinomial Naive Bayes...')
twogram_MNB.fit(X_train, y_train)
print('Classifying 2-gram Multinomial Naive Bayes...')
y_prediction = twogram_MNB.predict(X_test)
report = classification_report(y_test, y_prediction)
print(report)



