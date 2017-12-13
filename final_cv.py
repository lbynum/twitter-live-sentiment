import pickle
import pandas as pd

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import GridSearchCV


path_to_data = 'result.txt'
all_tweets_df = pd.read_table(path_to_data, names = ['ID', 'sentiment', 'tweet'])

# remove neutral tweets (for now)
all_tweets_df = all_tweets_df[all_tweets_df['sentiment'] != 'neutral']

# convert sentiment labels to binary (1 = positive, 0 = negative)
all_tweets_df['sentiment'] = (all_tweets_df['sentiment'] == 'positive').astype(int)

# split features and labels
X = all_tweets_df['tweet']
y = all_tweets_df['sentiment']

# separate the training and test data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    shuffle=True)

print('Training data dimensions n,d = {}'.format(X_train.shape))
print('Test data dimensions n,d = {}'.format(X_test.shape))



class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# embedding methods
embedding_tuple = ('vect', CountVectorizer(tokenizer=LemmaTokenizer(),
                             stop_words='english',
                             strip_accents='unicode',
                             max_df=0.5))
tfidf_tuple = ('tfidf', TfidfTransformer())

# classifiers
mnb_tuple = ('clf', MultinomialNB())
bnb_tuple = ('clf', BernoulliNB())
svm_tuple = ('clf', SVC())
logreg_tuple = ('clf', LogisticRegression())
sgd_tuple = ('clf', SGDClassifier())

### Classifier Pipelines
# Multinomial Naive Bayes
tfidf__mnb_classifier = Pipeline([embedding_tuple, tfidf_tuple, mnb_tuple])
bow__mnb_classifier = Pipeline([embedding_tuple, mnb_tuple])

# Bernoulli Naive Bayes
tfidf__bnb_classifier = Pipeline([embedding_tuple, tfidf_tuple, bnb_tuple])
bow__bnb_classifier = Pipeline([embedding_tuple, bnb_tuple])

# SVC (linear and rbf)
tfidf__svm_classifier = Pipeline([embedding_tuple, tfidf_tuple, svm_tuple])
bow__svm_classifier = Pipeline([embedding_tuple, svm_tuple])

# Logistic regression
tfidf__logreg_classifier = Pipeline([embedding_tuple, tfidf_tuple, logreg_tuple])
bow__logreg_classifier = Pipeline([embedding_tuple, logreg_tuple])

# Stochastic gradient descent
tfidf__sgd_classifier = Pipeline([embedding_tuple, tfidf_tuple, sgd_tuple])
bow__sgd_classifier = Pipeline([embedding_tuple, sgd_tuple])

# First specify parameter dictionaries for the various classifiers
parameters = {
    'vect__max_df': (0.5, 0.75),
    'vect__max_features': (None, 2000, 5000),
    'vect__ngram_range': ((1, 1), (1, 2), (2,2)),  # unigrams or bigrams
}

parameters_svm = {
    'vect__max_df': (0.5, 0.75),
    'vect__max_features': (None, 2000, 5000),
    'vect__ngram_range': ((1, 1), (1, 2), (2,2)),  # unigrams or bigrams
    'clf__kernel': ('linear', 'rbf'),
    'clf__C': (0.01, 0.1, 1, 10, 100),
    'clf__gamma': (0.01, 0.1, 1, 10, 100),
}

parameters_lr = {
    'vect__max_df': (0.5, 0.75),
    'vect__max_features': (None, 2000, 5000),
    'vect__ngram_range': ((1, 1), (1, 2), (2,2)),  # unigrams or bigrams
    'clf__C': (0.01, 0.1, 1, 10, 100),
}

# Multinomial Bayes with TF-IDF
print('Tuning Multinomial Bayes with TF-IDF')
grid_search = GridSearchCV(tfidf__mnb_classifier, parameters, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Multinomial Bayes with Bag of Words
print('Tuning Multinomial Bayes with Bag of Words')
grid_search = GridSearchCV(bow__mnb_classifier, parameters, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Logistic Regression with TF_IDF
print('Tuning Logistic Regression with TF_IDF')
grid_search = GridSearchCV(tfidf__logreg_classifier, parameters_lr, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters_lr.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Logistic Regression with Bag of Words
print('Logistic Regression with Bag of Words')
grid_search = GridSearchCV(bow__logreg_classifier, parameters_lr, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters_lr.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# SGD with TF_IDF
print('SGD with TF_IDF')
grid_search = GridSearchCV(tfidf__sgd_classifier, parameters, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# SGD with Bag of Words
print('SGD with Bag of Words')
grid_search = GridSearchCV(bow__sgd_classifier, parameters, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# SVM with TF_IDF
print('Tuning SVM with TF_IDF')
grid_search = GridSearchCV(tfidf__svm_classifier, parameters_svm, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters_svm.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# SVM with Bag of Words
print('SVM with Bag of Words')
grid_search = GridSearchCV(bow__svm_classifier, parameters_svm, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters_svm.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

Fit each of the classifiers and pickle the fitted models
print('Fitting Multinomial Bayes with Tf-idf')
tfidf_mnb_classifier.fit(X_train, y_train)
pred = tfidf_mnb_classifier.predict(X_test)
print('here is the mnb tf-idf accuracy, precision, recall: ', [accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred)])

print('Fitting Multinomial Bayes with bow')
bow_mnb_classifier.fit(X_train, y_train)
pred = bow_mnb_classifier.predict(X_test)
print('here is the mnb bow accuracy, precision, recall: ', [accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred)])

print('Fitting Logistic Regression with tf-idf')
tfidf_logreg_classifier.fit(X_train, y_train)
pred = tfidf_logreg_classifier.predict(X_test)
print('here is the log reg tf-idf accuracy, precision, recall: ', [accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred)])

print('Fitting Logistic Regression with bow')
bow_logreg_classifier.fit(X_train, y_train)
pred = bow_logreg_classifier.predict(X_test)
print('here is the log reg bow accuracy, precision, recall: ', [accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred)])

print('Fitting SGD with tf-idf')
tfidf_sgd_classifier.fit(X_train, y_train)
pred = tfidf_sgd_classifier.predict(X_test)
print('here is the sgd tf-idf accuracy, precision, recall: ', [accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred)])

print('Fitting SGD with bow')
bow_sgd_classifier.fit(X_train, y_train)
pred = bow_sgd_classifier.predict(X_test)
print('here is the sgd bow accuracy, precision, recall: ', [accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred)])

print('Fitting SVM with tf-idf')
tfidf_svm_classifier.fit(X_train, y_train)
pred = tfidf_svm_classifier.predict(X_test)
print('here is the svm tf-idf accuracy, precision, recall: ', [accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred)])

print('Fitting SVM with bow')
bow_svm_classifier.fit(X_train, y_train)
pred = bow_svm_classifier.predict(X_test)
print('here is the svm bow accuracy, precision, recall: ', [accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred)])

dictionary that contains model name as key and fitted model as value
models_dict = {'tfidf__mnb_classifier': tfidf_mnb_classifier, 'bow__mnb_classifier': bow_mnb_classifier, 'tfidf__logreg_classifier': tfidf_logreg_classifier,'bow__logreg_classifier': bow_logreg_classifier, 'tfidf_sgd_classifier': tfidf_sgd_classifier, 'bow__sgd_classifier': bow_sgd_classifier,'tfidf__svm_classifier': tfidf_svm_classifier, 'bow__svm_classifier': bow_svm_classifier}

for name, model in models_dict.items():
    pickle.dump(model, open('pickled_classifiers/' + name + '.pickle', 'wb'))
