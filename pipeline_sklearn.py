import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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
# define pipeline ##############################################################
# tweet_cleaner = FunctionToTransformer(
#     clean_tweet,
#     lemmatize_tweet
# )


tfidf_classifier = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(),
                             stop_words='english',
                             strip_accents='unicode',
                             max_df=0.5)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

bow_classifier = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(),
                             stop_words='english',
                             strip_accents='unicode',
                             max_df=0.5)),
    ('clf', MultinomialNB())
])

# run ##########################################################################
print('Training TF-IDF Multinomial Naive Bayes...')
tfidf_classifier.fit(X_train, y_train)
print('Classifying TF-IDF Multinomial Naive Bayes...')
y_prediction = tfidf_classifier.predict(X_test)
report = classification_report(y_test, y_prediction)
print(report)

print('Doing cross validation on tf-idf')
scores_acc = cross_val_score(tfidf_classifier, X, y, cv=10)
print 'accuracy', scores_acc
scores_prec = cross_val_score(tfidf_classifier, X, y, cv=10, scoring = 'precision')
print 'precision', scores_prec
scores_rec = cross_val_score(tfidf_classifier, X, y, cv=10, scoring = 'recall')
print 'recall', scores_rec

print('Training BOW Multinomial Naive Bayes...')
bow_classifier.fit(X_train, y_train)
print('Classifying BOW Multinomial Naive Bayes...')
y_prediction = bow_classifier.predict(X_test)
report = classification_report(y_test, y_prediction)
print(report)

print('Doing cross validation on BOG')
scores_acc = cross_val_score(bow_classifier, X, y, cv=10)
print 'accuracy', scores_acc
scores_prec = cross_val_score(bow_classifier, X, y, cv=10, scoring = 'precision')
print 'precision', scores_prec
scores_rec = cross_val_score(bow_classifier, X, y, cv=10, scoring = 'recall')
print 'recall', scores_rec

# hyperparameter tuning
parameters = {
    'vect__max_df': (0.5, 0.75),
    'vect__max_features': (None, 2000, 5000),
    'vect__ngram_range': ((1, 1), (1, 2), (2,2)),  # unigrams or bigrams
}

grid_search = GridSearchCV(tfidf_classifier, parameters, n_jobs=-1, verbose=1)

print("Performing grid search...")
grid_search.fit(X_train, y_train)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
print('finished with hyperparameter tuning')

y_prediction = grid_search.predict(X_test)




