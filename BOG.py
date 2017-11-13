import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

def main():
    df = pd.read_table('result.txt', sep='\t', delimiter=None, header='infer', names = ['id', 'sentiment', 'tweet'])
    df = df[df.sentiment != 'neutral'] # remove rows where sentiment is "neural"
    df['sentiment'] = (df['sentiment'] == 'positive').astype(int)
    X = df['tweet']
    y = df['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
    cv = CountVectorizer() # initialize count vectorizer 
    X_traincv = cv.fit_transform(x_train)
    X_testcv = cv.transform(x_test)
    print(X_traincv)
    print(X_traincv.toarray())
    #features_train = X_traincv.toarray()
    #features_names = cv.get_feature_names()
    #n,d = features_train.shape
    #print cv.inverse_transform(features_train[0])

    # USING NAIVE BAYES
    clf = MultinomialNB()
    clf.fit(X_traincv, y_train)
    predicted1 = clf.predict(X_testcv)
    print('naive bayes')
    print metrics.accuracy_score(y_test, predicted1)
    print metrics.precision_score(y_test, predicted1)
    print metrics.recall_score(y_test,predicted1)
    print confusion_matrix(y_test, predicted1)

    # USING SVMS
    clf2= svm.SVC()
    clf2.fit(X_traincv, y_train)
    predicted2 = clf2.predict(X_testcv)
    print('default svms')
    print metrics.accuracy_score(y_test, predicted2)
    print metrics.precision_score(y_test, predicted2)
    print metrics.recall_score(y_test,predicted2)
    print confusion_matrix(y_test, predicted2)

    clf3 = LogisticRegression()
    clf3.fit(X_traincv, y_train)
    predicted3 = clf3.predict(X_testcv)
    print('logistic regression')
    print metrics.accuracy_score(y_test, predicted3)
    print metrics.precision_score(y_test, predicted3)
    print metrics.recall_score(y_test,predicted3)
    print confusion_matrix(y_test, predicted3)


if __name__ == "__main__":
    main()