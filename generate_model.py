from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from cleanup import ApplyCleanup

def predict_text(clf, text):
    vectorized_text = vectorizer.transform([ApplyCleanup(text)])
    return clf.predict_proba(vectorized_text)

data = pd.read_csv('data.csv', error_bad_lines=False, encoding='latin1')
data.drop('Unnamed: 0', axis=1, inplace=True)
data = data[['label', 'text']]
data['text'] = data['text'].apply(str)
data['label'] = data['label'].apply(int)

data['text'] = data['text'].apply(lambda x : ApplyCleanup(x))
data.dropna(inplace=True)
data = data.drop_duplicates()

data = shuffle(data)

text = data['text']
label = data['label']

xtrain, xtest, ytrain, ytest = train_test_split(
    text, label, random_state=0, test_size=0.2)

xtrain.to_csv('train-text.csv', encoding='latin1', index=False, header=['label'])

def get_vectorizer(corpus, preprocessor=None, tokenizer=None):
    #vectorizer = CountVectorizer(ngram_range=(2,4),analyzer='char')
    vectorizer = CountVectorizer(min_df=5)
    vectorizer.fit(corpus)
    return vectorizer, vectorizer.get_feature_names()


def data_for_training():
    vectorizer, feature_names = get_vectorizer(xtrain)

    X_train_no_HD = vectorizer.transform(xtrain).toarray()
    X_test_no_HD = vectorizer.transform(xtest).toarray()

    return X_train_no_HD, ytrain, X_test_no_HD, ytest, feature_names, vectorizer


X_train, y_train, X_test, y_test, feature_names, vectorizer = data_for_training()

class_weights = dict(enumerate(class_weight.compute_class_weight(
    'balanced', np.unique(y_train), y_train)))
#class_weights[4] = class_weights.pop(3)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 100 else s[:97] + "..."

# #############################################################################
# Benchmark classifiers

clf_log1 = LogisticRegression(C=0.5, class_weight=class_weights, dual=False,
                              fit_intercept=True, intercept_scaling=1, max_iter=1000,
                              multi_class='multinomial', n_jobs=1, penalty='l2', random_state=None,
                              solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

# clf_log2 = LogisticRegression(C=1, class_weight=class_weights, dual=False,
#          fit_intercept=True, intercept_scaling=1, max_iter=20000,
#          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

clf_log1.fit(X_train, y_train)

import joblib
joblib.dump(clf_log1, "model.save")

pred = clf_log1.predict(X_test)
target_names = ['New Car Enquiry','Test Drive Enquiry','Breakdown', 'Feedback', 'Quality']
from sklearn import metrics
print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))
print(precision_score(y_test, clf_log1.predict(X_test), average='weighted'))


print("=" * 50)

print(predict_text(clf_log1, 'I was thinking of buying a the tata tiago'))
print(predict_text(clf_log1, 'I am looking to test drive the tata tiago'))
print(predict_text(clf_log1, 'My tata tiago broke down in the middle of the road'))
print(predict_text(clf_log1, 'I am satsified with the service i received at the garage'))
print(predict_text(
    clf_log1, 'There is a lot of noise coming out of the engine of my tata tiago'))
print('-' * 50)
