from cleanup import ApplyCleanup
from sklearn.feature_extraction.text import CountVectorizer

# Generate the Vectorizer from our corpus
def get_vectorizer(corpus, preprocessor=None, tokenizer=None):
    vectorizer = CountVectorizer(min_df=5)
    vectorizer.fit(corpus)
    return vectorizer, vectorizer.get_feature_names()

import pandas as pd
text = pd.read_csv('train-text.csv', error_bad_lines=False, encoding='latin1')
text = text['label'].apply(str)

vectorizer, feature_names = get_vectorizer(text)
import joblib

clf_log1 = joblib.load('model.save') 

def predict_text(clf, text):
    vectorized_text = vectorizer.transform([ApplyCleanup(text)])
    return clf.predict_proba(vectorized_text)

def PredictResults(text):
    return predict_text(clf_log1, text)
