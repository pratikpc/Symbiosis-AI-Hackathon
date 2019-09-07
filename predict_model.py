from cleanup import ApplyCleanup
from sklearn.feature_extraction.text import CountVectorizer

def predict_text(clf, text):
    vectorized_text = vectorizer.transform([ApplyCleanup(text)])
    return clf.predict_proba(vectorized_text)
def get_vectorizer(corpus, preprocessor=None, tokenizer=None):
    #vectorizer = CountVectorizer(ngram_range=(2,4),analyzer='char')
    vectorizer = CountVectorizer(min_df=5)
    vectorizer.fit(corpus)
    return vectorizer, vectorizer.get_feature_names()

import pandas as pd
text = pd.read_csv('train-text.csv', error_bad_lines=False, encoding='latin1')
text = text['label'].apply(str)

vectorizer, feature_names = get_vectorizer(text)
import joblib

clf_log1 = joblib.load('model.save') 

def PredictResults(text):
    return predict_text(clf_log1, text)

# print("=" * 50)

# print(predict_text(clf_log1, 'I was thinking of buying a the tata tiago'))
# print(predict_text(clf_log1, 'I am looking to test drive the tata tiago'))
# print(predict_text(clf_log1, 'My tata tiago broke down in the middle of the road'))
# print(predict_text(clf_log1, 'I am satsified with the service i received at the garage'))
# print(predict_text(
#     clf_log1, 'There is a lot of noise coming out of the engine of my tata tiago'))
# print('-' * 50)
