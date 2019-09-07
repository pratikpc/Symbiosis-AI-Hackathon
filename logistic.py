import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
stop_words = set(stopwords.words('english')) 
stop_words.add('tata')
stop_words.add('motors')
stop_words.add('motor')
stop_words.add('car')
stop_words.add('truck')
stop_words.add('vehicle')
stop_words.add('honda')
stop_words.add('mazda')
stop_words.add('hello')
stop_words.add('sir')

nlp=spacy.load('en')

data =  pd.read_csv('data.csv', error_bad_lines=False, encoding='latin1')
data.drop('Unnamed: 0', axis=1, inplace=True)
data = data[['label', 'text']]

data = data[data['text'].notnull()]

from sklearn.utils import shuffle
data = shuffle(data)

text = data['text']
label = data['label']

def ReadEnglishDictionary():
  with open('english.txt', 'r') as f:
    english_words = f.readlines()
  english_words = set([WordNetLemmatizer().lemmatize(x.strip().lower().replace('\'s', '')) for x in english_words] )
  english_words = dict.fromkeys(english_words, None)
  return english_words

english_words = ReadEnglishDictionary()

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(text, label, random_state=0, test_size=0.2)


def get_vectorizer(corpus, preprocessor=None, tokenizer=None):
    #vectorizer = CountVectorizer(ngram_range=(2,4),analyzer='char')
    vectorizer = CountVectorizer(min_df=5)
    vectorizer.fit(corpus)
    return vectorizer, vectorizer.get_feature_names()

from sklearn.linear_model import LogisticRegression

def data_for_training():
    vectorizer, feature_names = get_vectorizer(xtrain)
    
    X_train_no_HD = vectorizer.transform(xtrain).toarray()
    X_test_no_HD = vectorizer.transform(xtest).toarray()
            
    return X_train_no_HD, ytrain, X_test_no_HD, ytest, feature_names, vectorizer

X_train, y_train, X_test, y_test, feature_names, vectorizer = data_for_training()

from sklearn.utils import class_weight
class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))
#class_weights[4] = class_weights.pop(3)

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 100 else s[:97] + "..."


def predict_text(clf, text):
    # To clean the text of unnecessary data
    def clean_text(text):
        text = text.lower()
        filtered_text = [lemmatize_stemming(word) for word in text.split() if word not in stop_words]
        text = ' '.join(filtered_text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '', text)
        text = re.sub(r"[^A-Za-z]", " ", text)
        filtered_size = [word for word in text.split() if len(word) > 2]
        filtered_dict = [word for word in filtered_size if word in english_words]
        filtered_dict = filtered_dict[:20]
        text = ' '.join(filtered_dict)
        print(text)
        return text
    
    
    def IsEnglishWord(word):
        return (word in english_words)

        
    
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
    
        return tag_dict.get(tag, wordnet.NOUN)
      
    def lemmatize_stemming(text):
       return  WordNetLemmatizer().lemmatize(text, get_wordnet_pos(text))


    vectorized_text = vectorizer.transform([clean_text(text)])
    return clf.predict(vectorized_text)


def concat_text(text_list):
    text = ''
    for line in text_list:
        text += line
    return text


# #############################################################################
# Benchmark classifiers

clf_log1 = LogisticRegression(C=0.5, class_weight=class_weights, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=1000,
          multi_class='multinomial', n_jobs=1, penalty='l2', random_state=None,
          solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

#clf_log2 = LogisticRegression(C=1, class_weight=class_weights, dual=False,
#          fit_intercept=True, intercept_scaling=1, max_iter=20000,
#          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)


clf_log1.fit(X_train, y_train)
from sklearn.metrics import precision_score
print(precision_score(y_test, clf_log1.predict(X_test), average='weighted'))


print("Size of text data: ", len(text))
        
        
print("=" * 50) 

      
print(predict_text(clf_log1, 'I was thinking of buying a the tata tiago'))
print(predict_text(clf_log1, 'I am looking to test drive the tata tiago'))
print(predict_text(clf_log1, 'My tata tiago broke down in the middle of the road'))
print(predict_text(clf_log1, 'I am satsified with the service i received at the garage'))
print(predict_text(clf_log1, 'There is a lot of noise coming out of the engine of my tata tiago'))
print('-' * 50)