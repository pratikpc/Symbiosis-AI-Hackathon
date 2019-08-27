import spacy
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
stop_words = set(stopwords.words('english')) 
stop_words.add('tata')
stop_words.add('car')
stop_words.add('truck')
stop_words.add('vehicle')
stop_words.add('honda')
stop_words.add('mazda')

nlp=spacy.load('en')

data =  pd.read_csv('data.csv', error_bad_lines=False, encoding='latin1')
data.drop('Unnamed: 0', axis=1, inplace=True)
data = data[['label', 'text']]

data = data[data['text'].notnull()]

from sklearn.utils import shuffle
data = shuffle(data)

text = data['text']
label = data['label']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(text, label, random_state=0, test_size=0.2)

def get_vectorizer(corpus, preprocessor=None, tokenizer=None):
    #vectorizer = CountVectorizer(ngram_range=(2,4),analyzer='char')
    vectorizer = CountVectorizer()
    vectorizer.fit(corpus)
    return vectorizer, vectorizer.get_feature_names()

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.utils.extmath import density

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
    return s if len(s) <= 80 else s[:77] + "..."


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
        text = ' '.join(filtered_size)
        return text

        
    
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
def benchmark(clf, X_train, y_train, X_test, y_test, target_names,
              print_report=True, show_feature_names=True, print_top10=True,
              print_cm=True):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("Train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("Test time:  %0.3fs" % test_time)

    print("Accuracy of fit on train set:   %0.3f" % clf.score(X_train, y_train))
    
    score = metrics.accuracy_score(y_test, pred)
    print("Accuracy on test set:   %0.3f" % score)
    #print("Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

    if hasattr(clf, 'coef_'):
        print("Dimensionality: %d" % clf.coef_.shape[1])
        print("Density: %f" % density(clf.coef_))

        if print_top10 and show_feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(['New Car Enquiry', 'Test Drive Enquiry','Breakdown', 'Feedback', 'Quality']):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join([feature_names[i] for i in top10]))))
        print()

    if print_report:
        print("Classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))
        print("Class names identified: ", clf.classes_)

    if print_cm:
        print("Confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
        #plot_confusion_matrix(metrics.confusion_matrix(y_test, pred), ['New Car Enquiry','Test Drive Enquiry','Breakdown', 'Feedback', 'Quality'])
        plt.show()
        
    with open('./all_data/tata_text/tata_enquire.txt', 'r') as file:
        lines = file.readlines()
        print(predict_text(clf, concat_text(lines)))
    
    with open('./all_data/tata_text/tata_test_drive.txt', 'r') as file:
        lines = file.readlines()
        print(predict_text(clf, concat_text(lines)))
        
    
    with open('./all_data/tata_text/tata_breakdown.txt', 'r') as file:
        lines = file.readlines()
        print(predict_text(clf, concat_text(lines)))
    
    with open('./all_data/tata_text/tata_feedback.txt', 'r') as file:
        lines = file.readlines()
        print(predict_text(clf, concat_text(lines)))
        
    with open('./all_data/tata_text/tata_quality.txt', 'r') as file:
        lines = file.readlines()
        print(predict_text(clf, concat_text(lines)))
    

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
    
clf_log1 = LogisticRegression(C=0.5, class_weight=class_weights, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=1000,
          multi_class='multinomial', n_jobs=1, penalty='l2', random_state=None,
          solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

clf_log2 = LogisticRegression(C=1, class_weight=class_weights, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=20000,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)


from sklearn.svm import LinearSVC
clf_linsvc = LinearSVC(penalty='l2', loss='squared_hinge', 
                dual=True, tol=0.0001, C=0.7, 
                multi_class='ovr', 
                fit_intercept=True, intercept_scaling=1, 
                class_weight=class_weights, verbose=0, random_state=None, max_iter=1000)

    
from sklearn.linear_model import SGDClassifier
clf_sgd = SGDClassifier(loss='squared_hinge', penalty='l2', alpha=0.0001, l1_ratio=0.3, fit_intercept=True, 
                    max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, 
                    random_state=None, learning_rate='adaptive', eta0=1, power_t=0.5, early_stopping=True, 
                    validation_fraction=0.2, n_iter_no_change=5, 
                    class_weight=class_weights, warm_start=False, average=False)

from sklearn.naive_bayes import ComplementNB
clf_cnb = ComplementNB()

from sklearn.naive_bayes import MultinomialNB
clf_mnb = MultinomialNB()


classify_list = [clf_log1, clf_log2, clf_linsvc, clf_sgd, clf_cnb, clf_mnb]

for index, clf in enumerate(classify_list):
    print("Evaluating Split {}".format(index + 1))
    target_names = ['New Car Enquiry', 'Test Drive Enquiry','Breakdown', 'Feedback', 'Quality']
    #print("Train Size: {}\nTest Size: {}".format(X_train.shape[0], X_test.shape[0]))
    results = []
    print('=' * 80)
    #kfold = model_selection.KFold(n_splits=2, random_state=0)
    #model = LinearDiscriminantAnalysis()
    results.append(benchmark(clf, X_train, y_train, X_test, y_test, target_names))

print("Size of text data: ", len(text))
        
        
print("=" * 50) 

for clf in classify_list:       
    print(predict_text(clf, 'I was thinking of buying a the tata tiago'))
    print(predict_text(clf, 'I am looking to test drive the tata tiago'))
    print(predict_text(clf, 'My tata tiago broke down in the middle of the road'))
    print(predict_text(clf, 'I am satsified with the service i received at the garage'))
    print(predict_text(clf, 'There is a lot of noise coming out of the engine of my tata tiago'))
    print('-' * 50)