# Importing dependencies
import pandas as pd
import re
from nltk.probability import FreqDist
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) # Creates a list of stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer

# Loading and splitting the data into texts and labels
train_df = pd.read_excel('Data_Train.xlsx')

# Shuffling the dataset
from sklearn.utils import shuffle
train_df = shuffle(train_df)

text = train_df.iloc[:,0]
labels = train_df.iloc[:,1]

from sklearn.model_selection import train_test_split
train_article_texts, test_article_texts, train_labels, test_labels = train_test_split(text, labels, test_size=0.2)

# To clean the text of unnecessary data
def clean_text(text):
    text = text.lower()
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
    text = re.sub(r"[^A-Za-z]", " ", text)
    return text

train_clean_article_texts = train_article_texts.apply(lambda x : clean_text(x))
test_clean_article_texts = test_article_texts.apply(lambda x : clean_text(x))


# Displays the most frequently used word
def count_words(data):
    words = ' '.join([text for text in data])
    words = words.split()
    fdist = FreqDist(words)
    words_df = pd.DataFrame({'words': list(fdist.keys()), 'keys': list(fdist.values())})
    
    freq = words_df.nlargest(columns='keys', n=50)
    
    plt.figure(figsize=(12,15)) 
    ax = sns.barplot(data=freq, x="keys", y="words") 
    ax.set(ylabel = 'Word') 
    plt.show()
    

# Removing stopwords from cleaned texts
def remove_stop_words(data):
    wn = WordNetLemmatizer()
    filtered_text = [wn.lemmatize(word) for word in data.split() if word not in stop_words]
    return ' '.join(filtered_text)
train_clean_stop_texts = train_clean_article_texts.apply(lambda x : remove_stop_words(x))
test_clean_stop_texts = test_clean_article_texts.apply(lambda x : remove_stop_words(x))

# Feature creation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf = TfidfVectorizer()
countVec = CountVectorizer() # Try this out

X_train_tfidf = tfidf.fit_transform(train_clean_stop_texts)
X_train_cv = countVec.fit_transform(train_clean_stop_texts)
X_test_tfidf = tfidf.transform(test_clean_stop_texts)
X_test_cv = countVec.transform(test_clean_stop_texts)

from sklearn.naive_bayes import MultinomialNB as MNB
# For Count Vectorizer
model = MNB()

model.fit(X_train_cv, train_labels)
y_pred = model.predict(X_test_cv)
print("Score for MNB with CV against train set: ", model.score(X_test_cv, test_labels))

import numpy as np
label_values = [(test_labels == 0).sum(), (test_labels == 1).sum(), (test_labels == 2).sum(), (test_labels == 3).sum()]
label_values = np.asarray(label_values)
print(f"Number of values in Politics, Technology, Entertainment, Business", label_values.sum())

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, y_pred)

import seaborn as sns
import matplotlib.pyplot as plt     

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Greens'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Politics', 'Technology', 'Entertainment', 'Business'])
ax.yaxis.set_ticklabels(['Politics', 'Technology', 'Entertainment', 'Business'])