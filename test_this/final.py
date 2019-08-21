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
test_df = pd.read_excel('Data_Test.xlsx')

# Shuffling the dataset
from sklearn.utils import shuffle
train_df = shuffle(train_df)


train_article_texts = train_df.iloc[:,0]
train_article_labels = train_df.iloc[:,1].values

test_article_texts = test_df.iloc[:,0]




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

# Before filtering
count_words(train_clean_article_texts)
count_words(test_clean_article_texts)
# After filtering
count_words(train_clean_stop_texts)
count_words(test_clean_stop_texts)


# Feature creation
from sklearn.feature_extraction.text import CountVectorizer

countVec = CountVectorizer() # Try this out

X_train = countVec.fit_transform(train_clean_article_texts)
X_test = countVec.transform(test_clean_article_texts)

#############################################################
# Creating the model
from sklearn.naive_bayes import MultinomialNB as MNB
# For Count Vectorizer
model = MNB()

model.fit(X_train, train_article_labels)
y_pred = model.predict(X_test)
print("Score for MNB with CV against train set: ", model.score(X_train, train_article_labels))
# 0.9788935500786575 7628x21564 with stemming
# 0.9813843733613005 7628x30847 without stemming

#Politics: 0
#Technology: 1
#Entertainment: 2
#Business: 3

#pd.DataFrame(y_pred).to_excel('LOL_Output.xlsx')