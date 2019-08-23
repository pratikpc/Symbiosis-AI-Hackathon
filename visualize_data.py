import pandas as pd
import numpy as np

breakdown_df = pd.read_csv('./all_data/label_data/breakdown.csv')
car_enq_df = pd.read_csv('./all_data/label_data/car_enquire.csv')
feedback = pd.read_csv('./all_data/label_data/feedback_tweets.csv')
reply = pd.read_csv('./all_data/label_data/reply_tweets.csv')
feedback = pd.concat([feedback, reply])
quality_df = pd.read_csv('./all_data/label_data/quality.csv')
testdrive_df = pd.read_csv('./all_data/label_data/form_testdrive.csv')

import re
from nltk.probability import FreqDist
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) # Creates a list of stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer

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
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '', text)
    text = re.sub(r"[^A-Za-z]", " ", text)
    return text

clean_breakdown = breakdown_df.iloc[:,0].apply(lambda x : clean_text(x))
clean_testdrive = testdrive_df.iloc[:,0].apply(lambda x: clean_text(x))
clean_feedback = feedback.iloc[:,0].apply(lambda x: clean_text(x))
clean_quality = quality_df.iloc[:,0].apply(lambda x: clean_text(x))
clean_enquire = car_enq_df.iloc[:,0].apply(lambda x: clean_text(x))


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
    
# Before filtering
count_words(clean_breakdown)
count_words(clean_testdrive)
count_words(clean_feedback)
count_words(clean_quality)
count_words(clean_enquire)


# Removing stopwords from cleaned texts
def remove_stop_words(data):
    wn = WordNetLemmatizer()
    filtered_text = [wn.lemmatize(word) for word in data.split() if word not in stop_words]
    filtered_size = [word for word in filtered_text if len(word) > 2]
    return ' '.join(filtered_size)

stop_breakdown = clean_breakdown.apply(lambda x : remove_stop_words(x))
stop_testdrive = clean_testdrive.apply(lambda x : remove_stop_words(x))
stop_feedback = clean_feedback.apply(lambda x : remove_stop_words(x))
stop_quality = clean_quality.apply(lambda x : remove_stop_words(x))
stop_enquire = clean_enquire.apply(lambda x : remove_stop_words(x))

#Removing yes from stop_testdrive
stop_testdrive = stop_testdrive[stop_testdrive != 'yes']

# After filtering
count_words(stop_breakdown)
count_words(stop_testdrive)
count_words(stop_feedback)
count_words(stop_quality)
count_words(stop_enquire)


pooled_data = stop_breakdown
pooled_data = pooled_data.append(stop_testdrive[:115])
pooled_data = pooled_data.append(stop_feedback[:115])
pooled_data = pooled_data.append(stop_quality[:115])
pooled_data = pooled_data.append(stop_enquire[:115])
