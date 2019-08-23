import pandas as pd

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

breakdown_df['text'] = breakdown_df['text'].apply(lambda x : clean_text(x))
testdrive_df['text'] = testdrive_df['text'].apply(lambda x: clean_text(x))
feedback['text'] = feedback['text'].apply(lambda x: clean_text(x))
quality_df['text'] = quality_df['text'].apply(lambda x: clean_text(x))
car_enq_df['text'] = car_enq_df['text'].apply(lambda x: clean_text(x))


#Removing yes from stop_testdrive
testdrive_df = testdrive_df[testdrive_df.text != 'yes']

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
count_words(breakdown_df['text'])
count_words(testdrive_df['text'])
count_words(feedback['text'])
count_words(quality_df['text'])
count_words(car_enq_df['text'])


# Removing stopwords from cleaned texts
def remove_stop_words(data):
    wn = WordNetLemmatizer()
    filtered_text = [wn.lemmatize(word) for word in data.split() if word not in stop_words]
    filtered_size = [word for word in filtered_text if len(word) > 2]
    return ' '.join(filtered_size)

breakdown_df['text'] = breakdown_df['text'].apply(lambda x : remove_stop_words(x))
testdrive_df['text'] = testdrive_df['text'].apply(lambda x: remove_stop_words(x))
feedback['text'] = feedback['text'].apply(lambda x: remove_stop_words(x))
quality_df['text'] = quality_df['text'].apply(lambda x: remove_stop_words(x))
car_enq_df['text'] = car_enq_df['text'].apply(lambda x: remove_stop_words(x))


# After filtering
count_words(breakdown_df['text'])
count_words(testdrive_df['text'])
count_words(feedback['text'])
count_words(quality_df['text'])
count_words(car_enq_df['text'])


pooled_data = breakdown_df
pooled_data = pooled_data.append(feedback[:114], ignore_index = True)
pooled_data = pooled_data.append(testdrive_df[:114], ignore_index = True)
pooled_data = pooled_data.append(quality_df[:114], ignore_index = True)
pooled_data = pooled_data.append(car_enq_df[:114], ignore_index = True)

