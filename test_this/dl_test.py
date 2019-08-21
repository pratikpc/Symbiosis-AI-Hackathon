# Importing dependencies
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) # Creates a list of stopwords


# Loading and splitting the data into texts and labels
train_df = pd.read_excel('Data_Train.xlsx')

data = train_df.iloc[:,0]
data_labels = train_df.iloc[:,1].values
        
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
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text

clean_data = data.apply(lambda x: clean_text(x))

def remove_stop_words(data):
    filtered_text = [word for word in data.split() if word not in stop_words]
    return ' '.join(filtered_text)

clean_stop_data = clean_data.apply(lambda x: remove_stop_words(x))


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

def get_sequence_tokens(text):
    
    tokenizer.fit_on_texts(text)
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    for line in text:
        token_list = tokenizer.texts_to_sequences([line])[0]
        input_sequences.append(token_list)
    return input_sequences, total_words

convo_sequences, total_words = get_sequence_tokens(clean_stop_data)
print(convo_sequences[0])
print(tokenizer.sequences_to_texts([convo_sequences[0]]))

from keras.preprocessing.sequence import pad_sequences

def generate_padded_sequences(input_sequences):
    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen=max_sequence_length,
                                             padding='pre'))
    
    return input_sequences, max_sequence_length

predictors, max_sequence_length = generate_padded_sequences(convo_sequences)

from keras.layers import LSTM, Embedding, Dropout, Dense
from keras import Sequential

def build_model(max_sequence_length, total_words):
    input_len = max_sequence_length
    model = Sequential()
    
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    
    model.add(Dense(4, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = build_model(max_sequence_length, total_words)
model.summary()

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
ohe_labels = to_categorical(data_labels)
X_train, X_test, Y_train, Y_test = train_test_split(predictors, ohe_labels, test_size=0.2, random_state=24)

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10)
model.evaluate(X_test, Y_test)

#model.save('model1.h5')

#Politics: 0
#Technology: 1
#Entertainment: 2
#Business: 3