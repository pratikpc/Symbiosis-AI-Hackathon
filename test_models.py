import pandas as pd
data = pd.read_csv('data.csv')
data = data.iloc[:,1:]
data = data.dropna()
data = data.sample(frac=1, random_state=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,0], data.iloc[:,1], test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
countVec = CountVectorizer()

X_train = countVec.fit_transform(X_train)
X_test = countVec.transform(X_test)

#############################################################
# Creating the model
from sklearn.linear_model import SGDClassifier

log_model = SGDClassifier(loss='log', validation_fraction=0.2, early_stopping=True, class_weight='balanced')
#log_model = SGDClassifier(loss='log', solver='liblinear', Cs=15, multi_class='auto', dual=True, cv=5, class_weight='balanced', max_iter=500, refit=True)

log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
import matplotlib.pyplot as plt
def plot_confusion(cm):
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells
    
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['car enquire', 'test drive', 'breakdown', 'feedback', 'quality']); ax.yaxis.set_ticklabels(['car enquire', 'test drive', 'breakdown', 'feedback', 'quality'])
    
#plot_confusion(cm)
print("Accuracy on test set: ", accuracy_score(y_test, y_pred_log))
print("Accuracy on train set: ", log_model.score(X_train, y_train))