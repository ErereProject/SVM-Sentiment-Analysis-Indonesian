import time

import pandas
from numpy import random
from pandas import DataFrame

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

import os
random.seed(7000)
home = os.getcwd()
corpus = pandas.read_csv("clean.csv")
train_y, test_y, train_x, test_x = train_test_split(corpus['sentiment'], corpus['text_final'], test_size=0.1)

tn_start = time.time()
encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

tfidf = TfidfVectorizer()
tfidf.fit(corpus['text_final'])
train_x_tfidf = tfidf.transform(train_x)
test_x_tfidf = tfidf.transform(test_x)

svm_model = SVC(kernel="poly", C=1000, coef0=2, degree=3, gamma=0.0001) ## These configurations have been tested to be the best
svm_model.fit(train_x_tfidf, train_y)
prediction = svm_model.predict(test_x_tfidf)
print(f"SVM Prediction = {f1_score(test_y, prediction)*100}%")
print({time.time()-tn_start})
label = ["Negative", "Positive"]
################## Wrong predicted data ##################
idx = test_x.index
df = DataFrame(columns=['Text', 'Prediction', 'Real'])
for i in range(len(test_y)):
    if int(test_y[i]) is not int(prediction[i]):
        df.loc[i, 'Text'] = corpus.loc[idx[i], 'text']
        df.loc[i, 'Prediction'] = label[prediction[i]]
        df.loc[i, 'Real'] = label[test_y[i]]
print(df)
df.to_csv(f"{home}/wrong_predicted.csv")