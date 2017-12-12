
# coding: utf-8

import numpy as np
import pandas as pd
import os
import re
import io
import glob
import collections

#format and transform email data
data = []
path = glob.glob('spam_data/0*.txt')
for filename in path:   
    file = open(filename, encoding = "ISO-8859-1")
    data.append(file.read().replace('\n',' ').lower())
    file.close()

#create bag of words. I pick the most frequent 100 words in these emails(I will get rid of the STOP words in later version.)
bagsofwords = [ collections.Counter(re.findall(r'\w+', txt)) for txt in data]
sumbags = sum(bagsofwords, collections.Counter())
sumbags.most_common(200)
BOW = list(sorted(sumbags, key=sumbags.get, reverse=True))[:100]

#This function is used to count the frequency of the words in BOW.
def count_words(data):
    count = {}
    for i in BOW:
        count[i] = 0
    for word in re.findall(r"[\w']+", data):
        if word in BOW:
            count[word] = count.get(word)+1
    return list(count.values())

#Create email feature data
X_data = np.array(count_words(data[0]))
for i in range(1,50):
    words = count_words(data[i])
    X_data = np.row_stack((X_data, words))

#Load labels
tmp = []
Y_data=[]
with open('./spam_data\\labels.txt', 'r') as f:
    for line in f:
        tmp.append(line.split(None, 1)[0])
f.close()
Y_data=np.array([int(x) for x in tmp])

#split emails into train data and test data
np.random.seed(0)
indices = np.random.permutation(len(X_data))
X_train = X_data[indices[:-25]]
y_train = Y_data[indices[:-25]]
X_test = X_data[indices[-25:]]
y_test = Y_data[indices[-25:]]

#Load my classifier in part1.
class MyNearestNeighborClassifier():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test, X_train):
        target = []
        for i in range(len(X_test)):
            distances = []
            tmp = []
            for j in range(len(X_train)):
                distance = np.sqrt(np.sum(np.square(X_test[i] - X_train[j])))
                distances.append([distance,j])
                distances.sort()
            for t in range(self.n_neighbors):
                tmp.append(self.y_train[distances[t][1]])
                label = collections.Counter(tmp).most_common(1)[0][0]
            target.append(label)       
        return target

#conpare the accurancy
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) 
MyknnPred = knn.predict(X_test)
MyCLF = MyNearestNeighborClassifier(n_neighbors =3)
MyCLF.fit(X_train, y_train)
y_pred = MyCLF.predict(X_test,X_train)
print('MyAccuracy:',accuracy_score(y_test,y_pred))
print("KnnAccuracy:",accuracy_score(y_test,MyknnPred))