
# coding: utf-8

# In[9]:

import numpy as np
import math
import collections
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[10]:

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


# In[11]:

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(iris_X_train, iris_y_train) 
MyknnPred = knn.predict(iris_X_test)


# In[12]:

MyCLF = MyNearestNeighborClassifier(n_neighbors =3)
MyCLF.fit(iris_X_train, iris_y_train)
y_pred =MyCLF.predict(iris_X_test,iris_X_train)
print('MyAccuracy:',accuracy_score(iris_y_test,y_pred))
print("KnnAccuracy:",accuracy_score(iris_y_test,MyknnPred))




