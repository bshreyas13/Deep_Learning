# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:37:13 2021

@author: shrey
"""
# Import the necessary libraries
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn import metrics 
import numpy as np

def SplitData(X,y):
    # splitting X and y into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1) 
    
    return X_train, X_test, y_train, y_test

def Classifier(X_train,X_test, y_train, y_test):
    # training the model on training set 
    gnb = GaussianNB() 
    gnb.fit(X_train, y_train) 
    # making predictions on the testing set 
    y_pred = gnb.predict(X_test)
    
    return y_pred

# pointer to load the Winsconsin Breast Cancer dataset 
cancer = load_breast_cancer() 
# store the feature matrix (X) and response vector (y) 
X = cancer.data 
y = cancer.target 
 
X_train, X_test, y_train, y_test = SplitData(X,y)
 
y_pred = Classifier(X_train, X_test, y_train, y_test)
  
# comparing actual response values (y_test) with predicted response values (y_pred) 
tn,fp,fn,tp = metrics.confusion_matrix(y_test, y_pred).ravel()
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
print("The confusion matrix:"+"\n", metrics.confusion_matrix(y_test, y_pred))
print(" The number of true positives:",tp)
print(" The number of false positives:",fp)
print(" The number of true negatives:",tn)
print(" The number of false negatives:",fn)

# Adding 30-dim Gaussian Random noise to features
# With 0 mean and various variances

X_noisy = np.zeros (X.shape)
noisemat = np.zeros (X.shape)
variance = [50,100,200,400,800]
for v in variance :
    for i in range(X.shape[0]):
        noise = np.random.normal(0,v,[30])
        noisemat[i] = noise 
        X_noisy[i] = X[i,:] + noise

    X_train, X_test, y_train, y_test = SplitData(X_noisy,y)
 
    y_pred = Classifier(X_train, X_test, y_train, y_test)
  
    # comparing actual response values (y_test) with predicted response values (y_pred) 
    tn,fp,fn,tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    print("Gaussian Naive Bayes model with noise (variance {}) accuracy(in %):{}".format(v,metrics.accuracy_score(y_test, y_pred)*100))
    print("The confusion matrix:"+"\n", metrics.confusion_matrix(y_test, y_pred))
    print(" The number of true positives:",tp)
    print(" The number of false positives:",fp)
    print(" The number of true negatives:",tn)
    print(" The number of false negatives:",fn)
