#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
print("number of features is {0}".format(len(features_train[0])))

from sklearn import tree
# added min_samples_split setting referring below site.
# https://jefflirion.github.io/udacity/Intro_to_Machine_Learning/Lesson3.html
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(features_train, labels_train)
predicts = clf.predict(features_test)

import numpy as np
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, predicts)
#accuracy = accuracy_score(predicts, labels_test)
print("accuracy is {0}".format(accuracy))
#########################################################


