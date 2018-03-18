#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 
import numpy as np
from sklearn.svm import SVC
clf = SVC(kernel='rbf',
	#C=10, #0.616
	#C=100, #0.616
	#C=1000, #0.821
	C=10000, #0.892
	)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

t0 = time()
predict = clf.predict(features_test)
print "predict time:", round(time() - t0, 3), "s"

# accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, predict)
print("accuracy is {0}".format(accuracy))
#print("\ndata={0}".format(features_test))
#print("\nlabel={0}".format(labels_test))
#print("\npredict={0}".format(predict))
print("\nprediction(10,26,50)=({0},{1},{2})".format(predict[10], predict[26], predict[50]))
#########################################################


