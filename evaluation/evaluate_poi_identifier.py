#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

# Lesson 15-27
# from Lesson14-17
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
score = clf.score(features, labels)
print "score is {0}".format(score)

# from Lesson14-18
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.3, random_state=42)
clf = clf.fit(features_train, labels_train)
score = clf.score(features_test, labels_test)
print "The score of test data is {0}".format(score)
# Lesson15-28
predict = clf.predict(features_test)
import numpy
unique, counts = numpy.unique(predict, return_counts=True)
predict_dict = dict(zip(unique, counts))
print "num of predicted pois in test data is {0}".format(predict_dict[1])
# Lesson15-29
print "num of test data is {0}".format(len(labels_test))
# Lesson15-30
unique, counts = numpy.unique(labels_test, return_counts=True)
labels_test_dict = dict(zip(unique, counts))
print "num of pois in the test data is {0}".format(labels_test_dict[1])
print "num of innocent people in the test data is {0}".format(labels_test_dict[0])
print "accuracy if no poi is predicted is {0}".format(float(labels_test_dict[0])/len(labels_test))
# Lesson15-31
print "predict={0}".format(predict)
print "labels_test={0}".format(labels_test)
true_positive=0
for i, j in zip(predict, labels_test):
    if i==1 and j==1:
        true_positive += 1
print "true_positive={0}".format(true_positive)
# Lesson15-32
from sklearn.metrics import precision_score
precision = precision_score(labels_test, predict)
print "precision={0}".format(precision)
# Lesson15-33
from sklearn.metrics import recall_score
recall = recall_score(labels_test, predict)
print "recall={0}".format(recall)
