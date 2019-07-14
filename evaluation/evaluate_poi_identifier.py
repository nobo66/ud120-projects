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

