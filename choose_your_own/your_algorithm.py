#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
import sys
from time import time
from sklearn.metrics import accuracy_score
import numpy as np

def predict_with_nearest_neighbors():
	from sklearn.neighbors.nearest_centroid import NearestCentroid
	clf = NearestCentroid()
	t0 = time()
	clf.fit(features_train, labels_train)
	print("training time:{0}s(nearest neighbors)".format(round(time() - t0, 3)))
	# predict
	predict = clf.predict(features_test)
	# accuracy
	accuracy = accuracy_score(labels_test, predict)
	file_name = "ac{0:.3f}_nn_01.png".format(accuracy)
	print("accuracy:{0:.3f}(nearest neighbors)".format(accuracy))
	show_result(clf, file_name)


def predict_with_random_forest():
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier()
	t0 = time()
	#clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf.fit(features_train, labels_train)
	print("training time:{0}s(random forest)".format(round(time() - t0, 3)))
	# predict
	predict = clf.predict(features_test)
	# accuracy
	accuracy = accuracy_score(labels_test, predict)
	file_name = "ac{0:.3f}_rf_01.png".format(accuracy)
	print("accuracy:{0:.3f}(random forest)".format(accuracy))
	show_result(clf, file_name)


def predict_with_ada_boost():
	from sklearn.ensemble import AdaBoostClassifier
	clf = AdaBoostClassifier()
	t0 = time()
	clf.fit(features_train, labels_train)
	print("training time:{0}s(ada boost)".format(round(time() - t0, 3)))
	# predict
	predict = clf.predict(features_test)
	# accuracy
	accuracy = accuracy_score(labels_test, predict)
	file_name = "ac{0:.3f}_ab_01.png".format(accuracy)
	print("accuracy:{0:.3f}(ada boost)".format(accuracy))
	show_result(clf, file_name)


def predict_with_decision_tree():
	from sklearn import tree
	clf = tree.DecisionTreeClassifier()
	t0 = time()
	clf.fit(features_train, labels_train)
	print("training time:{0}s(decision tree)".format(round(time() - t0, 3)))
	# predict
	predict = clf.predict(features_test)
	# accuracy
	accuracy = accuracy_score(labels_test, predict)
	file_name = "ac{0:.3f}_dt_01.png".format(accuracy)
	print("accuracy:{0:.3f}(decision tree)".format(accuracy))
	show_result(clf, file_name)


def predict_with_svm():
	from sklearn.svm import SVC
	clf = SVC()
	t0 = time()
	clf.fit(features_train, labels_train)
	print("training time:{0}s(svm)".format(round(time() - t0, 3)))
	# predict
	predict = clf.predict(features_test)
	# accuracy
	accuracy = accuracy_score(labels_test, predict)
	file_name = "ac{0:.3f}_svm_01.png".format(accuracy)
	print("accuracy:{0:.3f}(svm)".format(accuracy))
	show_result(clf, file_name)


def predict_with_naive_bayes():
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()
	t0 = time()
	clf.fit(features_train, labels_train)
	print("training time:{0}s(naive bayes)".format(round(time() - t0, 3)))
	# predict
	predict = clf.predict(features_test)
	# accuracy
	accuracy = accuracy_score(labels_test, predict)
	file_name = "ac{0:.3f}_nb_01.png".format(accuracy)
	print("accuracy:{0:.3f}(naive bayes)".format(accuracy))
	show_result(clf, file_name)


def show_result(clf, o_file_name="test.png"):
    try:
	plt.figure()
        prettyPicture(clf, features_test, labels_test, o_file_name)
    	from PIL import Image
    	im = Image.open("./"+o_file_name)
    	#im.show()
    except NameError:
    	print("ecception occured!!")
    	pass


predict_with_nearest_neighbors()
predict_with_random_forest()
predict_with_ada_boost()
predict_with_decision_tree()
predict_with_svm()
predict_with_naive_bayes()
