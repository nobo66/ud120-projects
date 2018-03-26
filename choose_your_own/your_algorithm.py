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

def predict_with_nearest_neighbors():
	from sklearn.neighbors.nearest_centroid import NearestCentroid
	import numpy as np
	clf = NearestCentroid()
	clf.fit(features_train, labels_train)
	# predict
	predict = clf.predict(features_test)
	# accuracy
	from sklearn.metrics import accuracy_score
	accuracy = accuracy_score(labels_test, predict)
	file_name = "test_nn_01_ac{0}.png".format(accuracy)
	print("The accuracy of nearest neighbors is {0}".format(accuracy))
	show_result(clf, file_name)


def predict_with_random_forest():
	from sklearn.ensemble import RandomForestClassifier
	import numpy as np
	clf = RandomForestClassifier()
	#clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf.fit(features_train, labels_train)
	# predict
	predict = clf.predict(features_test)
	# accuracy
	from sklearn.metrics import accuracy_score
	accuracy = accuracy_score(labels_test, predict)
	file_name = "test_rf_02_ac{0}.png".format(accuracy)
	print("The accuracy of random forest is {0}".format(accuracy))
	show_result(clf, file_name)


def predict_with_ada_boost():
	from sklearn.ensemble import AdaBoostClassifier
	import numpy as np
	clf = AdaBoostClassifier()
	clf.fit(features_train, labels_train)
	# predict
	predict = clf.predict(features_test)
	# accuracy
	from sklearn.metrics import accuracy_score
	accuracy = accuracy_score(labels_test, predict)
	file_name = "test_ab_01_ac{0}.png".format(accuracy)
	print("The accuracy of ada boost is {0}".format(accuracy))
	show_result(clf, file_name)


def show_result(clf, o_file_name="test.png"):
    try:
        prettyPicture(clf, features_test, labels_test, o_file_name)
    	from PIL import Image
    	im = Image.open("./"+o_file_name)
    	im.show()
    except NameError:
    	print("ecception occured!!")
    	pass


predict_with_ada_boost()
predict_with_random_forest()
predict_with_nearest_neighbors()
