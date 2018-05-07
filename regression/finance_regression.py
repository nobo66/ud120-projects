#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )

### list the features you want to look at--first item in the 
### list will be the "target" feature
# changed for Lesson7-44
#features_list = ["bonus", "salary"]
features_list = ["bonus", "long_term_incentive"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from sklearn import linear_model
import pdb
#pdb.set_trace()
reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)
#pdb.set_trace()
print("Params of LinearRegression={0}".format(reg.get_params()))
print("reg.coef_={0}".format(reg.coef_))
print("reg.intercept_={0}".format(reg.intercept_))

from sklearn.metrics import accuracy_score
pred = reg.predict(feature_test)
#score = accuracy_score(target_test, pred)
#print("score={0}".format(score))

# create combined train data
#combined_train = [[0.0 for i in range(len(2)] for j in range(len(feature_train))]
#for i in range(len(feature_train):
#	for j in range(len(2):
#		combined_train[i][0] = feature_
# score with train data
score_train = reg.score(feature_train, target_train)
print("score with train data={0}".format(score_train))

# score with test data
score_test = reg.score(feature_test, target_test)
print("score with test data={0}".format(score_test))

pdb.set_trace()





### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
