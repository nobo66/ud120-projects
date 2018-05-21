#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
# Lesson8-17
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

# Lesson8-18 Check outliers
for key, personal_data in data_dict.iteritems():
	if personal_data['salary'] >= 1000000 and personal_data['salary'] != "NaN" and \
	personal_data['bonus'] >= 5000000 and personal_data['bonus'] != "NaN":
		print("{0} is a person whose salary is over 1,000,000USD and bonus is over 5,000,000USD".format(key))
		print("data={0}".format(personal_data))


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


