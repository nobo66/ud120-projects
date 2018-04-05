#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print enron_data['METTS MARK']
print("number of data points is {0}".format(len(enron_data)))
print("number of features to each person is {0}".format(len(enron_data['METTS MARK'])))
num_of_POI = sum(1 for x in enron_data.values() if x['poi'] == 1)
num_of_POI = sum(x['poi'] == 1 for x in enron_data.values())
print("number of POI is {0}".format(num_of_POI))
print("number of POI is {0}".format(sum(1 for x in enron_data.values() if x['poi'] == 1)))
print("number of POI is {0}".format(sum(x['poi'] == 1 for x in enron_data.values())))
#print for x in enron_data.values()
print("the total value of the stock belonging to James Prentice is {0}".format(enron_data['PRENTICE JAMES']['total_stock_value']))
print("How many email messages do we have from Wesley Colwell to persons of interest?"\
	"\nAnswer:{0}".format(enron_data['COLWELL WESLEY']['from_this_person_to_poi']))
print("What's the value of stock options exercised by Jeffrey K Skilling?"
	"\nAnswer:{0}".format(enron_data['SKILLING JEFFREY K']['exercised_stock_options']))
print("How much of total_payments of Jeffery Lay?"\
	"\nAnswer:{0}".format(enron_data['SKILLING JEFFREY K']['total_payments']))
print("How much of total_payments of Kenneth Lay?"\
	"\nAnswer:{0}".format(enron_data['LAY KENNETH L']['total_payments']))
print("How much of total_payments of Andrew Fastow?"\
	"\nAnswer:{0}".format(enron_data['FASTOW ANDREW S']['total_payments']))
#print(""\
#	"\nAnswer:{0}".format(enron_data['']['']))
