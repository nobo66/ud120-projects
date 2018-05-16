#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    errors = []
    for i in range(len(predictions)):
        errors.append((net_worths[i] - predictions[i])**2)
    sorted_errors = sorted(errors)
    ten_percent = len(predictions)/10
    threshold = sorted_errors[-ten_percent]
    for i in range(len(predictions)):
        if errors[i] < threshold:
            cleaned_data.append((ages[i], net_worths[i], (net_worths[i] - predictions[i])**2)) 
    return cleaned_data

