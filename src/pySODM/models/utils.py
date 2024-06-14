import numpy as np
from datetime import timedelta

def int_to_date(actual_start_date, t):
    date = actual_start_date + timedelta(days=t)
    return date

def list_to_dict(y, shape_dictionary, retain_floats=False):
    """
    A function to reconstruct model states, given a flat array of values and a dictionary with the variables name and desired shapes

    Parameters
    ----------

    y: list
        A flat list of values

    shape_dictionary: dict
        A dictionary containing the desired names and shapes of the output dictionary
    
    Returns
    -------

    y_dict: dict
        Dictionary containing the reconstructed values.
    """

    restoredArray =[]
    offset=0
    for s in shape_dictionary.values():
        n = np.prod(s)
        if ((n == 1) & (retain_floats == True)):
            restoredArray.append(y[offset])
        elif ((n == 1) & (retain_floats == False)):
            restoredArray.append(y[offset].reshape(s))
        else:
            restoredArray.append(y[offset:(offset+n)].reshape(s))
        offset+=n
    return dict(zip(shape_dictionary.keys(), restoredArray))
