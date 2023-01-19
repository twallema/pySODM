import numpy as np
import pandas as pd

def int_to_date(actual_start_date, t):
    date = actual_start_date + pd.Timedelta(t, unit='D')
    return date


def list_to_dict(y, shape_dictionary):
    """
    A function to reconstruct a number of variables of different shapes, given a flat array of values and a dictionary with the variables name and desired shapes

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
        restoredArray.append(y[offset:(offset+n)].reshape(s))
        offset+=n

    return dict(zip(shape_dictionary.keys(), restoredArray))
