import numpy as np
import pandas as pd

def int_to_date(actual_start_date, t):
    date = actual_start_date + pd.Timedelta(t, unit='D')
    return date


def list_to_dict(y, shape_dictionary, retain_floats=True):
    """
    A function to reconstruct model states, given a flat array of values and a dictionary with the variables name and desired shapes
    Retains floats.

    Parameters
    ----------

    y: list
        A flat list of values

    shape_dictionary: dict
        A dictionary containing the desired names and shapes of the output dictionary
    
    retain_floats: bool
        If False, float/np.float are converted in a 0D np.ndarray. Default: True.

    Returns
    -------

    y_dict: dict
        Dictionary containing the reconstructed values.
    """

    restoredArray =[]
    offset=0
    for s in shape_dictionary.values():
        n = np.prod(s)
        # Reshape changes type of floats to np.ndarray which is not desirable
        if ((n == 1) & (retain_floats==True)):
            restoredArray.append(y[offset])
        else:
            restoredArray.append(y[offset:(offset+n)].reshape(s))
        offset+=n
    return dict(zip(shape_dictionary.keys(), restoredArray))