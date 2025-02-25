import numpy as np
from datetime import timedelta

# time map used in `utils/time_to_date` and `validation/date_to_diff`
time_unit_map = {
    'us': 1,                       
    'ms': 1000,                    # us per ms
    's': 1000*1000,                # us per s
    'min': 60*1000*1000,           # ...
    'h': 60*60*1000*1000,        
    'D': 24*60*60*1000*1000,     
    'W': 7*24*60*60*1000*1000,   
}

def time_to_date(t, start_date, time_unit):
    """ Convert timestep to datetime
    """
    return start_date + timedelta(microseconds=t*time_unit_map[time_unit])

def list_to_dict(y, shape_dictionary, retain_floats=False):
    """
    A function to reconstruct model states, given a flat array of values and a dictionary with the variables name and desired shapes
    Does not retains floats.

    Parameters
    ----------

    y: np.ndarray
        A 1D numpy array

    shape_dictionary: dict
        A dictionary containing the desired names and shapes of the output dictionary
    
    retain_floats: bool
        If False, states initially given to pySODM as a float are converted in a 0D np.ndarray before being handed off to the integration function.
        This results in an apparent 'change' in datatype of model states to the user of an unstratified model. Computationally there are no consequences. Default: False.

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


def cut_ICF_parameters_from_parameters(parameters_names_modeldeclaration, extra_params_TDPF, extra_params_ICF, parameters):
    """
    A function to cut the initial condition function's parameters from the model parameters dictionary

    input
    -----

    parameters_names_modeldeclaration: list
        Parameter names provided by the user in the model declaration
    
    extra_params_TDPF: list
        Parameter names of the time-dependent parameter functions
    
    extra_params_ICF: list
        Parameter names of the initial condition function

    parameters: dict
        Model parameters dictionary

    output
    ------

    parameters: dict
        Model parameters dictionary, minus the parameters unique to the initial condition function
    """

    # Compute union of TDPF parameters and parameters in model declaration
    union_TDPF_integrate = set(extra_params_TDPF) | set(parameters_names_modeldeclaration)
    # Compute the difference between initial condition pars and union TDPF + model declaration pars to obtain unique ICF pars
    unique_ICF = set(extra_params_ICF) - union_TDPF_integrate 
    # Retain all but unique ICF pars
    return {key: value for key, value in parameters.items() if key in union_TDPF_integrate or key not in unique_ICF}
