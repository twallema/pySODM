import copy
import inspect
import itertools
import numpy as np
import pandas as pd
import datetime
from collections import OrderedDict

def date_to_diff(actual_start_date, end_date):
    """
    Convert date string to int (i.e. number of days since day 0 of simulation,
    which is warmup days before actual_start_date)
    """
    return float((pd.to_datetime(end_date)-pd.to_datetime(actual_start_date))/pd.to_timedelta('1D'))

def validate_simulation_time(time, warmup):
    """Validates the simulation time of the sim() function. Various input types are converted to: time = [start_float, stop_float]"""

    actual_start_date=None
    if isinstance(time, float):
        time = [0-warmup, round(time)]
    elif isinstance(time, int):
        time = [0-warmup, time]
    elif isinstance(time, list):
        if not len(time) == 2:
            raise ValueError(f"'Time' must be of format: time=[start, stop]. You have supplied: time={time}.")
        else:
            # If they are all int or flat (or commonly occuring np.int64/np.float64)
            if all([isinstance(item, (int,float,np.int32,np.float32,np.int64,np.float64)) for item in time]):
                time = [round(item) for item in time]
                time[0] -= warmup
            # If they are all timestamps
            elif all([isinstance(item, pd.Timestamp) for item in time]):
                actual_start_date = time[0] - pd.Timedelta(days=warmup)
                time = [0, date_to_diff(actual_start_date, time[1])]
            # If they are all strings
            elif all([isinstance(item, str) for item in time]):
                time = [pd.Timestamp(item) for item in time]
                actual_start_date = time[0] - pd.Timedelta(days=warmup)
                time = [0, date_to_diff(actual_start_date, time[1])]
            else:
                raise ValueError(
                    f"List-like input of simulation start and stop must contain either all int/float or all str/datetime, not a combination of the two "
                    )
    elif isinstance(time, (str,datetime.datetime)):
        raise TypeError(
            "You have only provided one date as input 'time', how am I supposed to know when to start/end this simulation?"
        )
    else:
        raise TypeError(
                "Input argument 'time' must be a single number (int or float), a list of format: time=[start, stop], a string representing of a timestamp, or a timestamp"
            )

    if time[1] < time[0]:
        raise ValueError(
            "Start of simulation is chronologically after end of simulation"
        )
    elif time[0] == time[1]:
        # TODO: Might be usefull to just return the initial condition in this case?
        raise ValueError(
            "Start of simulation is the same as the end of simulation"
        )
    return time, actual_start_date

def validate_draw_function(draw_function, parameters, samples):
    """Validates the draw functions input and output. For use in the sim() function. """

    sig = inspect.signature(draw_function)
    keywords = list(sig.parameters.keys())
    # Verify that names of draw function are param_dict, samples_dict
    if keywords[0] != "param_dict":
        raise ValueError(
            f"The first parameter of a draw function should be 'param_dict'. Current first parameter: {keywords[0]}"
        )
    elif keywords[1] != "samples_dict":
        raise ValueError(
            f"The second parameter of a draw function should be 'samples_dict'. Current second parameter: {keywords[1]}"
        )
    elif len(keywords) > 2:
        raise ValueError(
            f"A draw function can only have two input arguments: 'param_dict' and 'samples_dict'. Current arguments: {keywords}"
        )
    # Call draw function
    cp_draws=copy.deepcopy(parameters)
    d = draw_function(parameters, samples)
    parameters = cp_draws
    if not isinstance(d, dict):
        raise TypeError(
            f"A draw function must return a dictionary. Found type {type(d)}"
        )
    if set(d.keys()) != set(parameters.keys()):
        raise ValueError(
            "Keys of model parameters dictionary returned by draw function do not match with the original dictionary.\n"
            "Missing keys: {0}. Redundant keys: {1}".format(set(parameters.keys()).difference(set(d.keys())), set(d.keys()).difference(set(parameters.keys())))
        )

def fill_initial_state_with_zero(state_names, initial_states):
    for state in state_names:
        if state in initial_states:
            state_values = initial_states[state]
    return state_values

def validate_dimensions(dimension_names, coordinates):
    """Checks if the combination of the `dimension_names` in the model defenition is compatible with the `coordinates` provided when initializing the model. Returns the dimension size of every dimension.
    """
    # Validate dimension
    if dimension_names:
        if not coordinates:
            raise ValueError(
                "dimension name provided in integrate function but no coordinates were given when model was initialised"
            )
        else:
            if set(dimension_names) != set(coordinates.keys()):
                raise ValueError(
                    "`dimension_names` do not match coordinates dictionary keys.\n"
                    "Missing dimension names: {0}. Redundant coordinates: {1}".format(set(dimension_names).difference(set(coordinates.keys())), set(coordinates.keys()).difference(set(dimension_names)))
                )
            else:
                dimension_size = []
                for i, (key,value) in enumerate(coordinates.items()):
                    try:
                        dimension_size.append(len(value))
                    except:
                        raise ValueError(
                            f"Unable to deduce dimension length from '{value}' of coordinate '{key}'"
                        )
    else:
        if coordinates:
            raise ValueError(
                f"You have provided a dictionary of coordinates with keys: {list(coordinates.keys())} upon model initialization but no variable 'dimension names' was found in the integrate function.\nDefine a variable 'dimension_names = {list(coordinates.keys())}' in the model definition."
            )
        else:
            dimension_size = [1]

    return dimension_size

def validate_state_dimensions(state_dimensions, coordinates, state_names):
    """Valide if length of `state_dimensions` is equal to the length of `state_names`. Check if the dimensions provided for every model state are existing dimensions."""

    # Length equal to state_names?   
    if len(state_dimensions) != len(state_names):
        raise ValueError(
            f"The length of `state_dimensions` ({len(state_dimensions)}) must match the length of `state_names` ({len(state_names)})"
        )
    # Contains only valid coordinates?
    for i,state_name in enumerate(state_names):
        if not all(x in coordinates.keys() for x in state_dimensions[i]):
            raise ValueError(
                f"The dimension names of model state '{state_name}', specified in position {i} of `state_dimensions` contains invalid coordinate names. Redundant names: {set(state_dimensions[i]).difference(set(coordinates.keys()))}"
        )

def build_state_sizes_dimensions(coordinates, state_names, state_dimensions):
    """A function returning three dictionaries: A dictionary containing, for every model state: 1) Its shape, 2) its dimensions, 3) its coordinates.
    """

    if not state_dimensions:
        if not coordinates:
            return dict(zip(state_names, len(state_names)*[(1,),] )), dict(zip(state_names, len(state_names)*[[],] )), dict(zip(state_names, len(state_names)*[[],] ))
        else:
            shape=[]
            dimension=[]
            coordinate=[]
            for key,value in coordinates.items():
                try:
                    shape.append(len(value))
                    dimension.append(key)
                    coordinate.append(value)
                except:
                    raise ValueError(
                            f"Unable to deduce dimension length from '{value}' of coordinate '{key}'"
                        )
            return dict(zip(state_names, len(state_names)*[tuple(shape),] )), dict(zip(state_names, len(state_names)*[dimension,])), dict(zip(state_names, len(state_names)*[coordinate,]))
    else:
        if not coordinates:
            raise ValueError(
                "`state_dimensions` found in the model defenition, however you have not provided `coordinates` when initializing the model. "
            )
        else:
            shapes=[]
            coords=[]
            for i,state_name in enumerate(state_names):
                dimensions = state_dimensions[i]
                if not dimensions:
                    shapes.append( (1,) )
                    coords.append( [] )
                else:
                    shape=[]
                    coord=[]
                    for dimension in dimensions:
                        try:
                            shape.append(len(coordinates[dimension]))
                            coord.append(coordinates[dimension])
                        except:
                            raise ValueError(
                                 f"Unable to deduce dimension length from '{coordinates[dimension]}' of coordinate '{dimension}'"
                            )
                    shapes.append(tuple(shape))
                    coords.append(coord)
            return dict(zip(state_names, shapes)), dict(zip(state_names, state_dimensions)),  dict(zip(state_names, coords))

def validate_parameter_function(func):
    # Validate the function passed to time_dependent_parameters
    sig = inspect.signature(func)
    keywords = list(sig.parameters.keys())
    if keywords[0] != "t":
        raise ValueError(
            "The first parameter of the parameter function should be 't'"
        )
    if keywords[1] != "states":
        raise ValueError(
            "The second parameter of the parameter function should be 'states'"
        )
    if keywords[2] != "param":
        raise ValueError(
            "The second parameter of the parameter function should be 'param'"
        )
    else:
        return keywords[3:]

def validate_time_dependent_parameters(parameter_names, parameter_stratified_names,time_dependent_parameters):

    extra_params = []
    all_param_names = parameter_names.copy()

    if parameter_stratified_names:
        if not isinstance(parameter_stratified_names[0], list):
            all_param_names += parameter_stratified_names
        else:
            for lst in parameter_stratified_names:
                all_param_names.extend(lst)
            
    for param, func in time_dependent_parameters.items():
        if param not in all_param_names:
            raise ValueError(
                "The specified time-dependent parameter '{0}' is not an "
                "existing model parameter".format(param))
        kwds = validate_parameter_function(func)
        extra_params.append(kwds)

    return extra_params

def validate_initial_states(state_shapes, initial_states):
    """
    A function to check the types and sizes of the model's initial states provided by the user.
    Automatically assumes non-specified states are equal to zero.
    All allowed input types converted to np.float64.

    Parameters
    ----------

    state_shapes: dict
        Contains the shape of every model state.
    
    initial_states: dict
        Dictionary containing the model's initial states. Keys: model states. Values: corresponding initial values.

    Returns
    -------
    
    initial_states: dict
        Dictionary containing the model's validated initial states.
        Types/Size checked, redundant initial states checked, states sorted according to `state_names`.
    """

    for state_name, state_shape in state_shapes.items():
        if state_name in initial_states:
            # if present, verify the length
            initial_states[state_name] = check_initial_states_shape(
                                        initial_states[state_name], state_shape, state_name, "initial state",
                                        )
        else:
            # Fill with zeros
            initial_states[state_name] = np.zeros(state_shape, dtype=np.float64)

    # validate the states (using `set` to ignore order)
    if set(initial_states.keys()) != set(state_shapes.keys()):
        raise ValueError(
            f"The specified initial states don't exactly match the predefined states. Redundant states: {set(initial_states.keys()).difference(set(state_shapes.keys()))}"
        )

    # sort the initial states to match the state_names
    initial_states = {state: initial_states[state] for state in state_shapes}

    return initial_states

def check_initial_states_shape(values, desired_shape, name, object_name):
    """ A function checking if the provided initial states have the correct shape
        Converts all values of initial states to type np.float64
    """

    # If the model doesn't have dimensions, initial states can be defined as: np.array([int/float]), [int/float], int or float
    # However these still need to converted to a np.array internally
    if list(desired_shape) == [1]:
        if not isinstance(values, (list,int,float,np.int32,np.int64,np.float32,np.float64,np.ndarray)):
            raise TypeError(
                f"{object_name} {name} must be of type int, float, or list. found {type(values)}"
            )
        else:
            if isinstance(values,(int,float)):
                values = np.asarray([values,])
        values = np.asarray(values, dtype=np.float64)

        if values.shape != desired_shape:
            raise ValueError(
                "The desired shape of model state '{name}' is {desired_shape}, but provided {obj} '{name}' "
                "has length {val}".format(
                    desired_shape=desired_shape, obj=object_name, name=name, val=values.shape
                )
            )
    else:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != desired_shape:
            raise ValueError(
                "The desired shape of model state '{name}' is {desired_shape}, but provided {obj} '{name}' "
                "has length {val}".format(
                    desired_shape=desired_shape, obj=object_name, name=name, val=values.shape
                )
            )

    return values

def check_duplicates(lst, name):
    """A function raising an error if lst contains duplicates"""
    seen = set()
    dupes = [x for x in lst if x in seen or seen.add(x)]
    if dupes:
        raise ValueError(
            f"List '{name}' contains duplicates: {dupes}"
        )

def merge_parameter_names_parameter_stratified_names(parameter_names, parameter_stratified_names):
    """ A function to merge the 'parameter_names' and 'parameter_stratified_names' lists"""
    merged_params = parameter_names.copy()
    if parameter_stratified_names:
        if not isinstance(parameter_stratified_names[0], list):
            if len(parameter_stratified_names) == 1:
                merged_params += parameter_stratified_names
            else:
                for stratified_names in parameter_stratified_names:
                    merged_params += [stratified_names,]
        else:
            for stratified_names in parameter_stratified_names:
                merged_params += stratified_names
    return merged_params

def validate_apply_transitionings_signature(apply_transitionings, parameter_names_merged, state_names):
    """A function checking the inputs of the apply_transitionings function"""

    # First argument should always be 't'
    sig = inspect.signature(apply_transitionings)
    keywords = list(sig.parameters.keys())
    if keywords[0] != "t":
        raise ValueError(
            "The first argument of the 'apply_transitionings' function should always be 't'"
        )
    elif keywords[1] != 'tau':
        raise ValueError(
            "The second argument of the 'apply_transitionings' function should always be 'tau'"
        )
    elif keywords[2] != 'transitionings':
        raise ValueError(
            "The third argument of the 'apply_transitionings' function should always be 'transitionings'"
        )

    # Verify all parameters and state follow after 't'/'tau'/'transitionings'
    if set(parameter_names_merged + state_names) != set(keywords[3:]):
        # Extract redundant and missing parameters
        redundant_all = set(keywords[3:]).difference(set(parameter_names_merged + state_names))
        missing_all = set(parameter_names_merged + state_names).difference(set(keywords[1:]))
        # Let's split the missing variables in parameters/states for extra clarity
        missing_states = [name for name in missing_all if name in state_names]
        missing_parameters = [name for name in missing_all if name in parameter_names_merged]
        raise ValueError(
            "The provided state names and parameters don't match the parameters and states of the apply_transitionings function. "
            "Missing parameters: {0}. Missing states: {1}. "
            "Redundant: {2}. ".format(missing_parameters, missing_states, list(redundant_all))
        )

def validate_integrate_or_compute_rates_signature(integrate_func, parameter_names_merged, state_names, _function_parameters):
    """A function checking the inputs of the integrate/compute_rates function"""

    # First argument should always be 't'
    sig = inspect.signature(integrate_func)
    keywords = list(sig.parameters.keys())
    if keywords[0] != "t":
        raise ValueError(
            "The first argument of the integrate/compute_rates function should always be 't'"
        )

    # Verify all parameters and state follow after 't'
    if set(parameter_names_merged + state_names) != set(keywords[1:]):
        # Extract redundant and missing parameters
        redundant_all = set(keywords[1:]).difference(set(parameter_names_merged + state_names))
        missing_all = set(parameter_names_merged + state_names).difference(set(keywords[1:]))
        # Let's split the missing variables in parameters/states for extra clarity
        missing_states = [name for name in missing_all if name in state_names]
        missing_parameters = [name for name in missing_all if name in parameter_names_merged]
        raise ValueError(
            "The provided state names and parameters don't match the parameters and states of the integrate/compute_rates function. "
            "Missing parameters: {0}. Missing states: {1}. "
            "Redundant: {2}. ".format(missing_parameters, missing_states, list(redundant_all))
        )

    all_params = parameter_names_merged.copy()
    # additional parameters from time-dependent parameter functions
    # are added to specified_params after the above check
    if _function_parameters:
        _extra_params = [item for sublist in _function_parameters for item in sublist]
        # Remove duplicate arguments in time dependent parameter functions
        _extra_params = OrderedDict((x, True) for x in _extra_params).keys()
        all_params += _extra_params
        len_before = len(all_params)
        # Line below removes duplicate arguments with integrate defenition
        all_params = OrderedDict((x, True) for x in all_params).keys()
        len_after = len(all_params)
        # Line below computes number of integrate arguments used in time dependent parameter functions
        n_duplicates = len_before - len_after
        _n_function_params = len(_extra_params) - n_duplicates
    else:
        _n_function_params = 0
        _extra_params = []
    
    return all_params, list(_extra_params)

def validate_provided_parameters(all_params, parameters):
    """Verify all parameters (parameters + stratified parameters + TDPF parameters) are provided by the user. Verify 't' is not a TDPF parameter."""

    # Validate the params
    if set(parameters.keys()) != set(all_params):
        raise ValueError(
            "The specified parameters don't exactly match the predefined parameters. "
            "Redundant parameters: {0}. Missing parameters: {1}".format(
            set(parameters.keys()).difference(set(all_params)),
            set(all_params).difference(set(parameters.keys())))
        )
    parameters = {param: parameters[param] for param in all_params}

    # After building the list of all model parameters, verify no parameters 't' were used
    if 't' in parameters:
        raise ValueError(
        "Parameter name 't' is reserved for the timestep of scipy.solve_ivp.\nPlease verify no model parameters named 't' are present in the model parameters dictionary or in the time-dependent parameter functions."
            )
    
    return parameters

def check_stratpar_size(values, name, object_name,dimension_names,desired_size):
    """Checks the size of stratified parameters. Converts stratified parameters to a numpy array."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(
            "A {obj} value should be a 1D array, but {obj} '{name}' is"
            "{val}-dimensional".format(
                obj=object_name, name=name, val=values.ndim
            )
        )
    if len(values) != desired_size:
        raise ValueError(
            "The coordinates provided for dimension '{dimension_names}' indicates a "
            "dimension size of {desired_size}, but {obj} '{name}' "
            "has length {val}".format(
                dimension_names=dimension_names, desired_size=desired_size,
                obj=object_name, name=name, val=len(values)
            )
        )
    return values

def validate_parameter_stratified_sizes(parameter_stratified_names, dimension_names, coordinates, parameters):
    """Check if the sizes of the stratified parameters are correct"""

    if not isinstance(parameter_stratified_names[0], list):
        if len(list(coordinates.keys())) > 1:
            raise ValueError(
                f"The model has more than one dimension ({len(list(coordinates.keys()))}). I cannot deduce the dimension of your statified parameter from the provided `parameter_stratified_names`: {parameter_stratified_names}. "
                f"Make sure `parameter_stratified_names` is a list, containing {len(list(coordinates.keys()))} (no. dimensions) sublists. Each sublist corresponds to a dimension in `dimension_names`. "
                "Place your stratified parameter in the correct sublist so I know what model dimension to match it with. If a dimension has no stratified parameter, provide an empty list."
            )
        else:
            for param in parameter_stratified_names:
                parameters[param] = check_stratpar_size(
                                        parameters[param], param, "stratified parameter",dimension_names[0],len(coordinates[dimension_names[0]])
                                        )
    else:
        # Number of sublists equal to number of dimensions?
        if len(parameter_stratified_names) != len(list(coordinates.keys())):
            raise ValueError(
                f"The number of sublists in `parameter_stratified_names` ({len(parameter_stratified_names)}) must be equal to the number of coordinates ({len(list(coordinates.keys()))})"
            )
        for j,stratified_names in enumerate(parameter_stratified_names):
            for param in stratified_names:
                parameters[param] = check_stratpar_size(
                                        parameters[param], param, "stratified parameter",dimension_names[j],len(coordinates[dimension_names[j]])
                                        )
    return parameters

def validate_integrate(initial_states, parameters, integrate, state_shapes):
    """ Call _create_fun to check if the integrate function (1) works, (2) the differentials have the correct shape
    """

    try:
        # Call the integrate function
        dstates = integrate(1, **initial_states, **parameters)
        # Flatten
        out=[]
        for d in dstates:
            out.extend(list(np.ravel(d)))
    except:
        raise ValueError(
            "An error was encountered while calling your integrate function."
        )
    # Flatten initial states
    y0=[]
    for v in initial_states.values():
        y0.extend(list(np.ravel(v)))
    # Assert length equality
    if len(out) != len(y0):
        raise ValueError(
            "The total length of the differentials returned by your `integrate()` function do not appear to have the correct length. "
            "Verify the differentials are in the same order as `state_names`. Verify every differential has the same size as the state it corresponds to. "
            f"State shapes: {state_shapes}"
        )

def validate_compute_rates(compute_rates, initial_states, states_shape, parameter_names_merged, parameters):
    """A function to call compute_rates and check if the output type and contents are alright"""

    # Throw TDPF parameters out of parameters dictionary
    parameters_wo_TDPF_pars={k:v for k, v in parameters.items() if k in parameter_names_merged}
    # 'func' in class 'SDEModel' of 'base.py' automatically converts states to np.array
    # However, we do not wish to validate the output of 'func' but rather of its consituent functions: compute_rates, apply_transitionings
    initial_states_copy={k: v[:] for k, v in initial_states.items()}
    for k,v in initial_states.items():
        initial_states_copy[k] = np.asarray(initial_states[k])
    # Call the function with initial values to check if the function returns the right format of dictionary
    rates = compute_rates(10, **initial_states_copy, **parameters_wo_TDPF_pars)
    # Check if a dictionary is returned
    if not isinstance(rates, dict):
        raise TypeError("Output of function 'compute_rates' should be of type dictionary")
    # Check if all states present are valid model parameters
    for state in rates.keys():
        if not state in initial_states.keys():
            raise ValueError(
                f"Rate found for an invalid model state '{state}'"
            )
    # Check if all rates are given as a list
    for state_name, rate_list in rates.items():
        if not isinstance(rate_list, list):
            raise ValueError(
                f"Rate(s) of model state '{state_name}' are not containted within a list."
            )
    # Check if the sizes are correct
    for state_name, rate_list in rates.items():
        for i,rate in enumerate(rate_list):
            if not isinstance(rate, np.ndarray):
                raise TypeError(f"the rate of the {i}th transitioning for state '{state_name}' coming from function `compute_rates` is not of type 'np.ndarray' but {type(rate)}")
            if rate.shape != states_shape[state_name]:
                raise ValueError(f"State {state_name} has size {states_shape[state_name]}, but rate of the {i}th transitioning for state '{state_name}' has shape {list(rate.shape)}")
    return rates

def validate_apply_transitionings(apply_transitionings, rates, initial_states, states_shape, parameter_names_merged, parameters):
    """A function to call apply_transitionings and check if the output type and contents are alright"""
    
    # Throw TDPF parameters out of parameters dictionary
    parameters_wo_TDPF_pars={k:v for k, v in parameters.items() if k in parameter_names_merged}
    # Update states
    new_states = apply_transitionings(10, 1, rates, **initial_states, **parameters_wo_TDPF_pars)
    # Check
    if len(list(new_states)) != len(initial_states.keys()):
        raise ValueError(f"The number of outputs of function 'apply_transitionings_func' ({len(list(new_states))}) is not equal to the number of states ({len(initial_states.keys())})")
    for i, new_state in enumerate(list(new_states)):
        if not isinstance(new_state, np.ndarray):
            raise TypeError(f"Output state of function 'apply_transitionings_func' in position {i} is not a np.ndarray")
        if new_state.shape != states_shape[list(initial_states.keys())[i]]:
                raise ValueError(f"State {list(initial_states.keys())[i]} has size {states_shape[list(initial_states.keys())[i]]}, but rate of the {i}th transitioning for state '{list(initial_states.keys())[i]}' has shape {list(rate.shape)}")