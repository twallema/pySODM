import copy
import inspect
import itertools
import numpy as np
import pandas as pd
from collections import OrderedDict

def date_to_diff(actual_start_date, end_date):
    """
    Convert date string to int (i.e. number of days since day 0 of simulation,
    which is warmup days before actual_start_date)
    """
    return float((pd.to_datetime(end_date)-pd.to_datetime(actual_start_date))/pd.to_timedelta('1D'))

def validate_simulation_time(time, warmup):
    """Validates the simulation time of the sim() function. """

    actual_start_date=None
    if isinstance(time, float):
        time = [0-warmup, round(time)]
    elif isinstance(time, int):
        time = [0-warmup, time]
    elif isinstance(time, list):
        if not len(time) == 2:
            raise ValueError(f"Length of list-like input of simulation start and stop is two. You have supplied: time={time}. 'Time' must be of format: time=[start, stop].")
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
                raise TypeError(
                    f"List-like input of simulation start and stop must contain either all int/float or all strings or all pd.Timestamps "
                    )
    else:
        raise TypeError(
                "Input argument 'time' must be a single number (int or float), a list of format: time=[start, stop], a string representing of a timestamp, or a timestamp"
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

def validate_stratifications(stratification_names, coordinates):
    """Checks if the combination of the `stratification_names` in the model defenition is compatible with the `coordinates` provided when initializing the model. Returns the stratification size of every stratification.
    """
    # Validate stratification
    if stratification_names:
        if not coordinates:
            raise ValueError(
                "Stratification name provided in integrate function but no coordinates were given when model was initialised"
            )
        else:
            if set(stratification_names) != set(coordinates.keys()):
                raise ValueError(
                    "Stratification names do not match coordinates dictionary keys.\n"
                    "Missing stratification names: {0}. Redundant coordinates: {1}".format(set(stratification_names).difference(set(coordinates.keys())), set(coordinates.keys()).difference(set(stratification_names)))
                )
            else:
                stratification_size = []
                for i, (key,value) in enumerate(coordinates.items()):
                    try:
                        stratification_size.append(len(value))
                    except:
                        raise ValueError(
                            f"Unable to deduce stratification length from '{value}' of coordinate '{key}'"
                        )
    else:
        if coordinates:
            raise ValueError(
                f"You have provided a dictionary of coordinates with keys: {list(coordinates.keys())} upon model initialization but no variable 'stratification names' was found in the integrate function.\nDefine a variable 'stratification_names = {list(coordinates.keys())}' in the model definition."
            )
        else:
            stratification_size = [1]

    return stratification_size

def validate_state_stratifications(state_stratifications, coordinates, state_names):
    """Valide if length of `state_stratifications` is equal to the length of `state_names`. Check if the stratifications provided for every model state are existing stratifications."""

    # Length equal to state_names?   
    if len(state_stratifications) != len(state_names):
        raise ValueError(
            f"The length of `state_stratifications` ({len(state_stratifications)}) must match the length of `state_names` ({len(state_names)})"
        )
    # Contains only valid coordinates?
    for i,state_name in enumerate(state_names):
        if not all(x in coordinates.keys() for x in state_stratifications[i]):
            raise ValueError(
                f"The stratification names of model state '{state_name}', specified in position {i} of `state_stratifications` contains invalid coordinate names. Redundant names: {set(state_stratifications[i]).difference(set(coordinates.keys()))}"
        )

def build_state_sizes(coordinates, state_names, state_stratifications):
    """A function returning a dictionary containing, for every model state, the correct shape.
    """

    if not state_stratifications:
        if not coordinates:
            return dict(zip(state_names, len(state_names)*[(1,),] ))
        else:
            shape=[]
            for key,value in coordinates.items():
                try:
                    shape.append(len(value))
                except:
                    raise ValueError(
                            f"Unable to deduce stratification length from '{value}' of coordinate '{key}'"
                        )
            return dict(zip(state_names, len(state_names)*[tuple(shape),] ))
    else:
        if not coordinates:
            raise ValueError(
                "`state_stratifications` found in the model defenition, however you have not provided `coordinates` when initializing the model. "
            )
        else:
            shapes=[]
            for i,state_name in enumerate(state_names):
                stratifications = state_stratifications[i]
                if not stratifications:
                    shapes.append( (1,) )
                else:
                    shape=[]
                    for stratification in stratifications:
                        try:
                            shape.append(len(coordinates[stratification]))
                        except:
                            raise ValueError(
                                 f"Unable to deduce stratification length from '{coordinates[stratification]}' of coordinate '{stratification}'"
                            )
                    shapes.append(tuple(shape))
            return dict(zip(state_names, shapes))


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

def validate_initial_states(state_names, initial_states, stratification_size, coordinates):
    """
    A function to check the types and sizes of the model's initial states provided by the user.
    Automatically assumes non-specified states are equal to zero.

    Parameters
    ----------

    state_names: list
        Contains model state names (type: str)
    
    initial_states: dict
        Dictionary containing the model's initial states. Keys: model states. Values: corresponding initial values.

    stratification_size: list
        Contains the number of coordinates of every stratification
    
    coordinates: dict
        Keys: stratification name, Values: coordinates.

    Returns
    -------
    
    initial_states: dict
        Dictionary containing the model's validated initial states.
        Types/Size checked, redundant initial states checked, states sorted according to `state_names`.
    """

    for state in state_names:
        if state in initial_states:
            # if present, verify the length
            initial_states[state] = check_initial_states_size(
                                        initial_states[state], state, "initial state", stratification_size, coordinates
                                        )
        else:
            # Fill with zeros
            initial_states[state] = np.zeros(stratification_size)

    # validate the states (using `set` to ignore order)
    if set(initial_states.keys()) != set(state_names):
        raise ValueError(
            f"The specified initial states don't exactly match the predefined states. Redundant states: {set(initial_states.keys()).difference(set(state_names))}"
        )

    # sort the initial states to match the state_names
    initial_states = {state: initial_states[state] for state in state_names}

    return initial_states

def check_initial_states_size(values, name, object_name, stratification_size, coordinates):
    """A function checking the size of an initial state
    """
    # If the model doesn't have stratifications, initial states can be defined as: np.array([int/float]), [int/float], int or float
    # However these still need to converted to a np.array internally
    if stratification_size == [1]:
        if not isinstance(values, (list,int,float,np.int32,np.int64,np.float32,np.float64,np.ndarray)):
            raise TypeError(
                f"{object_name} {name} must be of type int, float, or list. found {type(values)}"
            )
        else:
            if isinstance(values,(int,float)):
                values = np.asarray([values,])
        values = np.asarray(values)

        if list(values.shape) != stratification_size:
            raise ValueError(
                "The abscence of model coordinates indicate a desired "
                "model states size of {strat_size}, but {obj} '{name}' "
                "has length {val}".format(
                    strat_size=stratification_size,
                    obj=object_name, name=name, val=list(values.shape)
                )
            )

    else:
        values = np.asarray(values)
        if list(values.shape) != stratification_size:
            raise ValueError(
                "The coordinates provided for the stratifications '{strat}' indicate a "
                "model states size of {strat_size}, but {obj} '{name}' "
                "has length {val}".format(
                    strat=list(coordinates.keys()), strat_size=stratification_size,
                    obj=object_name, name=name, val=list(values.shape)
                )
            )

    return values

def validate_stratified_parameters(values, name, object_name,i,stratification_size,coordinates):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(
            "A {obj} value should be a 1D array, but {obj} '{name}' is"
            "{val}-dimensional".format(
                obj=object_name, name=name, val=values.ndim
            )
        )
    if len(values) != stratification_size[i]:
        raise ValueError(
            "The coordinates provided for stratification '{strat}' indicates a "
            "stratification size of {strat_size}, but {obj} '{name}' "
            "has length {val}".format(
                strat=list(coordinates.keys())[i], strat_size=stratification_size[i],
                obj=object_name, name=name, val=len(values)
            )
        )

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

def validate_ODEModel(initial_states, parameters, coordinates, stratification_size, state_names, parameter_names,
                        parameter_stratified_names, _function_parameters, _create_fun, integrate_func):
    """
    This does some basic validation of the model + initialization:

    1) Validation of the integrate function to ensure it matches with
    the specified `state_names`, `parameter_names`, etc.
    This is actually a validation of the model class itself, but it is
    easier to do this only on initialization of a model instance.

    2) Validation of the actual initialization with initial values for the
    states and parameter values.

    """

    # Validate Model class definition (the integrate function)
    # First argument should always be 't'
    sig = inspect.signature(integrate_func)
    keywords = list(sig.parameters.keys())
    if keywords[0] != "t":
        raise ValueError(
            "The first argument of the integrate function should always be 't'"
        )
    # Add parameters and stratified parameters to one list: specified_params
    specified_params = merge_parameter_names_parameter_stratified_names(parameter_names, parameter_stratified_names)
    # Check if all the states and parameters are present
    if set(specified_params + state_names) != set(keywords[1:]):
        # Extract redundant and missing parameters
        redundant_all = set(keywords[1:]).difference(set(specified_params + state_names))
        missing_all = set(specified_params + state_names).difference(set(keywords[1:]))
        # Let's split the missing variables in parameters/states for extra clarity
        for state_name in state_names:
            missing_states = [name for name in missing_all if name in state_names]
        for parameter_name in parameter_names:
            missing_parameters = [name for name in missing_all if name in parameter_names]
        raise ValueError(
            "The provided state names and parameters don't match the parameters and states of the integrate function. "
            "Missing parameters: {0}. Missing states: {1}. "
            "Redundant: {2}. ".format(missing_parameters, missing_states, list(redundant_all))
        )
    # additional parameters from time-dependent parameter functions
    # are added to specified_params after the above check
    if _function_parameters:
        _extra_params = [item for sublist in _function_parameters for item in sublist]
        # Remove duplicate arguments in time dependent parameter functions
        _extra_params = OrderedDict((x, True) for x in _extra_params).keys()
        specified_params += _extra_params
        len_before = len(specified_params)
        # Line below removes duplicate arguments with integrate defenition
        specified_params = OrderedDict((x, True) for x in specified_params).keys()
        len_after = len(specified_params)
        # Line below computes number of integrate arguments used in time dependent parameter functions
        n_duplicates = len_before - len_after
        _n_function_params = len(_extra_params) - n_duplicates
    else:
        _n_function_params = 0
        _extra_params = []

    # Validate the params
    if set(parameters.keys()) != set(specified_params):
        raise ValueError(
            "The specified parameters don't exactly match the predefined parameters. "
            "Redundant parameters: {0}. Missing parameters: {1}".format(
            set(parameters.keys()).difference(set(specified_params)),
            set(specified_params).difference(set(parameters.keys())))
        )

    parameters = {param: parameters[param] for param in specified_params}

    # After building the list of all model parameters, verify no parameters 't' were used
    if 't' in parameters:
        raise ValueError(
        "Parameter name 't' is reserved for the timestep of scipy.solve_ivp.\nPlease verify no model parameters named 't' are present in the model parameters dictionary or in the time-dependent parameter functions."
            )

    # the size of the stratified parameters
    if parameter_stratified_names:
        i = 0
        if not isinstance(parameter_stratified_names[0], list):
            if len(parameter_stratified_names) == 1:
                for param in parameter_stratified_names:
                    validate_stratified_parameters(
                            parameters[param], param, "stratified parameter",i,stratification_size,coordinates
                        )
                i = i + 1
            else:
                for param in parameter_stratified_names:
                    validate_stratified_parameters(
                            parameters[param], param, "stratified parameter",i,stratification_size,coordinates
                        )
                i = i + 1
        else:
            for stratified_names in parameter_stratified_names:
                for param in stratified_names:
                    validate_stratified_parameters(
                        parameters[param], param, "stratified parameter",i,stratification_size,coordinates
                    )
                i = i + 1

    # Size/type of the initial states
    # Redundant states
    # Fill in unspecified states with zeros
    initial_states = validate_initial_states(state_names, initial_states, stratification_size, coordinates)


    # Call integrate function with initial values to check if the function returns all states
    fun = _create_fun(None)
    y0 = list(itertools.chain(*initial_states.values()))
    while np.array(y0).ndim > 1:
        y0 = list(itertools.chain(*y0))
    check = True
    try:
        result = fun(pd.Timestamp('2020-09-01'), np.array(y0), parameters)
    except:
        try:
            result = fun(1, np.array(y0), parameters)
        except:
            check = False
    if check:
        if len(result) != len(y0):
            raise ValueError(
                "The return value of the integrate function does not have the correct length."
            )

    return initial_states, parameters, _n_function_params, list(_extra_params)

def validate_SDEModel(initial_states, parameters, coordinates, stratification_size, state_names, parameter_names,
                        parameter_stratified_names, _function_parameters, compute_rates_func, apply_transitionings_func):
    """
    This does some basic validation of the model + initialization:

    1) Validation of the integrate function to ensure it matches with
    the specified `state_names`, `parameter_names`, etc.
    This is actually a validation of the model class itself, but it is
    easier to do this only on initialization of a model instance.

    2) Validation of the actual initialization with initial values for the
    states and parameter values.

    """

    #############################
    ## Validate the signatures ##
    #############################

    # Compute_rates function
    # ~~~~~~~~~~~~~~~~~~~~~~

    sig = inspect.signature(compute_rates_func)
    keywords = list(sig.parameters.keys())
    if keywords[0] != "t":
        raise ValueError(
            "The first argument of the 'compute_rates' function should always be 't'"
        )
    # Add parameters and stratified parameters to one list: specified_params
    specified_params = merge_parameter_names_parameter_stratified_names(parameter_names, parameter_stratified_names)
    # Check if all the states and parameters are present
    if set(specified_params + state_names) != set(keywords[1:]):
        # Extract redundant and missing parameters
        redundant_all = set(keywords[1:]).difference(set(specified_params + state_names))
        missing_all = set(specified_params + state_names).difference(set(keywords[1:]))
        # Let's split the missing variables in parameters/states for extra clarity
        for state_name in state_names:
            missing_states = [name for name in missing_all if name in state_names]
        for parameter_name in parameter_names:
            missing_parameters = [name for name in missing_all if name in parameter_names]
        raise ValueError(
            "The provided state names and parameters don't match the parameters and states of the compute_rates function. "
            "Missing parameters: {0}. Missing states: {1}. "
            "Redundant: {2}. ".format(missing_parameters, missing_states, list(redundant_all))
        )
    # Save a copy to validate the model later on
    specified_params_wo_TDPF_pars = specified_params.copy()
    # additional parameters from time-dependent parameter functions
    # are added to specified_params after the above check
    if _function_parameters:
        _extra_params = [item for sublist in _function_parameters for item in sublist]
        # Remove duplicate arguments in time dependent parameter functions
        _extra_params = OrderedDict((x, True) for x in _extra_params).keys()
        specified_params += _extra_params
        len_before = len(specified_params)
        # Line below removes duplicate arguments with integrate defenition
        specified_params = OrderedDict((x, True) for x in specified_params).keys()
        len_after = len(specified_params)
        # Line below computes number of integrate arguments used in time dependent parameter functions
        n_duplicates = len_before - len_after
        _n_function_params = len(_extra_params) - n_duplicates
    else:
        _n_function_params = 0
        _extra_params = []

    # Validate the params
    if set(parameters.keys()) != set(specified_params):
        raise ValueError(
            "The specified parameters don't exactly match the predefined parameters. "
            "Redundant parameters: {0}. Missing parameters: {1}".format(
            set(parameters.keys()).difference(set(specified_params)),
            set(specified_params).difference(set(parameters.keys())))
        )

    parameters_compute_rates = {param: parameters[param] for param in specified_params}

    # apply_transitionings function
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    sig = inspect.signature(apply_transitionings_func)
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

    # Add parameters and stratified parameters to one list: specified_params
    specified_params = merge_parameter_names_parameter_stratified_names(parameter_names, parameter_stratified_names)
    # Check if all the states and parameters are present
    if set(specified_params + state_names) != set(keywords[3:]):
        # Extract redundant and missing parameters
        redundant_all = set(keywords[3:]).difference(set(specified_params + state_names))
        missing_all = set(specified_params + state_names).difference(set(keywords[3:]))
        # Let's split the missing variables in parameters/states for extra clarity
        for state_name in state_names:
            missing_states = [name for name in missing_all if name in state_names]
        for parameter_name in parameter_names:
            missing_parameters = [name for name in missing_all if name in parameter_names]
        raise ValueError(
            "The provided state names and parameters don't match the parameters and states of the apply_transitionings function. "
            "Missing parameters: {0}. Missing states: {1}. "
            "Redundant: {2}. ".format(missing_parameters, missing_states, list(redundant_all))
        )
    # additional parameters from time-dependent parameter functions
    # are added to specified_params after the above check
    if _function_parameters:
        _extra_params = [item for sublist in _function_parameters for item in sublist]
        # Remove duplicate arguments in time dependent parameter functions
        _extra_params = OrderedDict((x, True) for x in _extra_params).keys()
        specified_params += _extra_params
        len_before = len(specified_params)
        # Line below removes duplicate arguments with integrate defenition
        specified_params = OrderedDict((x, True) for x in specified_params).keys()
        len_after = len(specified_params)
        # Line below computes number of integrate arguments used in time dependent parameter functions
        n_duplicates = len_before - len_after
        _n_function_params = len(_extra_params) - n_duplicates
    else:
        _n_function_params = 0
        _extra_params = []
    # Validate the params
    if set(parameters.keys()) != set(specified_params):
        raise ValueError(
            "The specified parameters don't exactly match the predefined parameters. "
            "Redundant parameters: {0}. Missing parameters: {1}".format(
            set(parameters.keys()).difference(set(specified_params)),
            set(specified_params).difference(set(parameters.keys())))
        )

    parameters = {param: parameters[param] for param in specified_params}

    # Assert equality as a sanity check
    if set(parameters.keys()) != set(parameters_compute_rates.keys()):
        raise ValueError(
            "The model parameters derived from the 'compute_rates' function do not match the model parameters of the 'apply_transitionings' function."
            "Different keys: {0}".format(
            set(parameters.keys()).difference(set(parameters_compute_rates.keys())))
        )

    # After building the list of all model parameters, verify no parameters 't'/'tau' or 'transitionings' were used
    if 't' in parameters:
        raise ValueError(
            "Parameter name 't' is reserved for the simulation time.\nPlease verify no model parameters named 't' are present in the model parameters dictionary or in the time-dependent parameter functions."
            )
    if 'tau' in parameters:
        raise ValueError(
            "Parameter name 'tau' is reserved for the simulation time.\nPlease verify no model parameters named 't' are present in the model parameters dictionary or in the time-dependent parameter functions."
            )
    if 'transitionings' in parameters:
        raise ValueError(
            "Parameter name 'transitionings' is reserved for the simulation time.\nPlease verify no model parameters named 't' are present in the model parameters dictionary or in the time-dependent parameter functions."
            )
    ###############################################################################
    ## Validate the initial_states / stratified params having the correct length ##
    ###############################################################################

    # the size of the stratified parameters
    if parameter_stratified_names:
        i = 0
        if not isinstance(parameter_stratified_names[0], list):
            if len(parameter_stratified_names) == 1:
                for param in parameter_stratified_names:
                    validate_stratified_parameters(
                            parameters[param], param, "stratified parameter",i,stratification_size,coordinates
                        )
                i = i + 1
            else:
                for param in parameter_stratified_names:
                    validate_stratified_parameters(
                            parameters[param], param, "stratified parameter",i,stratification_size,coordinates
                        )
                i = i + 1
        else:
            for stratified_names in parameter_stratified_names:
                for param in stratified_names:
                    validate_stratified_parameters(
                        parameters[param], param, "stratified parameter",i,stratification_size,coordinates
                    )
                i = i + 1

    # Size/type of the initial states
    # Redundant states
    # Fill in unspecified states with zeros
    initial_states = validate_initial_states(state_names, initial_states, stratification_size, coordinates, None)


    #########################################################################
    # Validate the 'compute_rates' and 'apply_transitionings' by calling it #
    #########################################################################

    # 'func' in class 'SDEModel' of 'base.py' automatically converts states to np.array
    # However, we do not wish to validate the output of 'func' but rather of its consituent functions: compute_rates, apply_transitionings
    initial_states_copy={k: v[:] for k, v in initial_states.items()}
    for k,v in initial_states.items():
        initial_states[k] = np.asarray(initial_states[k])

    # compute_rates
    # ~~~~~~~~~~~~~
    # TODO: Throw out _extra_params here
    parameters_wo_TDPF_pars = {param: parameters[param] for param in specified_params_wo_TDPF_pars}
    # Call the function with initial values to check if the function returns the right format of dictionary
    rates = compute_rates_func(10, **initial_states, **parameters_wo_TDPF_pars)
    # Check if a dictionary is returned
    if not isinstance(rates, dict):
        raise TypeError("Output of function 'compute_rates' should be of type dictionary")
    # Check if all states present are valid model parameters
    for state in rates.keys():
        if not state in state_names:
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
            if list(rate.shape) != stratification_size:
                raise ValueError(f"The provided coordinates indicate a state size of {stratification_size}, but rate of the {i}th transitioning for state '{state_name}' has shape {list(rate.shape)}")
    
    # apply_transitionings
    # ~~~~~~~~~~~~~~~~~~~~

    new_states = apply_transitionings_func(10, 1, rates, **initial_states, **parameters_wo_TDPF_pars)
    # Check
    if len(list(new_states)) != len(state_names):
        raise ValueError(f"The number of outputs of function 'apply_transitionings_func' ({len(list(new_states))}) is not equal to the number of states ({len(state_names)})")
    for i, new_state in enumerate(list(new_states)):
        if not isinstance(new_state, np.ndarray):
            raise TypeError(f"Output state of function 'apply_transitionings_func' in position {i} is not a np.ndarray")
        if list(new_state.shape) != stratification_size:
            raise ValueError(f"The provided coordinates indicate a state size of {stratification_size}, but the {i}th output of function 'apply_transitionings_func' has shape {list(new_state.shape)}")
    
    # Reset initial states
    initial_states={k: v[:] for k, v in initial_states_copy.items()}

    return initial_states, parameters, _n_function_params, list(_extra_params)
