import copy
import math
import inspect
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta

def check_formatting_names(states_names, parameters_names, parameters_stratified_names, dimensions_names):
    """
    A function checking the format of the 'states', 'parameters', 'stratified_parameters' and 'dimensions' provided by the user in their model class
    """

    # states_names/parameters_names
    for val, name in zip([states_names, parameters_names], ['states', 'parameters']):
        ## list?
        assert isinstance(val, list), f"'{name}' must be a list. found '{type(val)}'."
        ## containing strings
        if not all(isinstance(v, str) for v in val):
            raise TypeError(f"not all elements in '{name}' are of type str.")
    
    # dimensions
    if dimensions_names:
        # list?
        assert isinstance(dimensions_names, list), f"'dimensions' must be a list. found '{type(dimensions_names)}'."
        # containing only strings
        if not all(isinstance(v, str) for v in dimensions_names):
            raise TypeError(f"not all elements in 'dimensions' are of type str.")
        # exract number of dimensions
        n_dims = len(dimensions_names)

    # parameters_stratified_names
    if parameters_stratified_names:
        if not dimensions_names:
            raise TypeError(f"a model without dimensions cannot have stratified parameters.")
        elif n_dims == 1:
            # list?
            assert isinstance(parameters_stratified_names, list), f"'stratified_parameters' must be a list. found '{type(parameters_stratified_names)}'."
            # containing only strings?
            if not all(isinstance(v, str) for v in parameters_stratified_names):
                raise TypeError(f"not all elements in 'stratified_parameters' are of type str.")
        elif n_dims >= 2:
            # list?
            assert isinstance(parameters_stratified_names, list), f"'stratified_parameters' must be a list. found '{type(parameters_stratified_names)}'."
            # containing n_dims sublists?
            assert len(parameters_stratified_names) == n_dims, f"'stratified_parameters' must be a list containing {n_dims} sublists."
            if not all(isinstance(sublist, list) for sublist in parameters_stratified_names):
                raise TypeError(f"'stratified_parameters' must be a list containing {n_dims} sublists.")
            # each containing only strings OR being empty?
            for sublist in parameters_stratified_names:
                if ((not all(isinstance(v, str) for v in sublist)) & (sublist != [])):
                    raise TypeError(f"'stratified_parameters' must be a list containing {n_dims} sublists. each sublist must either be empty or contain only str.")


from pySODM.models.utils import time_unit_map
def date_to_diff(start_date, end_date, time_unit):
    """
    Computes the amount of `time_unit` steps between `start_date` and `end_date`
    Rounds upward, f.i. there are approx. 4.5 weeks in a month, but this will be rounded to 5 weeks of simulation if `time_unit` is set to weeks
    """

    # count number of days
    delta_us = (end_date - start_date).total_seconds() * 1e6
    
    # check time_unit
    if time_unit not in time_unit_map:
        raise ValueError(f"Invalid time unit '{time_unit}'. Choose from {list(time_unit_map.keys())}")
    
    # compute number of time units
    num_units = math.ceil(delta_us / time_unit_map[time_unit])

    # generate the datetime range for the output
    date_range = [start_date + timedelta(microseconds=i*time_unit_map[time_unit]) for i in range(num_units+1)]

    return [0, len(date_range)-1], date_range


def validate_simulation_time(time, time_unit):
    """Validates the simulation time of the sim() function. Various input types are converted to: time = [start_float, stop_float]"""

    date_range=[None,]
    if isinstance(time, float):
        time = [0, round(time)]
    elif isinstance(time, int):
        time = [0, time]
    elif isinstance(time, list):
        if not len(time) == 2:
            raise ValueError(f"wrong length of list-like simulation start and stop (length: {len(time)}). correct format: time=[start, stop] (length: 2).")
        else:
            # If they are all int or flat (or commonly occuring np.int/np.float)
            if all([isinstance(item, (int,float,np.int32,np.float32,np.int64,np.float64)) for item in time]):
                time = [round(item) for item in time]
            # If they are all datetime
            elif all([isinstance(item, datetime) for item in time]):
                check_chronology(time)
                time, date_range = date_to_diff(time[0], time[1], time_unit)
            # If they are all strings: assume format is YYYY-MM-DD and convert to a datetime
            elif all([isinstance(item, str) for item in time]):
                time = [datetime.strptime(item,"%Y-%m-%d") for item in time]
                check_chronology(time)
                time, date_range = date_to_diff(time[0], time[1], time_unit)
            else:
                types = [type(t) for t in time]
                raise ValueError(
                    "simulation start and stop must have the format: time=[start, stop]."
                    " 'start' and 'stop' must have the same datatype: int/float, str ('yyyy-mm-dd'), or datetime."
                    f" mixing of types is not allowed. you supplied: {types} "
                    )
    else:
        raise TypeError(
                "'time' must be 1) a single int/float representing the end of the simulation, 2) a list of format: time=[start, stop]."
            )

    # check chronology (for all cases we haven't checked yet)
    check_chronology(time)

    return time, date_range


def check_chronology(time):
    """ Check the simulation start and end chronology. Works on datetimes as well as int/float.
    """
    if time[1] < time[0]:
        raise ValueError(
            "start of simulation is chronologically after end of simulation"
        )
    elif time[0] == time[1]:
        raise ValueError(
            "start of simulation is the same as the end of simulation"
        )
    pass


def validate_solution_methods_ODE(rtol, method, tau):
    """
    Validates the input arguments of the ODE.sim() function

    input
    -----

    rtol: float
        Relative solver tolerance

    method: str
        Solver method: 'RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'

    tau: int/float
        Discrete integration size of timestep. 
    """

    if not isinstance(rtol, float):
        raise TypeError(
            "relative solver tolerance 'rtol' must be of type float"
            )
    if not isinstance(method, str):
        raise TypeError(
            "solver method 'method' must be of type string"
            )
    if method not in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        raise ValueError(
            f"invalid solution method '{method}'. valid methods: 'RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA'"
        )
    if tau != None:
        if not isinstance(tau, (int,float)):
            raise TypeError(
                "discrete timestep 'tau' must be of type int or float"
            )

def validate_solution_methods_JumpProcess(method, tau):
    """
    Validates the input arguments of the JumpProcess.sim() function

    method: str
        Solver method: 'SSA' or 'tau_leap'

    tau: int/float
        If method == 'tau_leap' --> leap size
    """

    # Input checks on solution method and timestep
    if not isinstance(method, str):
        raise TypeError(
            "solver method 'method' must be of type string"
            )
    if method not in ['tau_leap', 'SSA']:
        raise ValueError(
            f"invalid solution method '{method}'. valid methods: 'SSA' and 'tau_leap'"
        )
    if not isinstance(tau, (int,float)):
        raise TypeError(
            "discrete timestep 'tau' must be of type int or float"
            )
    
def validate_draw_function(draw_function, draw_function_kwargs, parameters, initial_states_function, state_shapes, initial_states_function_args):
    """
    Validates the draw function's input and output. Used in the sim() functions of the ODE and JumpProcess classes (base.py).
    
    Makes a call to `validate_initial_states` if an initial condition function is used

    input
    -----

    draw_function: function
        a function altering parameters in the model parameters dictionary between consecutive simulations
        has the dictionary of model parameters ('parameters') as its first obligatory input, followed by a variable number of inputs

    draw_function_kwargs: dict
        a dictionary containing the aforementioned additional inputs to the 'draw_function'
    
    parameters: dict
        the dictionary of model parameters

    initial_states_function: callable or None
        a function returning a dictionary of initial states

    state_shapes: dict
        contains the shape of every model state
    
    initial_states_function_args: list
        contains the names of the initial condition functions arguments
    """

    # check that the draw_function is a function
    if not callable(draw_function):
        raise TypeError(
            f"a 'draw function' must be callable (a function)"
        )
    # check that it's first argument is named 'parameters'
    args = list(inspect.signature(draw_function).parameters.keys())
    if args[0] != "parameters":
        raise ValueError(
            f"your draw function '{draw_function.__name__}' must have 'parameters' as its first input. Its current inputs are: '{args}'"
    )
    # check that `draw_function_kwargs` is a 'dict'
    if not isinstance(draw_function_kwargs, dict):
        raise TypeError(
            f"your `draw_function_kwargs` must be of type 'dict' but are of type '{type(draw_function_kwargs)}'"
        )
    # if draw_functions_kwargs is an empty dict and draw_function has additional kwargs the user has most likely forgotten to pass draw_function_kwargs to the sim() function
    if ((len(args[1:]) > 0) & (len(list(draw_function_kwargs.keys())) == 0)):
        raise ValueError(
            f"the draw function '{draw_function.__name__}' has {len(args[1:])} arguments in addition to the mandatory 'parameters' argument\n"
            f"have you forgotten to pass `draw_function_kwargs` to the sim() function?"
        )
    # check that it's keys have the same name as the inputs of draw_function that follow `parameters`
    if set(args[1:]) != set(list(draw_function_kwargs.keys())):
        raise ValueError(
            f"incorrect arguments passed to draw function '{draw_function.__name__}'\n"
            "keys missing in 'draw_function_kwargs': {0}. redundant keys: {1}".format(set(args[1:]).difference(list(draw_function_kwargs.keys())), set(list(draw_function_kwargs.keys())).difference(set(args[1:])))
        )
    
    # call draw function and check its outputs
    output = draw_function(copy.deepcopy(parameters), **draw_function_kwargs)
    # check if it returns one output
    if not isinstance(output, dict):
        raise TypeError(f"a draw function must return one dictionary containing the model's parameters. found type '{type(output)}'")
    # verify keys are the same on input/output 'parameters'
    if set(output.keys()) != set(parameters.keys()):
        raise ValueError(
            f"a draw function must return the dictionary of model parameters"
            f"keys in output dictionary of draw function '{draw_function.__name__}' must match the keys in input dictionary 'parameters'.\n"
            "keys missing in draw function output: {0}. redundant keys: {1}".format(set(parameters.keys()).difference(set(output.keys())), set(output.keys()).difference(set(parameters.keys())))
        )
    # if an initial condition function is used:
    if initial_states_function:
        # call it with the updated parameters and verify user didn't do anything stupid compromising the IC
        initial_states = initial_states_function(**{key: output[key] for key in initial_states_function_args})
        # verify the initial states sizes
        try:
            _ = validate_initial_states(initial_states, state_shapes)
        except Exception as e:
            error_message = f"draw function --> drawn parameter dictionary --> initial condition function --> invalid initial state.\nfound error: {str(e)}"
            raise RuntimeError(error_message) from e

def fill_initial_state_with_zero(state_names, initial_states):
    """ A function filling the undefined initial states with zeros """
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

def validate_dimensions_per_state(dimensions_per_state, coordinates, state_names):
    """Valide if length of `dimensions_per_state` is equal to the length of `state_names`. Check if the dimensions provided for every model state are existing dimensions."""

    # Length equal to state_names?   
    if len(dimensions_per_state) != len(state_names):
        raise ValueError(
            f"The length of `dimensions_per_state` ({len(dimensions_per_state)}) must match the length of `state_names` ({len(state_names)})"
        )
    # Contains only valid coordinates?
    for i,state_name in enumerate(state_names):
        if not all(x in coordinates.keys() for x in dimensions_per_state[i]):
            raise ValueError(
                f"The dimension names of model state '{state_name}', specified in position {i} of `dimensions_per_state` contains invalid coordinate names. Redundant names: {set(dimensions_per_state[i]).difference(set(coordinates.keys()))}"
        )

def build_state_sizes_dimensions(coordinates, state_names, dimensions_per_state):
    """A function returning three dictionaries: A dictionary containing, for every model state: 1) Its shape, 2) its dimensions, 3) its coordinates.
    """

    if not dimensions_per_state:
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
                "`dimensions_per_state` found in the model defenition, however you have not provided `coordinates` when initializing the model. "
            )
        else:
            shapes=[]
            coords=[]
            for i,state_name in enumerate(state_names):
                dimensions = dimensions_per_state[i]
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
            return dict(zip(state_names, shapes)), dict(zip(state_names, dimensions_per_state)),  dict(zip(state_names, coords))

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

def get_initial_states_fuction_parameters(initial_states):
    """
    A function checking the type of the initial condition. If it's a function, returns the arguments of the function. If it's a dictionary, returns an empty list.

    input
    -----

    initial_states: dict or callable
        Dictionary containing the model's initial states. Keys: model states. Values: corresponding initial values.
        Or a function generating such dictionary.
    """

    # check if type is right
    if ((not isinstance(initial_states, dict)) & (not callable(initial_states))):
        raise TypeError(f"initial states must be: 1) a dictionary, 2) a callable function generating a dictionary. found '{type(initial_states)}'")
    
    # if it's a callable function, call it and check if it returns a dictionary
    if callable(initial_states):
        # get arguments
        args = list(inspect.signature(initial_states).parameters.keys())
    else:
        args = []

    return args

def validate_initial_states(initial_states, state_shapes):
    """
    A function to check the types and sizes of the model's initial states provided by the user.
    Automatically assumes non-specified states are equal to zero.

    input
    -----

    initial_states: dict
        Dictionary containing the model's initial states. Keys: model states. Values: corresponding initial values.

    state_shapes: dict
        Contains the shape of every model state.

    output
    ------
    
    initial_states: dict
        Dictionary containing the model's validated initial states.
        Types/Size checked, redundant initial states checked, states sorted according to `state_names`.
    """

    # check if type is right (redundant; already checked in `get_initial_states_fuction_parameters`)
    if not isinstance(initial_states, dict):
        raise TypeError("initial states should be a dictionary by this point. contact Tijs Alleman if this error occurs.")

    # validate the states shape; if not present initialise with zeros
    for state_name, state_shape in state_shapes.items():
        if state_name in initial_states:
            # if present, verify the length
            initial_states[state_name] = check_initial_states_shape(
                                        initial_states[state_name], state_shape, state_name, "initial state",
                                        )
        else:
            # Fill with zeros
            initial_states[state_name] = np.zeros(state_shape, dtype=np.float64)

    # validate all states are present (using `set` to ignore order)
    if set(initial_states.keys()) != set(state_shapes.keys()):
        raise ValueError(
            f"The specified initial states don't exactly match the predefined states. Redundant states: {set(initial_states.keys()).difference(set(state_shapes.keys()))}"
        )

    # sort the initial states to match the state_names
    initial_states = {state: initial_states[state] for state in state_shapes}

    return initial_states

def evaluate_initial_condition_function(initial_states, parameters):
    """
    A function to check the user's input type of the initial states (dict or callable).
    If callable, evaluates the user's initial state function to obtain an initial states dictionary.

    input
    -----

    initial_states: dict
        Dictionary containing the model's initial states. Keys: model states. Values: corresponding initial values.

    parameters: dict
        Contains the model's parameters.

    output
    ------

    initial_states: dict
        Dictionary containing the model's validated initial states.
        Types/Size checked, redundant initial states checked, states sorted according to `state_names`.

    initial_states_function: callable
        A function generating a valid initial state
        If 'initial_states' was already a callable: equal to 'initial_states'
        If 'initial_states' was a dictionary: a dummy function returning that dictionary
    
    initial_states_function_args: list
        Contains the arguments of the initial_states_function.
    """
    
    # check if type is right (redundant; already checked in `get_initial_states_fuction_parameters`)
    if ((not isinstance(initial_states, dict)) & (not callable(initial_states))):
        raise TypeError("initial states must be: 1) a dictionary, 2) a callable function generating a dictionary")
    
    # if it's a callable function, call it and check if it returns a dictionary
    initial_states_is_function = False
    if callable(initial_states):
        initial_states_is_function = True
        # get arguments
        initial_states_function_args = list(inspect.signature(initial_states).parameters.keys())
        # save function
        initial_states_function = initial_states
        # construct initial states
        initial_states = initial_states_function(**{key: parameters[key] for key in initial_states_function_args})

    # define a dummy initial_states_function
    if not initial_states_is_function:
        initial_states_function = None
        initial_states_function_args = []

    return initial_states, initial_states_function, initial_states_function_args

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

def check_overlap(lst_1, lst_2, name_1, name_2):
    """A function raising an error if there is overlap between both lists"""
    overlap = set(lst_1) & set(lst_2)  # Find common elements between the lists
    if overlap:
        raise ValueError(f"overlap detected between '{name_1}' and '{name_2}': {list(overlap)}")

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
            "The input arguments of the integrate/compute_rates function should be 't', followed by the model's 'states' and 'parameters'. "
            "Missing parameters: {0}. Missing states: {1}. "
            "Redundant arguments: {2}. ".format(missing_parameters, missing_states, list(redundant_all))
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
        all_params = list(OrderedDict((x, True) for x in all_params).keys())
        len_after = len(all_params)
        # Line below computes number of integrate arguments used in time dependent parameter functions
        n_duplicates = len_before - len_after
        _n_function_params = len(_extra_params) - n_duplicates
    else:
        _n_function_params = 0
        _extra_params = []
    
    return all_params, list(_extra_params)

def validate_provided_parameters(all_params, parameters, state_names):
    """
    Verify all parameters (parameters + stratified parameters + TDPF parameters + initial condition parameters) are provided by the user.
    Verify 't' is not used as a parameter.
    Verify no state name is used as a parameter
    """

    # Validate the type
    if not isinstance(parameters, dict):
        raise TypeError("'parameters' should be a dictionary.")
    
    # Validate the parameter dictionary provided by the user (does it contain missing/redundant parameters)
    if set(parameters.keys()) != set(all_params):
        raise ValueError(
            "The specified parameters don't exactly match the predefined parameters. "
            "Redundant parameters: {0}. Missing parameters: {1}".format(
            set(parameters.keys()).difference(set(all_params)),
            set(all_params).difference(set(parameters.keys())))
        )
    
    # Build a dictionary of parameters: key: parameter name, value: parameter value
    parameters = {param: parameters[param] for param in all_params}

    # After building the list of all model parameters, verify no parameters named 't' were used
    if 't' in parameters:
        raise ValueError(
        "'t' is not a valid model parameter name. please verify no model parameters named 't' are present in your model, its time-dependent parameter functions or initial condition function."
            )
    
    # Lastly, verify no state name was used a parameter
    overlap = set(state_names) & set(parameters.keys())  # Find common elements between the lists
    if overlap:
        raise ValueError(f"overlapping names of model parameters and states is not allowed. found overlapping names: {list(overlap)}")
    
    return parameters

def check_stratpar_size(values, name, object_name,dimension_names,desired_size):
    """Checks the size of stratified parameters. Converts stratified parameters to a numpy array."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(
            "A {obj} value should be a 1D array, but {obj} '{name}' is "
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
    # 'func' in class 'JumpProcess' of 'base.py' automatically converts states to np.array
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