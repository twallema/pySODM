import inspect
import itertools
import numpy as np
import pandas as pd
from collections import OrderedDict

def fill_initial_state_with_zero(state_names, initial_states):
    for state in state_names:
        if state in initial_states:
            state_values = initial_states[state]
    return state_values

def validate_stratifications(stratification_names, coordinates):
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

def validate_time_dependent_parameters(parameter_names, parameters_stratified_names,time_dependent_parameters):

    extra_params = []
    all_param_names = parameter_names.copy()

    if parameters_stratified_names:
        if not isinstance(parameters_stratified_names[0], list):
            all_param_names += parameters_stratified_names
        else:
            for lst in parameters_stratified_names:
                all_param_names.extend(lst)
            
    for param, func in time_dependent_parameters.items():
        if param not in all_param_names:
            raise ValueError(
                "The specified time-dependent parameter '{0}' is not an "
                "existing model parameter".format(param))
        kwds = validate_parameter_function(func)
        extra_params.append(kwds)

    return extra_params

def validate_ODEModel(initial_states, parameters, coordinates, stratification_size, state_names, parameter_names,
                        parameters_stratified_names, _function_parameters, _create_fun, integrate_func, state_2d=None):
    """
    This does some basic validation of the model + initialization:

    1) Validation of the integrate function to ensure it matches with
    the specified `state_names`, `parameter_names`, etc.
    This is actually a validation of the model class itself, but it is
    easier to do this only on initialization of a model instance.

    2) Validation of the actual initialization with initial values for the
    states and parameter values.
    TODO: For now, we require that those are passed in the exact same
    order, but this requirement could in principle be relaxed, if we ensure
    to pass the states and parameters as keyword arguments and not as
    positional arguments to the `integrate` function.
    """

    # Validate Model class definition (the integrate function)
    sig = inspect.signature(integrate_func)
    keywords = list(sig.parameters.keys())
    if keywords[0] != "t":
        raise ValueError(
            "The first argument of the integrate function should always be 't'"
        )
    elif keywords[1] == "l":
        # Tau-leaping Gillespie
        discrete = True
        start_index = 2
    else:
        # ODE model
        discrete = False
        start_index = 1

    # Get names of states and parameters that follow after 't' or 't' and 'l'
    N_states = len(state_names)
    integrate_states = keywords[start_index : start_index + N_states]
    if integrate_states != state_names:
        raise ValueError(
            "The states in the 'integrate' function definition do not match "
            "the state_names: {0} vs {1}".format(integrate_states, state_names)
        )
    integrate_params = keywords[start_index + N_states :]
    specified_params = parameter_names.copy()

    if parameters_stratified_names:
        if not isinstance(parameters_stratified_names[0], list):
            if len(parameters_stratified_names) == 1:
                specified_params += parameters_stratified_names
            else:
                for stratified_names in parameters_stratified_names:
                    specified_params += [stratified_names,]
        else:
            for stratified_names in parameters_stratified_names:
                specified_params += stratified_names

    if integrate_params != specified_params:
        raise ValueError(
            "The parameters in the 'integrate' function definition do not match "
            "the parameter_names + parameters_stratified_names + stratification: "
            "{0} vs {1}".format(integrate_params, specified_params)
        )

    # additional parameters from time-dependent parameter functions
    # are added to specified_params after the above check

    if _function_parameters:
        extra_params = [item for sublist in _function_parameters for item in sublist]

        # TODO check that it doesn't duplicate any existing parameter (completed?)
        # Line below removes duplicate arguments in time dependent parameter functions
        extra_params = OrderedDict((x, True) for x in extra_params).keys()
        specified_params += extra_params
        len_before = len(specified_params)
        # Line below removes duplicate arguments with integrate defenition
        specified_params = OrderedDict((x, True) for x in specified_params).keys()
        len_after = len(specified_params)
        # Line below computes number of integrate arguments used in time dependent parameter functions
        n_duplicates = len_before - len_after
        _n_function_params = len(extra_params) - n_duplicates
    else:
        _n_function_params = 0

    # Validate the params
    if set(parameters.keys()) != set(specified_params):
        raise ValueError(
            "The specified parameters don't exactly match the predefined parameters. "
            "Redundant parameters: {0}. Missing parameters: {1}".format(
            set(parameters.keys()).difference(set(specified_params)),
            set(specified_params).difference(set(parameters.keys())))
        )

    parameters = {param: parameters[param] for param in specified_params}

    # After building the list of all model parameters, verify no parameters 'l' or 't' were used
    if 't' in parameters:
        raise ValueError(
        "Parameter name 't' is reserved for the timestep of scipy.solve_ivp.\nPlease verify no model parameters named 't' are present in the model parameters dictionary or in the time-dependent parameter functions."
            )
    if discrete == True:
        if 'l' in parameters:
            raise ValueError(
                "Parameter name 'l' is reserved for the leap size of the tau-leaping Gillespie algorithm.\nPlease verify no model parameters named 'l' are present in the model parameters dictionary or in the time-dependent parameter functions."
            )

    # Validate the initial_states / stratified params having the correct length

    def validate_stratified_parameters(values, name, object_name,i):
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

    def validate_initial_states(values, name, object_name):
        values = np.asarray(values)
        if state_2d:
            if name in state_2d:
                if list(values.shape) != [stratification_size[0],stratification_size[0]]:
                    raise ValueError(
                        "{obj} {name} was defined as a two-dimensional state "
                        "but has size {size}, instead of {desired_size}"
                        .format(obj=object_name,name=name,size=list(values.shape),desired_size=[stratification_size[0],stratification_size[0]])
                        )
        else:
            if list(values.shape) != stratification_size:
                raise ValueError(
                    "The coordinates provided for the stratifications '{strat}' indicate a "
                    "model states size of {strat_size}, but {obj} '{name}' "
                    "has length {val}".format(
                        strat=list(coordinates.keys()), strat_size=stratification_size,
                        obj=object_name, name=name, val=list(values.shape)
                    )
                )

    # the size of the stratified parameters
    if parameters_stratified_names:
        i = 0
        if not isinstance(parameters_stratified_names[0], list):
            if len(parameters_stratified_names) == 1:
                for param in parameters_stratified_names:
                    validate_stratified_parameters(
                            parameters[param], param, "stratified parameter",i
                        )
                i = i + 1
            else:
                for param in parameters_stratified_names:
                    validate_stratified_parameters(
                            parameters[param], param, "stratified parameter",i
                        )
                i = i + 1
        else:
            for stratified_names in parameters_stratified_names:
                for param in stratified_names:
                    validate_stratified_parameters(
                        parameters[param], param, "stratified parameter",i
                    )
                i = i + 1

    # the size of the initial states + fill in defaults
    for state in state_names:
        if state in initial_states:
            # if present, check that the length is correct
            validate_initial_states(
                initial_states[state], state, "initial state"
            )

        else:
            # otherwise add default of 0
            initial_states[state] = np.zeros(stratification_size)

    # validate the states (using `set` to ignore order)
    if set(initial_states.keys()) != set(state_names):
        raise ValueError(
            "The specified initial states don't exactly match the predefined states"
        )
    # sort the initial states to match the state_names
    initial_states = {state: initial_states[state] for state in state_names}

    # Call integrate function with initial values to check if the function returns all states
    fun = _create_fun(None,discrete)
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
    
    return initial_states, parameters, _n_function_params, discrete



def validate_SDEModel(initial_states, parameters, coordinates, stratification_size, state_names, parameter_names,
                        parameters_stratified_names, _function_parameters, compute_rates_func, apply_transitionings_func):
    """
    This does some basic validation of the model + initialization:

    1) Validation of the integrate function to ensure it matches with
    the specified `state_names`, `parameter_names`, etc.
    This is actually a validation of the model class itself, but it is
    easier to do this only on initialization of a model instance.

    2) Validation of the actual initialization with initial values for the
    states and parameter values.
    TODO: For now, we require that those are passed in the exact same
    order, but this requirement could in principle be relaxed, if we ensure
    to pass the states and parameters as keyword arguments and not as
    positional arguments to the `integrate` function.
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
    else:
        start_index = 1

    # Get names of states and parameters that follow after 't'
    N_states = len(state_names)
    compute_rates_states = keywords[start_index : start_index + N_states]
    if compute_rates_states != state_names:
        raise ValueError(
            "The states in the 'compute_rates' function definition do not match "
            "the provided 'state_names': {0} vs {1}".format(compute_rates_states, state_names)
        )
    compute_rates_params = keywords[start_index + N_states :]
    specified_params = parameter_names.copy()

    if parameters_stratified_names:
        if not isinstance(parameters_stratified_names[0], list):
            if len(parameters_stratified_names) == 1:
                specified_params += parameters_stratified_names
            else:
                for stratified_names in parameters_stratified_names:
                    specified_params += [stratified_names,]
        else:
            for stratified_names in parameters_stratified_names:
                specified_params += stratified_names

    if compute_rates_params != specified_params:
        raise ValueError(
            "The parameters in the 'compute_rates' function definition do not match "
            "the provided 'parameter_names' + 'parameters_stratified_names': "
            "{0} vs {1}".format(compute_rate_params, specified_params)
        )

    # additional parameters from time-dependent parameter functions
    # are added to specified_params after the above check

    if _function_parameters:
        extra_params = [item for sublist in _function_parameters for item in sublist]

        # TODO check that it doesn't duplicate any existing parameter (completed?)
        # Line below removes duplicate arguments in time dependent parameter functions
        extra_params = OrderedDict((x, True) for x in extra_params).keys()
        specified_params += extra_params
        len_before = len(specified_params)
        # Line below removes duplicate arguments with integrate defenition
        specified_params = OrderedDict((x, True) for x in specified_params).keys()
        len_after = len(specified_params)
        # Line below computes number of integrate arguments used in time dependent parameter functions
        n_duplicates = len_before - len_after
        _n_function_params = len(extra_params) - n_duplicates
    else:
        _n_function_params = 0

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
    else:
        start_index = 3

    # Get names of states and parameters that follow after 't'
    N_states = len(state_names)
    apply_transitionings_states = keywords[start_index : start_index + N_states]
    if apply_transitionings_states != state_names:
        raise ValueError(
            "The states in the 'apply_transitionings' function definition do not match "
            "the provided 'state_names': {0} vs {1}".format(apply_transitionings_states, state_names)
        )
    apply_transitionings_params = keywords[start_index + N_states :]
    specified_params = parameter_names.copy()

    if parameters_stratified_names:
        if not isinstance(parameters_stratified_names[0], list):
            if len(parameters_stratified_names) == 1:
                specified_params += parameters_stratified_names
            else:
                for stratified_names in parameters_stratified_names:
                    specified_params += [stratified_names,]
        else:
            for stratified_names in parameters_stratified_names:
                specified_params += stratified_names

    if apply_transitionings_params != specified_params:
        raise ValueError(
            "The parameters in the 'apply_transitionings' function definition do not match "
            "the provided 'parameter_names' + 'parameters_stratified_names': "
            "{0} vs {1}".format(apply_transitionings_params, specified_params)
        )

    # additional parameters from time-dependent parameter functions
    # are added to specified_params after the above check

    if _function_parameters:
        extra_params = [item for sublist in _function_parameters for item in sublist]

        # TODO check that it doesn't duplicate any existing parameter (completed?)
        # Line below removes duplicate arguments in time dependent parameter functions
        extra_params = OrderedDict((x, True) for x in extra_params).keys()
        specified_params += extra_params
        len_before = len(specified_params)
        # Line below removes duplicate arguments with integrate defenition
        specified_params = OrderedDict((x, True) for x in specified_params).keys()
        len_after = len(specified_params)
        # Line below computes number of integrate arguments used in time dependent parameter functions
        n_duplicates = len_before - len_after
        _n_function_params = len(extra_params) - n_duplicates
    else:
        _n_function_params = 0

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

    # After building the list of all model parameters, verify no parameters 't' was used
    if 't' in parameters:
        raise ValueError(
            "Parameter name 't' is reserved for the simulation time.\nPlease verify no model parameters named 't' are present in the model parameters dictionary or in the time-dependent parameter functions."
            )

    ###############################################################################
    ## Validate the initial_states / stratified params having the correct length ##
    ###############################################################################

    def validate_stratified_parameters(values, name, object_name,i):
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

    def validate_initial_states(values, name, object_name):
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

    # the size of the stratified parameters
    if parameters_stratified_names:
        i = 0
        if not isinstance(parameters_stratified_names[0], list):
            if len(parameters_stratified_names) == 1:
                for param in parameters_stratified_names:
                    validate_stratified_parameters(
                            parameters[param], param, "stratified parameter",i
                        )
                i = i + 1
            else:
                for param in parameters_stratified_names:
                    validate_stratified_parameters(
                            parameters[param], param, "stratified parameter",i
                        )
                i = i + 1
        else:
            for stratified_names in parameters_stratified_names:
                for param in stratified_names:
                    validate_stratified_parameters(
                        parameters[param], param, "stratified parameter",i
                    )
                i = i + 1

    # the size of the initial states + fill in defaults
    for state in state_names:
        if state in initial_states:
            # if present, check that the length is correct
            validate_initial_states(
                initial_states[state], state, "initial state"
            )
        else:
            # otherwise add default of 0
            initial_states[state] = np.zeros(stratification_size)

    # validate the states (using `set` to ignore order)
    if set(initial_states.keys()) != set(state_names):
        raise ValueError(
            "The specified initial states don't exactly match the predefined states. "
            "Redundant states: {0}".format(
            set(initial_states.keys()).difference(set(state_names)))
        )

    # sort the initial states to match the state_names
    initial_states = {state: initial_states[state] for state in state_names}

    #########################################################################
    # Validate the 'compute_rates' and 'apply_transitionings' by calling it #
    #########################################################################

    # compute_rates
    # ~~~~~~~~~~~~~

    # Call the function with initial values to check if the function returns the right format of dictionary
    rates = compute_rates_func(10, *initial_states.values(), *list(parameters.values())[:-_n_function_params])
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
                raise TypeError(f"Rate of the {i}th transitioning for state {state_name} is not of type 'np.ndarray' but {type(rate)}")
            if list(rate.shape) != stratification_size:
                raise ValueError(f"The provided coordinates indicate a state size of {stratification_size}, but rate of the {i}th transitioning for state '{state_name}' has shape {list(rate.shape)}")
    
    # apply_transitionings
    # ~~~~~~~~~~~~~~~~~~~~

    new_states = apply_transitionings_func(10, 1, rates, *initial_states.values(), *list(parameters.values())[:-_n_function_params])

    # Check
    if len(list(new_states)) != len(state_names):
        raise ValueError(f"The number of outputs of function 'apply_transitionings_func' ({len(list(new_states))}) is not equal to the number of states ({len(state_names)})")
    for i, new_state in enumerate(list(new_states)):
        if not isinstance(new_state, np.ndarray):
            raise TypeError(f"Output state of function 'apply_transitionings_func' in position {i} is not a np.ndarray")
        if list(new_state.shape) != stratification_size:
            raise ValueError(f"The provided coordinates indicate a state size of {stratification_size}, but the {i}th output of function 'apply_transitionings_func' has shape {list(new_state.shape)}")

    return initial_states, parameters, _n_function_params
