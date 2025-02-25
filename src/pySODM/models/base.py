import random
import itertools
import xarray
import copy
import numpy as np
from multiprocessing import get_context
from functools import partial
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp
from typing import List, Callable, Dict, Optional, Any, Union, Tuple
from pySODM.models.utils import time_to_date, list_to_dict, cut_ICF_parameters_from_parameters, time_unit_map
from pySODM.models.validation import merge_parameter_names_parameter_stratified_names, validate_draw_function, validate_simulation_time, validate_dimensions, \
                                        validate_time_dependent_parameters, validate_integrate, check_duplicates, build_state_sizes_dimensions, validate_dimensions_per_state, \
                                            validate_initial_states, validate_integrate_or_compute_rates_signature, validate_provided_parameters, validate_parameter_stratified_sizes, \
                                                validate_apply_transitionings_signature, validate_compute_rates, validate_apply_transitionings, validate_solution_methods_ODE, validate_solution_methods_JumpProcess, \
                                                    get_initial_states_fuction_parameters, check_overlap, evaluate_initial_condition_function, check_formatting_names

class JumpProcess:
    """
    Make a class with `JumpProcess` as input to build a pySODM model.

    ### Your class must contain

    - states: list
        - names of the model’s states.

    - parameters: list
        - names of the model’s parameters.

    - compute_rates: staticmethod (callable)
        - function to compute the rates of transitioning between the model states. 
        - must be a static method (decorated with @staticmethod).
        - signature: `def compute_rates(t, states, parameters, stratified_parameters)`
            - t (float): timestep
            - states: model’s states
            - parameters and stratified parameters        
        - returns: dict
            - keys: model states, values: list containing the rates of the possible transitionings of the state.
            - a model state can have multiple transitionings.
    
    - apply_transitionings: callable
        - function to update the states with the number of drawn transitionings.
        - signature: `def apply_transitionings(t, tau, transitionings, states, parameters, stratified_parameters)`
            - t (float): timestep
            - tau (float): solver timestep
            - transitionings (dict): drawn transitionings between the states
            - states: model’s states
            - parameters and stratified parameters        

    - (optional) dimensions: list
        - names of the model’s dimensions. number of dimensions determines the dimensions of the model’s states.
            - no dimensions: states are 0-D (scalar)
            - 1 dimension: states are 1-D (np.ndarray)
            - 2 dimensions: states are 2-D (np.ndarray)

    - (optional) stratified_parameters: list
        - names of the stratified parameters. stratified parameters are one-dimensional parameter associated with a model dimension.
            - 0 dimensions: not possible to have stratified parameters
            - 1 dimension: list containing strings - ['stratpar_1', 'stratpar_2']
            - 2+ dimensions: list contains n sublists, where n must equal the number of model dimensions. each sublist contains names of stratified parameters associated with the dimension in the corresponding position in dimensions 
            - example for 3 dimensions model: [['stratpar_1', 'stratpar_2'],[],['stratpar_3']], first dimension in dimensions has two stratified parameters stratpar_1 and stratpar_2, second dimension has no stratified parameters, third dimensions has one stratified parameter stratpar_3.
    
    - (optional) dimensions_per_state: list
        - specify the dimension of each model state seperately. allows you to define models with states of different sizes.
        - if `dimensions_per_state` is not provided, all model states will have the same number of dimensions, equal to the number of model dimensions specified using dimensions.
        - if specified, `dimensions_per_state` must contain n sublists, where n is the number of model states (`n = len(states)`). if a model state has no dimensions (i.e. it is a float), specify an empty list.

    ### Minimal example
    
    ```
    class MY_MODEL(JumpProcesses):

    states = ['Y1', 'Y2']
    parameters = ['alpha']

    # define the rates of the system's transitionings
    @staticmethod
    def compute_rates(t, Y1, Y2, alpha):
        return {'Y1': [alpha, ]}

    # apply the sampled number of transitionings
    @staticmethod
    def apply_transitionings(t, tau, transitionings, Y1, Y2, alpha):

        Y1_new = Y1 - transitionings['Y1'][0]
        Y2_new = Y2 + transitionings['Y2'][0]

        return Y1_new, Y2_new
    ```
    """

    states = None
    parameters = None
    stratified_parameters = None
    dimensions = None
    dimensions_per_state = None

    def __init__(self,
                 initial_states: Union[Dict[str, Union[int, float, np.ndarray]], Callable[..., Dict[str, Union[int, float, np.ndarray]]]],
                 parameters: Dict[str, Any],
                 coordinates: Optional[Dict[str, list]]=None,
                 time_dependent_parameters: Optional[Dict[str, Callable[[Union[float, datetime], Dict[str, Union[int, float, np.ndarray]], Any, tuple], Any]]]=None) -> None:
        """
        Initialise a jump process model solved using Gillespie's SSA or Tau-Leaping

        Parameters
        ----------

        - initial_states : dict or callable 
            - dict: contains the initial values of all non-zero model states, f.i. for an SIR model: {'S': 1000, 'I': 1}. Initialising zeros is not required.
            - function: an initial condition function (ICF) generating a dictionary can be provided. arguments of the ICF must be added to `parameters`.

        - parameters : dict
            - keys: names of all parameters, stratified_parameters, TDPF and ICF parameters. values: associated parameter values. 

        - (optional) coordinates: dict
            - for each `dimension_name` in the model, specifies the coordinates associated with that dimension.
            - f.e. {'spatial_units': ['city_1','city_2','city_3']}        

        - (optional) time_dependent_parameters : dict
            - keys: name of the parameter you want to impose a time-dependency on. values: time-dependent parameter function.
            - a pySODM-compatible TDPF has the signature `fun(t, states, param, other_parameter_1, ...)`, where:
                - t (float or datetime): current simulation timestep
                - states (dict): dictionary containing the model states at time `t`
                - param (any): value of the parameter the time dependency acts on
        """
            
        # Add a suffix _names to all user-defined name declarations 
        self.states_names = self.states
        self.parameters_names = self.parameters
        self.parameters_stratified_names = self.stratified_parameters
        self.dimensions_names = self.dimensions
        self.states = initial_states
        parameters = parameters
        self.coordinates = coordinates
        self.time_dependent_parameters = time_dependent_parameters

        # Check formatting of state, parameters, dimension names user has defined in his model class
        check_formatting_names(self.states_names, self.parameters_names, self.parameters_stratified_names, self.dimensions_names)

        # Merge parameter_names and parameter_stratified_names
        self.parameters_names_modeldeclaration = merge_parameter_names_parameter_stratified_names(self.parameters_names, self.parameters_stratified_names)

        # Duplicates in lists containing names of states/parameters/stratified parameters/dimensions?
        check_duplicates(self.states_names, 'state_names')
        check_duplicates(self.parameters_names, 'parameter_names')
        check_duplicates(self.parameters_names_modeldeclaration, 'parameter_names + parameter_stratified_names')
        if self.dimensions_names:
            check_duplicates(self.dimensions_names, 'dimension_names')

        # Overlapping state and parameter names?
        check_overlap(self.states_names, self.parameters_names_modeldeclaration, 'state_names', 'parameter_names + parameter_stratified_names')

        # Validate and compute the dimension sizes
        self.dimension_size = validate_dimensions(self.dimensions_names, self.coordinates)

        # Validate dimensions_per_state
        if self.dimensions_per_state:
            validate_dimensions_per_state(self.dimensions_per_state, self.coordinates, self.states_names)
        
        # Build a dictionary containing the size of every state; build a dictionary containing the dimensions of very state; build a dictionary containing the coordinates of every state
        self.state_shapes, self.dimensions_per_state, self.state_coordinates = build_state_sizes_dimensions(self.coordinates, self.states_names, self.dimensions_per_state)

        # Validate the time-dependent parameter functions (TDPFs) and extract the names of their input arguments
        if time_dependent_parameters:
            self._function_parameters = validate_time_dependent_parameters(self.parameters_names, self.parameters_stratified_names, self.time_dependent_parameters)
        else:
            self._function_parameters = []

        # Verify the signature of the compute_rates function; extract the additional parameters of the TDPFs
        all_params, self._extra_params_TDPF = validate_integrate_or_compute_rates_signature(self.compute_rates, self.parameters_names_modeldeclaration, self.states_names, self._function_parameters)

        # Verify the signature of the apply_transitionings function
        validate_apply_transitionings_signature(self.apply_transitionings, self.parameters_names_modeldeclaration, self.states_names)

        # Get additional parameters of the IC function
        self._extra_params_ICF = get_initial_states_fuction_parameters(self.states)
        all_params.extend(self._extra_params_ICF)
        
        # Verify all parameters were provided
        self.parameters = validate_provided_parameters(set(all_params), parameters, self.states_names)

        # Validate the shapes of the initial states, fill non-defined states with zeros
        self.initial_states, self.initial_states_function, self.initial_states_function_args = evaluate_initial_condition_function(self.states, self.parameters)
        self.initial_states = validate_initial_states(self.initial_states, self.state_shapes)

        # Validate the size of the stratified parameters (Perhaps move this way up front?)
        if self.parameters_stratified_names:
            self.parameters = validate_parameter_stratified_sizes(self.parameters_stratified_names, self.dimensions_names, coordinates, self.parameters)

        # Call the compute_rates function, check if it works and check the sizes of the differentials in the output
        rates = validate_compute_rates(self.compute_rates, self.initial_states, self.state_shapes, self.parameters_names_modeldeclaration, self.parameters)
        validate_apply_transitionings(self.apply_transitionings, rates, self.initial_states, self.state_shapes, self.parameters_names_modeldeclaration, self.parameters)

    # Overwrite integrate class
    @staticmethod
    def compute_rates():
        """to overwrite in subclasses"""
        raise NotImplementedError

    @staticmethod
    def apply_transitionings():
        """to overwrite in subclasses"""
        raise NotImplementedError

    @staticmethod
    def _SSA(states, rates):
        """
        Stochastic simulation algorithm by Gillespie
        Based on: https://lewiscoleblog.com/gillespie-algorithm

        Inputs
        ------

        states: dict
            Dictionary containing the values of the model states.

        rates: dict
            Dictionary containing the rates associated with each transitionings of the model states.

        Returns
        -------

        transitionings: dict
            Dictionary containing the number of transitionings for every model state

        tau: int/float
            Timestep
        """

        # Calculate overall reaction rate
        R=0
        for k,v in rates.items():
            for i,lst in enumerate(v):
                R += np.sum(states[k]*lst, axis=None)

        if R != 0:
            # Compute tau
            u1 = np.random.random()
            tau = 1/R * np.log(1/u1)

            # Compute the propensities
            propensities = {k: v[:] for k, v in rates.items()}
            for k,v in propensities.items():
                for i,lst in enumerate(v):
                    propensities[k][i] = states[k]*lst/R

            # Flatten propensities
            flat_propensities=[]
            for element in list(itertools.chain(*propensities.values())):
                flat_propensities += (list(element.flatten()))

            # Compute transitioning index
            idx = flat_propensities.index(random.choices(flat_propensities, flat_propensities))

            # Construct transitionings
            transitionings = {k: v[:] for k, v in propensities.items()}
            index = 0
            for i, (k,v) in enumerate(transitionings.items()):
                for j,lst in enumerate(v):
                    # Set to zeros and flatten
                    shape=transitionings[k][j].shape
                    transitionings[k][j] = np.zeros(shape).flatten()
                    # Find index
                    for l in range(len(transitionings[k][j])):
                        if index == idx:
                            transitionings[k][j][l] = 1
                        index += 1    
                    transitionings[k][j] = np.reshape(transitionings[k][j], shape)
        else:
            # Default to tau=1
            tau = 1
            # Set all transitionings to zero
            # Construct transitionings
            transitionings = {k: v[:] for k, v in rates.items()}
            index = 0
            for i, (k,v) in enumerate(transitionings.items()):
                for j,lst in enumerate(v):
                    transitionings[k][j] = np.zeros(transitionings[k][j].shape)

        return transitionings, tau

    @staticmethod
    def _multinomial_rvs(count, p):
        """
        Sample from the multinomial distribution with multiple p vectors.

        Retrieved from: https://stackoverflow.com/questions/55818845/fast-vectorized-multinomial-in-python

        * count must be an (n-1)-dimensional numpy array.
        * p must an n-dimensional numpy array, n >= 1.  The last axis of p
        must hold the sequence of probabilities for a multinomial distribution.

        The return value has the same shape as p.
        """
        out = np.zeros(p.shape, dtype=int)
        ps = p.cumsum(axis=-1)
        # Conditional probabilities
        with np.errstate(divide='ignore', invalid='ignore'):
            condp = p / ps
        condp[np.isnan(condp)] = 0.0
        for i in range(p.shape[-1]-1, 0, -1):
            binsample = np.random.binomial(count, condp[..., i])
            out[..., i] = binsample
            count -= binsample
        out[..., 0] = count
        return out

    def _tau_leap(self, states, rates, tau):
        """
        Tau-leaping algorithm by Gillespie
        Loops over the transitionings, extracts the values of the states and the rates, passes them to `_multinomial_rvs()`

        Inputs
        ------

        states: dict
            Dictionary containing the values of the model states.

        rates: dict
            Dictionary containing the rates associated with each transitionings of the model states.

        tau: int/float
            Leap size/simulation timestep
        
        Returns
        -------

        transitionings: dict
            Dictionary containing the number of transitionings for every model state
        
        tau: int/float
            Leap size/simulation timestep
        """

        transitionings = rates.copy()
        for k,rate in rates.items():
            p = 1 - np.exp(-tau*np.stack(rate, axis=-1, dtype=np.float64))
            s=np.zeros(p.shape[:-1], dtype=np.float64)
            for i in range(p.shape[-1]-1):
                s += p[...,i]
            s = 1 - s
            s = np.expand_dims(s, axis=-1)
            p = np.concatenate((p,s), axis=-1)
            trans = self._multinomial_rvs(np.array(states[k], np.int64), p)
            transitionings[k] = [np.take(trans, i, axis = -1) for i in range(trans.shape[-1])]

        return transitionings, tau
 
    def _create_fun(self, start_date, time_unit, method='SSA', tau_input=0.5):
        
        def func(t, y, pars={}):
                """As used by scipy -> flattend in, flattend out"""

                # ------------------------------------------------
                # Deflatten y and reconstruct dictionary of states 
                # ------------------------------------------------

                states = list_to_dict(y, self.state_shapes, retain_floats=False)

                # --------------------------------------------
                # Reconstruct datetime simtime (if applicable)
                # --------------------------------------------

                if start_date is not None:
                    # datetime simtime
                    date = time_to_date(float(t), start_date, time_unit)
                else:
                    # integer simtime
                    date = t

                # --------------------------------------
                # update time-dependent parameter values
                # --------------------------------------

                params = pars.copy()
                if self.time_dependent_parameters:
                    for i, (param, param_func) in enumerate(self.time_dependent_parameters.items()):
                        func_params = {key: params[key] for key in self._function_parameters[i]}
                        params[param] = param_func(date, states, pars[param], **func_params)

                # -------------------------------------------------------------
                #  throw parameters of TDPFs out of model parameters dictionary
                # -------------------------------------------------------------

                params = {k:v for k,v in params.items() if ((k not in self._extra_params_TDPF) or (k in self.parameters_names_modeldeclaration))}

                # -------------
                # compute rates
                # -------------

                rates = self.compute_rates(date, **states, **params)

                # --------------
                # Gillespie step
                # --------------

                if method == 'SSA':
                    transitionings, tau = self._SSA(states, rates)
                elif method == 'tau_leap':
                    transitionings, tau = self._tau_leap(states, rates, tau_input)
                    
                # -------------
                # update states 
                # -------------

                dstates = self.apply_transitionings(t, tau, transitionings, **states, **params)

                # -------
                # Flatten
                # -------

                out=[]
                for d in dstates:
                    out.extend(list(np.ravel(d)))

                return np.asarray(out), tau

        return func

    def _solve_discrete(self, fun, t_eval, y, args):
        # Preparations
        y = np.reshape(y, [len(y),1])
        y_prev=y
        # Simulation loop
        t_lst=[t_eval[0]]
        t = t_eval[0]
        while t < t_eval[-1]:
            out, tau = fun(t, y_prev, args)
            y_prev = out
            out = np.reshape(out,[len(out),1])
            y = np.append(y,out,axis=1)
            t = t + tau
            t_lst.append(t)
        # Interpolate output y to times t_eval
        y_eval = np.zeros([y.shape[0], len(t_eval)])
        for row_idx in range(y.shape[0]):
            y_eval[row_idx,:] = np.interp(t_eval, t_lst, y[row_idx,:])

        return {'y': y_eval, 't': t_eval}

    def _sim_single(self, time, start_date=None, time_unit=None, method='tau_leap', tau=0.5):

        # Initialize wrapper
        fun = self._create_fun(start_date, time_unit, method, tau)

        # Prepare time
        t0, t1 = time
        t_eval = np.arange(start=t0, stop=t1 + 1, step=1)

        # Update initial condition if a function was provided by user
        if self.initial_states_function:
            ## Call the initial state function to update the initial state
            initial_states = self.initial_states_function(**{key: self.parameters[key] for key in self.initial_states_function_args})
            ## Check the initial states size and fill states not provided with zeros
            initial_states = validate_initial_states(initial_states, self.state_shapes)
            ## Throw out parameters belonging (uniquely) to the initial condition function
            params = cut_ICF_parameters_from_parameters(self.parameters_names_modeldeclaration, self._extra_params_TDPF, self._extra_params_ICF, self.parameters)
        else:
            ## Do nothing
            initial_states = self.initial_states
            params = self.parameters

        # Flatten initial states
        y0=[]
        for v in initial_states.values():
            y0.extend(list(np.ravel(v)))

        # Run the time loop
        output = self._solve_discrete(fun, t_eval, np.asarray(y0), params)

        return _output_to_xarray_dataset(output, self.state_shapes, self.dimensions_per_state, self.state_coordinates, start_date, time_unit)

    def _mp_sim_single(self, drawn_parameters, seed, time, start_date, time_unit, method, tau):
        """
        A Multiprocessing-compatible wrapper for _sim_single, assigns the drawn dictionary and runs _sim_single
        """
        # set sampled parameters
        self.parameters.update(drawn_parameters)
        # set seed
        np.random.seed(seed)
        # simulate model
        return self._sim_single(time, start_date, time_unit, method, tau)

    def sim(self,
            time: Union[int, float, List[Union[int, float]], List[Union[str, datetime]]],
            N: int=1,
            draw_function: Optional[Callable[[Dict[str, Any], *tuple[Any, ...]], Dict[str, Any]]]=None,
            draw_function_kwargs: Dict={},
            processes: Optional[int]=None,
            method: str='tau_leap',
            tau: Union[int, float] = 1,
            time_unit: str='D',
            ) -> xarray.Dataset:

        """
        Interate a JumpProcess model over a specified time interval.
        Can optionally perform `N` repeated simulations with sampling of model parameters using a sampling function `draw_function`.

        input
        -----

        - time : The start and stop "time" for the simulation run.
            - 1) int/float: Interpreted as `[0, time]`. Floats are automatically rounded.
            - 2) list of int/float of type: [start_time, stop_time]: Interpreted as [start_time, stop_time]. Time axis in xarray simulation output is named 'time'. Floats are automatically rounded.
            - 3) list of datetime.datetime or str of type: ['YYYY-MM-DD', 'YYYY-MM-DD']: Interpreted as [start_date, stop_date]. Time axis in xarray simulation output is named 'date'. Floats are automatically rounded.

        - N : int
            - Number of repeated simulations (default: 1)

        - draw_function : function
            - A function altering parameters in the model parameters dictionary between consecutive simulations, usefull to propagate uncertainty, perform sensitivity analysis
            - Has the dictionary of model parameters ('parameters') as its first obligatory input, followed by a variable number of additional inputs.

        - draw_function_kwargs : dictionary
            - A dictionary containing the additional input arguments of the draw function (all inputs except 'parameters').

        - processes: int
            - Number of cores to distribute the `N` draws over (default: 1)

        - method: str
            - Stochastic simulation method. Either 'Stochastic Simulation Algorithm' ('SSA'), or its tau-leaping approximation ('tau_leap'). (default: 'tau_leap').
    
        - tau: int/float
            - Timestep used by the tau-leaping algorithm (default: 0.5)

        - time_unit: str
            - If using datetime simulation time, defines the time unit (default: days)
            - Valid options are: 'us' , 'ms' , 's', 'min', 'h', 'D', 'W'

        output
        ------

        - output: xarray.Dataset
            - Simulation output
        """
        
        # Input checks on solution settings
        validate_solution_methods_JumpProcess(method, tau)
        # Input checks on supplied simulation time
        time, date_range = validate_simulation_time(time, time_unit)
        # Input checks related to draw functions
        if draw_function:
            # validate function
            validate_draw_function(draw_function, draw_function_kwargs, copy.deepcopy(self.parameters), self.initial_states_function, self.state_shapes, self.initial_states_function_args)

        # save a copy before altering to reset after simulation
        cp_pars = copy.deepcopy(self.parameters)

        # Construct list of drawn dictionaries
        drawn_parameters=[]
        for _ in range(N):      
            if draw_function:
                drawn_parameters.append(draw_function(copy.deepcopy(self.parameters), **draw_function_kwargs))
            else:
                drawn_parameters.append({})

        # Run simulations
        if processes: # Needed 
            with get_context("fork").Pool(processes) as p:      # 'fork' instead of 'spawn' to run on Apple Silicon
                seeds = np.random.randint(0, 2**32, size=N)     # requires manual reseeding of the random number generators used in the stochastic algorithms in every child process
                output = p.starmap(partial(self._mp_sim_single, time=time, start_date=date_range[0], time_unit=time_unit, method=method, tau=tau), zip(drawn_parameters, seeds))
        else:
            output=[]
            for pars in drawn_parameters:
                output.append(self._mp_sim_single(pars, np.random.randint(0, 2**32, size=1), time, start_date=date_range[0], time_unit=time_unit, method=method, tau=tau))

        # Append results
        out = output[0]
        for xarr in output[1:]:
            out = xarray.concat([out, xarr], "draws")

        # Reset parameter dictionary
        self.parameters = cp_pars

        return out

################
## ODE Models ##
################

class ODE:
    """
    Make a class with `ODE` as input to build a pySODM model.

    ### Your class must contain

    - states: list
        - names of the model’s states.

    - parameters: list
        - names of the model’s parameters.

    - integrate: staticmethod (callable)
        - function computing the differentials of every model state.
        - must be a static method (decorated with @staticmethod).
        - signature: `def integrate(t, states, parameters, stratified_parameters)`
            - t (float): timestep
            - states: model’s states
            - parameters and stratified parameters
        - returns: a differential for every model state, in the same order as in `states`.

    - (optional) dimensions: list
        - names of the model’s dimensions. number of dimensions determines the dimensions of the model’s states.
            - no dimensions: states are 0-D (scalar)
            - 1 dimension: states are 1-D (np.ndarray)
            - 2 dimensions: states are 2-D (np.ndarray)

    - (optional) stratified_parameters: list
        - names of the stratified parameters. stratified parameters are one-dimensional parameter associated with a model dimension.
            - 0 dimensions: not possible to have stratified parameters
            - 1 dimension: list containing strings - ['stratpar_1', 'stratpar_2']
            - 2+ dimensions: list contains n sublists, where n must equal the number of model dimensions. each sublist contains names of stratified parameters associated with the dimension in the corresponding position in dimensions 
            - example for 3 dimensions model: [['stratpar_1', 'stratpar_2'],[],['stratpar_3']], first dimension in dimensions has two stratified parameters stratpar_1 and stratpar_2, second dimension has no stratified parameters, third dimensions has one stratified parameter stratpar_3.
    
    - (optional) dimensions_per_state: list
        - specify the dimension of each model state seperately. allows you to define models with states of different sizes.
        - if `dimensions_per_state` is not provided, all model states will have the same number of dimensions, equal to the number of model dimensions specified using dimensions.
        - if specified, `dimensions_per_state` must contain n sublists, where n is the number of model states (`n = len(states)`). if a model state has no dimensions (i.e. it is a float), specify an empty list.

    ### Minimal example
    
    ```
    # Define the model equations
    class MY_MODEL(ODE):

        states = ['Y1', 'Y2']
        parameters = ['alpha']

        @staticmethod
        def integrate(t, Y1, Y2, alpha):
            return -alpha*Y1, alpha*Y2
    ```
    """

    states = None
    parameters = None
    stratified_parameters = None
    dimensions = None
    dimensions_per_state = None
    # TODO: states, parameters, dimensions --> list containing str (check input!)

    def __init__(self,
                 initial_states: Union[Dict[str, Union[int, float, np.ndarray]], Callable[..., Dict[str, Union[int, float, np.ndarray]]]],
                 parameters: Dict[str, Any],
                 coordinates: Optional[Dict[str, list]]=None,
                 time_dependent_parameters: Optional[Dict[str, Callable[[Union[float, datetime], Dict[str, Union[int, float, np.ndarray]], Any, tuple], Any]]]=None) -> None:
        
        """
        Initialise an ordinary differential equations model

        Parameters
        ----------

        - initial_states : dict or callable 
            - dict: contains the initial values of all non-zero model states, f.i. for an SIR model: {'S': 1000, 'I': 1}. Initialising zeros is not required.
            - function: an initial condition function (ICF) generating a dictionary can be provided. arguments of the ICF must be added to `parameters`.

        - parameters : dict
            - keys: names of all parameters, stratified_parameters, TDPF and ICF parameters. values: associated parameter values. 

        - (optional) coordinates: dict
            - for each `dimension_name` in the model, specifies the coordinates associated with that dimension.
            - f.e. {'spatial_units': ['city_1','city_2','city_3']}        

        - (optional) time_dependent_parameters : dict
            - keys: name of the parameter you want to impose a time-dependency on. values: time-dependent parameter function.
            - a pySODM-compatible TDPF has the signature `fun(t, states, param, other_parameter_1, ...)`, where:
                - t (float or datetime): current simulation timestep
                - states (dict): dictionary containing the model states at time `t`
                - param (any): value of the parameter the time dependency acts on
        """

        # Add a suffix _names to all user-defined name declarations 
        self.states_names = self.states
        self.parameters_names = self.parameters
        self.parameters_stratified_names = self.stratified_parameters
        self.dimensions_names = self.dimensions
        self.states = initial_states
        parameters = parameters
        self.coordinates = coordinates
        self.time_dependent_parameters = time_dependent_parameters

        # Check formatting of state, parameters, dimension names user has defined in his model class
        check_formatting_names(self.states_names, self.parameters_names, self.parameters_stratified_names, self.dimensions_names)
         
        # Merge parameter_names and parameter_stratified_names
        self.parameters_names_modeldeclaration = merge_parameter_names_parameter_stratified_names(self.parameters_names, self.parameters_stratified_names)

        # Duplicates in lists containing names of states/parameters/stratified parameters/dimensions?
        check_duplicates(self.states_names, 'state_names')
        check_duplicates(self.parameters_names, 'parameter_names')
        check_duplicates(self.parameters_names_modeldeclaration, 'parameter_names + parameter_stratified_names')
        if self.dimensions_names:
            check_duplicates(self.dimensions_names, 'dimension_names')

        # Overlapping state and parameter names?
        check_overlap(self.states_names, self.parameters_names_modeldeclaration, 'state_names', 'parameter_names + parameter_stratified_names')

        # Validate and compute the dimension sizes
        self.dimension_size = validate_dimensions(self.dimensions_names, self.coordinates)

        # Validate dimensions_per_state
        if self.dimensions_per_state:
            validate_dimensions_per_state(self.dimensions_per_state, self.coordinates, self.states_names)
        
        # Build a dictionary containing the size of every state; build a dictionary containing the dimensions of very state; build a dictionary containing the coordinates of every state
        self.state_shapes, self.dimensions_per_state, self.state_coordinates = build_state_sizes_dimensions(self.coordinates, self.states_names, self.dimensions_per_state)

        # Validate the time-dependent parameter functions (TDPFs) and extract the names of their input arguments
        if time_dependent_parameters:
            self._function_parameters = validate_time_dependent_parameters(self.parameters_names, self.parameters_stratified_names, self.time_dependent_parameters)
        else:
            self._function_parameters = []

        # Verify the signature of the integrate function; extract the additional parameters of the TDPFs
        all_params, self._extra_params_TDPF = validate_integrate_or_compute_rates_signature(self.integrate, self.parameters_names_modeldeclaration, self.states_names, self._function_parameters)

        # Get additional parameters of the IC function
        self._extra_params_ICF = get_initial_states_fuction_parameters(self.states)
        all_params.extend(self._extra_params_ICF)
        
        # Verify all parameters were provided
        self.parameters = validate_provided_parameters(set(all_params), parameters, self.states_names)

        # Validate the shapes of the initial states, fill non-defined states with zeros
        self.initial_states, self.initial_states_function, self.initial_states_function_args = evaluate_initial_condition_function(self.states, self.parameters)
        self.initial_states = validate_initial_states(self.initial_states, self.state_shapes)

        # Validate the size of the stratified parameters (Perhaps move this way up front?) --> will deprecate
        if self.parameters_stratified_names:
            self.parameters = validate_parameter_stratified_sizes(self.parameters_stratified_names, self.dimensions_names, coordinates, self.parameters)

        # Call the integrate function, check if it works and check the sizes of the differentials in the output
        validate_integrate(self.initial_states, dict(zip(self.parameters_names_modeldeclaration,[self.parameters[k] for k in self.parameters_names_modeldeclaration])), self.integrate, self.state_shapes)

    # Overwrite integrate class
    @staticmethod
    def integrate():
        """to overwrite in subclasses"""
        raise NotImplementedError

    def _create_fun(self, start_date, time_unit):
        """Convert integrate statement to scipy-compatible function"""
  
        def func(t, y, pars={}):
            """As used by scipy -> flattend in, flattend out"""

            # ------------------------------------------------
            # Deflatten y and reconstruct dictionary of states 
            # ------------------------------------------------

            states = list_to_dict(y, self.state_shapes)

            # --------------------------------------------
            # Reconstruct datetime simtime (if applicable)
            # --------------------------------------------

            if start_date is not None:
                # datetime simtime
                date = time_to_date(float(t), start_date, time_unit)
            else:
                # integer simtime
                date = t

            # --------------------------------------
            # update time-dependent parameter values
            # --------------------------------------

            params = pars.copy()
            if self.time_dependent_parameters:
                for i, (param, param_func) in enumerate(self.time_dependent_parameters.items()):
                    func_params = {key: params[key] for key in self._function_parameters[i]}
                    params[param] = param_func(date, states, pars[param], **func_params)

            # -------------------------------------------------------------
            #  throw parameters of TDPFs out of model parameters dictionary
            # -------------------------------------------------------------

            params = {k:v for k,v in params.items() if ((k not in self._extra_params_TDPF) or (k in self.parameters_names_modeldeclaration))}

            # -------------------
            # perform integration
            # -------------------

            dstates = self.integrate(date, **states, **params)

            # -------
            # Flatten
            # -------

            out=[]
            for d in dstates:
                out.extend(list(np.ravel(d)))

            return out

        return func

    def _solve_discrete(self, fun, tau, t_eval, y, args):
        # Preparations
        y = np.reshape(y, [len(y),1])
        y_prev=y
        # Simulation loop
        t_lst=[t_eval[0]]
        t = t_eval[0]
        while t < t_eval[-1]:
            out = fun(t, y_prev, args)
            out = np.reshape(out,[len(out),1])*tau
            y = np.append(y,y_prev+out,axis=1)
            y_prev += out
            t = t + tau
            t_lst.append(t)
        # Interpolate output y to times t_eval
        y_eval = np.zeros([y.shape[0], len(t_eval)])
        for row_idx in range(y.shape[0]):
            y_eval[row_idx,:] = np.interp(t_eval, t_lst, y[row_idx,:])
            
        return {'y': y_eval, 't': t_eval}

    def _sim_single(self, time, start_date=None, time_unit=None, method='RK23', rtol=5e-3, tau=None):
        """
        Integrates the model over the interval time = [t0, t1] and builds the xarray output. Integration is either continuous (default) or discrete (tau != None).
        """

        # Create scipy-compatible wrapper
        fun = self._create_fun(start_date, time_unit)

        # Construct vector of timesteps
        t0, t1 = time
        t_eval = np.arange(start=t0, stop=t1 + 1, step=1)

        # Update initial condition if a function was provided by user
        if self.initial_states_function:
            ## Call the initial state function to update the initial state
            initial_states = self.initial_states_function(**{key: self.parameters[key] for key in self.initial_states_function_args})
            ## Check the initial states size and fill states not provided with zeros
            initial_states = validate_initial_states(initial_states, self.state_shapes)
            ## Throw out parameters belonging (uniquely) to the initial condition function
            params = cut_ICF_parameters_from_parameters(self.parameters_names_modeldeclaration, self._extra_params_TDPF, self._extra_params_ICF, self.parameters)
        else:
            ## Do nothing
            initial_states = self.initial_states
            params = self.parameters

        # Flatten initial states
        y0=[]
        for v in initial_states.values():
            y0.extend(list(np.ravel(v)))

        # Discrete/continuous
        if tau:
            output = self._solve_discrete(fun, tau, t_eval, y0, args=params)
        else:
            output = solve_ivp(fun, time, y0, args=[params], t_eval=t_eval, method=method, rtol=rtol)

        # Map to variable names
        return _output_to_xarray_dataset(output, self.state_shapes, self.dimensions_per_state, self.state_coordinates, start_date, time_unit)

    def _mp_sim_single(self, drawn_parameters, time, start_date, time_unit, method, rtol, tau):
        """
        A `multiprocessing`-compatible wrapper for `_sim_single`, assigns the drawn dictionaries and runs `_sim_single`
        """
        # set sampled parameters
        self.parameters.update(drawn_parameters)
        # simulate model
        out = self._sim_single(time, start_date, time_unit, method, rtol, tau)
        return out
    
    def sim(self,
            time: Union[int, float, List[Union[int, float]], List[Union[str, datetime]]],
            N: int=1,
            draw_function: Optional[Callable[[Dict[str, Any], *tuple[Any, ...]], Dict[str, Any]]]=None,
            draw_function_kwargs: Dict={},
            processes: Optional[int]=None,
            method: str='RK45',
            rtol: float=1e-4,
            tau: Optional[Union[int,float]]=None,
            time_unit: str='D',
            ) -> xarray.Dataset:

        """
        Integrate an ODE model over a specified time interval.
        Can optionally perform `N` repeated simulations with sampling of model parameters using a function `draw_function`.

        input
        -----

        - time : The start and stop "time" for the simulation run.
            - 1) int/float: Interpreted as `[0, time]`. Floats are automatically rounded.
            - 2) list of int/float of type: [start_time, stop_time]: Interpreted as [start_time, stop_time]. Time axis in xarray simulation output is named 'time'. Floats are automatically rounded.
            - 3) list of datetime.datetime or str of type: ['YYYY-MM-DD', 'YYYY-MM-DD']: Interpreted as [start_date, stop_date]. Time axis in xarray simulation output is named 'date'. Floats are automatically rounded.

        - N : int
            - Number of repeated simulations (default: 1)

        - draw_function : function
            - A function altering parameters in the model parameters dictionary between consecutive simulations, usefull to propagate uncertainty, perform sensitivity analysis
            - Has the dictionary of model parameters ('parameters') as its first obligatory input, followed by a variable number of additional inputs.

        - draw_function_kwargs : dictionary
            - A dictionary containing the additional input arguments of the draw function (all inputs except 'parameters').

        - processes: int
            - Number of cores to distribute the `N` draws over (default: 1)

        - method: str
            - Method used by Scipy `solve_ivp` for integration of differential equations. Default: 'RK45'.

        - rtol: float
            - Relative tolerance of Scipy `solve_ivp`. Default: 1e-4.
        
        - tau: int/float
            - If `tau != None`, the integrator (`scipy.solve_ivp()`) is overwritten and a discrete timestepper with timestep `tau` is used (default: None)

        - time_unit: str
            - If using datetime simulation time, defines the time unit (default: days)
            - Valid options are: 'us' , 'ms' , 's', 'min', 'h', 'D', 'W'

        output
        ------

        - output: xarray.Dataset
            - Simulation output
        """

        # Input checks on solution settings
        validate_solution_methods_ODE(rtol, method, tau)
        # Input checks on supplied simulation time
        time, date_range = validate_simulation_time(time, time_unit)
        # Input checks on draw functions
        if draw_function:
            # validate function
            validate_draw_function(draw_function, draw_function_kwargs, copy.deepcopy(self.parameters), self.initial_states_function, self.state_shapes, self.initial_states_function_args)
        # provinding 'N' but no draw function: wasteful of resources
        if ((N != 1) & (draw_function==None)):
            raise ValueError('attempting to perform N={0} repeated simulations without using a draw function'.format(N))

        # save a copy before altering to reset after simulation
        cp_pars = copy.deepcopy(self.parameters)

        # Construct list of drawn parameters and initial states
        drawn_parameters=[]
        for _ in range(N):
            if draw_function:
                drawn_parameters.append(draw_function(copy.deepcopy(self.parameters), **draw_function_kwargs))
            else:
                drawn_parameters.append({})

        # Run simulations
        if processes: # Needed 
            with get_context("fork").Pool(processes) as p:
                output = p.map(partial(self._mp_sim_single, time=time, start_date=date_range[0], time_unit=time_unit, method=method, rtol=rtol, tau=tau), drawn_parameters)
        else:
            output=[]
            for pars in drawn_parameters:
                output.append(self._mp_sim_single(pars, time, date_range[0], time_unit, method=method, rtol=rtol, tau=tau))

        # Append results
        out = output[0]
        for xarr in output[1:]:
            out = xarray.concat([out, xarr], "draws")

        # Reset parameter dictionary
        self.parameters = cp_pars

        return out

def _output_to_xarray_dataset(output, state_shapes, dimensions_per_state, state_coordinates, start_date=None, time_unit=None):
    """
    Convert array (returned by scipy) to an xarray Dataset with the right coordinates and variable names

    Parameters
    ----------

    output: dict
        Keys: "y" (states) and "t" (timesteps)
        Size of `y`: (total number of states, number of timesteps)
        Size of `t`: number of timesteps

    state_shapes: dict
        Keys: state names. Values: tuples with state shape

    dimensions_per_state: dict
        Keys: state names. Values: list containing dimensions associated with state

    state_coordinates: dict
        Keys: state names. Values: list containing coordinates of every dimension the state is associated with

    actual_start_date: datetime
        Used to determine if the time output should be returned as dates

    Returns
    -------

    output: xarray.Dataset
        Simulation results
    """

    # Convert scipy's output to a dictionary of states with the correct sizes
    new_state_shapes={}
    for k,v in state_shapes.items():
        v=list(v)
        v.append(len(output["t"]))
        new_state_shapes.update({k: tuple(v)})
    output_flat = np.ravel(output["y"])
    data_variables = list_to_dict(output_flat, new_state_shapes)
    # Move time axis to first position (yes, obviously I have tried this  with np.reshape in `list_to_dict` but this didn't work for n-D states)
    for k,v in data_variables.items():
        data_variables.update({k: np.moveaxis(v, [-1,], [0,])})

    # Append the time dimension
    new_dimensions_per_state={}
    for k,v in dimensions_per_state.items():
        v_acc = v.copy()
        if start_date is not None:
            v_acc = ['date',] + v_acc
            new_dimensions_per_state.update({k: v_acc})
        else:
            v_acc = ['time',] + v_acc
            new_dimensions_per_state.update({k: v_acc})

    # Append the time coordinates
    new_state_coordinates={}
    for k,v in state_coordinates.items():
        v_acc=v.copy()
        if start_date is not None:
            v_acc = [[start_date + timedelta(microseconds=float(i*time_unit_map[time_unit])) for i in output["t"]],] + v_acc
            new_state_coordinates.update({k: v_acc})
        else:
            v_acc = [output["t"],] + v_acc
            new_state_coordinates.update({k: v_acc})

    # Build the xarray dataset
    data = {}
    for var, arr in data_variables.items():
        if len(new_dimensions_per_state[var]) == 1:
            arr = np.ravel(arr)
        xarr = xarray.DataArray(arr, dims=new_dimensions_per_state[var], coords=new_state_coordinates[var])
        data[var] = xarr

    return xarray.Dataset(data)