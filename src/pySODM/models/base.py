# Prevents numpy from using multiple threads to perform computations on large matrices
import os
os.environ["OMP_NUM_THREADS"] = "1"
# Packages
import random
import itertools
import xarray
import copy
import numpy as np
import pandas as pd
import numba as nb
import multiprocessing as mp
from functools import partial
from scipy.integrate import solve_ivp
from pySODM.models.utils import int_to_date, list_to_dict
from pySODM.models.validation import merge_parameter_names_parameter_stratified_names, validate_draw_function, validate_simulation_time, validate_dimensions, \
                                        validate_time_dependent_parameters, validate_integrate, check_duplicates, build_state_sizes_dimensions, validate_state_dimensions, \
                                            validate_initial_states, validate_integrate_or_compute_rates_signature, validate_provided_parameters, validate_parameter_stratified_sizes, \
                                                validate_apply_transitionings_signature, validate_compute_rates, validate_apply_transitionings

class SDEModel:
    """
    Initialise a stochastic differential equations model

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states, f.i. for an SIR model,
        e.g. {'S': 1000, 'I': 1}
        initialising zeros is not required
    parameters : dictionary
        values of all parameters, stratified_parameters, TDPF parameters
    time_dependent_parameters : dictionary, optional
        Optionally specify a function for time dependency on the parameters. The
        signature of the function should be `fun(t, states, param, other_parameter_1, ...)` taking
        the time, the initial parameter value, and potentially additional
        keyword argument, and should return the new parameter value for
        time `t`.
    coordinates: dictionary, optional
        Specify for each 'dimension_name' the coordinates to be used.
        These coordinates can then be used to acces data easily in the output xarray.
        Example: {'spatial_units': ['city_1','city_2','city_3']}
    """

    state_names = None
    parameter_names = None
    dimension_names = None
    parameter_stratified_names = None
    state_dimensions = None

    def __init__(self, states, parameters, coordinates=None, time_dependent_parameters=None):

        # Do not undergo manipulation during model initialization
        self.coordinates = coordinates
        self.time_dependent_parameters = time_dependent_parameters

        # Merge parameter_names and parameter_stratified_names
        self.parameter_names_merged = merge_parameter_names_parameter_stratified_names(self.parameter_names, self.parameter_stratified_names)

        # Duplicates in lists containing names of states/parameters/stratified parameters/dimensions?
        check_duplicates(self.state_names, 'state_names')
        check_duplicates(self.parameter_names, 'parameter_names')
        check_duplicates(self.parameter_names_merged, 'parameter_names + parameter_stratified_names')
        if self.dimension_names:
            check_duplicates(self.dimension_names, 'dimension_names')

        # Validate and compute the dimension sizes
        self.dimension_size = validate_dimensions(self.dimension_names, self.coordinates)

        # Validate state_dimensions
        if self.state_dimensions:
            validate_state_dimensions(self.state_dimensions, self.coordinates, self.state_names)
        
        # Build a dictionary containing the size of every state; build a dictionary containing the dimensions of very state; build a dictionary containing the coordinates of every state
        self.state_shapes, self.state_dimensions, self.state_coordinates = build_state_sizes_dimensions(self.coordinates, self.state_names, self.state_dimensions)

        # Validate the shapes of the initial states, fill non-defined states with zeros
        self.initial_states = validate_initial_states(self.state_shapes, states)

        # Validate the time-dependent parameter functions (TDPFs) and extract the names of their input arguments
        if time_dependent_parameters:
            self._function_parameters = validate_time_dependent_parameters(self.parameter_names, self.parameter_stratified_names, self.time_dependent_parameters)
        else:
            self._function_parameters = []

        # Verify the signature of the compute_rates function; extract the additional parameters of the TDPFs
        all_params, self._extra_params = validate_integrate_or_compute_rates_signature(self.compute_rates, self.parameter_names_merged, self.state_names, self._function_parameters)

        # Verify the signature of the apply_transitionings function
        validate_apply_transitionings_signature(self.apply_transitionings, self.parameter_names_merged, self.state_names)

        # Verify all parameters were provided
        self.parameters = validate_provided_parameters(all_params, parameters)

        # Validate the size of the stratified parameters (Perhaps move this way up front?)
        if self.parameter_stratified_names:
            self.parameters = validate_parameter_stratified_sizes(self.parameter_stratified_names, self.dimension_names, coordinates, self.parameters)

        # Call the compute_rates function, check if it works and check the sizes of the differentials in the output
        rates = validate_compute_rates(self.compute_rates, self.initial_states, self.state_shapes, self.parameter_names_merged, self.parameters)
        validate_apply_transitionings(self.apply_transitionings, rates, self.initial_states, self.state_shapes, self.parameter_names_merged, self.parameters)

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

    def _tau_leap(self, states, rates, tau):
        """
        Tau-leaping algorithm by Gillespie
        Loops over the model states, extracts the values of the states and the rates, passes them to `_draw_transitionings()`

        Inputs
        ------

        states: dict
            Dictionary containing the values of the model states.

        rates: dict
            Dictionary containing the rates associated with each transitionings of the model states.

        tau: int/float
            Timestep
        
        Returns
        -------

        transitionings: dict
            Dictionary containing the number of transitionings for every model state
        
        tau: int/float
            Timestep
        
        """

        transitionings={k: v[:] for k, v in rates.items()}
        for k,rate in rates.items():
            transitionings[k] = self._draw_transitionings(states[k], nb.typed.List(rate), tau)

        return transitionings, tau

    @staticmethod
    @nb.jit(nopython=True)
    def _draw_transitionings(states, rates, tau):
        """
        Draw the transitionings from a multinomial distribution
        The use of the multinomial distribution is necessary to avoid states with two transitionings from becoming negative

        Inputs
        ------
        states: np.ndarray
            n-dimensional matrix representing a given model state (passed down from `_tau_leap()`)

        rates: list
            List containing the transitioning rates of the considered model state. Elements of rates are np.ndarrays with the same dimensions as states.

        tau: int/float
            Timestep

        Returns
        -------
        Transitionings: list
            List containing the drawn transitionings of the considered model state. Elements of rates are np.ndarrays with the same dimensions as states.

        """
        
        # Step 1: flatten states and rates
        size = states.shape
        states = states.flatten()
        transitioning_out = rates
        transitioning = [rate.flatten() for rate in rates]
        rates  = [rate.flatten() for rate in rates]

        # Step 2: loop + draw
        for i, state in enumerate(states):
            p=np.zeros(len(rates),np.float64)
            # Make a list with the transitioning probabilities
            for j in range(len(rates)):
                p[j] = 1 - np.exp(-tau*rates[j][i])
            # Append the chance of no transitioning
            p = np.append(p, 1-np.sum(p))
            # Sample from a multinomial and omit chance of no transition
            samples = np.random.multinomial(int(state), p)[:-1]
            # Assign to transitioning
            for j,sample in enumerate(samples):
                transitioning[j][i] = sample
        # Reshape transitioning
        for j,trans in enumerate(transitioning):
            transitioning_out[j] = np.reshape(transitioning[j], size)
            
        return transitioning_out

    def _create_fun(self, actual_start_date, method='SSA', tau_input=0.5):
        
        def func(t, y, pars={}):
                """As used by scipy -> flattend in, flattend out"""

                # ------------------------------------------------
                # Deflatten y and reconstruct dictionary of states 
                # ------------------------------------------------

                states = list_to_dict(y, self.state_shapes, retain_floats=False)

                # --------------------------------------
                # update time-dependent parameter values
                # --------------------------------------

                params = pars.copy()

                if self.time_dependent_parameters:
                    if actual_start_date is not None:
                        date = int_to_date(actual_start_date, t)
                    else:
                        date = t
                    for i, (param, param_func) in enumerate(self.time_dependent_parameters.items()):
                        func_params = {key: params[key] for key in self._function_parameters[i]}
                        params[param] = param_func(date, states, pars[param], **func_params)

                # -------------------------------------------------------------
                #  throw parameters of TDPFs out of model parameters dictionary
                # -------------------------------------------------------------

                params = {k:v for k,v in params.items() if ((k not in self._extra_params) or (k in self.parameter_names_merged))}

                # -------------
                # compute rates
                # -------------

                rates = self.compute_rates(t, **states, **params)

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

    def _sim_single(self, time, actual_start_date=None, method='tau_leap', tau=0.5, output_timestep=1):

        # Initialize wrapper
        fun = self._create_fun(actual_start_date, method, tau)

        # Prepare time
        t0, t1 = time
        t_eval = np.arange(start=t0, stop=t1 + 1, step=output_timestep)

        # Flatten initial states
        y0=[]
        for v in self.initial_states.values():
            y0.extend(list(np.ravel(v)))

        # Run the time loop
        output = self._solve_discrete(fun, t_eval, np.asarray(y0), self.parameters)

        return _output_to_xarray_dataset(output, self.state_shapes, self.state_dimensions, self.state_coordinates, actual_start_date)

    def _mp_sim_single(self, drawn_parameters, time, actual_start_date, method, tau, output_timestep):
        """
        A Multiprocessing-compatible wrapper for _sim_single, assigns the drawn dictionary and runs _sim_single
        """
        self.parameters.update(drawn_parameters)
        return self._sim_single(time, actual_start_date, method, tau, output_timestep)

    def sim(self, time, warmup=0, N=1, draw_function=None, samples=None, processes=None, method='tau_leap', tau=0.5, output_timestep=1):

        """
        Run a model simulation for the given time period. Can optionally perform N repeated simulations of time days.
        Can change the values of model parameters at every repeated simulation by drawing samples from a dictionary `samples` using a function `draw_function`

        Parameters
        ----------

        time : 1) int/float, 2) list of int/float of type '[start_time, stop_time]', 3) list of pd.Timestamp or str of type '[start_date, stop_date]',
            The start and stop "time" for the simulation run.
            1) Input is converted to [0, time]. Floats are automatically rounded.
            2) Input is interpreted as [start_time, stop_time]. Time axis in xarray output is named 'time'. Floats are automatically rounded.
            3) Input is interpreted as [start_date, stop_date]. Time axis in xarray output is named 'date'. Floats are automatically rounded.

        warmup : int
            Number of days to simulate prior to start time or date

        N : int
            Number of repeated simulations (default: 1)

        draw_function : function
            A function with as obligatory inputs the dictionary of model parameters and a dictionary of samples
            The function can alter parameters in the model parameters dictionary between repeated simulations `N`.
            Usefull to propagate uncertainty, perform sensitivity analysis.

        samples : dictionary
            Sample dictionary used by `draw_function`.

        processes: int
            Number of cores to distribute the `N` draws over

        method: str
            Stochastic simulation method. Either 'Stochastic Simulation Algorithm' ('SSA'), or its tau-leaping approximation ('tau_leap').
    
        tau: int/float
            Timestep used by the tau-leaping algorithm (default: 0.5)

        output_timestep: int/flat
            Interpolate model output to every `output_timestep` time
            For datetimes: expressed in days

        Returns
        -------

        output: xarray.Dataset
            Simulation output
        """

        # Input checks on solution method and timestep
        if not isinstance(method, str):
            raise TypeError(
                "solver method 'method' must be of type string"
                )
        if not isinstance(tau, (int,float)):
            raise TypeError(
                "discrete timestep 'tau' must be of type int or float"
                )

        # Input checks on supplied simulation time
        time, actual_start_date = validate_simulation_time(time, warmup)

        # Input check on draw function
        if draw_function:
            validate_draw_function(draw_function, self.parameters, samples)

        # Copy parameter dictionary --> dict is global
        cp = copy.deepcopy(self.parameters)
        # Construct list of drawn dictionaries
        drawn_dictionaries=[]
        for n in range(N):
            cp_draws=copy.deepcopy(self.parameters)        
            if draw_function:
                out={} # Need because of global dictionaries and voodoo magic
                out.update(draw_function(self.parameters,samples))
                drawn_dictionaries.append(out)
            else:
                drawn_dictionaries.append({})
            self.parameters=cp_draws    

        # Run simulations
        if processes: # Needed 
            with mp.Pool(processes) as p:
                output = p.map(partial(self._mp_sim_single, time=time, actual_start_date=actual_start_date, method=method, tau=tau, output_timestep=output_timestep), drawn_dictionaries)
        else:
            output=[]
            for dictionary in drawn_dictionaries:
                output.append(self._mp_sim_single(dictionary, time, actual_start_date, method=method, tau=tau, output_timestep=output_timestep))

        # Append results
        out = output[0]
        for xarr in output[1:]:
            out = xarray.concat([out, xarr], "draws")

        # Reset parameter dictionary
        self.parameters = cp

        return out

################
## ODE Models ##
################

class ODEModel:
    """
    Initialise an ordinary differential equations model

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states, f.i. for an SIR model,
        e.g. {'S': 1000, 'I': 1}
        initialising zeros is not required
    parameters : dictionary
        values of all parameters, stratified_parameters, TDPF parameters
    time_dependent_parameters : dictionary, optional
        Optionally specify a function for time dependency on the parameters. The
        signature of the function should be `fun(t, states, param, other_parameter_1, ...)` taking
        the time, the initial parameter value, and potentially additional
        keyword argument, and should return the new parameter value for
        time `t`.
    coordinates: dictionary, optional
        Specify for each 'dimension_name' the coordinates to be used.
        These coordinates can then be used to acces data easily in the output xarray.
        Example: {'spatial_units': ['city_1','city_2','city_3']}
    """

    state_names = None
    parameter_names = None
    dimension_names = None
    parameter_stratified_names = None
    state_dimensions = None

    def __init__(self, states, parameters, coordinates=None, time_dependent_parameters=None):

        # Do not undergo manipulation during model initialization
        self.coordinates = coordinates
        self.time_dependent_parameters = time_dependent_parameters

        # Merge parameter_names and parameter_stratified_names
        self.parameter_names_merged = merge_parameter_names_parameter_stratified_names(self.parameter_names, self.parameter_stratified_names)

        # Duplicates in lists containing names of states/parameters/stratified parameters/dimensions?
        check_duplicates(self.state_names, 'state_names')
        check_duplicates(self.parameter_names, 'parameter_names')
        check_duplicates(self.parameter_names_merged, 'parameter_names + parameter_stratified_names')
        if self.dimension_names:
            check_duplicates(self.dimension_names, 'dimension_names')

        # Validate and compute the dimension sizes
        self.dimension_size = validate_dimensions(self.dimension_names, self.coordinates)

        # Validate state_dimensions
        if self.state_dimensions:
            validate_state_dimensions(self.state_dimensions, self.coordinates, self.state_names)
        
        # Build a dictionary containing the size of every state; build a dictionary containing the dimensions of very state; build a dictionary containing the coordinates of every state
        self.state_shapes, self.state_dimensions, self.state_coordinates = build_state_sizes_dimensions(self.coordinates, self.state_names, self.state_dimensions)

        # Validate the shapes of the initial states, fill non-defined states with zeros
        self.initial_states = validate_initial_states(self.state_shapes, states)

        # Validate the time-dependent parameter functions (TDPFs) and extract the names of their input arguments
        if time_dependent_parameters:
            self._function_parameters = validate_time_dependent_parameters(self.parameter_names, self.parameter_stratified_names, self.time_dependent_parameters)
        else:
            self._function_parameters = []

        # Verify the signature of the integrate function; extract the additional parameters of the TDPFs
        all_params, self._extra_params = validate_integrate_or_compute_rates_signature(self.integrate, self.parameter_names_merged, self.state_names, self._function_parameters)

        # Verify all parameters were provided
        self.parameters = validate_provided_parameters(all_params, parameters)

        # Validate the size of the stratified parameters (Perhaps move this way up front?)
        if self.parameter_stratified_names:
            self.parameters = validate_parameter_stratified_sizes(self.parameter_stratified_names, self.dimension_names, coordinates, self.parameters)

        # Call the integrate function, check if it works and check the sizes of the differentials in the output
        validate_integrate(self.initial_states, dict(zip(self.parameter_names_merged,[self.parameters[k] for k in self.parameter_names_merged])), self.integrate, self.state_shapes)

    # Overwrite integrate class
    @staticmethod
    def integrate():
        """to overwrite in subclasses"""
        raise NotImplementedError

    def _create_fun(self, actual_start_date):
        """Convert integrate statement to scipy-compatible function"""
  
        def func(t, y, pars={}):
            """As used by scipy -> flattend in, flattend out"""

            # ------------------------------------------------
            # Deflatten y and reconstruct dictionary of states 
            # ------------------------------------------------

            states = list_to_dict(y, self.state_shapes)

            # --------------------------------------
            # update time-dependent parameter values
            # --------------------------------------

            params = pars.copy()

            if self.time_dependent_parameters:
                if actual_start_date is not None:
                    date = int_to_date(actual_start_date, t)
                else:
                    date = t
                for i, (param, param_func) in enumerate(self.time_dependent_parameters.items()):
                    func_params = {key: params[key] for key in self._function_parameters[i]}
                    params[param] = param_func(date, states, pars[param], **func_params)

            # -------------------------------------------------------------
            #  throw parameters of TDPFs out of model parameters dictionary
            # -------------------------------------------------------------

            params = {k:v for k,v in params.items() if ((k not in self._extra_params) or (k in self.parameter_names_merged))}

            # -------------------
            # perform integration
            # -------------------

            dstates = self.integrate(t, **states, **params)

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

    def _sim_single(self, time, actual_start_date=None, method='RK23', rtol=5e-3, output_timestep=1, tau=None):
        """
        Integrates the model over the interval time = [t0, t1] and builds the xarray output. Integration is either continuous (default) or discrete (tau != None).
        """

        # Create scipy-compatible wrapper
        fun = self._create_fun(actual_start_date)

        # Construct vector of timesteps
        t0, t1 = time
        t_eval = np.arange(start=t0, stop=t1 + output_timestep, step=output_timestep)

        # Flatten initial states
        y0=[]
        for v in self.initial_states.values():
            y0.extend(list(np.ravel(v)))

        if tau:
            output = self._solve_discrete(fun, tau, t_eval, y0, args=self.parameters)
        else:
            output = solve_ivp(fun, time, y0, args=[self.parameters], t_eval=t_eval, method=method, rtol=rtol)

        # Map to variable names
        return _output_to_xarray_dataset(output, self.state_shapes, self.state_dimensions, self.state_coordinates, actual_start_date)

    def _mp_sim_single(self, drawn_parameters, time, actual_start_date, method, rtol, output_timestep, tau):
        """
        A `multiprocessing`-compatible wrapper for `_sim_single`, assigns the drawn dictionary and runs `_sim_single`
        """
        self.parameters.update(drawn_parameters)
        out = self._sim_single(time, actual_start_date, method, rtol, output_timestep, tau)
        return out

    def sim(self, time, warmup=0, N=1, draw_function=None, samples=None, processes=None, method='RK23', rtol=1e-3, tau=None, output_timestep=1):
        """
        Run a model simulation for a given time period. Can optionally perform `N` repeated simulations with sampling of model parameters from a dictionary `samples` using a function `draw_function`. Can perform discrete timestepping if `tau != None`.

        Parameters
        ----------

        time : 1) int/float, 2) list of int/float of type '[start_time, stop_time]', 3) list of pd.Timestamp or str of type '[start_date, stop_date]',
            The start and stop "time" for the simulation run.
            1) Input is converted to [0, time]. Floats are automatically rounded.
            2) Input is interpreted as [start_time, stop_time]. Time axis in xarray output is named 'time'. Floats are automatically rounded.
            3) Input is interpreted as [start_date, stop_date]. Time axis in xarray output is named 'date'. Floats are automatically rounded.

        warmup : int
            Number of days to simulate prior to start time or date

        N : int
            Number of repeated simulations (default: 1)

        draw_function : function
            A function with as obligatory inputs the dictionary of model parameters and a dictionary of samples
            The function can alter parameters in the model parameters dictionary between repeated simulations `N`.
            Usefull to propagate uncertainty, perform sensitivity analysis.

        samples : dictionary
            Sample dictionary used by `draw_function`.

        processes: int
            Number of cores to distribute the `N` draws over.

        method: str
            Method used by Scipy `solve_ivp` for integration of differential equations. Default: 'RK23'.

        rtol: float
            Relative tolerance of Scipy `solve_ivp`. Default: 1e-3.
        
        tau: int/float
            If `tau != None`, the integrator (`scipy.solve_ivp()`) is overwritten and a discrete timestepper with timestep `tau` is used. (default:None)

        output_timestep: int/flat
            Interpolate model output to every `output_timestep` time
            For datetimes: expressed in days

        Returns
        -------

        output: xarray.Dataset
            Simulation output
        """

        # Input checks on solver settings
        if not isinstance(rtol, float):
            raise TypeError(
                "relative solver tolerance 'rtol' must be of type float"
                )
        if not isinstance(method, str):
            raise TypeError(
                "solver method 'method' must be of type string"
                )
        if tau != None:
            if not isinstance(tau, (int,float)):
                raise TypeError(
                    "discrete timestep 'tau' must be of type int or float"
                )
            print(f"performing discrete timestepping with tau = {tau}\n")

        # Input checks on supplied simulation time
        time, actual_start_date = validate_simulation_time(time, warmup)

        # Input check on draw function
        if draw_function:
            validate_draw_function(draw_function, self.parameters, samples)
        
        # Provinding 'N' but no draw function: wastefull
        if ((N != 1) & (draw_function==None)):
            raise ValueError('performing N={0} repeated simulations without a `draw_function` is mighty wastefull of computational resources'.format(N))

        # Copy parameter dictionary --> dict is global
        cp = copy.deepcopy(self.parameters)
        # Construct list of drawn dictionaries
        drawn_dictionaries=[]
        for n in range(N):
            cp_draws=copy.deepcopy(self.parameters)
            if draw_function:
                out={} # Need because of global dictionaries and voodoo magic
                out.update(draw_function(self.parameters,samples))
                drawn_dictionaries.append(out)
            else:
                drawn_dictionaries.append({})
            self.parameters=cp_draws

        # Run simulations
        if processes: # Needed 
            with mp.Pool(processes) as p:
                output = p.map(partial(self._mp_sim_single, time=time, actual_start_date=actual_start_date, method=method, rtol=rtol, output_timestep=output_timestep, tau=tau), drawn_dictionaries)
        else:
            output=[]
            for dictionary in drawn_dictionaries:
                output.append(self._mp_sim_single(dictionary, time, actual_start_date, method=method, rtol=rtol, output_timestep=output_timestep, tau=tau))

        # Append results
        out = output[0]
        for xarr in output[1:]:
            out = xarray.concat([out, xarr], "draws")

        # Reset parameter dictionary
        self.parameters = cp

        return out


def _output_to_xarray_dataset(output, state_shapes, state_dimensions, state_coordinates, actual_start_date=None):
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

    state_dimensions: dict
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
    new_state_dimensions={}
    for k,v in state_dimensions.items():
        v_acc = v.copy()
        if actual_start_date is not None:
            v_acc = ['date',] + v_acc
            new_state_dimensions.update({k: v_acc})
        else:
            v_acc = ['time',] + v_acc
            new_state_dimensions.update({k: v_acc})

    # Append the time coordinates
    new_state_coordinates={}
    for k,v in state_coordinates.items():
        v_acc=v.copy()
        if actual_start_date is not None:
            v_acc = [actual_start_date + pd.to_timedelta(output["t"], unit='D'),] + v_acc
            new_state_coordinates.update({k: v_acc})
        else:
            v_acc = [output["t"],] + v_acc
            new_state_coordinates.update({k: v_acc})

    # Build the xarray dataset
    data = {}
    for var, arr in data_variables.items():
        if len(new_state_dimensions[var]) == 1:
            arr = np.ravel(arr)
        xarr = xarray.DataArray(arr, dims=new_state_dimensions[var], coords=new_state_coordinates[var])
        data[var] = xarr

    return xarray.Dataset(data)