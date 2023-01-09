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
from pySODM.models.utils import int_to_date
from pySODM.models.validation import merge_parameter_names_parameter_stratified_names, validate_draw_function, validate_simulation_time, validate_stratifications, validate_time_dependent_parameters, validate_ODEModel, validate_SDEModel, check_duplicates

class SDEModel:
    """
    Initialise a stochastic differential equations model

    Parameters
    ----------
    To initialise the model, provide following inputs:

    states : dictionary
        contains the initial values of all non-zero model states, f.i. for an SEIR model,
        e.g. {'S': [1000, 1000, 1000], 'E': [1, 1, 1]
        initialising zeros is not required
    parameters : dictionary
        values of all parameters (both stratified and not)
    time_dependent_parameters : dictionary, optional
        Optionally specify a function for time-dependent parameters. The
        signature of the function should be ``fun(t, states, param, ...)`` taking
        the time, the initial parameter value, and potentially additional
        keyword argument, and should return the new parameter value for
        time `t`.
    coordinates: dictionary, optional
        Specify for each 'stratification_name' the coordinates to be used.
        These coordinates can then be used to acces data easily in the output xarray.
        Example: {'spatial_units': ['city_1','city_2','city_3']}
    """

    state_names = None
    parameter_names = None
    parameter_stratified_names = None
    stratification_names = None

    def __init__(self, states, parameters, coordinates=None, time_dependent_parameters=None):

        self.parameters = parameters
        self.coordinates = coordinates
        self.time_dependent_parameters = time_dependent_parameters

        # Merge parameter_names and parameter_stratified_names
        self.parameter_names_merged = merge_parameter_names_parameter_stratified_names(self.parameter_names, self.parameter_stratified_names)

        # Duplicates in lists containing names of states/parameters/stratified parameters/stratifications?
        check_duplicates(self.state_names, 'state_names')
        try:
            check_duplicates(self.parameter_names_merged, 'parameter_names + parameter_stratified_names')
        except:
            check_duplicates(self.parameter_names, 'parameter_names')
        if self.stratification_names:
            check_duplicates(self.stratification_names, 'stratification_names')

        # Validate and compute the stratification sizes
        self.stratification_size = validate_stratifications(self.stratification_names, self.coordinates)

        # Validate the time-dependent parameter functions
        if time_dependent_parameters:
            self._function_parameters = validate_time_dependent_parameters(self.parameter_names, self.parameter_stratified_names, self.time_dependent_parameters)
        else:
            self._function_parameters = []
        
        # Validate the model
        self.initial_states, self.parameters, self._n_function_params, self._extra_params = validate_SDEModel(states, parameters, coordinates, self.stratification_size, self.state_names,
                                                                                                              self.parameter_names, self.parameter_stratified_names, self._function_parameters,
                                                                                                              self.compute_rates, self.apply_transitionings)

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

                # -------------------------------------------------------------
                # Flatten y and construct dictionary of states and their values
                # -------------------------------------------------------------

                # for the moment assume sequence of parameters, vars,... is correct
                size_lst=[len(self.state_names)]
                for size in self.stratification_size:
                    size_lst.append(size)
                y_reshaped = y.reshape(tuple(size_lst))
                states = dict(zip(self.state_names, y_reshaped))

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

                return np.array(dstates).flatten(), tau

        return func

    def _solve_discrete(self, fun, t_eval, y, args):
        # Preparations
        y=np.asarray(y) # otherwise error in func : y.reshape does not work
        y=np.reshape(y,[y.size,1])
        y_prev=y
        # Simulation loop
        t_lst=[t_eval[0]]
        t = t_eval[0]
        while t < t_eval[-1]:
            out, tau = fun(t, y_prev, args)
            y_prev=out
            out = np.reshape(out,[out.size,1])
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

        # Flatten the initial condition
        y0 = list(itertools.chain(*self.initial_states.values()))
        while np.array(y0).ndim > 1:
            y0 = list(itertools.chain(*y0))

        # Run the time loop
        output = self._solve_discrete(fun, t_eval, list(itertools.chain(*self.initial_states.values())), self.parameters)

        return self._output_to_xarray_dataset(output, actual_start_date)

    def _mp_sim_single(self, drawn_parameters, time, actual_start_date, method, tau, output_timestep):
        """
        A Multiprocessing-compatible wrapper for _sim_single, assigns the drawn dictionary and runs _sim_single
        """
        self.parameters.update(drawn_parameters)
        return self._sim_single(time, actual_start_date, method, tau, output_timestep)

    def sim(self, time, warmup=0, N=1, draw_function=None, samples=None, method='tau_leap', tau=0.5, output_timestep=1, processes=None):

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
            Number of repeated simulations (useful for stochastic models). One by default.

        draw_function : function
            A function which takes as its input the dictionary of model parameters and a samples dictionary
            and the dictionary of sampled parameter values and assings these samples to the model parameter dictionary ad random.

        samples : dictionary
            Sample dictionary used by draw_function. Does not need to be supplied if samples_dict is not used in draw_function.

        method: str
            Stochastic simulation method. Either 'Stochastic Simulation Algorithm' (SSA), or its tau-leaping approximation (tau_leap).
        
        tau: int/float
            Timestep used by the tau-leaping algorithm

        output_timestep: int/flat
            Interpolate model output to every `output_timestep` time
            For datetimes: expressed in days

        processes: int
            Number of cores to distribute the N draws over.

        Returns
        -------
        xarray.Dataset

        """

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

    def _output_to_xarray_dataset(self, output, actual_start_date=None):
        """
        Convert array (returned by scipy) to an xarray Dataset with the right coordinates and variable names
        """

        if self.coordinates:
            dims = list(self.coordinates.keys()).copy()
        else:
            dims = []
        
        if actual_start_date is not None:
            dims.append('date')
            coords = {"date": actual_start_date + pd.to_timedelta(output["t"], unit='D')}
        else:
            dims.append('time')
            coords = {"time": output["t"]}

        if self.coordinates:
            coords.update(self.coordinates)

        size_lst = [len(self.state_names)]
        if self.coordinates:
            for size in self.stratification_size:
                size_lst.append(size)
        size_lst.append(len(output["t"]))


        y_reshaped = output["y"].reshape(tuple(size_lst))
        zip_star = zip(self.state_names, y_reshaped)
     
        data = {}
        for var, arr in zip_star:
            xarr = xarray.DataArray(arr, coords=coords, dims=dims)
            data[var] = xarr
        
        return xarray.Dataset(data)

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
        contains the initial values of all non-zero model states, f.i. for an SEIR model,
        e.g. {'S': [1000, 1000, 1000], 'E': [1, 1, 1]
        initialising zeros is not required
    parameters : dictionary
        values of all parameters (both stratified and not)
    time_dependent_parameters : dictionary, optional
        Optionally specify a function for time-dependent parameters. The
        signature of the function should be ``fun(t, states, param, ...)`` taking
        the time, the initial parameter value, and potentially additional
        keyword argument, and should return the new parameter value for
        time `t`.
    coordinates: dictionary, optional
        Specify for each 'stratification_name' the coordinates to be used.
        These coordinates can then be used to acces data easily in the output xarray.
        Example: {'spatial_units': ['city_1','city_2','city_3']}
    """

    state_names = None
    parameter_names = None
    parameter_stratified_names = None
    stratification_names = None
    state_2d = None

    def __init__(self, states, parameters, coordinates=None, time_dependent_parameters=None):

        self.parameters = parameters
        self.coordinates = coordinates
        self.time_dependent_parameters = time_dependent_parameters

        # Merge parameter_names and parameter_stratified_names
        self.parameter_names_merged = merge_parameter_names_parameter_stratified_names(self.parameter_names, self.parameter_stratified_names)

        # Duplicates in lists containing names of states/parameters/stratified parameters/stratifications?
        check_duplicates(self.state_names, 'state_names')
        try:
            check_duplicates(self.parameter_names_merged, 'parameter_names + parameter_stratified_names')
        except:
            check_duplicates(self.parameter_names, 'parameter_names')
        if self.stratification_names:
            check_duplicates(self.stratification_names, 'stratification_names')
        if self.state_2d:
            check_duplicates(self.state_2d, 'state_2d')

        # Validate and compute the stratification sizes
        self.stratification_size = validate_stratifications(self.stratification_names, self.coordinates)

        # Validate the time-dependent parameter functions
        if time_dependent_parameters:
            self._function_parameters = validate_time_dependent_parameters(self.parameter_names, self.parameter_stratified_names, self.time_dependent_parameters)
        else:
            self._function_parameters = []

        # Validate the model
        self.initial_states, self.parameters, self._n_function_params, self._extra_params = validate_ODEModel(states, parameters, coordinates, self.stratification_size, self.state_names,
                                                                                                            self.parameter_names, self.parameter_stratified_names, self._function_parameters,
                                                                                                            self._create_fun, self.integrate, self.state_2d)

        # Experimental: added to use 2D states for the Economic IO model
        if self.state_2d:
            self.split_point = (len(self.state_names) - 1) * self.stratification_size[0]

    # Overwrite integrate class
    @staticmethod
    def integrate():
        """to overwrite in subclasses"""
        raise NotImplementedError

    def _create_fun(self, actual_start_date):
        """Convert integrate statement to scipy-compatible function"""
  
        def func(t, y, pars={}):
            """As used by scipy -> flattend in, flattend out"""

            # -------------------------------------------------------------
            # Flatten y and construct dictionary of states and their values
            # -------------------------------------------------------------

            if not self.state_2d:
                # for the moment assume sequence of parameters, vars,... is correct
                size_lst=[len(self.state_names)]
                for size in self.stratification_size:
                    size_lst.append(size)
                y_reshaped = y.reshape(tuple(size_lst))
                states = dict(zip(self.state_names, y_reshaped))
            else:
                # incoming y -> different reshape for 1D vs 2D variables  (2)
                y_1d, y_2d = np.split(y, [self.split_point])
                y_1d = y_1d.reshape(((len(self.state_names) - 1), self.stratification_size[0]))
                y_2d = y_2d.reshape((self.stratification_size[0], self.stratification_size[0]))
                states  = dict(zip(self.state_names, [y_1d,y_2d]))

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

            if not self.state_2d:
                dstates = self.integrate(t, **states, **params)
                return np.array(dstates).flatten()
            else:
                dstates = self.integrate(t, *y_1d, y_2d, **params)
                return np.concatenate([np.array(state).flatten() for state in dstates])

        return func

    def _sim_single(self, time, actual_start_date=None, method='RK23', rtol=5e-3, output_timestep=1):
        """"""
        fun = self._create_fun(actual_start_date)

        t0, t1 = time
        t_eval = np.arange(start=t0, stop=t1 + 1, step=output_timestep)

        if self.state_2d:
            for state in self.state_2d:
                self.initial_states[state] = self.initial_states[state].flatten()

        # Scipy can only take flattened list of states
        # TODO: rearrange y0 in order of self.state_names --> ALREADY DONE IN INITIALIZATION!
        y0 = list(itertools.chain(*self.initial_states.values()))
        while np.array(y0).ndim > 1:
            y0 = list(itertools.chain(*y0))

        output = solve_ivp(fun, time, y0, args=[self.parameters], t_eval=t_eval, method=method, rtol=rtol)

        # map to variable names
        return self._output_to_xarray_dataset(output, actual_start_date)

    def _mp_sim_single(self, drawn_parameters, time, actual_start_date, method, rtol, output_timestep):
        """
        A Multiprocessing-compatible wrapper for _sim_single, assigns the drawn dictionary and runs _sim_single
        """
        self.parameters.update(drawn_parameters)
        out = self._sim_single(time, actual_start_date, method, rtol, output_timestep)
        return out

    def sim(self, time, warmup=0, N=1, draw_function=None, samples=None, method='RK23', output_timestep=1, rtol=1e-3, processes=None):
        """
        Run a model simulation for the given time period. Can optionally perform `N` repeated simulations of time days.
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
            Number of repeated simulations (useful for stochastic models). One by default.

        draw_function : function
            A function which takes as its input the dictionary of model parameters and a samples dictionary
            and the dictionary of sampled parameter values and assings these samples to the model parameter dictionary ad random.

        samples : dictionary
            Sample dictionary used by draw_function. Does not need to be supplied if samples_dict is not used in draw_function.

        processes: int
            Number of cores to distribute the N draws over.

        method: str
            Method used by Scipy `solve_ivp` for integration of differential equations. Default: 'RK23'.
        
        output_timestep: int/flat
            Interpolate model output to every `output_timestep` time
            For datetimes: expressed in days

        rtol: float
            Relative tolerance of Scipy `solve_ivp`. Default: 1e-3. Quick and dirty: 5e-3.

        Returns
        -------
        xarray.Dataset

        """

        # Input checks on solver settings
        if not isinstance(rtol, float):
            raise TypeError(
                "Relative solver tolerance 'rtol' must be of type float"
            )
        if not isinstance(method, str):
            raise TypeError(
                "Solver method 'method' must be of type string"
            )

        # Input checks on supplied simulation time
        time, actual_start_date = validate_simulation_time(time, warmup)

        # Input check on draw function
        if draw_function:
            validate_draw_function(draw_function, self.parameters, samples)
           
        # Copy parameter dictionary --> dict is global
        cp = copy.deepcopy(self.parameters)
        
        # Parallel case: https://www.delftstack.com/howto/python/python-pool-map-multiple-arguments/#parallel-function-execution-with-multiple-arguments-using-the-pool.starmap-method
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
                output = p.map(partial(self._mp_sim_single, time=time, actual_start_date=actual_start_date, method=method, rtol=rtol, output_timestep=output_timestep), drawn_dictionaries)
        else:
            output=[]
            for dictionary in drawn_dictionaries:
                output.append(self._mp_sim_single(dictionary, time, actual_start_date, method=method, rtol=rtol, output_timestep=output_timestep))

        # Append results
        out = output[0]
        for xarr in output[1:]:
            out = xarray.concat([out, xarr], "draws")

        # Reset parameter dictionary
        self.parameters = cp

        return out

    def _output_to_xarray_dataset(self, output, actual_start_date=None):
        """
        Convert array (returned by scipy) to an xarray Dataset with the right coordinates and variable names
        """

        if self.coordinates:
            dims = list(self.coordinates.keys()).copy()
        else:
            dims = []
        
        if actual_start_date is not None:
            dims.append('date')
            coords = {"date": actual_start_date + pd.to_timedelta(output["t"], unit='D')}
        else:
            dims.append('time')
            coords = {"time": output["t"]}

        if self.coordinates:
            coords.update(self.coordinates)

        size_lst = [len(self.state_names)]
        if self.coordinates:
            for size in self.stratification_size:
                size_lst.append(size)
        size_lst.append(len(output["t"]))

        if not self.state_2d:
            y_reshaped = output["y"].reshape(tuple(size_lst))
            zip_star = zip(self.state_names, y_reshaped)
        else:
            # assuming only 1 2D variable!
            size_lst[0] = size_lst[0]-1
            y_1d, y_2d = np.split(output["y"], [self.split_point])
            y_1d_reshaped = y_1d.reshape(tuple(size_lst))
            y_2d_reshaped = y_2d.reshape(self.stratification_size[0], self.stratification_size[0],len(output["t"]))
            zip_star=zip(self.state_names[:-1],y_1d_reshaped)

        data = {}
        for var, arr in zip_star:
            xarr = xarray.DataArray(arr, coords=coords, dims=dims)
            data[var] = xarr
        
        if self.state_2d:
            if actual_start_date is not None:
                xarr = xarray.DataArray(y_2d_reshaped,coords=coords,dims=[list(self.coordinates.keys())[0],list(self.coordinates.keys())[0],'date'])
            else:
                xarr = xarray.DataArray(y_2d_reshaped,coords=coords,dims=[list(self.coordinates.keys())[0],list(self.coordinates.keys())[0],'time'])
            data[self.state_names[-1]] = xarr

        return xarray.Dataset(data)