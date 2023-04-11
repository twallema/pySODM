import inspect
import itertools
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import norm, weibull_min, triang, gamma
from scipy.special import gammaln
from pySODM.models.utils import list_to_dict
from pySODM.optimization.utils import _thetas_to_thetas_dict
from pySODM.models.validation import validate_initial_states

##############################
## Log-likelihood functions ##
##############################

def ll_gaussian(ymodel, ydata, sigma):
    """Loglikelihood of a Gaussian distribution (minus constant terms).
    
    Parameters
    ----------
    ymodel: np.ndarray
        mean values of the Gaussian distribution (i.e. "mu" values), as predicted by the model
    ydata: np.ndarray
        data time series values to be matched with the model predictions
    sigma: float/list of floats/np.ndarray
        standard deviation(s) of the Gaussian distribution around the mean value 'ymodel'
        Two options are possible: 1) one error per model dimension, applied uniformly to all datapoints corresponding to that dimension OR
        2) one error for every datapoint, corresponding to a weighted least-squares estimator

    Returns
    -------
    ll: float
        Loglikelihood belonging to the comparison of the data points and the model prediction for its particular parameter values
    """

    if not sigma.shape == ymodel.shape:
        # Expand first dimensions on 'alpha' to match the axes
        sigma = np.array(sigma)[np.newaxis, ...]   
    # Check for zeros (TODO: move to a higher layer)
    if len(sigma[sigma<=0]) != 0:
        raise ValueError(
            'argument `sigma` of `ll_gaussian` contains values smaller than or equal to zero'
        )
    return - 1/2 * np.sum((ydata - ymodel) ** 2 / sigma**2 + np.log(2*np.pi*sigma**2))

def ll_poisson(ymodel, ydata):
    """Loglikelihood of Poisson distribution
    
    Parameters
    ----------
    ymodel: np.ndarray
        mean values of the Poisson distribution (i.e. "lambda" values), as predicted by the model
    ydata: np.ndarray
        data time series values to be matched with the model predictions

    Returns
    -------
    ll: float
        Loglikelihood belonging to the comparison of the data points and the model prediction.
    """

    # Convert datatype to float
    ymodel = ymodel.astype('float64')
    ydata = ydata.astype('float64')
    # Raise ymodel or ydata if there are negative values present
    if ((np.min(ymodel) < 0) | (np.min(ydata) < 0)):
        offset_value = (-1 - 1e-6)*np.min([np.min(ymodel), np.min(ydata)])
        ymodel += offset_value
        ydata += offset_value
        #warnings.warn(f"One or more values in the prediction were negative thus the prediction was offset, minimum predicted value: {min(ymodel)}")
    elif ((np.min(ymodel) == 0) | (np.min(ydata) == 0)):
        ymodel += 1e-6
        ydata += 1e-6

    return - np.sum(ymodel) + np.sum(ydata*np.log(ymodel)) - np.sum(gammaln(ydata))

def ll_negative_binomial(ymodel, ydata, alpha):
    """Loglikelihood of negative binomial distribution

    https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Negative_Binomial_Regression.pdf
    https://content.wolfram.com/uploads/sites/19/2013/04/Zwilling.pdf
    https://www2.karlin.mff.cuni.cz/~pesta/NMFM404/NB.html
    https://www.jstor.org/stable/pdf/2532104.pdf

    Parameters
    ----------
    ymodel: np.ndarray
        mean values of the negative binomial distribution, as predicted by the model
    ydata: np.ndarray
        data time series values to be matched with the model predictions
    alpha: float/list
        Dispersion. The variance in the dataseries is equal to 1/dispersion and hence dispersion is bounded [0,1].
 
    Returns
    -------
    ll: float
        Loglikelihood belonging to the comparison of the data points and the model prediction.
    """

    # Expand first dimensions on 'alpha' to match the axes
    alpha = np.array(alpha)[np.newaxis, ...]
    # Convert datatype to float
    ymodel = ymodel.astype('float64')
    ydata = ydata.astype('float64')
    # Raise ymodel or ydata if there are negative values present
    if ((np.min(ymodel) < 0) | (np.min(ydata) < 0)):
        offset_value = (-1 - 1e-6)*np.min([np.min(ymodel), np.min(ydata)])
        ymodel += offset_value
        ydata += offset_value
        #warnings.warn(f"One or more values in the prediction were negative thus the prediction was offset, minimum predicted value: {min(ymodel)}")
    elif ((np.min(ymodel) == 0) | (np.min(ydata) == 0)):
        ymodel += 1e-6
        ydata += 1e-6

    return np.sum(ydata*np.log(ymodel)) - np.sum((ydata + 1/alpha)*np.log(1+alpha*ymodel)) + np.sum(ydata*np.log(alpha)) + np.sum(gammaln(ydata+1/alpha)) - np.sum(gammaln(ydata+1))- np.sum(ydata.shape[0]*gammaln(1/alpha))

#####################################
## Log prior probability functions ##
#####################################

def log_prior_uniform(x, bounds):
    """ Uniform log prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    bounds: tuple
        Tuple containg the upper and lower bounds of the parameter value.

    Returns
    -------
    Log probability of sample x in light of a uniform prior distribution.

    """
    prob = 1/(bounds[1]-bounds[0])
    condition = bounds[0] <= x <= bounds[1]
    if condition == True:
        # Can also be set to zero: value doesn't matter much because its constant
        return np.log(prob)
    else:
        return -np.inf

def log_prior_custom(x, args):
    """ Custom log prior distribution: computes the probability of a sample in light of a list containing samples from a previous MCMC run

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    args: tuple
        Tuple containg the density of each bin in the first position and the bounds of the bins in the second position.
        Contains a weight given to the custom prior in the third position of the tuple.

    Returns
    -------
    Log probability of sample x in light of a list with previously sampled parameter values.

    Example use:
    ------------
    # Posterior of 'my_par' in samples_dict['my_par']
    density_my_par, bins_my_par = np.histogram(samples_dict['my_par'], bins=20, density=True)
    density_my_par_norm = density_my_par/np.sum(density_my_par)
    prior_fcn = prior_custom
    prior_fcn_args = (density_my_par_norm, bins_my_par, weight)
    # Prior_fcn and prior_fcn_args must then be passed on to the function log_probability
    """
    density, bins, weight = args
    if x <= bins.min() or x >= bins.max():
        return -np.inf
    else:
        idx = np.digitize(x, bins)
        return weight*np.log(density[idx-1])

def log_prior_normal(x,norm_params):
    """ Normal log prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    norm_params: tuple
        Tuple containg mu and stdev.

    Returns
    -------
    Log probability of sample x in light of a normal prior distribution.

    """
    mu,stdev=norm_params
    return np.sum(norm.logpdf(x, loc = mu, scale = stdev))

def log_prior_triangle(x,triangle_params):
    """ Triangle log prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    triangle_params: tuple
        Tuple containg lower bound, upper bound and mode of the triangle distribution.

    Returns
    -------
    Log probability of sample x in light of a triangle prior distribution.

    """
    low,high,mode = triangle_params
    return triang.logpdf(x, loc=low, scale=high, c=mode)

def log_prior_gamma(x,gamma_params):
    """ Gamma log prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    gamma_params: tuple
        Tuple containg gamma parameters alpha and beta.

    Returns
    -------
    Log probability of sample x in light of a gamma prior distribution.

    """
    a,b = gamma_params
    return gamma.logpdf(x, a=a, scale=1/b)

def log_prior_weibull(x,weibull_params):
    """ Weibull log prior distribution

    Parameters
    ----------
    x: float
        Parameter value whos probability we want to test.
    weibull_params: tuple
        Tuple containg weibull parameters k and lambda.

    Returns
    -------
    Log probability of sample x in light of a weibull prior distribution.

    """
    k,lam = weibull_params
    return gamma.logpdf(x, k, shape=lam, loc=0 )    

#############################################
## Computing the log posterior probability ##
#############################################

class log_posterior_probability():
    """ Computation of log posterior probability

    A generic implementation to compute the log posterior probability of a model given some data, computed as the sum of the log prior probabilities and the log likelihoods.
    # TODO: fully document docstring
    """
    def __init__(self, model, parameter_names, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args,
                 weights=None, log_prior_prob_fnc=None, log_prior_prob_fnc_args=None, initial_states=None, aggregation_function=None, labels=None):

        ############################################################################################################
        ## Validate lengths of data, states, log_likelihood_fnc, log_likelihood_fnc_args, weights, initial states ##
        ############################################################################################################

        # Check type of `weights`
        if isinstance(weights, (list,np.ndarray)):
            if isinstance(weights, np.ndarray):
                if weights.ndim > 1:
                    raise TypeError("Expected a 1D np.array for input argument `weights`")
        else:
            if weights:
                raise TypeError("Expected a list/1D np.array for input argument `weights`")
            
        if weights is None:
            if any(len(lst) != len(data) for lst in [states, log_likelihood_fnc, log_likelihood_fnc_args]):
                raise ValueError(
                    "The number of datasets ({0}), model states ({1}), log likelihood functions ({2}), and the extra arguments of the log likelihood function ({3}) must be of equal".format(len(data),len(states), len(log_likelihood_fnc), len(log_likelihood_fnc_args))
                    )
            else:
                weights = len(data)*[1,]

        if not initial_states:
            if any(len(lst) != len(data) for lst in [states, log_likelihood_fnc, weights, log_likelihood_fnc_args]):
                raise ValueError(
                    "The number of datasets ({0}), model states ({1}), log likelihood functions ({2}), the extra arguments of the log likelihood function ({3}), and weights ({4}) must be of equal".format(len(data),len(states), len(log_likelihood_fnc), len(log_likelihood_fnc_args), len(weights))
                    )
        else:
            # Validate initial states
            for i, initial_states_dict in enumerate(initial_states):
                initial_states[i] = validate_initial_states(model.state_shapes, initial_states_dict)
            # Check 
            if any(len(lst) != len(data) for lst in [states, log_likelihood_fnc, weights, log_likelihood_fnc_args, initial_states]):
                raise ValueError(
                    "The number of datasets ({0}), model states ({1}), log likelihood functions ({2}), the extra arguments of the log likelihood function ({3}), weights ({4}) and initial states ({5}) must be of equal".format(len(data),len(states), len(log_likelihood_fnc), len(log_likelihood_fnc_args), len(weights), len(initial_states))
                    )
        self.initial_states = initial_states

        ###########################
        ## Validate the datasets ##
        ###########################

        # Get additional axes beside time axis in dataset.
        data, self.time_index, self.additional_axes_data = validate_dataset(data)
        # Extract start- and enddate of simulations
        self.start_sim = min([df.index.get_level_values(self.time_index).unique().min() for df in data])
        self.end_sim = max([df.index.get_level_values(self.time_index).unique().max() for df in data])

        ########################################
        ## Validate the calibrated parameters ##
        ########################################

        parameter_sizes, self.parameter_shapes = validate_calibrated_parameters(parameter_names, model.parameters)

        ################################################
        ## Construct expanded bounds, pars and labels ##
        ################################################

        # Expand parameter names 
        self.parameter_names_postprocessing = expand_parameter_names(self.parameter_shapes)
        # Expand bounds
        if len(bounds) == len(parameter_names):
            check_bounds(bounds)
            self.expanded_bounds = expand_bounds(parameter_sizes, bounds)
        elif len(bounds) == sum(parameter_sizes.values()):
            check_bounds(bounds)
            self.expanded_bounds = bounds
        else:
            raise Exception(
                f"The number of provided bounds ({len(bounds)}) must either:\n\t1) equal the number of calibrated parameters '{parameter_names}' ({len(parameter_names)}) or,\n\t2) equal the element-expanded number of calibrated parameters '{self.parameter_names_postprocessing}'  ({len(self.parameter_names_postprocessing)})"
            )
        # Expand labels
        if labels:
            if len(labels) == len(parameter_names):
                self.expanded_labels = expand_labels(self.parameter_shapes, labels)
            elif len(labels) == sum(parameter_sizes.values()):
                self.expanded_labels = labels
            else:
                raise Exception(
                    f"The number of provided labels ({len(labels)}) must either:\n\t1) equal the number of calibrated parameters '{parameter_names}' ({len(parameter_names)}) or,\n\t2) equal the element-expanded number of calibrated parameters '{self.parameter_names_postprocessing}'  ({len(self.parameter_names_postprocessing)})"
                )
        else:
            self.expanded_labels = self.parameter_names_postprocessing

        ####################################################################
        ## Input check on number of prior functions + potential expansion ##
        ####################################################################

        self.log_prior_prob_fnc, self.log_prior_prob_fnc_args = validate_expand_log_prior_prob(log_prior_prob_fnc, log_prior_prob_fnc_args, parameter_sizes, self.expanded_bounds)

        ######################################################
        ## Check number of aggregation functions and expand ##
        ######################################################

        if aggregation_function:
            aggregation_function = validate_aggregation_function(aggregation_function, len(data))
        self.aggregation_function = aggregation_function

        ############################################
        ## Compare data and model dimensions ##
        ############################################

        out = create_fake_xarray_output(model.state_dimensions, model.state_coordinates, model.initial_states, self.time_index)
        self.coordinates_data_also_in_model, self.aggregate_over = compare_data_model_coordinates(out, data, states, aggregation_function, self.additional_axes_data)

        ########################################
        ## Input checks on log_likelihood_fnc ##
        ########################################

        self.n_log_likelihood_extra_args = validate_log_likelihood_funtion(log_likelihood_fnc)
        self.log_likelihood_fnc_args = validate_log_likelihood_function_extra_args(data, self.n_log_likelihood_extra_args, self.additional_axes_data, self.coordinates_data_also_in_model,
                                                                                self.time_index, log_likelihood_fnc_args, log_likelihood_fnc)

        # Assign attributes to class
        self.model = model
        self.data = data
        self.states = states
        self.parameter_names = parameter_names
        self.log_likelihood_fnc = log_likelihood_fnc
        self.weights = weights

    @staticmethod
    def compute_log_prior_probability(thetas, log_prior_prob_fnc, log_prior_prob_fnc_args):
        """
        Loops over the log_prior_probability functions and their respective arguments to compute the prior probability of every model parameter in theta.
        """
        lp=[]
        for idx,fnc in enumerate(log_prior_prob_fnc):
            theta = thetas[idx]
            args = log_prior_prob_fnc_args[idx]
            lp.append(fnc(theta,args))
        return sum(lp)

    def compute_log_likelihood(self, out, states, df, weights, log_likelihood_fnc, log_likelihood_fnc_args,
                                time_index, n_log_likelihood_extra_args, aggregate_over, additional_axes_data, coordinates_data_also_in_model,
                                aggregation_function):
        """
        Matches the model output of the desired states to the datasets provided by the user and then computes the log likelihood using the user-specified function.
        """

        total_ll=0

        # Apply aggregation function
        if aggregation_function:
            out_copy = aggregation_function(out[states])
        else:
            out_copy = out[states]
        # Reduce dimensions on the model prediction
        for dimension in out.dims:
            if dimension in aggregate_over:
                out_copy = out_copy.sum(dim=dimension)
        # Interpolate to right times/dates
        interp = out_copy.interp({time_index: df.index.get_level_values(time_index).unique().values}, method="linear")
        # Select right axes
        if not additional_axes_data:
            # Only dates must be matched
            ymodel = np.expand_dims(interp.sel({time_index: df.index.get_level_values(time_index).unique().values}).values, axis=1)
            ydata = np.expand_dims(df.squeeze().values,axis=1)
            # Check if shapes are consistent
            if ymodel.shape != ydata.shape:
                raise Exception(f"Shapes of model prediction {ymodel.shape} and data {ydata.shape} do not correspond; np.arrays 'ymodel' and 'ydata' must be of the same size")
            if n_log_likelihood_extra_args == 0:
                total_ll += weights*log_likelihood_fnc(ymodel, ydata)
            else:
                total_ll += weights*log_likelihood_fnc(ymodel, ydata, *[log_likelihood_fnc_args,])
        else:
            # First reorder model output 
            dims = [time_index,] + additional_axes_data
            interp = interp.transpose(*dims)
            ymodel = interp.sel({k:coordinates_data_also_in_model[jdx] for jdx,k in enumerate(additional_axes_data)}).values
            # Automatically reorder the dataframe so that time/date is first index (stack only works for 2 indices sadly --> pandas to xarray --> reorder --> to numpy)
            df = df.to_xarray()
            df = df.transpose(*dims)
            ydata = df.to_numpy()
            # Check if shapes are consistent
            if ymodel.shape != ydata.shape:
                raise Exception(f"Shapes of model prediction {ymodel.shape} and data {ydata.shape} do not correspond; np.arrays 'ymodel' and 'ydata' must be of the same size")
            if n_log_likelihood_extra_args == 0:
                total_ll += weights*log_likelihood_fnc(ymodel, ydata)
            else:
                total_ll += weights*log_likelihood_fnc(ymodel, ydata, *[log_likelihood_fnc_args,])
    
        return total_ll

    def __call__(self, thetas, simulation_kwargs={}):
        """
        This function manages the internal bookkeeping (assignment of model parameters, model simulation) and then computes and sums the log prior probabilities and log likelihoods to compute the log posterior probability.
        """

        # Compute log prior probability 
        lp = self.compute_log_prior_probability(thetas, self.log_prior_prob_fnc, self.log_prior_prob_fnc_args)

        # Restrict thetas to user-provided bounds
        for i,theta in enumerate(thetas):
            if theta > self.expanded_bounds[i][1]:
                thetas[i] = self.expanded_bounds[i][1]
            elif theta < self.expanded_bounds[i][0]:
                thetas[i] = self.expanded_bounds[i][0]

        # Unflatten thetas
        thetas_dict = list_to_dict(thetas, self.parameter_shapes)

        # Assign and remove warmup
        if 'warmup' in thetas_dict.keys():
            simulation_kwargs.update({'warmup': float(thetas_dict['warmup'])})
            del thetas_dict['warmup']

        # Assign model parameters
        self.model.parameters.update(thetas_dict)

        if not self.initial_states:
            # Perform simulation only once
            out = self.model.sim([self.start_sim,self.end_sim], **simulation_kwargs)
            # Loop over dataframes
            for idx,df in enumerate(self.data):
                # Get aggregation function
                if self.aggregation_function:
                    aggfunc = self.aggregation_function[idx]
                else:
                    aggfunc = None
                # Compute log likelihood
                lp += self.compute_log_likelihood(out, self.states[idx], df, self.weights[idx], self.log_likelihood_fnc[idx], self.log_likelihood_fnc_args[idx], 
                                                  self.time_index, self.n_log_likelihood_extra_args[idx], self.aggregate_over[idx], self.additional_axes_data[idx],
                                                  self.coordinates_data_also_in_model[idx], aggfunc)
        else:
            # Loop over dataframes
            for idx,df in enumerate(self.data):
                # Replace initial condition
                self.model.initial_states.update(self.initial_states[idx])
                # Perform simulation
                out = self.model.sim([self.start_sim,self.end_sim], **simulation_kwargs)
                # Get aggregation function
                if self.aggregation_function:
                    aggfunc = self.aggregation_function[idx]
                else:
                    aggfunc = None
                # Compute log likelihood
                lp += self.compute_log_likelihood(out, self.states[idx], df, self.weights[idx], self.log_likelihood_fnc[idx], self.log_likelihood_fnc_args[idx], 
                                                  self.time_index, self.n_log_likelihood_extra_args[idx], self.aggregate_over[idx], self.additional_axes_data[idx],
                                                  self.coordinates_data_also_in_model[idx], aggfunc)
        return lp

#############################################
## Validation of log posterior probability ##
#############################################

def validate_dataset(data):
    """
    Validates a dataset:
        - Correct datatype?
        - No Nan's?
        - Is the index level 'date'/'time present? (obligated)
    Extracts and returns the additional dimensions in dataset besides the time axis.

    Parameters
    ----------

    data: list
        List containing the datasets (pd.Series, pd.Dataframe, xarray.DataArray, xarray.Dataset)
    
    Returns
    -------
    data: list
        List containing the datasets. xarray.DataArray have been converted to pd.DataFrame

    additional_axes_data: list
        Contains the index levels beside 'date'/'time' present in the dataset

    time_index: str
        'date': datetime-like time index. 'time': float-like time index.
    """

    additional_axes_data=[] 
    time_index=[]
    for idx, df in enumerate(data):
        # Is dataset either a pd.Series, pd.Dataframe or xarray.Dataset?
        if not isinstance(df, (pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset)):
            raise TypeError(
                f"{idx}th dataset is of type {type(df)}. expected pd.Series, pd.DataFrame or xarray.DataArray, xarray.Dataset"
            )
        # If it is an xarray dataset, convert it to a pd.Dataframe
        if isinstance(df, (xr.DataArray, xr.Dataset)):
            df = df.to_dataframe()
            data[idx] = df
        # If it is a pd.DataFrame, does it have one column?
        if isinstance(df, pd.DataFrame):
            if len(df.columns) != 1:
                raise ValueError(
                    f"{idx}th dataset is a pd.DataFrame with {len(df.columns)} columns. expected one column."
                )
            else:
                # Convert to a series for ease
                data[idx] = df.squeeze()
        # Does data contain NaN values anywhere?
        if np.isnan(df).values.any():
            raise ValueError(
                f"{idx}th dataset contains nans"
                )
        # Does data have 'date' or 'time' as index level? (required)
        if (('date' not in df.index.names) & ('time' not in df.index.names)):
            raise ValueError(
                f"Index of {idx}th dataset does not have 'date' or 'time' as index level (index levels: {df.index.names})."
                )
        elif (('date' in df.index.names) & ('time' not in df.index.names)):
            time_index.append('date')
            additional_axes_data.append([name for name in df.index.names if name != 'date'])
        elif (('date' not in df.index.names) & ('time' in df.index.names)):
            time_index.append('time')
            additional_axes_data.append([name for name in df.index.names if name != 'time'])
        else:
            raise ValueError(
                f"Index of {idx}th dataset has both 'date' and 'time' as index level (index levels: {df.index.names})."
                )
    # Are all time index levels equal to 'date' or 'time'?
    if all(index == 'date' for index in time_index):
        time_index='date'
    elif all(index == 'time' for index in time_index):
        time_index='time'
    else:
        raise ValueError(
            "Some datasets have 'time' as time index, other have 'date as time index. pySODM does not allow mixing."
        )
    return data, time_index, additional_axes_data


def validate_calibrated_parameters(parameters_function, parameters_model):
    """
    Validates the parameters the user wants to calibrate. Parameters must: 1) Be valid model parameters, 2) Have one value, or, have multiple values (list or np.ndarray).
    Construct a dictionary containing the parameter names as key and their sizes (type: tuple) as values.
    An exception is coded for 'warmup' (= number of days between data and simulation start)
   
    Parameters
    ----------

    parameters_function: list
        Contains the names parameters the user wants to calibrate (type: str)

    parameters_model: dict
        Model parameters dictionary.

    Returns
    -------

    parameters_sizes: dict
        Dictionary containing the size (=number of entries) of every model parameter.

    parameters_shapes: dict
        Dictionary containing the shape of every model parameter.    
    """

    parameters_sizes = []
    parameters_shapes = []
    for param_name in parameters_function:
        # Check if the parameter exists
        if ((param_name not in parameters_model)&(param_name!='warmup')):
            raise Exception(
                f"To be calibrated model parameter '{param_name}' is not a valid model parameter!"
            )
        elif param_name == 'warmup':
            parameters_shapes.append((1,))
            parameters_sizes.append(1)
        else:
            # Check the datatype: only int, float, list of int/float, np.array
            if isinstance(parameters_model[param_name], bool):
                raise TypeError(
                        f"pySODM supports the calibration of model parameters of type int, float, list (containing int/float) or 1D np.ndarray. Model parameter '{param_name}' is of type '{type(model.parameters[param_name])}'"
                    )
            elif isinstance(parameters_model[param_name], (int,float)):
                parameters_shapes.append((1,))
                parameters_sizes.append(1)
            elif isinstance(parameters_model[param_name], np.ndarray):
                parameters_shapes.append(parameters_model[param_name].shape)
                parameters_sizes.append(parameters_model[param_name].size)
            elif isinstance(parameters_model[param_name], list):
                if not all([isinstance(item, (int,float)) for item in parameters_model[param_name]]):
                    raise TypeError(
                        f"To be calibrated model parameter '{param_name}' of type list must only contain int or float!"
                    )
                else:
                    parameters_shapes.append(np.array(parameters_model[param_name]).shape)
                    parameters_shapes.append(np.array(parameters_model[param_name]).size)
            else:
                raise TypeError(
                        f"pySODM supports the calibration of model parameters of type int, float, list (containing int/float) or 1D np.ndarray. Model parameter '{param_name}' is of type '{type(model.parameters[param_name])}'"
                        )

    return dict(zip(parameters_function, parameters_sizes)), dict(zip(parameters_function, parameters_shapes))

def expand_parameter_names(parameter_shapes):
    """A function to expand the names of parameters with multiple entries
    """
    expanded_names = []
    for i, name in enumerate(parameter_shapes.keys()):
        if parameter_shapes[name] == (1,):
            expanded_names.append(name)
        else:
            for index in itertools.product(*[list(range(i)) for i in itertools.chain(parameter_shapes[name])]):
                n = name + '_{'
                for val in index:
                    n += str(val)+','
                n=n[:-1]
                n+='}'
                expanded_names.append(n)
    return expanded_names

def check_bounds(bounds):
    """
    A function to check the elements in 'bounds': type(list/tuple), length (2), ub > lb
    """
    for i,bound in enumerate(bounds):
        # Check type
        assert isinstance(bound, (list,tuple)), f'parameter bound (position {i}) is not a list or tuple'
        # Check length
        assert len(bound)==2, f'parameter bound (position {i}) contains {len(bound)} entries instead of 2'
        # Check ub > lb
        assert bound[0] < bound[1], f'upper-bound value must be greater than lower-bound value (position {i})'

def expand_bounds(parameter_sizes, bounds):
    """"
    A function to expand the bounds of parameters with multiple elements
    """
    expanded_bounds = []
    for i, name in enumerate(parameter_sizes.keys()):
        if parameter_sizes[name] == 1:
            expanded_bounds.append(bounds[i])
        else:
            for _ in range(parameter_sizes[name]):
                expanded_bounds.append(bounds[i])
    return expanded_bounds

def expand_labels(parameter_shapes, labels):
    """A function to expand the labels of parameters with multiple entries
    """
    expanded_labels = []
    for i, name in enumerate(parameter_shapes.keys()):
        if parameter_shapes[name] == (1,):
            expanded_labels.append(labels[i])
        else:
            for index in itertools.product(*[list(range(i)) for i in itertools.chain(parameter_shapes[name])]):
                n = labels[i] + '_{'
                for val in index:
                    n += str(val)+','
                n=n[:-1]
                n+='}'
                expanded_labels.append(n)
    return expanded_labels

def validate_expand_log_prior_prob(log_prior_prob_fnc, log_prior_prob_fnc_args, parameter_sizes, expanded_bounds):
    """ 
    Validation of the log prior probability function and its arguments.
    Expansion for parameters with multiple entries.
    """

    if ((not log_prior_prob_fnc) & (not log_prior_prob_fnc_args)):
        # Setup uniform priors
        expanded_log_prior_prob_fnc = len(expanded_bounds)*[log_prior_uniform,]
        expanded_log_prior_prob_fnc_args = expanded_bounds
    elif ((log_prior_prob_fnc != None) & (not log_prior_prob_fnc_args)):
        raise Exception(
            f"Invalid input. Prior probability functions provided but no prior probability function arguments."
        )
    elif ((not log_prior_prob_fnc) & (log_prior_prob_fnc_args != None)):
        raise Exception(
            f"Invalid input. Prior probability function arguments provided but no prior probability functions."
        )
    else:
        if any(len(lst) != len(log_prior_prob_fnc) for lst in [log_prior_prob_fnc_args]):
            raise ValueError(
                f"The number of prior functions ({len(log_prior_prob_fnc)}) and the number of sets of prior function arguments ({len(log_prior_prob_fnc_args)}) must be of equal length"
            ) 
        if ((len(log_prior_prob_fnc) != len(parameter_sizes.keys()))&(len(log_prior_prob_fnc) != len(expanded_bounds))):
            raise Exception(
                f"The number of provided log prior probability functions ({len(log_prior_prob_fnc)}) must either:\n\t1) equal the number of calibrated parameters ({len(parameter_sizes.keys())}) or,\n\t2) equal the element-expanded number of calibrated parameters  ({sum(parameter_sizes.values())})"
                )
        if len(log_prior_prob_fnc) != len(expanded_bounds):
            # Expand
            expanded_log_prior_prob_fnc = []
            expanded_log_prior_prob_fnc_args = []
            for i, name in enumerate(parameter_sizes.keys()):
                if parameter_sizes[name] == 1:
                    expanded_log_prior_prob_fnc.append(log_prior_prob_fnc[i])
                    expanded_log_prior_prob_fnc_args.append(log_prior_prob_fnc_args[i])
                else:
                    for _ in range(parameter_sizes[name]):
                        expanded_log_prior_prob_fnc.append(log_prior_prob_fnc[i])
                        expanded_log_prior_prob_fnc_args.append(log_prior_prob_fnc_args[i])

    return expanded_log_prior_prob_fnc, expanded_log_prior_prob_fnc_args

def get_coordinates_data_also_in_model(data_index_diff, i, model_coordinates, data):
    """ A function to retrieve, for every dataset, and for every model dimension, the coordinates present in the data and also in the model.
    
    Parameters
    ----------

    additional_axes_data: list
        Contains the names of the model dimensions besides 'time'/'date' present in the data

    model_coordinates: dict
        Model output coordinates. Keys: dimensions, values: coordinates.

    data: list
        Containing pd.Series or pd.Dataframes of data.

    Returns
    -------

    coordinates_data_also_in_model: list
        Contains a list for every dataset. Contains a list for every model dimension besides 'date'/'time', containing the coordinates present in the data and also in the model.

    """

    tmp1=[]
    for data_dim in data_index_diff:
        tmp2=[]
        # Model has no dimensions: data can only contain 'date' or 'time'
        if not model_coordinates:
            raise Exception(
                f"Your model has no dimensions. Remove all coordinates except 'time' or 'date' ({data_index_diff}) from dataset {i} by slicing or grouping."
            )
        # Verify the axes in additional_axes_data are valid model dimensions
        if data_dim not in list(model_coordinates.keys()):
            raise Exception(
                f"{i}th dataset coordinate '{data_dim}' is not a model dimension. Remove the coordinate '{data_dim}' from dataset {i} by slicing or grouping."
            )
        else:
            # Verify all coordinates in the dataset can be found in the model
            coords_model = model_coordinates[data_dim]
            coords_data = list(data[i].index.get_level_values(data_dim).unique().values)
            for coord in coords_data:
                if coord not in coords_model:
                    raise Exception(
                        f"coordinate '{coord}' of dimension '{data_dim}' in the {i}th dataset was not found in the model coordinates of dimension '{data_dim}': {coords_model}"
                        )
                else:
                    tmp2.append(coord)
        tmp1.append(tmp2)
    return tmp1

def get_dimensions_sum_over(additional_axes_data, model_coordinates):
    """ A function to compute the model dimensions that are not present in the dataset.

    Parameters
    ----------

    additional_axes_data: list
        The axes present in the dataset, excluding the time dimensions 'time'/'date'
    
    model_coordinates: dict
        Dictionary of model coordinates. Key: dimension name. Value: corresponding coordinates.

    Returns
    -------
    dimensions_sum_over: list
        Contains a list per provided dataset. Contains the model dimensions not present in the dataset. pySODM will automatically sum over these dimensions.
    """

    tmp=[]
    if model_coordinates:
        for model_strat in list(model_coordinates.keys()):
            if model_strat not in additional_axes_data:
                tmp.append(model_strat)
        return tmp
    else:
        return []

def validate_aggregation_function(aggregation_function, n_datasets):
    """ A function to validate the number of aggregation functions provided by the user
        Valid options are: a function, a list containing one function (both applied to all datasets) or one function per dataset.
    
    Parameters
    ----------
    aggregation_function: callable function or list
        An aggregation functions receives as input an xarray.DataArray, obtained by extracting the model output at the state the user wishes to calibrate.
        e.g. input_to_aggregation_function = simulation_output[state_name]
        The aggregation function performs some operation on this data (f.i. aggregating over certain age groups or spatial levels) and returns an xarray.DataArray.
        NO INPUT CHECKS ARE PERFORMED ON THE AGGREGATION FUNCTIONS THEMSELVES!
    
    n_datasets: int
        Number of datasets

    Returns
    -------
    aggregation_function: list
        An expanded list of aggregation functions containing one aggregation function per dataset.

    """
    if isinstance(aggregation_function, list):
        if ((not len(aggregation_function) == n_datasets) & (not len(aggregation_function) == 1)):
            raise ValueError(
                f"number of aggregation functions must be equal to one or the number of datasets"
            )
        if len(aggregation_function) == 1:
            for i in range(n_datasets-1):
                aggregation_function.append(aggregation_function[0])
    elif inspect.isfunction(aggregation_function):
        aggfunc=[]
        for i in range(n_datasets):
            aggfunc.append(aggregation_function)
        aggregation_function = aggfunc
    else:
        raise ValueError(
            f"Valid formats of aggregation functions are: 1) a list containing one function, 2) a list containing a number of functions equal to the number of datasets, 3) a callable function."
        )
    return aggregation_function

def create_fake_xarray_output(state_dimensions, state_coordinates, initial_states, time_index):
    """ 
    A function to "replicate" the xarray.Dataset output of a simulation
    Made to omit the need to call the sim function in the initialization of the log_posterior_probability

    Parameters
    ----------
    state_dimensions: dict
        Keys: model states. Values: List containing dimensions associated with state.

    state_coordinates: dict
        Keys: model states. Values: List containing coordinates associated with each dimension associated with state.

    initial_states: dict
        Dictionary of initial model states
    
    time_index: str
        'date' or 'time'
        Could be ommitted if we wanted.

    Returns
    -------
    out: xarray.Dataset
        Model "output" 
    """

    # Append the time dimension
    new_state_dimensions={}
    for k,v in state_dimensions.items():
        v_acc = v.copy()
        v_acc = [time_index,] + v_acc
        new_state_dimensions.update({k: v_acc})

    # Append the time coordinates
    new_state_coordinates={}
    for k,v in state_coordinates.items():
        v_acc=v.copy()
        if time_index == 'time':
            v_acc = [[0,],] + v_acc
        elif time_index == 'date':
            v_acc = [[pd.Timestamp('2000-01-01'),],] + v_acc
        new_state_coordinates.update({k: v_acc})

    # Build the xarray dataset
    data = {}
    for var, arr in initial_states.items():
        if arr.ndim >= 1:
            if state_dimensions[var]:
                arr = arr[np.newaxis, ...]
        xarr = xr.DataArray(arr, dims=new_state_dimensions[var], coords=new_state_coordinates[var])
        data[var] = xarr

    return xr.Dataset(data)


def compare_data_model_coordinates(output, data, calibration_state_names, aggregation_function, additional_axes_data):
    """
    A function to check if data and model dimensions/coordinates can be aligned correctly (subject to possible aggregation functions introduced by the user).

    Parameters
    ----------
    output: xarray.Dataset
        Simulation output. Generated using `create_fake_xarray_output()`
    
    data: list
        List containing the datasets

    calibration_state_names: list
        Contains the names (type: str) of the model states to match with each dataset in `data`

    aggregation_function: list
        List of length len(data). Contains: None (no aggregation) or aggregation functions.
    
    additional_axes_data: list
        Axes in dataset, excluding the 'time'/'date' axes.

    Returns
    -------
    coordinates_data_also_in_model: list
        List of length len(data). Contains, per dataset, the coordinates in the data that are also coordinates of the model.

    aggregate_over: list
        List of length len(data). Contains, per dataset, the remaining model dimensions not present in the dataset. These are then automatically summed over while calculating the log likelihood.
    """

    # Validate
    coordinates_data_also_in_model=[]
    aggregate_over=[]
    # Loop over states/datasets we'd like to match
    for i, (state_name, df) in enumerate(zip(calibration_state_names, data)):
        # Call the aggregation function
        if aggregation_function:
            new_output = aggregation_function[i](output[state_name])
        else:
            new_output = output[state_name]
        # Create a dictionary containing, for every dimensions that is not 'time'/'date'
        # key: list of dimensions, value: list of corresponding coordinates
        dimensions = list(new_output.dims)
        dimensions = [d for d in dimensions if ((d != 'time')&(d!='date'))]
        coordinates=[]
        for dim in dimensions:
            coordinates.append(new_output.coords[dim].values)
        dims_coords = dict(zip(dimensions, coordinates))

        # Compute the coordinates present in both data and model (we'll have to match these)
        coordinates_data_also_in_model.append(get_coordinates_data_also_in_model(additional_axes_data[i], i, dims_coords, data))
        
        # Construct a list containing (per dataset) the axes we need to sum the model output over prior to matching the data
        aggregate_over.append(get_dimensions_sum_over(additional_axes_data[i], dims_coords))

    return coordinates_data_also_in_model, aggregate_over

def validate_log_likelihood_funtion(log_likelihood_function):
    """
    A function to validate the log likelihood function's arguments and return the number of extra arguments of the log likelihood function.

    Parameters
    ----------
    log_likelihood_function: callable function
        The log likelihood function. F.i. Gaussian, Poisson, etc.

    Returns
    -------
    n_log_likelihood_extra_args: int
        Number of "extra arguments" of the log likelihood function
    """

    # Check that log_likelihood_fnc always has ymodel as the first argument and ydata as the second argument
    # Find out how many additional arguments are needed for the log_likelihood_fnc (f.i. sigma for gaussian model, alpha for negative binomial)
    n_log_likelihood_extra_args=[]
    for idx,fnc in enumerate(log_likelihood_function):
        sig = inspect.signature(fnc)
        keywords = list(sig.parameters.keys())
        if keywords[0] != 'ymodel':
            raise ValueError(
            "The first parameter of log_likelihood function in position {0} is not equal to 'ymodel' but {1}".format(idx, keywords[0])
        )
        if keywords[1] != 'ydata':
            raise ValueError(
            "The second parameter of log_likelihood function in position {0} is not equal to 'ydata' but {1}".format(idx, keywords[1])
        )
        extra_args = len([arg for arg in keywords if ((arg != 'ymodel')&(arg != 'ydata'))])
        n_log_likelihood_extra_args.append(extra_args)

    # Support for more than one extra argument of the log likelihood function is not available
    for i in range(len(n_log_likelihood_extra_args)):
        if n_log_likelihood_extra_args[i] > 1:
            raise ValueError(
                "Support for log likelihood functions with more than one additional argument is not implemented. Raised for log likelihood function {0}".format(log_likelihood_function[i])
                )
    return n_log_likelihood_extra_args


def validate_log_likelihood_function_extra_args(data, n_log_likelihood_extra_args, additional_axes_data, coordinates_data_also_in_model, time_index, log_likelihood_fnc_args,
                                                log_likelihood_fnc):

    # Input checks on the additional arguments of the log likelihood functions
    for idx, df in enumerate(data):
        if n_log_likelihood_extra_args[idx] == 0:
            if ((isinstance(log_likelihood_fnc_args[idx], int)) | (isinstance(log_likelihood_fnc_args[idx], float)) | (isinstance(log_likelihood_fnc_args[idx], np.ndarray))):
                raise ValueError(
                    "the likelihood function {0} used for the {1}th dataset has no extra arguments. Expected an empty list as argument. You have provided an {2}.".format(log_likelihood_fnc[idx], idx, type(log_likelihood_fnc_args[idx]))
                    )
            elif log_likelihood_fnc_args[idx]:
                raise ValueError(
                    "the likelihood function {0} used for the {1}th dataset has no extra arguments. Expected an empty list as argument. You have provided a non-empty list.".format(log_likelihood_fnc[idx], idx)
                    )
        else:
            if not additional_axes_data[idx]:
                # ll_poisson, ll_gaussian, ll_negative_binomial take int/float, but ll_gaussian can also take an error for every datapoint (= weighted least-squares)
                # Thus, its additional argument must be a np.array of the same dimensions as the data
                if not isinstance(log_likelihood_fnc_args[idx], (int,float,np.ndarray,pd.Series)):
                    raise ValueError(
                        f"arguments of the {idx}th dataset log likelihood function '{log_likelihood_fnc[idx]}' cannot be of type {type(log_likelihood_fnc_args[idx])}."
                        "accepted types are int, float, np.ndarray and pd.Series"
                    )
                else:
                    if isinstance(log_likelihood_fnc_args[idx], np.ndarray):
                        if log_likelihood_fnc_args[idx].shape != df.values.shape:
                            raise ValueError(
                                f"the shape of the np.ndarray with the arguments of the log likelihood function '{log_likelihood_fnc[idx]}' for the {idx}th dataset ({log_likelihood_fnc_args[idx].shape}) don't match the number of datapoints ({df.values.shape})"
                            )
                        else:
                            log_likelihood_fnc_args[idx] = np.expand_dims(log_likelihood_fnc_args[idx], axis=1)
                    if isinstance(log_likelihood_fnc_args[idx], pd.Series):
                        if not log_likelihood_fnc_args[idx].index.equals(df.index):
                            raise ValueError(
                                f"index of pd.Series containing arguments of the {idx}th log likelihood function must match the index of the {idx}th dataset"
                            )
                        else:
                            log_likelihood_fnc_args[idx] = np.expand_dims(log_likelihood_fnc_args[idx].values,axis=1)

            elif len(additional_axes_data[idx]) == 1:
                if not isinstance(log_likelihood_fnc_args[idx],(list,np.ndarray,pd.Series)):
                    raise TypeError(
                            f"arguments of the {idx}th dataset log likelihood function '{log_likelihood_fnc[idx]}' cannot be of type {type(log_likelihood_fnc_args[idx])}."
                            "accepted types are list, np.ndarray or pd.Series "
                    )
                else:
                    # list
                    if isinstance(log_likelihood_fnc_args[idx], list):
                        if not len(df.index.get_level_values(additional_axes_data[idx][0]).unique()) == len(log_likelihood_fnc_args[idx]):
                            raise ValueError(
                                f"length of list/1D np.array containing arguments of the log likelihood function '{log_likelihood_fnc[idx]}' ({len(log_likelihood_fnc_args[idx])}) must equal the length of the dimension axes '{additional_axes_data[idx][0]}' ({len(df.index.get_level_values(additional_axes_data[idx][0]).unique())}) in the {idx}th dataset."
                                )
                    # np.ndarray
                    if isinstance(log_likelihood_fnc_args[idx], np.ndarray):
                        if log_likelihood_fnc_args[idx].ndim != 1:
                            raise ValueError(
                                f"np.ndarray containing arguments of the log likelihood function '{log_likelihood_fnc[idx]}' for dataset {idx} must be 1 dimensional"
                            )
                        elif not len(df.index.get_level_values(additional_axes_data[idx][0]).unique()) == len(log_likelihood_fnc_args[idx]):
                            raise ValueError(
                                f"length of list/1D np.array containing arguments of the log likelihood function '{log_likelihood_fnc[idx]}' must equal the length of the dimension axes '{additional_axes_data[idx][0]}' ({len(df.index.get_level_values(additional_axes_data[idx][0]).unique())}) in the {idx}th dataset."
                            )
                    # pd.Series
                    if isinstance(log_likelihood_fnc_args[idx], pd.Series):
                        if not log_likelihood_fnc_args[idx].index.equals(df.index):
                            raise ValueError(
                                f"index of pd.Series containing arguments of the {idx}th log likelihood function must match the index of the {idx}th dataset"
                            )
                        else:
                            # Make sure time index is in first position
                            val = log_likelihood_fnc_args[idx].to_xarray()
                            dims = [time_index,]+additional_axes_data[idx]
                            val = val.transpose(*dims)
                            log_likelihood_fnc_args[idx] = val.to_numpy()         
            else:
                # Compute desired shape in case of one parameter per stratfication
                desired_shape=[]
                for lst in coordinates_data_also_in_model[idx]:
                    desired_shape.append(len(lst))
                # Input checks
                if not isinstance(log_likelihood_fnc_args[idx], (np.ndarray, pd.Series)):
                    raise TypeError(
                        f"arguments of the {idx}th dataset log likelihood function '{log_likelihood_fnc[idx]}' cannot be of type {type(log_likelihood_fnc_args[idx])}."
                        "accepted types are np.ndarray and pd.Series"
                    )
                else:
                    if isinstance(log_likelihood_fnc_args[idx], np.ndarray):
                        shape = list(log_likelihood_fnc_args[idx].shape)
                        if shape != desired_shape:
                            raise ValueError(
                                f"Shape of np.array containing arguments of the log likelihood function '{log_likelihood_fnc[idx]}' for dataset {idx} must equal {desired_shape}. You provided {shape}"
                            )
                    if isinstance(log_likelihood_fnc_args[idx], pd.Series):
                        if not log_likelihood_fnc_args[idx].index.equals(df.index):
                            raise ValueError(
                                f"index of pd.Series containing arguments of the {idx}th log likelihood function must match the index of the {idx}th dataset"
                            )
                        else:
                            # Make sure time index is in first position
                            val = log_likelihood_fnc_args[idx].to_xarray()
                            dims = [time_index,]+additional_axes_data[idx]
                            val = val.transpose(*dims)
                            log_likelihood_fnc_args[idx] = val.to_numpy()
    
    return log_likelihood_fnc_args
