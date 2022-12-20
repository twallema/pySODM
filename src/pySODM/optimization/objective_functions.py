import inspect
import pandas as pd
import numpy as np
from scipy.stats import norm, weibull_min, triang, gamma
from scipy.special import gammaln
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
        Two options are possible: 1) one error per model stratification, applied uniformly to all datapoints corresponding to that stratification OR
        2) one error for every datapoint, corresponding to a weighted least-squares estimator

    Returns
    -------
    ll: float
        Loglikelihood belonging to the comparison of the data points and the model prediction for its particular parameter values
    """

    if not sigma.shape == ymodel.shape:
        # Expand first dimensions on 'alpha' to match the axes
        sigma = np.array(sigma)[np.newaxis, ...]   

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
    condition = bounds[0] < x < bounds[1]
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
                 weights=None, log_prior_prob_fnc=None, log_prior_prob_fnc_args=None, initial_states=None, labels=None):

        #######################
        ## Construct weights ##
        #######################

        if not weights:
            if any(len(lst) != len(data) for lst in [states, log_likelihood_fnc, log_likelihood_fnc_args]):
                raise ValueError(
                    "The number of datasets ({0}), model states ({1}), log likelihood functions ({2}), and the extra arguments of the log likelihood function ({3}) must be of equal".format(len(data),len(states), len(log_likelihood_fnc), len(log_likelihood_fnc_args))
                    )
            else:
                weights = len(data)*[1,]

        ##################################
        ## Construct initial conditions ##
        ##################################
        #####################################################################################################
        ## Check provided number of datasets, states, weights, log_likelihood functions and initial states ##
        #####################################################################################################
        
        if not initial_states:
            if any(len(lst) != len(data) for lst in [states, log_likelihood_fnc, weights, log_likelihood_fnc_args]):
                raise ValueError(
                    "The number of datasets ({0}), model states ({1}), log likelihood functions ({2}), the extra arguments of the log likelihood function ({3}), and weights ({4}) must be of equal".format(len(data),len(states), len(log_likelihood_fnc), len(log_likelihood_fnc_args), len(weights))
                    )
        else:
            # Validate intial conditions provided
            for i,initial_states_dict in enumerate(initial_states):
                for state in model.state_names:
                    if state in initial_states_dict:
                        # if present, check that the length is correct
                        initial_states_dict[state] = validate_initial_states(initial_states_dict[state], state, "initial state", model.stratification_size, model.coordinates, None)
                    else:
                        # otherwise add default of 0
                        initial_states_dict[state] = np.zeros(model.stratification_size)

                # check if the states match with model states
                if set(initial_states_dict.keys()) != set(model.state_names):
                    raise ValueError(
                        "The initial states in position {0} don't exactly match the model's predefined states. "
                        "Redundant states: {1}".format(
                        i,set(initial_states_dict.keys()).difference(set(model.state_names)))
                    )

                # sort the initial states to match the state_names
                initial_states_dict = {state: initial_states_dict[state] for state in model.state_names}

                # Assign 
                initial_states[i] = initial_states_dict

            if any(len(lst) != len(data) for lst in [states, log_likelihood_fnc, weights, log_likelihood_fnc_args, initial_states]):
                raise ValueError(
                    "The number of datasets ({0}), model states ({1}), log likelihood functions ({2}), the extra arguments of the log likelihood function ({3}), weights ({4}) and initial states ({5}) must be of equal".format(len(data),len(states), len(log_likelihood_fnc), len(log_likelihood_fnc_args), len(weights), len(initial_states))
                    )

        self.initial_states = initial_states

        ####################
        ## Checks on data ##
        ####################

        self.additional_axes_data=[] 
        for idx, df in enumerate(data):
            # Is dataset either a pd.Series or a pd.Dataframe?
            if not isinstance(df, (pd.Series,pd.DataFrame)):
                raise TypeError(
                    f"{idx}th dataset is of type {type(df)}. expected pd.Series or pd.DataFrame"
                )
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
                self.time_index = 'date'
                self.additional_axes_data.append([name for name in df.index.names if name != 'date'])
            elif (('date' not in df.index.names) & ('time' in df.index.names)):
                self.time_index = 'time'
                self.additional_axes_data.append([name for name in df.index.names if name != 'time'])
            else:
                raise ValueError(
                    f"Index of {idx}th dataset has both 'date' and 'time' as index level (index levels: {df.index.names})."
                    )

        # Extract start- and enddate of simulations
        index_min=[]
        index_max=[]
        for idx, df in enumerate(data):
            index_min.append(df.index.get_level_values(self.time_index).unique().min())
            index_max.append(df.index.get_level_values(self.time_index).unique().max())
        self.start_sim = min(index_min)
        self.end_sim = max(index_max)

        ########################################
        ## Checks on parameters to calibrated ##
        ########################################

        idx_par_with_multiple_entries=[]
        for i,param_name in enumerate(parameter_names):
            # Check if the parameter exists
            if param_name not in model.parameters:
                raise Exception(
                    f"'{param_name}' is not a valid model parameter!"
                )
            # Check the datatype: only int, float, list of int/float, 1D np.array
            if isinstance(model.parameters[param_name], bool):
                raise TypeError(
                        f"pySODM supports the calibration of model parameters of type int, float, list (containing int/float) or 1D np.ndarray. Model parameter '{param_name}' is of type '{type(model.parameters[param_name])}'"
                    )
            elif ((not isinstance(model.parameters[param_name], int)) & (not isinstance(model.parameters[param_name], float))):
                if isinstance(model.parameters[param_name], np.ndarray):
                    if model.parameters[param_name].ndim != 1:
                        raise NotImplementedError(
                            f"Only 1D numpy arrays can be calibrated using pySDOM. Parameter '{param_name}' is a {model.parameters[param_name].ndim}-dimensional np.ndarray!"
                        )
                    else:
                        idx_par_with_multiple_entries.append(i)
                elif isinstance(model.parameters[param_name], list):
                    if not all([isinstance(item, (int,float)) for item in model.parameters[param_name]]):
                        raise TypeError(
                            f"Model parameter '{param_name}' of type list must only contain int or float!"
                        )
                    else:
                        idx_par_with_multiple_entries.append(i)
                else:
                    raise TypeError(
                        f"pySODM supports the calibration of model parameters of type int, float, list (containing int/float) or 1D np.ndarray. Model parameter '{param_name}' is of type '{type(model.parameters[param_name])}'"
                    )

        #######################################
        ## Construct bounds, pars and labels ##
        #######################################

        # Compute expanded size of bounds
        expanded_length=0
        for i, param_name in enumerate(parameter_names):
            if i in idx_par_with_multiple_entries:
                expanded_length += len(model.parameters[param_name])
            else:
                expanded_length += 1

        # Construct parameters_postprocessing
        parameter_names_postprocessing = []
        for i, param_name in enumerate(parameter_names):
            if i in idx_par_with_multiple_entries:
                l = len(model.parameters[param_name])
                for j in range(l):
                    parameter_names_postprocessing.append(param_name+'_'+str(j))
            else:
                parameter_names_postprocessing.append(param_name)

        if len(bounds) == len(parameter_names):
            # Expand bounds
            new_bounds = []
            for i, param_name in enumerate(parameter_names):
                if i in idx_par_with_multiple_entries:
                    l = len(model.parameters[param_name])
                    for j in range(l):
                        new_bounds.append(bounds[i])
                else:
                    new_bounds.append(bounds[i])
        elif len(bounds) == expanded_length:
            new_bounds=bounds
        else:
            raise Exception(
                f"The number of provided bounds ({len(bounds)}) must either:\n\t1) equal the number of calibrated parameters '{parameter_names}' ({len(parameter_names)}) or,\n\t2) equal the element-expanded number of calibrated parameters '{parameter_names_postprocessing}'  ({len(parameter_names_postprocessing)})"
            )

        # Expand labels if provided
        if labels:
            if len(labels) == len(parameter_names):
                new_labels=[]
                for i, param_name in enumerate(parameter_names):
                    if i in idx_par_with_multiple_entries:
                        l = len(model.parameters[param_name])
                        for j in range(l):
                            new_labels.append(labels[i]+'_'+str(j))
                    else:
                        new_labels.append(labels[i])
                labels=new_labels
            elif len(labels) == expanded_length:
                pass
            else:
                raise Exception(
                    f"The number of provided labels ({len(labels)}) must either:\n\t1) equal the number of calibrated parameters '{parameter_names}' ({len(parameter_names)}) or,\n\t2) equal the element-expanded number of calibrated parameters '{parameter_names_postprocessing}'  ({len(parameter_names_postprocessing)})"
                )
        else:
            labels = parameter_names_postprocessing
        
        # Assign to objective function
        self.expanded_bounds=new_bounds
        self.expanded_labels=labels
        self.parameter_names_postprocessing=parameter_names_postprocessing

        ####################################################################
        ## Input check on number of prior functions + potential expansion ##
        ####################################################################

        if ((not log_prior_prob_fnc) & (not log_prior_prob_fnc_args)):
            # Setup uniform priors
            log_prior_prob_fnc = len(self.expanded_bounds)*[log_prior_uniform,]
            log_prior_prob_fnc_args = self.expanded_bounds
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
            if ((len(log_prior_prob_fnc) != len(parameter_names))&(len(log_prior_prob_fnc) != len(parameter_names_postprocessing))):
                raise Exception(
                    f"The number of provided log prior probability functions ({len(log_prior_prob_fnc)}) must either:\n\t1) equal the number of calibrated parameters '{parameter_names}' ({len(parameter_names)}) or,\n\t2) equal the element-expanded number of calibrated parameters '{parameter_names_postprocessing}'  ({len(parameter_names_postprocessing)})"
                    )
            if len(log_prior_prob_fnc) != len(self.expanded_bounds):
                # Expand
                new_log_prior_prob_fnc = []
                new_log_prior_prob_fnc_args = []
                for i, param_name in enumerate(parameter_names):
                    if i in idx_par_with_multiple_entries:
                        l = len(model.parameters[param_name])
                        for j in range(l):
                            new_log_prior_prob_fnc.append(log_prior_prob_fnc[i])
                            new_log_prior_prob_fnc_args.append(log_prior_prob_fnc_args[i])
                    else:
                        new_log_prior_prob_fnc.append(log_prior_prob_fnc[i])
                        new_log_prior_prob_fnc_args.append(log_prior_prob_fnc_args[i])
                log_prior_prob_fnc = new_log_prior_prob_fnc
                log_prior_prob_fnc_args = new_log_prior_prob_fnc_args
            else:
                pass

        # Assign to class
        self.log_prior_prob_fnc = log_prior_prob_fnc
        self.log_prior_prob_fnc_args = log_prior_prob_fnc_args

        ############################################
        ## Compare data and model stratifications ##
        ############################################

        self.coordinates_data_also_in_model=[]
        for i, data_index_diff in enumerate(self.additional_axes_data):
            tmp1=[]
            for data_dim in data_index_diff:
                tmp2=[]
                # Model has no stratifications: data can only contain 'date' or 'time'
                if not model.coordinates:
                    raise Exception(
                        f"Your model has no stratifications. Remove all coordinates except 'time' or 'date' ({data_index_diff}) from dataset {i} by slicing or grouping."
                    )
                # Verify the axes in additional_axes_data are valid model dimensions
                if data_dim not in list(model.coordinates.keys()):
                    raise Exception(
                        f"{i}th dataset coordinate '{data_dim}' is not a model stratification. Remove the coordinate '{data_dim}' from dataset {i} by slicing or grouping."
                    )
                else:
                    # Verify all coordinates in the dataset can be found in the model
                    coords_model = model.coordinates[data_dim]
                    coords_data = list(data[i].index.get_level_values(data_dim).unique().values)
                    for coord in coords_data:
                        if coord not in coords_model:
                            raise Exception(
                                f"coordinate '{coord}' of stratification '{data_dim}' in the {i}th dataset was not found in the model coordinates of stratification '{data_dim}': {coords_model}"
                             )
                        else:
                            tmp2.append(coord)
                tmp1.append(tmp2)
            self.coordinates_data_also_in_model.append(tmp1)

        # Construct a list containing (per dataset) the axes we need to sum the model output over prior to matching the data
        # Is the difference between the data axes and model axes (excluding time/date)
        self.aggregate_over=[]
        for i, data_index_diff in enumerate(self.additional_axes_data):
            tmp=[]
            if model.coordinates:
                for model_strat in list(model.coordinates.keys()):
                    if model_strat not in data_index_diff:
                        tmp.append(model_strat)
                self.aggregate_over.append(tmp)
            else:
                self.aggregate_over.append(tmp)

        ########################################
        ## Input checks on log_likelihood_fnc ##
        ########################################

        # Check that log_likelihood_fnc always has ymodel as the first argument and ydata as the second argument
        # Find out how many additional arguments are needed for the log_likelihood_fnc (f.i. sigma for gaussian model, alpha for negative binomial)
        n_log_likelihood_extra_args=[]
        for idx,fnc in enumerate(log_likelihood_fnc):
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
        self.n_log_likelihood_extra_args = n_log_likelihood_extra_args

        # Support for more than one extra argument of the log likelihood function is not available
        for i in range(len(n_log_likelihood_extra_args)):
            if n_log_likelihood_extra_args[i] > 1:
                raise ValueError(
                    "Support for log likelihood functions with more than one additional argument is not implemented. Raised for log likelihood function {0}".format(log_likelihood_fnc[i])
                    )

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
                if not self.additional_axes_data[idx]:
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

                elif len(self.additional_axes_data[idx]) == 1:
                    if not isinstance(log_likelihood_fnc_args[idx],(list,np.ndarray,pd.Series)):
                        raise TypeError(
                             f"arguments of the {idx}th dataset log likelihood function '{log_likelihood_fnc[idx]}' cannot be of type {type(log_likelihood_fnc_args[idx])}."
                             "accepted types are list, np.ndarray or pd.Series "
                        )
                    else:
                        # list
                        if isinstance(log_likelihood_fnc_args[idx], list):
                            if not len(df.index.get_level_values(self.additional_axes_data[idx][0]).unique()) == len(log_likelihood_fnc_args[idx]):
                                raise ValueError(
                                    f"length of list/1D np.array containing arguments of the log likelihood function '{log_likelihood_fnc[idx]}' ({len(log_likelihood_fnc_args[idx])}) must equal the length of the stratification axes '{self.additional_axes_data[idx][0]}' ({len(df.index.get_level_values(self.additional_axes_data[idx][0]).unique())}) in the {idx}th dataset."
                                    )
                        # np.ndarray
                        if isinstance(log_likelihood_fnc_args[idx], np.ndarray):
                            if log_likelihood_fnc_args[idx].ndim != 1:
                                raise ValueError(
                                    f"np.ndarray containing arguments of the log likelihood function '{log_likelihood_fnc[idx]}' for dataset {idx} must be 1 dimensional"
                                )
                            elif not len(df.index.get_level_values(self.additional_axes_data[idx][0]).unique()) == len(log_likelihood_fnc_args[idx]):
                                raise ValueError(
                                    f"length of list/1D np.array containing arguments of the log likelihood function '{log_likelihood_fnc[idx]}' must equal the length of the stratification axes '{self.additional_axes_data[idx][0]}' ({len(df.index.get_level_values(self.additional_axes_data[idx][0]).unique())}) in the {idx}th dataset."
                                )
                        # pd.Series
                        if isinstance(log_likelihood_fnc_args[idx], pd.Series):
                            if not log_likelihood_fnc_args[idx].index.equals(df.index):
                                raise ValueError(
                                    f"index of pd.Series containing arguments of the {idx}th log likelihood function must match the index of the {idx}th dataset"
                                )
                            else:
                                # Make sure time index is in first position
                                log_likelihood_fnc_args[idx] = self.series_to_ndarray(log_likelihood_fnc_args[idx].reorder_levels([self.time_index,]+self.additional_axes_data[idx]))                 
                else:
                    # Compute desired shape in case of one parameter per stratfication
                    desired_shape=[]
                    for lst in self.coordinates_data_also_in_model[idx]:
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
                                log_likelihood_fnc_args[idx] = self.series_to_ndarray(log_likelihood_fnc_args[idx].reorder_levels([self.time_index,]+self.additional_axes_data[idx]))

        # Find out if 'warmup' needs to be estimated
        self.warmup_position=None
        if 'warmup' in parameter_names:
            self.warmup_position=parameter_names.index('warmup')

        # Assign attributes to class
        self.model = model
        self.data = data
        self.states = states
        self.parameter_names = parameter_names
        self.log_likelihood_fnc = log_likelihood_fnc
        self.log_likelihood_fnc_args = log_likelihood_fnc_args
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

    @staticmethod
    def series_to_ndarray(df):
            shape = [len(df.index.get_level_values(i).unique().values) for i in range(df.index.nlevels)]
            return df.to_numpy().reshape(shape)

    def compute_log_likelihood(self, out, states, df, weights, log_likelihood_fnc, log_likelihood_fnc_args, time_index, n_log_likelihood_extra_args, aggregate_over, additional_axes_data, coordinates_data_also_in_model):
        """
        Matches the model output of the desired states to the datasets provided by the user and then computes the log likelihood using the user-specified function.
        """

        total_ll=0

        # Reduce dimensions on the model prediction
        out_copy = out[states]
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
            # Automatically reorder the dataframe so that time/date is first index
            ydata = self.series_to_ndarray(df.reorder_levels([time_index,]+additional_axes_data))
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

        # Add exception for estimation of warmup
        if self.warmup_position:
            simulation_kwargs.update({'warmup': thetas[self.warmup_position]})
            # Convert thetas for model parameters to a dictionary with key-value pairs
            thetas_dict, n = _thetas_to_thetas_dict([x for (i,x) in enumerate(thetas) if i != self.warmup_position], [x for x in self.parameter_names if x != "warmup"], self.model.parameters)
        else:
            # Convert thetas for model parameters to a dictionary with key-value pairs
            thetas_dict, n = _thetas_to_thetas_dict(thetas, self.parameter_names, self.model.parameters)

        # Assign thetas for model parameters to the model object
        for param,value in thetas_dict.items():
            self.model.parameters.update({param : value})

        # Compute log prior probability 
        lp = self.compute_log_prior_probability(thetas, self.log_prior_prob_fnc, self.log_prior_prob_fnc_args)

        if not self.initial_states:
            # Perform simulation only once
            out = self.model.sim([self.start_sim,self.end_sim], **simulation_kwargs)
            # Loop over dataframes
            for idx,df in enumerate(self.data):
                # Compute log likelihood
                lp += self.compute_log_likelihood(out, self.states[idx], df, self.weights[idx], self.log_likelihood_fnc[idx], self.log_likelihood_fnc_args[idx], 
                                                  self.time_index, self.n_log_likelihood_extra_args[idx], self.aggregate_over[idx], self.additional_axes_data[idx],
                                                  self.coordinates_data_also_in_model[idx])
        else:
            # Loop over dataframes
            for idx,df in enumerate(self.data):
                # Replace initial condition
                self.model.initial_states.update(self.initial_states[idx])
                # Perform simulation
                out = self.model.sim([self.start_sim,self.end_sim], **simulation_kwargs)
                # Compute log likelihood
                lp += self.compute_log_likelihood(out, self.states[idx], df, self.weights[idx], self.log_likelihood_fnc[idx], self.log_likelihood_fnc_args[idx], 
                                                  self.time_index, self.n_log_likelihood_extra_args[idx], self.aggregate_over[idx], self.additional_axes_data[idx],
                                                  self.coordinates_data_also_in_model[idx])
        return lp