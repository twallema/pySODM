import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Tuple, Union, Dict, Any
from pySODM.models.utils import list_to_dict
from pySODM.optimization.objective_functions import validate_calibrated_parameters


def add_poisson_noise(output: xr.Dataset) -> xr.Dataset:
    """A function to add poisson noise to a simulation result

    Parameters
    ----------

    output: xarray.Dataset
        Simulation output

    Returns
    -------

    output: xarray.Dataset
        Simulation output, but every value was replaced with a poisson estimate.
    """
    new_output = output.copy()
    # Loop over variables in xarray
    for varname, da in new_output.data_vars.items():
        # Replace very value with a poison draw
        values = np.random.poisson(da.values)
        new_output[varname].values = values
    return new_output


def add_gaussian_noise(output: xr.Dataset, sigma: float, relative: bool=True) -> xr.Dataset:
    """A function to add absolute gaussian noise to a simulation result

    Parameters
    ----------

    output: xarray.Dataset
        Simulation output

    sigma: float
        Standard deviation. Must be larger than or equal to 0. Equal for all datapoints.

    relative: bool
        Add noise relative to magnitude of simulation output.

    Returns
    -------

    output: xarray.Dataset
        Simulation output, but every value was replaced with a gaussian estimate.

    """
    new_output = output.copy()
    # Loop over variables in xarray
    for varname, da in new_output.data_vars.items():
        # Replace very value with a normal draw
        if relative == False:
            values = np.random.normal(da.values, sigma)
        elif relative == True:
            values = np.random.normal(da.values, np.abs(sigma*da.values))
        new_output[varname].values = values
    return new_output


def add_negative_binomial_noise(output: xr.Dataset, alpha: float) -> xr.Dataset:
    """A function to add negative binomial noise to a simulation result

    Parameters
    ----------

    output: xarray.Dataset
        Simulation output

    alpha: float
        Overdispersion factor. Must be larger than or equal to 0. Reduces to Poisson noise for alpha --> 0.

    Returns
    -------

    output: xarray.Dataset
        Simulation output, but every value was replaced with a negative binomial estimate.
    """

    new_output = output.copy()
    # Loop over variables in xarray
    for varname, da in new_output.data_vars.items():
        # Replace very value with a negative_binomial draw
        values = np.random.negative_binomial(
            1/alpha, (1/alpha)/(da.values + (1/alpha)))
        new_output[varname].values = values
    return new_output


def assign_theta(param_dict: Dict[str, Any], parameter_names: List[str], thetas: Union[List[float], np.ndarray]) -> Dict[str, Any]:
    """ A generic function to assign the output of a PSO/Nelder-Mead calibration to the model parameters dictionary

    Parameters
    ----------
    param_dict : dict
        Model parameters dictionary

    parameter_names : list (of strings)
        Names of model parameters estimated using PSO

    thetas : list (of floats)
        Result of a PSO or Nelder-Mead calibration, results must correspond to the order of the parameter names list (pars)

    Returns
    -------

    param_dict : dict
        Model parameters dictionary with values of parameters 'pars' set to the obtained PSO estimate in vector 'theta'

    """

    _, parameter_shapes = validate_calibrated_parameters(parameter_names, param_dict)
    thetas_dict = list_to_dict(np.array(thetas), parameter_shapes, retain_floats=True)
    for _, (param, value) in enumerate(thetas_dict.items()):
        param_dict.update({param: value})
    return param_dict


def variance_analysis(data: pd.Series, window_length: str, half_life: float) -> Tuple[pd.DataFrame, plt.Axes]:

    """ A function to analyze the relationship between the variance and the mean in a timeseries of data
        ================================================================================================

        1. An exponential moving average (EMA) of the data is computed.
        2. Data and EMA are binned in windows of length `window_length`.
        3. The mean of the EMA is computed in each bin.
        4. The variance of the data in relation to the mean of the EMA is computed in each bin.
        5. Several statistical models are then fitted to the relationship between the mean and variance. The statistical models are: gaussian (var = c), poisson (var = mu), quasi-poisson (var = theta*mu), negative binomial (var = mu + alpha*mu**2).

        To assure quality of the output, user must,

        1. Make sure the EMA of the data follows the data sufficiently closely.
        2. Make sure each window contains multiple datapoints (3 is a minimum).
        3. Assess how their results change when `window_length` and `halflife` are adjusted.

        Please consult section 3.1 of the Supplementary Materials in Alleman et al. 2023 Applied Mathematical Modeling for a rigorous description of the procedure.

        Parameters
        ----------

            series: pd.Series
                Timeseries of data to be analyzed. The series must have a pd.Timestamp index labeled 'date' for the time dimension.
                Additionally, this function supports the addition of one more dimension (f.i. space) using a multiindex.
                This function is not intended to study the variance of datasets containing multiple datapoints on the same date. 

            window_length: str
                Length of each window, given as a valid pd.Timedelta frequency.
                Valid options are: 'W': weekly, 'M': monthly, etc. Or multiples thereof: '2W': biweekly, etc.

            halflife: float
                Halflife of the exponential moving average.

        Output
        ------

            result: pd.Dataframe
                Contains the estimated parameter(s) and the Akaike Information Criterion (AIC) of the fitted statistical model.
                If two index levels are present (thus 'date' and 'other index level'), the result pd.Dataframe contains the result stratified per 'other index level'.

            ax: axes object
                Contains a plot of the estimated mean versus variance, togheter with the fitted statistical models. The best-fitting model is less transparent than the other models.
       """

    #################
    ## Bookkeeping ##
    #################

    # Input checks
    if 'date' not in data.index.names:
        raise ValueError(
            "Indexname 'date' not found. Make sure the time dimension index is named 'date'. Current index dimensions: {0}".format(
                data.index.names)
        )
    if len(data.index.names) > 2:
        raise ValueError(
            "The maximum number of index dimensions is two and must always include a time dimension named 'date'. Valid options are thus: 'date', or ['date', 'something else']. Current index dimensions: {0}".format(
                data.index.names)
        )
    # Relevant parameters
    if len(data.index.names) == 1:
        secundary_index = False
        secundary_index_name = None
        secundary_index_values = None
    else:
        secundary_index = True
        secundary_index_name = data.index.names[data.index.names != 'date']
        secundary_index_values = data.index.get_level_values(
            data.index.names[data.index.names != 'date'])

    ###########################################
    ## Define variance models and properties ##
    ###########################################

    def gaussian(mu, var): return var*np.ones(len(mu))
    def poisson(mu, dummy): return mu
    def quasi_poisson(mu, theta): return mu*theta
    def negative_binomial(mu, alpha): return mu + alpha*mu**2
    models = [gaussian, poisson, quasi_poisson, negative_binomial]
    n_model_pars = [1, 0, 1, 1]
    model_names = ['gaussian', 'poisson', 'quasi-poisson', 'negative binomial']

    ##########################################################
    ## Define error function for parameter estimation (SSE) ##
    ##########################################################

    def error(model_par, model, mu_data, var_data): return sum(
        (model(mu_data, model_par) - var_data)**2)

    #################################
    ## Approximate mu, var couples ##
    #################################

    # needed to generate data to calibrate our variance model to
    if not secundary_index:
        rolling_mean = data.ewm(halflife=half_life, adjust=False).mean()
        mu_data = (data.groupby(
            [pd.Grouper(freq=window_length, level='date')]).mean())
        var_data = (((data-rolling_mean) **
                    2).groupby([pd.Grouper(freq=window_length, level='date')]).mean())
    else:
        rolling_mean = data.groupby(level=secundary_index_name, group_keys=False).apply(
            lambda x: x.ewm(halflife=half_life, adjust=False).mean())
        mu_data = (data.groupby([pd.Grouper(
            freq=window_length, level='date')] + [secundary_index_values]).mean())
        var_data = (((data-rolling_mean)**2).groupby([pd.Grouper(
            freq=window_length, level='date')] + [secundary_index_values]).mean())

    # Protect calibration against nan values
    merge = pd.merge(mu_data, var_data, right_index=True,
                     left_index=True).dropna()
    mu_data = merge.iloc[:, 0]
    var_data = merge.iloc[:, 1]

    ###################################
    ## Preallocate results dataframe ##
    ###################################

    if not secundary_index:
        results = pd.DataFrame(index=model_names, columns=[
                               'theta', 'AIC'], dtype=np.float64)
    else:
        iterables = [data.index.get_level_values(
            secundary_index_name).unique(), model_names]
        index = pd.MultiIndex.from_product(
            iterables, names=[secundary_index_name, 'model'])
        results = pd.DataFrame(index=index, columns=[
                               'theta', 'AIC'], dtype=np.float64)

    ########################
    ## Perform estimation ##
    ########################

    if not secundary_index:
        for i, model in enumerate(models):
            opt = minimize(error, 0, args=(
                model, mu_data.values, var_data.values))
            results.loc[model_names[i], 'theta'] = opt['x'][0]
            n = len(mu_data.values)
            results.loc[model_names[i], 'AIC'] = n * \
                np.log(opt['fun']/n) + 2*n_model_pars[i]
    else:
        for index in secundary_index_values.unique():
            for i, model in enumerate(models):
                opt = minimize(error, 0, args=(model, mu_data.loc[slice(
                    None), index].values, var_data.loc[slice(None), index].values))
                results.loc[(index, model_names[i]), 'theta'] = opt['x'][0]
                n = len(mu_data.loc[slice(None), index].values)
                results.loc[(index, model_names[i]), 'AIC'] = n * \
                    np.log(opt['fun']/n) + 2*n_model_pars[i]

    ##########################
    ## Make diagnostic plot ##
    ##########################
    from itertools import compress
    linestyles = ['-', '-.', ':', '--']

    if not secundary_index:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(mu_data, var_data, color='black', alpha=0.5,
                   linestyle='None', facecolors='none', s=60, linewidth=2)
        mu_model = np.linspace(start=0, stop=max(mu_data))
        # Find model with lowest AIC
        best_model = list(
            compress(model_names, results['AIC'].values == min(results['AIC'].values)))[0]
        for idx, model in enumerate(models):
            if model_names[idx] == best_model:
                ax.plot(mu_model, model(
                    mu_model, results.loc[model_names[idx], 'theta']), linestyles[idx], color='black', linewidth='2')
            else:
                ax.plot(mu_model, model(
                    mu_model, results.loc[model_names[idx], 'theta']), linestyles[idx], color='black', linewidth='2', alpha=0.2)
            model_names[idx] += ' (AIC: {:.0f})'.format(
                results.loc[model_names[idx], 'AIC'])
        ax.grid(False)
        ax.set_ylabel('Estimated variance')
        ax.set_xlabel('Estimated mean')
        ax.legend(['data', ]+model_names, bbox_to_anchor=(0.05, 1),
                  loc='upper left', fontsize=12)

    else:
        # Compute figure size
        ncols = 3
        nrows = int(np.ceil(len(secundary_index_values.unique())/ncols))
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12))
        i = 0
        j = 0
        for k, index in enumerate(secundary_index_values.unique()):
            # Determine right plot index
            if ((k % ncols == 0) & (k != 0)):
                j = 0
                i += 1
            elif k != 0:
                j += 1
            # Plot data
            ax[i, j].scatter(mu_data.loc[slice(None), index].values, var_data.loc[slice(
                None), index].values, color='black', alpha=0.5, facecolors='none', linestyle='None', s=60, linewidth=2)
            # Find best model
            best_model = list(compress(model_names, results.loc[(index, slice(
                None)), 'AIC'].values == min(results.loc[(index, slice(None)), 'AIC'].values)))[0]
            # Plot models
            mu_model = np.linspace(start=0, stop=max(
                mu_data.loc[slice(None), index].values))
            for l, model in enumerate(models):
                if model_names[l] == best_model:
                    ax[i, j].plot(mu_model, model(mu_model, results.loc[(
                        index, model_names[l]), 'theta']), linestyles[l], color='black', linewidth='2')
                else:
                    ax[i, j].plot(mu_model, model(mu_model, results.loc[(
                        index, model_names[l]), 'theta']), linestyles[l], color='black', linewidth='2', alpha=0.2)
            # Format axes
            ax[i, j].grid(False)
            # Add xlabels and ylabels
            if j == 0:
                ax[i, j].set_ylabel('Estimated variance')
            if i == nrows-1:
                ax[i, j].set_xlabel('Estimated mean')
            # Add a legend
            title = secundary_index_name + ': ' + str(index)
            ax[i, j].legend(['data', ]+model_names, bbox_to_anchor=(0.05, 1),
                            loc='upper left', fontsize=7, title=title, title_fontsize=8)

    return results, ax
