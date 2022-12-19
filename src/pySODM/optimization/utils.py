from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_poisson_noise(output):
    """A function to add poisson noise to a simulation result

    Parameters
    ----------

    output: xarray
        Simulation output

    Returns
    -------

    output: xarray
        Simulation output, but every value was replaced with a poisson estimate.
    """

    # Loop over variables in xarray
    for varname, da in output.data_vars.items():
        # Replace very value with a poison draw
        values = np.random.poisson(da.values)
        output[varname].values = values
    return output

def add_gaussian_noise(output, sigma, relative=True):
    """A function to add absolute gaussian noise to a simulation result

    Parameters
    ----------

    output: xarray
        Simulation output

    sigma: float
        Standard deviation. Must be larger than or equal to 0. Equal for all datapoints.

    relative: bool
        Add noise relative to magnitude of simulation output.

    Returns
    -------

    output: xarray
        Simulation output, but every value was replaced with a gaussian estimate.

    """

    # Loop over variables in xarray
    for varname, da in output.data_vars.items():
        # Replace very value with a normal draw
        if relative == False:
            values = np.random.normal(da.values, sigma)
        elif relative == True:
            values = np.random.normal(da.values, np.abs(sigma*da.values))
        output[varname].values = values
    return output

def add_negative_binomial_noise(output, alpha):
    """A function to add negative binomial noise to a simulation result

    Parameters
    ----------

    output: xarray
        Simulation output

    alpha: float
        Overdispersion factor. Must be larger than or equal to 0. Reduces to Poisson noise for alpha --> 0.

    Returns
    -------

    output: xarray
        Simulation output, but every value was replaced with a negative binomial estimate.
    """

    # Loop over variables in xarray
    for varname, da in output.data_vars.items():
        # Replace very value with a negative_binomial draw
        values = np.random.negative_binomial(
            1/alpha, (1/alpha)/(da.values + (1/alpha)))
        output[varname].values = values
    return output

def assign_theta(param_dict, parameter_names, thetas):
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
    warmup : int
        Offset between simulation start and start of data collection
        Because 'warmup' does not reside in the model parameters dictionary, this argument is only returned if 'warmup' is in the parameter name list 'pars'

    param_dict : dict
        Model parameters dictionary with values of parameters 'pars' set to the obtained PSO estimate in vector 'theta'

    """

    # Find out if 'warmup' needs to be estimated
    warmup_position = None
    if 'warmup' in parameter_names:
        warmup_position = parameter_names.index('warmup')
        warmup = thetas[warmup_position]
        parameter_names = [x for x in parameter_names if x != "warmup"]
        thetas = [x for (i, x) in enumerate(thetas) if i != warmup_position]

    thetas_dict, n = _thetas_to_thetas_dict(thetas, parameter_names, param_dict)
    for i, (param, value) in enumerate(thetas_dict.items()):
        param_dict.update({param: value})

    if warmup_position:
        return warmup, param_dict
    else:
        return param_dict

def _thetas_to_thetas_dict(thetas, parameter_names, model_parameter_dictionary):
    """
    Add a docstring
    """
    dict = {}
    idx = 0
    total_n_values = 0
    for param in parameter_names:
        try:
            dict[param] = np.array(
                thetas[idx:idx+len(model_parameter_dictionary[param])], np.float64)
            total_n_values += len(dict[param])
            idx = idx + len(model_parameter_dictionary[param])
        except:
            if ((isinstance(model_parameter_dictionary[param], float)) | (isinstance(model_parameter_dictionary[param], int))):
                dict[param] = thetas[idx]
                total_n_values += 1
                idx = idx + 1
            else:
                raise ValueError(
                    'Calibration parameters must be either of type int, float, list (containing int/float) or 1D np.array')
    return dict, total_n_values

def variance_analysis(data, resample_frequency):
    """ A function to analyze the relationship between the variance and the mean in a timeseries of data
        ================================================================================================

        The timeseries is binned and the mean and variance of the datapoints within this bin are estimated.
        Several statistical models are then fitted to the relationship between the mean and variance.
        The statistical models are: gaussian (var = c), poisson (var = mu), quasi-poisson (var = theta*mu), negative binomial (var = mu + alpha*mu**2)

        Parameters
        ----------

            series: pd.Series
                Timeseries of data to be analyzed. The series must have a pd.Timestamp index labeled 'date' for the time dimension.
                Additionally, this function supports the addition of one more dimension (f.i. space) using a multiindex.
                This function is not intended to study the variance of datasets containing multiple datapoints on the same date. 

            resample_frequency: str
                This function approximates the average and variance in the timeseries data by binning the timeseries. The resample frequency denotes the number of days in each bin.
                Valid options are: 'W': weekly, '2W': biweekly, 'M': monthly, etc.

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
        rolling_mean = data.ewm(span=7, adjust=False).mean()
        mu_data = (data.groupby(
            [pd.Grouper(freq=resample_frequency, level='date')]).mean())
        var_data = (((data-rolling_mean) **
                    2).groupby([pd.Grouper(freq=resample_frequency, level='date')]).mean())
    else:
        rolling_mean = data.groupby(level=secundary_index_name, group_keys=False).apply(
            lambda x: x.ewm(span=7, adjust=False).mean())
        mu_data = (data.groupby([pd.Grouper(
            freq=resample_frequency, level='date')] + [secundary_index_values]).mean())
        var_data = (((data-rolling_mean)**2).groupby([pd.Grouper(
            freq=resample_frequency, level='date')] + [secundary_index_values]).mean())

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
