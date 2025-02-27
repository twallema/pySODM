# Optimization

## pySODM.optimization.objective_functions

### Log posterior probability

***class* log_posterior_probability(model, parameter_names, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args, start_sim=None, weights=None, log_prior_prob_fnc=None, log_prior_prob_fnc_args=None, initial_states=None, aggregation_function=None, labels=None)**

**Parameters:**

* **model** (object) - An initialized `pySODM.models.base.ODE` or `pySODM.models.base.JumpProcess` model.
* **parameter_names** (list) - Names of model parameters (type: str) to calibrate. Model parameters must be of type float (0D), list containing float (1D), or np.ndarray (nD).
* **bounds** (list) - Lower and upper bound of calibrated parameters. Provided as a list or tuple containing lower and upper bound: example: `bounds = [(lb_1, ub_1), ..., (lb_n, ub_n)]`.
* **data** (list) - Contains the datasets (type: pd.Series/pd.DataFrame) the model should be calibrated to. If there is only one dataset use `data = [df,]`. Dataframe must contain an index named `time` or `date`. Stratified data can be incorporated using a `pd.Multiindex`, whose index levels must have names corresponding to valid model dimensions, and whose indices must be valid dimension coordinates.
* **states** (list) - Names of the model states (type: str) the respective datasets should be matched to. Must have the same length as `data`.
* **log_likelihood_fnc** (list) - Contains a log likelihood function for every provided dataset. Must have the same length as `data`.
* **log_likelihood_fnc_args** (list) - Contains the arguments of the log likelihood functions. If the log likelihood function has no arguments (such as `ll_poisson`), provide an empty list. Must have the same length as `data`.
* **start_sim** (int/float or str/datetime) - optional - Can be used to alter the start of the simulation. By default, the start of the simulation is chosen as the earliest time/date found in the datasets. 
* **weights** (list) - optional - Contains the weights of every dataset in the final log posterior probability. Defaults to one for every dataset.
* **log_prior_prob_fnc** (list) - optional - Contains a log prior probability function for every calibrated parameter. Must have the same length as `parameter_names`. If not provided, defaults the log prior probability function to a uniform distribution over `bounds`. The log prior probability functions available in
`pySODM.optimization.objective_functions` are `log_prior_uniform`, `log_prior_triangle`, `log_prior_normal`, `log_prior_gamma`, `log_prior_beta` and `log_prior_custom`.
* **log_prior_prob_fnc_args** (list) - optional - Contains the arguments of the prior probability functions, as a dictionary. Must have the same length as `parameter_names`. For example, if `log_prior_prob_fnc = [log_prior_normal,]` then `log_prior_prob_fnc_args = [{'avg': 0, 'stdev': 1},]` or `[{'avg': 0, 'stdev': 1, 'weight': 1},]`.
* **initial_states** (list) - optional - Contains a dictionary of initial states for every dataset.
* **aggregation_function** (callable function or list) - optional - A user-defined function to manipulate the model output before matching it to data. The function takes as input an `xarray.DataArray`, resulting from selecting the simulation output at the state we wish to match to the dataset (`model_output_xarray_Dataset['state_name']`), as its input. The output of the function must also be an `xarray.DataArray`. No checks are performed on the input or output of the aggregation function, use at your own risk. Illustrative use case: I have a spatially explicit epidemiological model and I desire to simulate it a fine spatial resolution. However, data is only available on a coarser level. Hence, I use an aggregation function to properly aggregate the spatial levels. I change the coordinates on the spatial dimensions in the model output. Valid inputs for the argument `aggregation_function`are: 1) one callable function --> applied to every dataset. 2) A list containing one callable function --> applied to every dataset. 3) A list containing a callable function for every dataset --> every dataset has its own aggregation function.
* **labels** (list) - optional - Contains a custom label for the calibrated parameters. Defaults to the names provided in `parameter_names`.
* **simulation_kwargs** (dict) - optional - Optional arguments to be passed to the model's [sim()](models.md) function when evaluating the posterior probability.

**Methods:**

* **__call__(thetas)**

    **Parameters:**
    * **thetas** (list/np.ndarray) - A flattened list containing the estimated parameter values.

    **Returns:**
    * **lp** (float) - Logarithm of the posterior probability.

### Log likelihood

***function* ll_normal(ymodel, ydata, sigma)**

>    **Parameters:**
>   * **ymodel** (list/np.ndarray) - Mean values of the normal distribution (i.e. "mu" values), as predicted by the model
>   * **ydata** (list/np.ndarray) - Data time series values to be matched with the model predictions.
>   * **sigma** (float/list of floats/np.ndarray) - Standard deviation(s) of the normal distribution around the data 'ydata'. Two options are possible: 1) One error per model dimension, applied uniformly to all datapoints corresponding to that dimension; OR 2) One error for every datapoint, corresponding to a weighted least-squares regression.

>   **Returns:**
>   * **ll** (float) - Loglikelihood associated with the comparison of the data points and the model prediction.  


***function* ll_lognormal(ymodel, ydata, sigma)**

>    **Parameters:**
>   * **ymodel** (list/np.ndarray) - Mean values of the lognormal distribution (i.e. "mu" values), as predicted by the model
>   * **ydata** (list/np.ndarray) - Data time series values to be matched with the model predictions.
>   * **sigma** (float/list of floats/np.ndarray) - Standard deviation(s) of the lognormal distribution around the data 'ydata'. Two options are possible: 1) One error per model dimension, applied uniformly to all datapoints corresponding to that dimension; OR 2) One error for every datapoint, corresponding to a weighted least-squares regression.

>   **Returns:**
>   * **ll** (float) - Loglikelihood associated with the comparison of the data points and the model prediction.  


>   ***function* ll_poisson(ymodel, ydata)**

>   **Parameters:**
>   * **ymodel** (list/np.ndarray) - Mean values of the Poisson distribution (i.e. "lambda" values), as predicted by the model.
>   * **ydata** (list/np.ndarray) - Data time series values to be matched with the model predictions.

>   **Returns:**
>   * **ll** (float) - Loglikelihood associated with the comparison of the data points and the model prediction.


***function* ll_negative_binomial(ymodel, ydata, alpha)**

>   **Parameters:**
>   * **ymodel** (list/np.ndarray) - Mean values of the Negative Binomial distribution, as predicted by the model.
>   * **ydata** (list/np.ndarray) - Data time series values to be matched with the model predictions.
>   * **alpha** (float/list) - Dispersion. Must be positive. If alpha goes to zero, the negative binomial distribution converges to the poisson distribution.

>   **Returns:**
>   * **ll** (float) - Loglikelihood associated with the comparison of the data points and the model prediction.  

### Log prior probability

***function* log_prior_uniform(x, bounds=None, weight=1)**

>   Uniform log prior distribution.

>   **Parameters:**

>    * **x** (float) - Parameter value. Passed internally by pySODM. 
>    * **bounds** (tuple) - Tuple containing the lower and upper bounds of the uniform probability distribution.
>    * **weight** (float) - optional - Regularisation weight (default: 1) -- does nothing.

>    **Returns:**
>    * **lp** (float) Log probability of x in light of a uniform prior distribution.

***function* log_prior_triangle(x, low=None, high=None, mode=None, weight=1)**

>   Triangular log prior distribution.

>    **Parameters:**
>    * **x** (float) - Parameter value. Passed internally by pySODM. 
>    * **low** (float) - Lower bound of the triangle distribution.
>    * **high** (float) - Upper bound of the triangle distribution.
>    * **mode** (float) - Mode of the triangle distribution.
>    * **weight** (float) - optional - Regularisation weight (default: 1).

>    **Returns:**
>    * **lp** (float) Log probability of sample x in light of a triangular prior distribution.

***function* log_prior_normal(x, avg=None, stdev=None, weight=1)**

>   Normal log prior distribution.

>    **Parameters:**
>    * **x** (float) - Parameter value. Passed internally by pySODM. 
>    * **avg** (float) - Average of the normal distribution.
>    * **stdev** (float) - Standard deviation of the normal distribution.
>    * **weight** (float) - optional - Regularisation weight (default: 1).

>    **Returns:**
>    * **lp** (float) Log probability of sample x in light of a normal prior distribution.

***function* log_prior_gamma(x, a=None, loc=None, scale=None, weight=1)**

>   Gamma log prior distribution.

>    **Parameters:**
>    * **x** (float) - Parameter value. Passed internally by pySODM. 
>    * **a** (float) - Parameter 'a' of `scipy.stats.gamma.logpdf`.
>    * **loc** (float) - Location parameter of `scipy.stats.gamma.logpdf`.
>    * **scale** (float) - Scale parameter of `scipy.stats.gamma.logpdf`.
>    * **weight** (float) - optional - Regularisation weight (default: 1).

>    **Returns:**
>    * **lp** (float) Log probability of sample x in light of a gamma prior distribution.

***function* log_prior_beta(x, a=None, b=None, loc=None, scale=None, weight=1)**

>   Beta log prior distribution.

>    **Parameters:**
>    * **x** (float) - Parameter value. Passed internally by pySODM. 
>    * **a** (float) - Parameter 'a' of `scipy.stats.beta.logpdf`.
>    * **b** (float) - Parameter 'b' of `scipy.stats.beta.logpdf`.
>    * **loc** (float) - Location parameter of `scipy.stats.beta.logpdf`.
>    * **scale** (float) - Scale parameter of `scipy.stats.beta.logpdf`.
>    * **weight** (float) - optional - Regularisation weight (default: 1).

>    **Returns:**
>    * **lp** (float) Log probability of sample x in light of a beta prior distribution.


***function* log_prior_custom(x, density=None, bins=None, weight=1)**

>    A custom log prior distribution: compute the probability of a sample in light of a list containing samples from a distribution

>    **Parameters:**
>    * **x** (float) - Parameter value. Passed internally by pySODM. 
>    * **density** (np.ndarray) - The values of the histogram (generated by `np.histogram()`).
>    * **bins** (np.ndarray) - The histogram's bin edges (generated by `np.histogram()`).
>    * **weight** (float) - optional - Regularisation weight (default: 1).

>    **Returns:**
>    * **lp** (float) Log probability of x in light of a custom distribution of data.

>    **Example use:**

>```python
>density_my_par, bins_my_par = np.histogram([sample_0, sample_1, ..., sample_n], bins=50, density=True) # convert to a list of samples to a binned PDF
>prior_fcn = prior_custom
>prior_fcn_args = (density_my_par_norm, bins_my_par, weight)
>```

## pySODM.optimization.nelder_mead

***function* optimize(func, x_start, step, bounds=None, args=(), kwargs={}, processes=1, no_improve_thr=1e-6, no_improv_break=100, max_iter=1000, alpha=1., gamma=2., rho=-0.5, sigma=0.5)**

>    Perform a [Nelder-Mead minimization](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method).

>    **Parameters:**
>    * **func** (function) - Callable function or class representing the objective function to be minimized. Recommended using `pySODM.optimization.log_posterior_probability`. 
>    * **x_start** (list or 1D np.ndarray) - Starting estimate for the search algorithm. Length must equal the number of provided bounds. 
>    * **step** (list or 1D np.ndarray) - (Relative) size of the initial search simplex. 
>    * **bounds** (list) - optional - The bounds of the design variable(s). In form `[(lb_1, ub_1), ..., (lb_n, ub_n)]`. If class `log_posterior_probability` is used as `func`, it already contains bounds. If bounds are provided these will overwrite the bounds available in the 'log_posterior_probability' object.
>    * **args** (tuple) - optional - Additional arguments passed to objective function.
>    * **kwargs** (dict) - optional - Additional keyworded arguments passed to objective function.
>    * **processes** (int) - optional - Number of cores to use.

>   **Hyperparameters:**
>    * **no_improve_thr** (float) - optional - Threshold relative iteration-to-iteration objective function value change to label the iteration as having no improvement.
>    * **no_improv_break** (int) - optional - Break after `no_improv_break` iterations without improvement.
>    * **max_iter** (int) - optional - Maximum number of iterations.
>    * **alpha** (float) - optional - Reflection coefficient
>    * **gamma** (float) - optional - Expansion coefficient
>    * **rho** (float) - optional - Contraction coefficient
>    * **sigma** (float) - optional - Shrink coefficient

>    **Returns:**
>    * **theta** (list) - Optimised parameter values.
>    * **score** (float) - Corresponding corresponding objective function value.

## pySODM.optimization.pso

***function* optimize(func, bounds=None, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, processes=1, swarmsize=100, max_iter=100, minstep=1e-12, minfunc=1e-12, omega=0.8, phip=0.8, phig=0.8,  debug=False, particle_output=False, transform_pars=None)**

>    Perform a [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization).

>    **Parameters:**
>    * **func** (function) - Callable function or class representing the objective function to be minimized. Recommended using `log_posterior_probability`.
>    * **bounds** (list) - optional - The bounds of the design variable(s). In form `[(lb_1, ub_1), ..., (lb_n, ub_n)]`. If class `log_posterior_probability` is used as `func`, it already contains bounds. If bounds are provided these will overwrite the bounds available in the 'log_posterior_probability' object.
>    * **args** (tuple) - optional - Additional arguments passed to objective function.
>    * **kwargs** (dict) - optional - Additional keyworded arguments passed to objective function.
>    * **ieqcons** (list) - A list of functions of length n such that ```ieqcons[j](x,*args) >= 0.0``` in a successfully optimized problem
>    * **f_ieqcons** (function) - Returns a 1-D array in which each element must be greater or equal to 0.0 in a successfully optimized problem. If f_ieqcons is specified, ieqcons is ignored
>    * **processes** (int) - optional - Number of cores to use.

>   **Hyperparameters:**
>    * **swarmsize** (int) - optional - Size of the swarm. "Number of particles"
>    * **max_iter** (int) - optional - Maximum number of iterations
>    * **minstep** (float) - optional - The minimum stepsize of swarm's best position before the search terminates
>    * **minfunc** (float) - optional - The minimum change of swarm's best objective value before the search terminates
>    * **omega** (float) - optional - Inertia weight. Must be smaller than one. 
>    * **phip** (float) - optional - Tendency to search away from the particles best known position.  A higher value means each particle has less confidence in it's own best value.
>    * **phig** (float) - optional - Tendency to search away from the swarm's best known position. A higher value means each particle has less confidence in the swarm's best value.
>    * **debug** (bool) - optional - If True, progress statements will be displayed every iteration
>    * **transform_pars** (func) - optional - Transform the parameter values. E.g. to integer values or to map to a list of possibilities

>    **Returns:**
>    * **theta** (list) - Optimised parameter values.
>    * **score** (float) - Corresponding corresponding objective function value.

## pySODM.optimization.mcmc

***function* run_EnsembleSampler(pos, max_n, identifier, objective_function, objective_function_args=None, objective_function_kwargs=None, moves=[(emcee.moves.DEMove(), 0.25),(emcee.moves.DESnookerMove(),0.25),(emcee.moves.KDEMove(bw_method='scott'), 0.50)], fig_path=None, samples_path=None, print_n=10, backend=None, processes=1, progress=True, settings_dict={})**

> Wrapper function to setup an `emcee.EnsembleSampler` and handle all backend-related tasks.

>    **Parameters:**
>    * **pos** (np.ndarray) - Starting position of the Markov Chains. We recommend using `perturbate_theta()`.
>    * **max_n** (int) - Maximum number of iterations.
>    * **identifier** (str) - Identifier of the expirement.
>    * **objective_function** (callable function) - Objective function. Recommended `log_posterior_probability`.
>    * **objective_function_args** (tuple) - optional - Arguments of the objective function. If using `log_posterior_probability` as objective function, use default `None`.
>    * **objective_function_kwargs** (dict) - optional - Keyworded arguments of the objective function. If using `log_posterior_probability` as objective function, use default `None`.
>    * **fig_path** (str) - optional - Location where the diagnostic figures (autocorrelation and trace plot) are saved.
>    * **samples_path** (str) - optional - Location where the `.hdf5` backend and settings `.json` should be saved.
>    * **print_n** (int) - optional - Print autocorrelation and trace plots every `print_n` iterations.
>    * **discard** (int) - optional - Number of iterations to remove from the beginning of the markov chains ("burn-in").
>    * **thin** (int) - optional - Retain only every `thin`-th iteration.
>    * **processes** (int) - optional - Number of cores to use.
>    * **settings_dict** (dict) - optional - Dictionary containing calibration settings or other usefull settings for long-term storage. Appended to output `samples_xr` as attributes. Valid datatypes for values: str, Number, ndarray, number, list, tuple, bytes. 

>    **Hyperparameters:**
>    * **moves** (list) - optional - Algorithm used for updating the coordinates of walkers in an ensemble sampler. By default, pySODM uses a shotgun approach by implementing a balanced cocktail of `emcee` moves. Consult the [emcee documentation](https://emcee.readthedocs.io/en/stable/user/moves/) for an overview of all moves.
>    * **backend** (str) - optional - Path to backend of a previous sampling run. If a backend is provided, the sampler is restarted from the last iteration of the previous run. Consult the [emcee documentation](https://emcee.readthedocs.io/en/stable/user/backends/).
>    * **progress** (bool) - optional - Enables the progress bar.

>    **Returns:**
>    * **sampler** (`emcee.EnsembleSampler`) - Emcee sampler object ([see](https://emcee.readthedocs.io/en/stable/user/sampler/)).
>    * **samples_xr** (`xarray.Dataset`) - Samples formatted in an xarray.Dataset. 
>       * scalar parameters: 
>           * dimensions: `['iteration', 'chain']`
>           * coordinates: `[samples_np.shape[0], samples_np.shape[1]]`
>       * n-dimensional parameters: 
>           * dimensions: `['iteration', 'chain', '{parname}_dim_0', ..., '{parname}_dim_n']`
>           * coordinates: `[samples_np.shape[0], samples_np.shape[1], parameter_shapes[parname][0], ..., parameter_shapes[parname][n]]`

***function* perturbate_theta(theta, pert, multiplier=2, bounds=None, verbose=None)**

> A function to perturbate a NM/PSO estimate and construct a matrix with initial positions for the MCMC chains

>    **Parameters:**
>    * **theta** (list or 1D np.ndarray) - List containing the parameter values to be perturbated.
>    * **pert** (list) - Relative perturbation per parameter in `theta`. Drawn from a uniform distribution (plus-minus pert).
>    * **multiplier** (int) - optional - Multiplier determining the total number of markov chains that will be run by `emcee`. Typically, total nr. chains = multiplier * nr. parameters. Minimum is two, at least five recommended.
>    * **bounds** (list) - optional - Lower and upper bound of calibrated parameters provided as tuples. `[(lb_1, ub_1), ..., (lb_n, ub_n)]`. Note: bounds must not be zero, because the perturbation is based on a percentage of the value, and any percentage of zero returns zero, causing an error regarding linear dependence of walkers.
>    * **verbose** (bool) - optional - Print user feedback to stdout

>    **Returns:**
>    * **ndim** (int) - Number of parameters. Equal to `len(theta)`.
>    * **nwalkers** (int) - Number of Markov chains.
>    * **pos** (np.ndarray) - Initial positions of the Markov chains. Dimensions: `[ndim, nwalkers]`.
 
## pySODM.optimization.utils

***function* add_poisson_noise(output)**

>   Replace every value x in a simulation output with a realization from a Poisson distribution with expectation value x

>    **Parameters:**
>    * **output** (xarray.Dataset) - Simulation output (obtained using the `sim()` function of `ODE` or `JumpProcess`).

>    **Returns:**
>    * **output** (xarray.Dataset) - Simulation output

***function* add_gaussian_noise(output, sigma, relative=True)**

>   Replace every value x in a simulation output with a realization from a normal distribution with average x and standard deviation `sigma` (relative=False) or x*`sigma` (relative=True)

>    **Parameters:**
>    * **output** (xarray.Dataset) - Simulation output (obtained using the `sim()` function of `ODE` or `JumpProcess`).
>    * **sigma** (float) - Standard deviation. Must be larger than or equal to 0.
>    * **relative** (bool) - Add noise relative to magnitude of simulation output.

>    **Returns:**
>    * **output** (xarray.Dataset) - Simulation output

 ***function* add_negative_binomial_noise(output, alpha)**

>   Replace every value x in a simulation output with a realization from a negative binomial distribution with expectation value x and overdispersion `alpha`

>    **Parameters:**
>    * **output** (xarray.Dataset) - Simulation output (obtained using the `sim()` function of `ODE` or `JumpProcess`).
>    * **alpha** (float) - Overdispersion factor. Must be larger than or equal to 0. Reduces to Poisson noise for alpha --> 0.

>    **Returns:**
>    * **output** (xarray.Dataset) - Simulation output

 ***function* assign_theta(param_dict, parameter_names, thetas)**

>   A function to assign a vector containing estimates of model parameters to the model parameters dictionary

>    **Parameters:**
>    * **param_dict** (dict) - Model parameters dictionary
>    * **parameter_names** (list) - Names of model parameters estimated using PSO
>    * **thetas** (list or 1D np.ndarray) - A list with values of model parameters. Values ordermust correspond to the order of `parameter_names`

>    **Returns:**
>    * **param_dict** (dict) - Model parameters dictionary with values of parameters `parameter_names` set to the values listed in `thetas`

***function* variance_analysis(data, window_length, half_life)**

>    A function to analyze the relationship between the variance and the mean in a timeseries of data, usefull when no measure of error is available.

>    The timeseries is binned into sets of length `window_length`. The mean and variance of the datapoints within each bin are estimated. Several statistical models are then fitted to the relationship between the mean and variance. The statistical models are: Gaussian ({math}`\sigma^2 = c`), Poisson ({math}`\sigma^2 = \mu`), Quasi-poisson ({math}`\sigma^2 = \alpha \mu`), Negative Binomial ({math}`\sigma^2 = \mu + \alpha \mu^2`)

>    **Parameters:**
>    * **data** (pd.Series or pd.DataFrame) - Timeseries of data to be analyzed. The series must have a pd.Timestamp index labeled 'date' for the time dimension. Additionally, this function supports the addition of one more dimension (f.i. space) using a pd.Multiindex.
>    * **window_length** (str) - The length of each bin. Examples of valid arguments are: 'W': weekly, '2W': biweekly, 'M': monthly, etc. [Consult the pandas docs for all valid options.](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
>    * **half_life** (str) - Halflife of the exponential moving average.

>    **Returns:**
>    * **result** (pd.Dataframe) - Contains the estimated parameter(s) and the Akaike Information Criterion (AIC) of the fitted statistical model. If two index levels are present (thus 'date' and 'other index level'), the result pd.Dataframe contains the result stratified per 'other index level'.
>    * **ax** (matplotlib.pyplot axis object) - Contains a plot of the estimated mean versus variance, togheter with the fitted statistical models. The best-fitting model is highlighted.