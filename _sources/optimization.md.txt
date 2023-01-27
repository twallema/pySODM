# Optimization

## objective_functions.py

### Log posterior probability

***class* log_posterior_probability(model, parameter_names, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args, weights=None, log_prior_prob_fnc=None, log_prior_prob_fnc_args=None, initial_states=None, aggregation_function=None, labels=None)**

**Parameters:**

* **model** (object) - An initialized ODEModel or SDEModel.
* **parameter_names** (list) - Names of model parameters (type: str) to calibrate. Model parameters must be of type float (0D), list containing float (1D), or np.ndarray (nD).
* **bounds** (list) - Lower and upper bound of calibrated parameters provided as lists/tuples containing lower and upper bound: example: `[(lb_1, ub_1), ..., (lb_n, ub_n)]`. Values falling outside these bounds will be restricted to the provided ranges before simulating the model.
* **data** (list) - Contains the datasets (type: pd.Series/pd.DataFrame) the model should be calibrated to. For one dataset use `[dataset,]`. Must contain a time index named `time` or `date`. Additional axes must be implemented using a `pd.Multiindex` and must bear the names/contain the coordinates of a valid model dimension.
* **states** (list) - Names of the model states (type: str) the respective datasets should be matched to.
* **log_likelihood_fnc** (list) - Contains a log likelihood function for every provided dataset.
* **log_likelihood_fnc_args** (list) - Contains the arguments of the log likelihood functions. If the log likelihood function has no arguments (`ll_poisson`), provide an empty list.
* **weights** (list) - optional - Contains the weights of every dataset in the final log posterior probability. Defaults to one for every dataset.
* **log_prior_prob_fnc** (list) - optional - Contains a prior probability function for every calibrated parameter. Defaults to a uniform prior using the provided bounds.
* **log_prior_prob_fnc_args** (list) - optional - Contains the arguments of the provided prior probability functions.
* **initial_states** (list) - optional - Contains a dictionary of initial states for every dataset. 
* **aggregation_function** (callable function or list) - optional - A user-defined function to manipulate the model output before matching it to data. The function takes as input an `xarray.DataArray`, resulting from selecting the simulation output at the state we wish to match to the dataset (`model_output_xarray_Dataset['state_name']`), as its input. The output of the function must also be an `xarray.DataArray`. No checks are performed on the input or output of the aggregation function, use at your own risk. Illustrative use case: I have a spatially explicit epidemiological model and I desire to simulate it a high spatial resolutioni. However, data is only available on a lower level of spatial resolution. Hence, I use an aggregation function to properly aggregate the spatial levels. I change the coordinates on the spatial dimensions in the model output. Valid inputs for the argument `aggregation_function`are: 1) one callable function --> applied to every dataset. 2) A list containing one callable function --> applied to every dataset. 3) A list containing a callable function for every dataset --> every dataset has its own aggregation function.
* **labels** (list) - optional - Contains a custom label for the calibrated parameters. Defaults to the names provided in `parameter_names`.

**Methods:**

* **__call__(thetas, simulation_kwargs={})**

    **Parameters:**
    * **thetas** (list/np.ndarray) - A flattened list containing the estimated parameter values.
    * **simulation_kwargs** (dict) - Optional arguments to be passed to the model's [sim()](models.md) function when evaluating the posterior probability.

    **Returns:**
    * **lp** (float) - Logarithm of the posterior probability.

### Log likelihood

***function* ll_gaussian(ymodel, ydata, sigma)**

>    **Parameters:**
>   * **ymodel** (list/np.ndarray) - Mean values of the Gaussian distribution (i.e. "mu" values), as predicted by the model
>   * **ydata** (list/np.ndarray) - Data time series values to be matched with the model predictions.
>   * **sigma** (float/list of floats/np.ndarray) - Standard deviation(s) of the Gaussian distribution around the data 'ydata'. Two options are possible: 1) One error per model dimension, applied uniformly to all datapoints corresponding to that dimension; OR 2) One error for every datapoint, corresponding to a weighted least-squares regression.

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

***function* log_prior_uniform(x, bounds)**

>   Uniform log prior distribution.

>   **Parameters:**

>    * **x** (float) - Parameter value whos probability we want to test.
>    * **bounds** (tuple) - Contains the upper and lower bounds of the parameter value.

>    **Returns:**
>    * **lp** (float) Log probability of sample x in light of a uniform prior distribution.

***function* log_prior_custom(x, args)**

>   Computes the probability of a sample in light of a list containing samples.

>    **Parameters:**
>    * **x** (float) - Parameter value whos probability we want to test.
>    * **args** (tuple) - Contains the density of each bin in the first position and the bounds of the bins in the second position. Contains a weight given to the custom prior in the third position of the tuple.

>    **Returns:**
>    * **lp** (float) Log probability of sample x in light of a custom prior distribution.

>    **Example use:**

>```python
>density_my_par, bins_my_par = np.histogram([sample_0, sample_1, ..., sample_n], bins=20, density=True)
>density_my_par_norm = density_my_par/np.sum(density_my_par)
>prior_fcn = prior_custom
>prior_fcn_args = (density_my_par_norm, bins_my_par, weight)
>```

***function* log_prior_normal(x, norm_pars)**

>   Normal log prior distribution.

>    **Parameters:**
>    * **x** (float) - Parameter value whos probability we want to test.
>    * **norm_pars** (tuple) - Tuple containg average and standard deviation of normal distribution.

>    **Returns:**
>    * **lp** (float) Log probability of sample x in light of a normal prior distribution.

***function* log_prior_triangle(x, triangle_pars)**

>   Triangular log prior distribution.

>    **Parameters:**
>    * **x** (float) - Parameter value whos probability we want to test.
>    * **triangle_pars** (tuple) - Tuple containg lower bound, upper bound and mode of the triangle distribution.

>    **Returns:**
>    * **lp** (float) Log probability of sample x in light of a triangular prior distribution.

***function* log_prior_gamma(x, gamma_pars)**

>   Gamma log prior distribution.

>    **Parameters:**
>    * **x** (float) - Parameter value whos probability we want to test.
>    * **gamma_pars** (tuple) - Tuple containg gamma parameters alpha and beta.

>    **Returns:**
>    * **lp** (float) Log probability of sample x in light of a gamma prior distribution.

***function* log_prior_weibull(x, weibull_params)**

>   Weibull log prior distribution.

>    **Parameters:**
>    * **x** (float) - Parameter value whos probability we want to test.
>    * **weibull_params** (tuple) - Contains the weibull parameters k and lambda.

>    **Returns:**
>    * **lp** (float) Log probability of sample x in light of a weibull prior distribution.

## nelder_mead.py

***function* optimize(func, x_start, step, bounds=None, args=(), kwargs={}, processes=1, no_improve_thr=1e-6, no_improv_break=100, max_iter=1000, alpha=1., gamma=2., rho=-0.5, sigma=0.5)**

>    Perform a [Nelder-Mead minimization](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method).

>    **Parameters:**
>    * **func** (function) - Callable function or class representing the objective function to be minimized. Recommended `log_posterior_probability`.
>    * **x_start** (list or 1D np.ndarray) - Starting estimate for the search algorithm. Length must equal the number of provided bounds. 
>    * **step** (list or 1D np.ndarray) - (Relative) size of the initial search simplex. 
>    * **bounds** (list) - optional - The bounds of the design variable(s). In form `[(lb_1, ub_1), ..., (lb_n, ub_n)]`. If class `log_posterior_probability` is used as `func`, it already contains bounds. If bounds are provided these will overwrite the bounds available in the 'log_posterior_probability' object.
>    * **args** (tuple) - optional - Additional arguments passed to objective function.
>    * **kwargs** (dict) - optional - Additional keyworded arguments passed to objective function. Example use: To compute our log posterior probability (class `log_posterior_probability`) with the 'RK45' method, we must change the `method` argument of the `sim` function, which is called in `log_posterior_probability`. To achieve this, we can supply the keyworded argument `simulation_kwargs` of `log_posterior_probability`, which passes its arguments on to the `sim` function. To this end, use `kwargs={'simulation_kwargs':{'method': 'RK45'}}`.
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
>    * **theta** (list) - Position 0: Estimated parameters. Position 1: corresponding score of `func`.

## pso.py

***function* optimize(func, bounds=None, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, processes=1, swarmsize=100, max_iter=100, minstep=1e-12, minfunc=1e-12, omega=0.8, phip=0.8, phig=0.8,  debug=False, particle_output=False, transform_pars=None)**

>    Perform a [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization). If a PSO with more options is desired, coupling to [pySwarms](https://pyswarms.readthedocs.io/en/latest/examples/tutorials/basic_optimization.html#Optimizing-a-function) is straightforward.

>    **Parameters:**
>    * **func** (function) - Callable function or class representing the objective function to be minimized. Recommended `log_posterior_probability`.
>    * **bounds** (list) - optional - The bounds of the design variable(s). In form `[(lb_1, ub_1), ..., (lb_n, ub_n)]`. If class `log_posterior_probability` is used as `func`, it already contains bounds. If bounds are provided these will overwrite the bounds available in the 'log_posterior_probability' object.
>    * **ieqcons** (list) - A list of functions of length n such that ```ieqcons[j](x,*args) >= 0.0``` in a successfully optimized problem
>    * **f_ieqcons** (function) - Returns a 1-D array in which each element must be greater or equal to 0.0 in a successfully optimized problem. If f_ieqcons is specified, ieqcons is ignored
>    * **args** (tuple) - optional - Additional arguments passed to objective function.
>    * **kwargs** (dict) - optional - Additional keyworded arguments passed to objective function. Example use: To compute our log posterior probability (class `log_posterior_probability`) with the 'RK45' method, we must change the `method` argument of the `sim` function, which is called in `log_posterior_probability`. To achieve this, we can supply the keyworded argument `simulation_kwargs` of `log_posterior_probability`, which passes its arguments on to the `sim` function. To this end, use `kwargs={'simulation_kwargs':{'method': 'RK45'}}`.
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
>    * **particle_output** (bool) - optional - If True, function additionally returns the best particles position and objective function score
>    * **transform_pars** (func) - optional - Transform the parameter values. E.g. to integer values or to map to a list of possibilities

>    **Returns:**
>    * **theta** (list) - Position 0: Estimated parameters. Position 1: Corresponding score of `func`. If `particle_output==True` then: Position 3: The best known position per particle. Position 4: Vorresponding score of `func`.

## mcmc.py

***function* run_EnsembleSampler(pos, max_n, identifier, objective_function, objective_function_args=None, objective_function_kwargs=None, moves=[(emcee.moves.DEMove(), 0.5),(emcee.moves.KDEMove(bw_method='scott'), 0.5)], fig_path=None, samples_path=None, print_n=10, backend=None, processes=1, progress=True, settings_dict={})**

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
>    * **processes** (int) - optional - Number of cores to use.
>    * **settings_dict** (dict) - optional - Dictionary containing calibration settings or other usefull settings for long-term storage. Saved in `.json` format. Appended to the samples dictionary generated by `emcee_sampler_to_dictionary()`. 

>    **Hyperparameters:**
>    * **moves** (list) - optional - Consult the [emcee documentation](https://emcee.readthedocs.io/en/stable/user/moves/).
>    * **backend** (`emcee.backends.HDFBackend`) - optional - Backend of a previous sampling experiment. If a backend is provided, the sampler is restarted from the last iteration of the previous run. Consult the [emcee documentation](https://emcee.readthedocs.io/en/stable/user/backends/).
>    * **progress** (bool) - optional - Enables the progress bar.

>    **Returns:**
>    * **sampler** (`emcee.EnsembleSampler`) - Emcee sampler object ([see](https://emcee.readthedocs.io/en/stable/user/sampler/)). To extract a dictionary of samples + settings, use `emcee_sampler_to_dictionary`.

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

***function* emcee_sampler_to_dictionary(samples_path, identifier, discard=0, thin=1, run_date=str(datetime.date.today()))**

> A function to discard and thin the samples available in the `emcee` sampler object and subsequently convert them to a dictionary of format: `{parameter_name: [sample_0, ..., sample_n]}`. Appends the dictionary of settings. Automatically saves the resulting dictionary in a .json format.

>    **Parameters:**
>    * **samples_path** (str) - Path to the .hdf5 `emcee` backend.
>    * **identifier** (str) - Identifier used for the calibration.
>    * **discard** (int) - optional - Number of samples to discard at the start of the Markov chain.
>    * **thin** (int) - optional - Thinning ratio of the Markov chain.
>    * **run_date** (datetime) - optional - Date of calibration.

Samples path, identifier and run_date are combined to find the right .hdf5 `emcee` backend and the `.json` containing the settings. 

>   **Returns:**
>   * **samples_dict** (dict) - Dictionary containing the discarded and thinned MCMC samples and settings.
 
## utils.py

***function* add_poisson_noise(output)**

>   Replace every value x in a simulation output with a realization from a Poisson distribution with expectation value x

>    **Parameters:**
>    * **output** (xarray.Dataset) - Simulation output (obtained using the `sim()` function of `ODEModel` or `SDEModel`).

>    **Returns:**
>    * **output** (xarray.Dataset) - Simulation output

***function* add_gaussian_noise(output, sigma, relative=True)**

>   Replace every value x in a simulation output with a realization from a normal distribution with average x and standard deviation `sigma` (relative=False) or x*`sigma` (relative=True)

>    **Parameters:**
>    * **output** (xarray.Dataset) - Simulation output (obtained using the `sim()` function of `ODEModel` or `SDEModel`).
>    * **sigma** (float) - Standard deviation. Must be larger than or equal to 0.
>    * **relative** (bool) - Add noise relative to magnitude of simulation output.

>    **Returns:**
>    * **output** (xarray.Dataset) - Simulation output

 ***function* add_negative_binomial_noise(output, alpha)**

>   Replace every value x in a simulation output with a realization from a negative binomial distribution with expectation value x and overdispersion `alpha`

>    **Parameters:**
>    * **output** (xarray.Dataset) - Simulation output (obtained using the `sim()` function of `ODEModel` or `SDEModel`).
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
>    * **warmup** (float) - Offset between simulation start and start of data collection. Because 'warmup' does not reside in the model parameters dictionary, this argument is only returned if 'warmup' is in the parameter name list 'pars'
>    * **param_dict** (dict) - Model parameters dictionary with values of parameters `parameter_names` set to the values listed in `thetas`

***function* variance_analysis(data, resample_frequency)**

>    A function to analyze the relationship between the variance and the mean in a timeseries of data, usefull when no measure of error is available.

>    The timeseries is binned into sets of length `resample_frequency`. The mean and variance of the datapoints within each bin are estimated. Several statistical models are then fitted to the relationship between the mean and variance. The statistical models are: Gaussian ({math}`\sigma^2 = c`), Poisson ({math}`\sigma^2 = \mu`), Quasi-poisson ({math}`\sigma^2 = \alpha \mu`), Negative Binomial ({math}`\sigma^2 = \mu + \alpha \mu^2`)

>    **Parameters:**
>    * **data** (pd.Series or pd.DataFrame) - Timeseries of data to be analyzed. The series must have a pd.Timestamp index labeled 'date' for the time dimension. Additionally, this function supports the addition of one more dimension (f.i. space) using a pd.Multiindex.
>    * **resample_frequency** (str) - The resample frequency determines the number of days in each bin. We recommend varying this parameter before drawing conclusions. Valid options are: 'W': weekly, '2W': biweekly, 'M': monthly, etc. [Consult the pandas docs](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).

>    **Returns:**
>    * **result** (pd.Dataframe) - Contains the estimated parameter(s) and the Akaike Information Criterion (AIC) of the fitted statistical model. If two index levels are present (thus 'date' and 'other index level'), the result pd.Dataframe contains the result stratified per 'other index level'.
>    * **ax** (matplotlib.pyplot axis object) - Contains a plot of the estimated mean versus variance, togheter with the fitted statistical models. The best-fitting model is highlighted.