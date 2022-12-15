# Models

Standard usage of pySODM involves building an ODE or SDE model. For practical examples of initializing models, we refer to the [tutorials](workflow.md).

##  *class* ODEModel

The ODEModel class inherits several attributes from the model class defined by the user.

**Inherits:**

* **state_names** (list) - Contains the names (type: str) of the states.
* **parameter_names** (list) - Contains the names (type: str) of the non-stratified parameters. Non-stratified parameters are not subject to input checks and can thus be of any datatype/size.
* **stratification_names** (list) - optional - Contains the names of the stratification axes. The names given here become the dimensions in the simulation output `xarray` Dataset.
* **parameter_stratified_names** (list) - optional - Contains the names of the stratified parameters. Stratified parameters are subject to input checks. They must be a list or a 1D np.ndarray with a length equal to the number of coordinates of the stratification axis. 
    * For one stratification axes: list contains strings - `['stratpar_1', 'stratpar_2']`
    * For multiple stratification axes: list contains lists, each sublist contains names of stratified parameters associated with that stratification axes - `[['stratpar_1', 'stratpar_2'],[],['stratpar_3']]`
* **state_2d** (list) - optional - Contains the names (type: str) of 2D states. Experimental, only available for ODE models and tested with one stratification axis.

Upon intialization of the model class, the following arguments must be provided.

**Parameters:**

* **states** (dict) - Keys: names of states. Values: values of states. The dictionary does not have to contain a key,value pair for all states listed in `state_names`. States that lack a key,value pair are filled with zeros upon initialization.
* **parameters** (dict) - Keys: names of parameters. Values: values of parameters. A key,value pair must be provided for all parameters listed in `parameter_names` and `parameter_stratified_names`. If time dependent parameter functions with additional parameters (aside from the obligatory `t`, `states`, `params`) are used, these parameters must be included as well. 
* **coordinates** (dict) - optional - Keys: names of stratifications (`stratification_names`). Values: desired coordinates for the stratification axis. Values provided here become the dimension's coordinates in the `xarray` Dataset.
* **time_dependent_parameters** (dict) - optional - Keys: name of the model parameter the time-dependent parameter function should be applied to. Must be a valid model parameter. Values: time-dependent parameter function. Time-dependent parameter functions must have `t` (simulation timestep), `states` (model states at timestep `t`) and `params` (model parameters dictionary) as the first three arguments.

**Methods:**

* **sim(time, N=1, draw_function=None, samples=None, processes=None, method='RK23', rtol=1e-3, output_timestep=1, warmup=0)**

    Simulate a model over a given time period using `scipy.integrate.solve_ivp()`. Can optionally perform `N` repeated simulations. Can change the values of model parameters at every repeated simulation by drawing samples from a dictionary `samples` using a function `draw_function`.

    **Parameters**

    * **time** - (int/float or list) - The start and stop "time" for the simulation run. Three possible inputs: 1) int/float, 2) list of int/float of type `[start_time, stop_time]`, 3) list of pd.Timestamp or str of type `[start_date, stop_date]`.
    * **N** - (int) - optional - Number of repeated simulations to perform.
    * **samples** - (dict) - optional - Dictionary containing samples of model parameters. Obligatory input to a *draw function*.   
    * **draw_function** - (function) - optional - A function consisting of two inputs: the model parameters dictionary `param_dict`, the previously documented samples dictionary `samples_dict`. Function must return the model parameters dictionary `param_dict`. Function can be used to update a model parameter's value during every repeated simulation. Usefull to propagate MCMC samples of model parameters or perform sensitivity analyses.
    * **processes** - (int) - optional - Number of cores to use when {math}`N > 1`.
    * **method** - (str) - optional - Integration methods of `scipy.integrate.solve_ivp()` (read the [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)).
    * **rtol** - (int/float) - optional - Relative tolerance of `scipy.integrate.solve_ivp()` (read the [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)).
    * **output_timestep** - (int/float) - optional - Interpolate model output to every `output_timestep` timesteps. For datetimes: expressed in days.
    * **warmup** - (float) - optional - Number of days to simulate prior to the simulation start. Usefull in epidemiological contexts when the time between the appearance of "patient zero" and the collection of data is unkown. 

    **Returns**

    * **out** - (xarray.Dataset) - Simulation output. Consult the xarray documentation [here](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html#xarray.Dataset). `xarray.Dataset.data_vars` are the model states. `xarray.Dataset.dimensions` are the time dimension plus the model's stratifications. The time dimension is named `time` if timesteps were numbers, the time dimension is named `date` if timesteps were dates. When {math}`N > 1` an additional dimension `draws` is added to the output accomodate the repeated simulations.

## *class* SDEModel

The SDEModel class inherits several attributes from the model class defined by the user.

**Inherits:**

* **state_names** (list) - Contains the names (type: str) of the states.
* **parameter_names** (list) - Contains the names (type: str) of the non-stratified parameters. Non-stratified parameters are not subject to input checks and can thus be of any datatype/size.
* **stratification_names** (list) - optional - Contains the names of the stratification axes. The names given here become the dimensions in the simulation output `xarray` Dataset.
* **parameter_stratified_names** (list) - optional - Contains the names of the stratified parameters. Stratified parameters are subject to input checks. They must be a list or a 1D np.ndarray with a length equal to the number of coordinates of the stratification axis. 
    * For one stratification axes: list contains strings - `['stratpar_1', 'stratpar_2']`
    * For multiple stratification axes: list contains lists, each sublist contains names of stratified parameters associated with that stratification axes - `[['stratpar_1', 'stratpar_2'],[],['stratpar_3']]`

Upon intialization of the model class, the following arguments must be provided.

**Parameters:**

* **states** (dict) - Keys: names of states. Values: values of states. The dictionary does not have to contain a key,value pair for all states listed in `state_names`. States that lack a key,value pair are filled with zeros upon initialization.
* **parameters** (dict) - Keys: names of parameters. Values: values of parameters. A key,value pair must be provided for all parameters listed in `parameter_names` and `parameter_stratified_names`. If time dependent parameter functions with additional parameters (aside from the obligatory `t`, `states`, `params`) are used, these parameters must be included as well. 
* **coordinates** (dict) - optional - Keys: names of stratifications (`stratification_names`). Values: desired coordinates for the stratification axis. Values provided here become the dimension's coordinates in the `xarray` Dataset.
* **time_dependent_parameters** (dict) - optional - Keys: name of the model parameter the time-dependent parameter function should be applied to. Must be a valid model parameter. Values: time-dependent parameter function. Time-dependent parameter functions must have `t` (simulation timestep), `states` (model states at timestep `t`) and `params` (model parameters dictionary) as the first three arguments.

**Methods:**

* **sim(time, N=1, draw_function=None, samples=None, processes=None, method='tau_leap', tau=0.5, output_timestep=1, warmup=0)**

    Simulate a model over a given time period stochastically using Gillespie's Stochastic Simulation Algorithm (SSA) or the approximate Tau-leaping method. Can optionally perform `N` repeated simulations. Can change the values of model parameters at every repeated simulation by drawing samples from a dictionary `samples` using a function `draw_function`.

    **Parameters**

    * **time** - (int/float or list) - The start and stop "time" for the simulation run. Three possible inputs: 1) int/float, 2) list of int/float of type `[start_time, stop_time]`, 3) list of pd.Timestamp or str of type `[start_date, stop_date]`.
    * **N** - (int) - optional - Number of repeated simulations to perform.
    * **samples** - (dict) - optional - Dictionary containing samples of model parameters. Obligatory input to a *draw function*.   
    * **draw_function** - (function) - optional - A function consisting of two inputs: the model parameters dictionary `param_dict`, the previously documented samples dictionary `samples_dict`. Function must return the model parameters dictionary `param_dict`. Function can be used to update a model parameter's value during every repeated simulation. Usefull to propagate MCMC samples of model parameters or perform sensitivity analyses.
    * **processes** - (int) - optional - Number of cores to use when {math}`N > 1`.
    * **method** - (str) - optional - 'SSA': Stochastic Simulation Algorithm. 'tau-leap': Tau-leaping algorithm. Consult the following [blog](https://lewiscoleblog.com/gillespie-algorithm) for more background information.
    * **tau** - (int/float) - optional - Leap value of the tau-leaping method. Consult the following [blog](https://lewiscoleblog.com/gillespie-algorithm) for more background information.
    * **output_timestep** - (int/float) - optional - Interpolate model output to every `output_timestep` timesteps. For datetimes: expressed in days.
    * **warmup** - (float) - optional - Number of days to simulate prior to the simulation start. Usefull in epidemiological contexts when the time between the appearance of "patient zero" and the collection of data is unkown. 

    **Returns**

    * **out** - (xarray.Dataset) - Simulation output. Consult the xarray documentation [here](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html#xarray.Dataset). `xarray.Dataset.data_vars` are the model states. `xarray.Dataset.dimensions` are the time dimension plus the model's stratifications. The time dimension is named `time` if timesteps were numbers, the time dimension is named `date` if timesteps were dates. When {math}`N > 1` an additional dimension `draws` is added to the output accomodate the repeated simulations.
