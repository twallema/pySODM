# Models

Standard usage of pySODM involves building an ODE or SDE model. For practical examples of initializing models, we refer to the [tutorials](quickstart.md).

## base.py

###  *class* ODEModel

The ODEModel class inherits several attributes from the model class defined by the user.

**Inherits:**

* **state_names** (list) - Names of the model's states.
* **parameter_names** (list) - Names of the model's parameters. Parameters are not subject to input checks and can thus be of any datatype/size.
* **integrate** (function) - Function computing the differentials of every model state. The integrate function must have the timestep `t` as its first input, followed by the `state_names`, `parameter_names` and `parameter_stratified_names` (their order is not important). The integrate function must return a differential for every model state of the correct size and **in the same order as `state_names`**. The integrate function must be a static method (include `@staticmethod`).
* **dimension_names** (list) - optional - Names of the possible model dimensions. The coordinates of the dimensions are specified during initialization of the model.
* **parameter_stratified_names** (list) - optional - Names of the *stratified* parameters. Stratified parameters must be of type list/1D np.array and their length, which must be equal to the number of coordinates of the dimension axis it corresponds to, is verified during model initialization. Their use is optional and mainly serves as a way for the user to structure his code.
    * If the model has one dimension: list contains strings - `['stratpar_1', 'stratpar_2']`
    * If the model has multiple dimensions: list contains *n* sublists, where *n* is the number of model dimensions (length of `dimension_names`). Each sublist contains names of stratified parameters associated with the dimension in the corresponding position in `dimension_names` - example: `[['stratpar_1', 'stratpar_2'],[],['stratpar_3']]`
* **state_dimensions** (list) - optional - Specify, for each model state in `state_names`, its dimensions. Allows users to define models with states of different sizes. If `state_dimensions` is not provided, all model states will have the same size, depending on the model's dimensions specified in `dimension_names`. If specified, `state_dimensions` must contain *n* sublists, where *n* is the number of model states (length of `state_names`). If some model states have no dimensions (i.e. it is a float), specify an empty list.

Upon intialization of the model class, the following arguments must be provided.

**Parameters:**

* **states** (dict) - Keys: names of states. Values: values of states. The dictionary does not have to contain a key,value pair for all states listed in `state_names`. States that lack a key,value pair are filled with zeros upon initialization.
* **parameters** (dict) - Keys: names of parameters. Values: values of parameters. A key,value pair must be provided for all parameters listed in `parameter_names` and `parameter_stratified_names`. If time dependent parameter functions with additional parameters (aside from the obligatory `t`, `states`, `params`) are used, these parameters must be included as well. 
* **coordinates** (dict) - optional - Keys: names of dimensions (`dimension_names`). Values: desired coordinates for the dimension axis. Values provided here become the dimension's coordinates in the `xarray` Dataset.
* **time_dependent_parameters** (dict) - optional - Keys: name of the model parameter the time-dependent parameter function should be applied to. Must be a valid model parameter. Values: time-dependent parameter function. Time-dependent parameter functions must have `t` (simulation timestep), `states` (model states at timestep `t`) and `params` (model parameters dictionary) as the first three arguments.

**Methods:**

* **sim(time, N=1, draw_function=None, samples=None, processes=None, method='RK23', rtol=1e-3, tau=None, output_timestep=1, warmup=0)**

    Simulate a model over a given time period using `scipy.integrate.solve_ivp()`. Can optionally perform `N` repeated simulations. Can change the values of model parameters at every repeated simulation by drawing samples from a dictionary `samples` using a function `draw_function`.

    **Parameters**

    * **time** - (int/float or list) - The start and stop "time" for the simulation run. Three possible inputs: 1) int/float, 2) list of int/float of type `[start_time, stop_time]`, 3) list of pd.Timestamp or str of type `[start_date, stop_date]`.
    * **N** - (int) - optional - Number of repeated simulations to perform.
    * **samples** - (dict) - optional - Dictionary containing samples of model parameters. Obligatory input to a *draw function*.   
    * **draw_function** - (function) - optional - A function consisting of two inputs: the model parameters dictionary `param_dict`, the previously documented samples dictionary `samples_dict`. Function must return the model parameters dictionary `param_dict`. Function can be used to update a model parameter's value during every repeated simulation. Usefull to propagate MCMC samples of model parameters or perform sensitivity analyses.
    * **processes** - (int) - optional - Number of cores to use when {math}`N > 1`.
    * **method** - (str) - optional - Integration methods of `scipy.integrate.solve_ivp()` (read the [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)).
    * **rtol** - (int/float) - optional - Relative tolerance of `scipy.integrate.solve_ivp()` (read the [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)).
    * **tau** - (int/float) - optional - If `tau != None`, the model is simulated using a simple discrete timestepper with fixed timestep `tau`. Overrides the `method` and `rtol` arguments. 
    * **output_timestep** - (int/float) - optional - Interpolate model output to every `output_timestep` timesteps. For datetimes: expressed in days.
    * **warmup** - (float) - optional - Number of days to simulate prior to the simulation start. Usefull in epidemiological contexts when the time between the appearance of "patient zero" and the collection of data is unkown. 

    **Returns**

    * **out** - (xarray.Dataset) - Simulation output. Consult the xarray documentation [here](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html#xarray.Dataset). `xarray.Dataset.data_vars` are the model states. `xarray.Dataset.dimensions` are the time dimension plus the model's dimensions. The time dimension is named `time` if timesteps were numbers, the time dimension is named `date` if timesteps were dates. When {math}`N > 1` an additional dimension `draws` is added to the output accomodate the repeated simulations.

### *class* SDEModel

The SDEModel class inherits several attributes from the model class defined by the user.

**Inherits:**

* **state_names** (list) - Names of the model's states.
* **parameter_names** (list) - Names of the model's parameters. Parameters are not subject to input checks and can thus be of any datatype/size.
* **compute_rates** (function) - Function returning the rates of transitioning between the model states. `compute_rates()` must have the timestep `t` as its first input, followed by the `state_names`, `parameter_names` and the `parameter_stratified_names` (their order is not important). `compute_rates()` must be a static method (include `@staticmethod`). The output of `compute_rates()` must be a dictionary. Its keys must be valid model states, a rate is only needed for the states undergoing a transitioning. The corresponding values must be a list containing the rates of the possible transitionings of the state. In this way, a model state can have multiple transitionings.
* **apply_transitionings** (function) - Function to update the states with the number of drawn transitionings. `apply_transitionings()` must have the timestep `t` as its first input, followed by the solver timestep `tau`, follwed by the dictionary containing the transitionings `transitionings`, then followed by the model states and parameters similarily to `compute_rates()`.
* **dimension_names** (list) - optional - Names of the possible model dimensions. The coordinates of the dimensions are specified during initialization of the model.
* **parameter_stratified_names** (list) - optional - Names of the *stratified* parameters. Stratified parameters must be of type list/1D np.array and their length, which must be equal to the number of coordinates of the dimension axis it corresponds to, is verified during model initialization. Their use is optional and mainly serves as a way for the user to structure his code.
    * If the model has one dimension: list contains strings - `['stratpar_1', 'stratpar_2']`
    * If the model has multiple dimensions: list contains *n* sublists, where *n* is the number of model dimensions (length of `dimension_names`). Each sublist contains names of stratified parameters associated with the dimension in the corresponding position in `dimension_names` - example: `[['stratpar_1', 'stratpar_2'],[],['stratpar_3']]`
* **state_dimensions** (list) - optional - Specify, for each model state in `state_names`, its dimensions. Allows users to define models with states of different sizes. If `state_dimensions` is not provided, all model states will have the same size, depending on the model's dimensions specified in `dimension_names`. If specified, `state_dimensions` must contain *n* sublists, where *n* is the number of model states (length of `state_names`). If some model states have no dimensions (i.e. it is a float), specify an empty list.

Upon intialization of the model class, the following arguments must be provided.

**Parameters:**

* **states** (dict) - Keys: names of states. Values: values of states. The dictionary does not have to contain a key,value pair for all states listed in `state_names`. States that lack a key,value pair are filled with zeros upon initialization.
* **parameters** (dict) - Keys: names of parameters. Values: values of parameters. A key,value pair must be provided for all parameters listed in `parameter_names` and `parameter_stratified_names`. If time dependent parameter functions with additional parameters (aside from the obligatory `t`, `states`, `params`) are used, these parameters must be included as well. 
* **coordinates** (dict) - optional - Keys: names of dimensions (`dimension_names`). Values: desired coordinates for the dimension axis. Values provided here become the dimension's coordinates in the `xarray` Dataset.
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

    * **out** - (xarray.Dataset) - Simulation output. Consult the xarray documentation [here](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html#xarray.Dataset). `xarray.Dataset.data_vars` are the model states. `xarray.Dataset.dimensions` are the time dimension plus the model's dimensions. The time dimension is named `time` if timesteps were numbers, the time dimension is named `date` if timesteps were dates. When {math}`N > 1` an additional dimension `draws` is added to the output accomodate the repeated simulations.
