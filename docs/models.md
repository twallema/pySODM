# Models

pySODM contains a class to build ordinary differential equation models (`ODE`) and a class to build stochastic jump process models (`JumpProcess`), both live in `models/base.py`. To learn more about initialising and simulating these models, see the [quickstart tutorial](quickstart.md).

## pySODM.models.base

###  *class* ODE

**Inherits from the user's model class:**

* **states** (list) - Names of the model's states.

* **parameters** (list) - Names of the model's parameters. Parameters are not subject to input checks and can have any type.

* **integrate** (function) - Function computing the differentials of every model state. The integrate function must have the timestep `t` as its first input, followed by the names of the model's states, parameters and stratified parameters (order not important). The integrate function must return a differential for every model state of the correct shape and **in the same order as the names `states`**. The integrate function must be a static method (include the decorator `@staticmethod`).

* **dimensions** (list) - optional - Names of the model's dimensions. Number of dimensions determines the dimensionality of the model's states. 
    * No dimensions: states are 0-D (scalar)
    * 1 dimension: states are 1-D (np.ndarray)
    * 2 dimensions: states are 2-D (np.ndarray)
    * etc. 

* **stratified_parameters** (list) - optional - Names of the *stratified* parameters. Their use is optional and mainly serves as a way for the user to structure his code.
    * 0 dimensions: not possible to have *stratified* parameters
    * 1 dimension: list containing strings - `['stratpar_1', 'stratpar_2']`
    * 2+ dimensions: List contains *n* sublists, where *n* is the number of model dimensions. Each sublist contains names of stratified parameters associated with the dimension in the corresponding position in `dimensions` - example for 3 dimensions model: `[['stratpar_1', 'stratpar_2'],[],['stratpar_3']]`, first dimension in `dimensions` has two stratified parameters `stratpar_1` and `stratpar_2`, second dimension has no stratified parameters, third dimensions has one stratified parameter `stratpar_3`.

* **dimensions_per_state** (list) - optional - Specify the dimensions of each model states. Allows users to define models with states of different sizes. If `dimensions_per_state` is not provided, all model states will have the same number of dimensions, equal to the number of model dimensions specified using `dimensions`. If specified, `dimensions_per_state` must contain *n* sublists, where *n* is the number of model states (`n = len(states)`). If a model state has no dimensions (i.e. it is a float), specify an empty list.

Below is a minimal example of a user-defined model class inheriting `ODE`.

```python
# Import the ODE class
from pySODM.models.base import ODE

# Define the model equations
class MY_MODEL(ODE):

    states = ['Y1', 'Y2']
    parameters = ['alpha']

    @staticmethod
    def integrate(t, Y1, Y2, alpha):
        return -alpha*Y1, alpha*Y2
```

To intialize this model, the following arguments must be provided.

**Parameters:**

* **initial_states** (dict or callable) - Initial condition. If a dictionary. Keys: names of model states. Values: initial values of model states. The dictionary does not need to contain all model states, missing states are filled with zeros upon initialization. If a callable (*initial condition function*): a function that creates such a dictionary.

* **parameters** (dict) - Model parameters. Keys: names of parameters. Values: values of parameters. A key,value pair must be provided for all parameter names listed in `parameters` and `stratified_parameters` of the model declaration. If *time dependent parameter functions* with additional parameters are used, these parameters must be included as well. If an *initial condition function* with parameters is used, these parameters must be included as well.

* **coordinates** (dict) - optional - Keys: names of dimensions (`dimensions`). Values: coordinates of the dimension. 

* **time_dependent_parameters** (dict) - optional - Keys: name of the model parameter the time-dependent parameter function should be applied to. Must be a valid model parameter. Values: time-dependent parameter function (callable function). Time-dependent parameter functions must have `t` (timestep/data), `states` (dictionary of model states at time `t`) and `params` (model parameters dictionary) as the first three arguments.

For our example,

```python
model = MY_MODEL(initial_states={'Y1': 1000, 'Y2': 0}, parameters={'alpha': 1})
```

Or using an *initial condition function*,

```python
def initial_condition_function(Y1_0):
    return {'Y1': Y1_0, 'Y2': 0}

model = MY_MODEL(initial_states=initial_condition_function, parameters={'alpha': 1, 'Y1_0': 1000})
```
The parameters of the initial condition function become a part of the model's parameters and can therefore be optimised. 

**Class methods:**

* **sim(time, N=1, draw_function=None, draw_function_kwargs={}, processes=None, method='RK23', rtol=1e-3, tau=None, time_unit='D')**

    Integrate a pySODM model using `scipy.integrate.solve_ivp()`. Can optionally perform `N` repeated simulations. Can change the values of model parameters at every consecutive simulation by manipulating the dictionary of model parameters `parameters` using a `draw_function`.

    **Parameters**

    * **time** - (int/float or list) - The start and stop "time" or "date" of the integration. Three possible inputs: 1) an int/float denoting the end of the integration, 2) a list of int/float `[start_time, stop_time]`, 3) a list of dates (type: `datetime` or a 'YYYY-MM-DD' string representation thereof) `[start_date, stop_date]`.

    * **N** - (int) - optional - Number of consecutive simulations to perform.

    * **draw_function** - (function) - optional - A function altering the model parameters dictionary `parameters` between consecutive simulations. Function must have `parameters` as its first input, followed by an arbitrary number of additional parameters. Function must return the model parameters dictionary `parameters` with its keys unaltered, meaning no parameters should be added or removed by a *draw function*.

    * **draw_function_kwargs** - (dict) - optional - Dictionary containing the parameters of the *draw function*, excluding `parameters`. Keys: names of additional parameters *draw function*, values: values of additional parameters *draw function*. Not subject to input checks regarding data type.  

    * **processes** - (int) - optional - Number of cores to use when {math}`N > 1`.

    * **method** - (str) - optional - Integration methods of `scipy.integrate.solve_ivp()`, [click to consult scipy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

    * **rtol** - (int/float) - optional - Relative tolerance of `scipy.integrate.solve_ivp()`, [click to consult scipy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

    * **tau** - (int/float) - optional - If `tau != None`, the model is simulated using a simple discrete timestepper with fixed timestep `tau`. Overrides the `method` and `rtol` arguments. 

    * **time_unit** - (str) - optional - Defines the unit of time. Valid options are: 'us' , 'ms' , 's', 'min', 'h', 'D', 'W'. Only relevant when using dates to index simulations.

    **Returns**

    * **out** - ([xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html#xarray.Dataset)) - Simulation output. `xarray.Dataset.data_vars` return the model's states. `xarray.Dataset.dimensions` returns the temporal dimension plus the model's dimensions. The time dimension is named `time` if timesteps were numbers, the time dimension is named `date` if timesteps were dates. When {math}`N > 1` an additional dimension `draws` is added to accomodate the consecutive simulations.

### *class* JumpProcess

**Inherits from the user's model class:**

* **states** (list) - Names of the model's states.

* **parameters** (list) - Names of the model's parameters. Parameters are not subject to input checks and can have any type.

* **compute_rates** (function) - Function returning the rates of transitioning between the model states. `compute_rates()` must have the timestep `t` as its first input, followed by the names `states`, `parameters` and `stratified_parameters` (their order is not important). `compute_rates()` must be a static method (include `@staticmethod`). The output of `compute_rates()` must be a dictionary. Its keys must be valid model states, a rate is only needed for the states undergoing a transitioning. The corresponding values must be a list containing the rates of the possible transitionings of the state. In this way, a model state can have multiple transitionings.

* **apply_transitionings** (function) - Function to update the states with the number of drawn transitionings. `apply_transitionings()` must have the timestep `t` as its first input, followed by the solver timestep `tau`, follwed by the dictionary containing the transitionings `transitionings`, then followed by the model states and parameters similarily to `compute_rates()`.

* **dimensions** (list) - optional - Names of the model's dimensions. Number of dimensions determines the dimensionality of the model's states. 
    * No dimensions: states are 0-D (scalar)
    * 1 dimension: states are 1-D (np.ndarray)
    * 2 dimensions: states are 2-D (np.ndarray)
    * etc. 

* **stratified_parameters** (list) - optional - Names of the *stratified* parameters. Their use is optional and mainly serves as a way for the user to structure his code.
    * 0 dimensions: not possible to have *stratified* parameters
    * 1 dimension: list containing strings - `['stratpar_1', 'stratpar_2']`
    * 2+ dimensions: List contains *n* sublists, where *n* is the number of model dimensions. Each sublist contains names of stratified parameters associated with the dimension in the corresponding position in `dimensions` - example for 3 dimensions model: `[['stratpar_1', 'stratpar_2'],[],['stratpar_3']]`, first dimension in `dimensions` has two stratified parameters `stratpar_1` and `stratpar_2`, second dimension has no stratified parameters, third dimensions has one stratified parameter `stratpar_3`.
    
* **dimensions_per_state** (list) - optional - Specify the dimensions of each model states. Allows users to define models with states of different sizes. If `dimensions_per_state` is not provided, all model states will have the same number of dimensions, equal to the number of model dimensions specified using `dimensions`. If specified, `dimensions_per_state` must contain *n* sublists, where *n* is the number of model states (`n = len(states)`). If a model state has no dimensions (i.e. it is a float), specify an empty list.

Below is a minimal example of a user-defined model class inheriting `JumpProcesses`.

```python
# Import the JumpProcesses class
from pySODM.models.base import JumpProcesses

class MY_MODEL(JumpProcesses):

    states = ['Y1', 'Y2']
    parameters = ['alpha']

    # define the rates of the system's transitionings
    @staticmethod
    def compute_rates(t, Y1, Y2, alpha):
        return {'Y1': [alpha, ]}

    # apply the sampled number of transitionings
    @staticmethod
    def apply_transitionings(t, tau, transitionings, Y1, Y2, alpha):

        Y1_new = Y1 - transitionings['Y1'][0]
        Y2_new = Y2 + transitionings['Y2'][0]

        return Y1_new, Y2_new
```

To intialize the user-defined model class, the following arguments must be provided.

**Parameters:**

* **initial_states** (dict or callable) - Initial condition. If a dictionary. Keys: names of model states. Values: initial values of model states. The dictionary does not need to contain all model states, missing states are filled with zeros upon initialization. If a callable (*initial condition function*): a function that creates such a dictionary.

* **parameters** (dict) - Model parameters. Keys: names of parameters. Values: values of parameters. A key,value pair must be provided for all parameter names listed in `parameters` and `stratified_parameters` of the model declaration. If *time dependent parameter functions* with additional parameters are used, these parameters must be included as well. 

* **coordinates** (dict) - optional - Keys: names of dimensions (`dimensions`). Values: coordinates of the dimension. 

* **time_dependent_parameters** (dict) - optional - Keys: name of the model parameter the time-dependent parameter function should be applied to. Must be a valid model parameter. Values: time-dependent parameter function (callable function). Time-dependent parameter functions must have `t` (timestep/data), `states` (dictionary of model states at time `t`) and `params` (model parameters dictionary) as the first three arguments.

For our example,

```python
model = MY_MODEL(initial_states={'Y1': 1000, 'Y2': 0}, parameters={'alpha': 1})
```

Or using an *initial condition function*,

```python
def initial_condition_function(Y1_0):
    return {'Y1': Y1_0, 'Y2': 0}

model = MY_MODEL(initial_states=initial_condition_function, parameters={'alpha': 1, 'Y1_0': 1000})
```
The parameters of the initial condition function become a part of the model's parameters and can therefore be optimised. 

**Methods:**

* **sim(time, N=1, draw_function=None, draw_function_kwargs={}, processes=None, method='tau_leap', tau=0.5, time_unit='D')**

    Integrate a model stochastically using Gillespie's Stochastic Simulation Algorithm (SSA) or the approximate Tau-leaping method. Can optionally perform `N` repeated simulations. Can change the values of model parameters at every consecutive simulation by manipulating the dictionary of model parameters `parameters` using a `draw_function`.

    **Parameters**

    * **time** - (int/float or list) - The start and stop "time" or "date" of the integration. Three possible inputs: 1) an int/float denoting the end of the integration, 2) a list of int/float `[start_time, stop_time]`, 3) a list of dates (type: `datetime` or a 'YYYY-MM-DD' string representation thereof) `[start_date, stop_date]`.

    * **N** - (int) - optional - Number of consecutive simulations to perform.

    * **draw_function** - (function) - optional - A function altering the model parameters dictionary `parameters` between consecutive simulations. Function must have `parameters` as its first input, followed by an arbitrary number of additional parameters. Function must return the model parameters dictionary `parameters` with its keys unaltered, meaning no parameters should be added or removed by a *draw function*.

    * **draw_function_kwargs** - (dict) - optional - Dictionary containing the parameters of the *draw function*, excluding `parameters`. Keys: names of additional parameters *draw function*, values: values of additional parameters *draw function*. Not subject to input checks regarding data type.  

    * **processes** - (int) - optional - Number of cores to use when {math}`N > 1`.

    * **method** - (str) - optional - 'SSA': Stochastic Simulation Algorithm. 'tau-leap': Tau-leaping algorithm. Consult the [following blog](https://lewiscoleblog.com/gillespie-algorithm) for more background information.
    * **tau** - (int/float) - optional - Leap value of the tau-leaping method. Consult the following [blog](https://lewiscoleblog.com/gillespie-algorithm) for more background information.

    * **time_unit** - (str) - optional - Defines the unit of time. Valid options are: 'us' , 'ms' , 's', 'min', 'h', 'D', 'W'. Only relevant when using dates to index simulations.

    **Returns**

    * **out** - ([xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html#xarray.Dataset)) - Simulation output. `xarray.Dataset.data_vars` return the model's states. `xarray.Dataset.dimensions` returns the temporal dimension plus the model's dimensions. The time dimension is named `time` if timesteps were numbers, the time dimension is named `date` if timesteps were dates. When {math}`N > 1` an additional dimension `draws` is added to accomodate the consecutive simulations.
