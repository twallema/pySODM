# Quickstart

## Set up a simple SIR disease model

Load pySODM's [`ODEModel` class](models.md) and define your model. pySODM can also be used to solve models using Gillespie's SSA or Tau-Leaping method, for an example, checkout the [Influenza 17-18 tutorial](influenza_1718.md). 

```python
# Import the ODEModel class
from pySODM.models.base import ODEModel

# Define the model equations
class SIR(ODEModel):
    """
    Simple SIR model
    """
    
    state_names = ['S','I','R']
    parameter_names = ['beta','gamma']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma):
        
        # Calculate total population
        N = S+I+R

        # Calculate differentials
        dS = -beta*S*I/N
        dI = beta*S*I/N - 1/gamma*I
        dR = 1/gamma*I

        return dS, dI, dR
```

To initialize the model, provide a dictionary containing the initial values of the model states and a dictionary containing all model parameters. Undefined initial states are automatically filled with zeros.

```python
model = SIR(states={'S': 1000, 'I': 1}, parameters={'beta': 0.35, 'gamma': 5})
```

Simulate the model using the `sim()` method. The solver method and tolerance, as well as the timesteps in the output can be tailored. pySODM supports the use of `datetime` types as timesteps.

```python
out = model.sim(121)
```

Results are sent to an `xarray.Dataset`, for more information on indexing and selecting data, [see](https://docs.xarray.dev/en/stable/user-guide/indexing.html).
```bash
<xarray.Dataset>
Dimensions:  (time: 122)
Coordinates:
  * time     (time) int64 0 1 2 3 4 5 6 7 8 ... 114 115 116 117 118 119 120 121
Data variables:
    S        (time) float64 1e+03 999.6 999.2 998.7 ... 287.2 287.2 287.2 287.2
    I        (time) float64 1.0 1.161 1.348 1.565 ... 0.1455 0.1316 0.1192
    R        (time) float64 0.0 0.2157 0.4662 0.7569 ... 713.6 713.7 713.7 713.7


```

## Stratifications: adding age groups to the SIR model

To transform our SIR model's states in 1D vectors, referred to as a *stratification* throughout the documentation, add the `stratification_names` keyword to the model declaration.

```python
# Import the ODEModel class
from pySODM.models.base import ODEModel

# Define the model equations
class stratified_SIR(ODEModel):
    """
    Simple SIR model with age groups
    """
    
    state_names = ['S','I','R']
    parameter_names = ['beta', 'gamma]
    stratification_names = ['age_groups']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma):
        
        # Calculate total population
        N = S+I+R

        # Calculate differentials
        dS = -beta*S*I/N
        dI = beta*S*I/N - 1/gamma*I
        dR = 1/gamma*I

        return dS, dI, dR
```

When initializing the model, additionally provide a dictionary containing coordinates for every stratification declared previously.

```python
model = stratified_SIR(states={'S': 1000*np.ones(4), 'I': np.ones(4)},
                       parameters={'beta': 0.35, 'gamma': 5},
                       coordinates={'age_groups': ['0-5','5-15', '15-65','65-120']})
out = model.sim(121)
print(out)                       
```

The dimension `'age_groups'` with coordinates `['0-5','5-15', '15-65','65-120']` is automatically added to the model output.
```bash
<xarray.Dataset>
Dimensions:     (time: 122, age_groups: 4)
Coordinates:
  * time        (time) int64 0 1 2 3 4 5 6 7 ... 114 115 116 117 118 119 120 121
  * age_groups  (age_groups) <U6 '0-5' '5-15' '15-65' '65-120'
Data variables:
    S           (age_groups, time) float64 1e+03 999.6 999.2 ... 287.2 287.2
    I           (age_groups, time) float64 1.0 1.161 1.348 ... 0.1316 0.1192
    R           (age_groups, time) float64 0.0 0.2157 0.4662 ... 713.7 713.7
```

## Advanced simulation features

### Draw function

The `sim()` method can be used to perform {math}`N` repeated simulations (keyword `N`). Additionally a *draw function* can be used to vary model parameters during every run, keywords `draw_function` and `samples`. A draw function to randomly draw `gamma` from a uniform simulation before every run is implemented as follows,

```python
def draw_function(param_dict, samples_dict):
    param_dict['gamma'] = np.random.uniform(low=1, high=5)
    return param_dict

out = model.sim(121, N=10, draw_function=draw_function, samples={})
print(out)   
```

An additional dimension `'draws'` has been added to the output to accomodate the results of the repeated simulations.

```bash
<xarray.Dataset>
Dimensions:     (time: 122, age_groups: 4, draws: 10)
Coordinates:
  * time        (time) int64 0 1 2 3 4 5 6 7 ... 114 115 116 117 118 119 120 121
  * age_groups  (age_groups) <U6 '0-5' '5-15' '15-65' '65-120'
Dimensions without coordinates: draws
Data variables:
    S           (draws, age_groups, time) float64 1e+03 999.6 ... 316.9 316.9
    I           (draws, age_groups, time) float64 1.0 1.007 ... 0.1644 0.1492
    R           (draws, age_groups, time) float64 0.0 0.3439 ... 684.0 684.0
```

### Time-dependent parameter function

Parameters can also be varied through the course of one simulation using a *time-dependent parameter function*, which can be an arbitrarily complex function. A generic time-dependent parameter function has the timestep, the dictionary of model states and the value of the varied parameter as its inputs. Additionally, the function can take any number of arguments.

```python
def vary_my_parameter(t, states, param, an_additional_parameter):
    """A function to vary a model parameter"""

    # Do some computation
    param = ...

    return param
```

All we need to do is use the `time_dependent_parameters` keyword to declare what parameter should be varied according to our function, and, the additional parameter of the function should be added to the parameters dictionary.

```python
model = SIR(states={'S': 1000, 'I': 1},
            parameters={'beta': 0.35, 'gamma': 5, 'an_additional_parameter': anything_you_want},
            time_dependent_parameters={'beta': vary_my_parameter})
```