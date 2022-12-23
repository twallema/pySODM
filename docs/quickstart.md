# Quickstart

This quickstart example is (partially) drafted from the [Modeling and Simulation Workflow](workflow.md) tutorial.

## Set up a simple SIR disease model

Load pySODM's [`ODEModel` class](models.md) and define your model. pySODM can also be used to solve models using Gillespie's SSA or Tau-Leaping method, checkout the [Influenza 17-18 tutorial](influenza_1718.md).

```python
# Import the ODEModel class
from pySODM.models.base import ODEModel

# Define the model equations
class SIR(ODEModel):
    """
    Simple SIR model without stratifications
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

Provide a dictionary containing the initial values of the model states and a dictionary containing all parameters to initialize the model.

```python
model = SIR(states=init_states={'S': 1000, 'I': 1}, parameters={'beta': 0.35, 'gamma': 5})
```

Simulate the model using the `sim()` method. The solver method and tolerance, as well as the frequency of the time axis output can be tweaked. pySODM supports handling of `datetime`.

```python
out = model.sim(121)
```

Results are sent to an `xarray.Dataset`. 
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

## Stratifications: add age groups to the SIR model

To transform our SIR model's states in 1D vectors, referred to as *stratification* throughout the documentation, add the `stratification_names` keyword to the model declaration.

```python
# Import the ODEModel class
from pySODM.models.base import ODEModel

# Define the model equations
class stratified_SIR(ODEModel):
    """
    Simple SIR model with age groups
    """
    
    state_names = ['S','I','R']
    parameter_names = ['beta']
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

As before, provide a dictionary containing the initial values of the model states and a dictionary containing all parameters to initialize the model. Additionally provide a dictionary containing coordinates for every stratification declared previously.

```python
model = stratified_SIR(states=init_states={'S': 1000, 'I': 1},
                       parameters={'beta': 0.35, 'gamma': 5},
                       coordinates={'age_groups': ['0-5','5-15','15-65','65-120']})
out = model.sim(121)
print(out)                       
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

### Time-dependent parameter function

## Calibrate a model

We refer to the [Modeling and Simulation Workflow](workflow.md) tutorial.