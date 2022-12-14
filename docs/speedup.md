## Speeding up models

### Quick-and-dirty

When using ODE models, playing around with the solution methods and relative tolerance of the solver is the easiest way to achieve speedups. Be aware, the default setting of `RK23` with a tolerance of `5e-03` is already quite "quick-and-dirty". When using SDE models with the tau-leaping method, try increasing the tau-leap size to speed up your models.

### JIT compilation

Just-in-time compilation, available through [numba](https://numba.pydata.org/) can be used to speed code up by means of precompilation. JIT compilation is typically applied to the `integrate` (`compute_rates` for SDE models) or time-dependent parameter functions. The amount of achievable speedup is different for every model. For our COVID-19 models, we  

Jit compiling the 1D PFR in the [enzyme kinetics](enzyme_kinetics.md) tutorial speeds up the code ... fold, while jit compiling the simple [SIR tutorial](workflow.md) speeds up the code ... fold.

### Avoiding large inputs in time-dependent parameter functions

Using large inputs directly in a time-dependent parameter function will force pySODM to read that input at every timestep in the integration. As the integrator typically takes thousands of steps, this slows the code down drastically. It is thus recommended to avoid the following syntax:

```python
def TDPF(t, states, params, a_large_dataset):
    t = pd.Timestamp(t.date())
    return a_large_dataset.loc[t]
```

Instead, we recommend wrapping the TDPF in a class and using the `@lru_cache()` decorator to place the function where the large dataset is evaluated in working memory.

```python
class make_TDPF():

    def __init__(self, a_large_dataset):
        self.data = a_large_dataset

    @lru_cache()
    def __call__(self, t):
        t = pd.Timestamp(t.date())
        return self.data.loc[t]
    
    def wrapper_function(self, t, states, param):
        return self.__call__(t)

# The time-dependent parameter function used in the model initialization
TDPF = make_TDPF(a_large_dataset).wrapper_function
```

