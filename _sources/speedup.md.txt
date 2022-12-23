## Speeding up models

### Quick-and-dirty: solver accuracy

When using ODE models, using a lower grade algorithm (RK23 < RK45 < DOP853) and/or a lower relative tolerance of the solver is the easiest way to achieve speedups. Be aware, the default solver settings of `RK23` with a tolerance of `5e-03` is already quite "quick-and-dirty". When using SDE models with the tau-leaping method, try increasing the tau-leap size to speed up your models.

### JIT compilation

Just-in-time compilation, made possible by [numba](https://numba.pydata.org/), can be used to speed up code. It is best applied to `integrate()` or time-dependent parameter functions. The amount of achievable speedup is different for every model. Generally speaking, models with for-loops in them, or models with large matrix computations will speed up quite nicely. Jit compiling the 1D PFR in the [enzyme kinetics](enzyme_kinetics.md) tutorial results in a 16-fold speedup, while jit compiling the PPBB enzyme kinetic model only speeds up the code by 6%.

### Avoid large inputs in time-dependent parameter functions

Using large inputs directly in a time-dependent parameter function will force pySODM to read that input at every timestep in the integration. As the integrator typically takes thousands of steps, the IO operations slow the code down drastically. It is thus recommended to avoid the following syntax:

```python
def TDPF(t, states, params, a_large_dataset):
    t = pd.Timestamp(t.date())
    return a_large_dataset.loc[t]
```

Instead, we recommend wrapping the TDPF in a class and using the `@lru_cache()` decorator to place the function where the large dataset is evaluated in working memory.

```python
from functools import lru_cache

class make_TDPF():

    # Assign large dataset to class
    def __init__(self, a_large_dataset):
        self.data = a_large_dataset

    # Place function where large dataset is used in memory
    @lru_cache()
    def __call__(self, t):
        t = pd.Timestamp(t.date())
        return self.data.loc[t]
    
    # Write a pySODM compatible wrapper 
    def wrapper_function(self, t, states, param):
        return self.__call__(t)

# The time-dependent parameter function used in the model initialization
TDPF = make_TDPF(a_large_dataset).wrapper_function
```

