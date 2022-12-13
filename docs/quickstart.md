## Quickstart

In this quickstart tutorial, we'll set up a simple (ODE) SIR disease model and calibrate its basic reproduction number to a synthetically generated dataset. We'll then asses what happens if the pathogen's infectivity is lowered. We'll use a simple model but cover the most important concepts and features of `pySODM`, you'll learn about building more complex models in other tutorials (REF). This quickstart tutorial serves as a template for a typical workflow:
1. Import dependencies
2. Load the dataset
3. Load/Define a model
4. Initialize the model
5. Perform a frequentist optimization
6. Perform a bayesian optimization
7. Visualize the result

The quickstart example can be reproduced using `~/tutorials/quickstart/quickstart.py`

### Import dependencies

I typically place all my dependencies together at the top of my script. However, for this demo, we'll import some common dependencies here and then import the pySODM code on the go. That way, the imports of the necessary pySODM code are located where they are required which is more illustrative than including them all here.

```
import numpy as np
import pandas as pd
from matplotlib.pyplot import plt
```

### Generating synthetic data

First, we'll generate a sythetic dataset of disease cases. We'll accomplish this by assuming the disease is generating cases exponentially with a doubling time of 10 days. Mathematically,

```{math}
n_{cases} = e^{t * \frac{log(2)}{t_d}}
```

We'll assume the first case was detected on December 1st, 2022 and data was collected on every weekday until December 21st, 2022. Then, we'll add observational noise to the synthetic data. For count based data, observational noise is typically the result of a poisson or negative binomial proces, depending on the occurence of overdispersion. For a poisson proces, the variance is equal to the mean: {math}`\sigma^2 = \mu`, while for a negative binomial proces the mean-variance relationship is quadratic: {math}`\sigma^2 = \mu + \alpha \mu^2`. For this example we'll assume the data are overdispersed so we'll resample the data using the negative binomial distribution. A dispersion factor `alpha=0.03` is used, which was representative for data collection during the COVID-19 pandemic in Belgium.

```
# Parameters
alpha = 0.03 # Overdispersion
t_d = 10 # Doubling time
# Sample data
dates = pd.date_range('2022-12-01','2023-01-21')
t = np.linspace(start=0, stop=len(dates)-1, num=len(dates))
y = np.random.negative_binomial(1/alpha, (1/alpha)/(np.exp(t*np.log(2)/td) + (1/alpha)))
# Place in a pd.Series
d = pd.Series(index=dates, data=y, name='CASES')
# Data collection only on weekdays only
d = d[d.index.dayofweek < 5]
```

Datasets used in an optimization must always be pandas Series or DataFrames. In the dataset, an index level named `time` (if the time axis consists of int/float) or `date` (if the time axis consists of dates) must always be present. In this tutorial, we'll use dates and thus name the index of our dataset `date`.  

```
# Index name must be data for calibration to work
d.index.name = 'date'
```

![synethetic_dataset](/_static/figs/quickstart_synthetic_dataset.png)

### Defining and initializing the model

In this demo we'll set up an SIR disease model governed by the following set of equations,

```{math}
\begin{eqnarray}
N &=& S + I + R, \\
\frac{dS}{dt} &=& - \beta S (I/N), \\
\frac{dI}{dt} &=& \beta S (I/N) - (1/\gamma)I, \\
\frac{dR}{dt} &=& (1/\gamma)I,
\end{eqnarray}
```
or schematically,

<img src="./_static/figs/quickstart_flowchart_SIR.png" width="400" />

The model has three states: 1) The number of individuals susceptible to the disease (S), 2) the number of infectious individuals (I), and 3) the number of removed individuals (R). The model has two parameters: 1) `beta`, the rate of transmission and, 2) `gamma`, the duration of infectiousness. 

To define the SIR model, first, the `ODEModel` class must be loaded from `~/src/models/base.by`. The `ODEModel` class is then passed on to our model class, `ODE_SIR`. Inside our model class, we'll have to define the names of the model states and parameters, as well as an `integrate()` function where the differentials are computed. There are some formatting requirements to the integrate function, which are checked upon model initialization,

1. The integrate function must have the timestep `t` as its first input
2. The timestep `t` is first followed by the states, then the parameters in the correct order
3. The integrate function must return a differential for every model state
4. The integrate function must be a static method (include `@staticmethod`)

```
# Import the ODEModel class
from pySODM.models.base import ODEModel

# Define the model equations
class ODE_SIR(ODEModel):
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

After defining our model, we'll initialize our model by supplying a dictionary of initial states and a dictionary of model parameters. In our example, we'll assume the disease spreads in a relatively small population of 1000 individuals (f.i. a university campus). At the start of the simulation we'll assume there is one "patient zero". We don't have to define the number of recovered individuals as undefined states are automatically set to zero by pySODM.

```
model = ODE_SIR(states=init_states={'S': 1000, 'I': 1}, parameters={'beta': 0.35, 'gamma': 5})
```

Initializing the model starts a series of input checks on the initial states, parameters and the model class which should prevent the most common mistakes during initialization. If you manage to break the code by slipping some input through the checks resulting in cryptic errors, please let me know by creating a [New issue](https://github.com/twallema/pySODM/issues/new/choose). 

After initializing the model, we can easily simulate the model using the `sim()` method of the `ODEModel` class. In the example below, dates are used as coordinates of the time axis.   

```
# Extract start- and enddate of dataset
start_date = d.index.min()
end_date = d.index.max()

# Simulate the model from the start of data collection to 100 days past data collection
out = model.sim([start_date, end_date+pd.Timedelta(days=100)])

# Visualize the result
fig,ax=plt.subplots(figsize=(12,4))
ax.plot(out['date'], out['S'], color='green', label='Susceptible')
ax.plot(out['date'], out['I'], color='red', label='Infectious')
ax.plot(out['date'], out['R'], color='black', label='Removed')
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.legend()
plt.show()
plt.close()
```

![SIR_date](/_static/figs/quickstart_SIR_date.png)

The `sim()` method can also be run with timesteps as coordinates of the time axis (int/float). By convention, the name of the time axis in the `xarray` output is equal to `date` when using dates and `time` when using timesteps.
```
# Simulate from t=0 until t=121
out = model.sim([0, 121])

# Is equivalent to:
out = model.sim(121)

# But now the time axis is named 'time'
fig,ax=plt.subplots(figsize=(12,4))
ax.plot(out['time'], out['S'], color='green', label='Susceptible')
ax.plot(out['time'], out['I'], color='red', label='Infectious')
ax.plot(out['time'], out['R'], color='black', label='Removed')
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.legend()
plt.show()
plt.close()
```

![SIR_time](/_static/figs/quickstart_SIR_time.png)

The user can acces the integration methods and relative tolerance of `scipy.solve_ivp()` by supplying the `method` and `rtol` arguments to the `sim()` function. The user can determine the output timestep frequency by changing the optional `output_timestep` argument. For more info, check out the docstring of the `sim()` function.

### Calibrating the model

#### The posterior probability 

Before we can have our computer find a set of model parameters that aligns the model with the data, we must instruct it what deviations between the data and model prediction are allowed, and this is referred to as an *objective function*. pySODM contains the necessary tools to setup an appropriate error function for optimization in its `log_posterior_probability` class (resides in `~/src/pySODM/optimization/objective_functions.py`).

In what follows we will set up and attempt to maximize the posterior probability {math}`p(\theta | data)`, which is is the probability of our model's parameters in light of the data. It contrasts with the likelihood function {math}`p(data | \theta)`, which is the probability of the data given the model's parameters. The two are related as follows by Bayes' theorem,

$$ p (\theta | data) = \frac{p(data | \theta) p(\theta)}{p(data)}. $$

Here, {math}`p(data)` is used for normalization and can be forgotten for all practical purposes. {math}`p(\theta)` is referred to as the prior probability of the model parameters and contains any prior beliefs about the probability density distribution of the parameters {math}`\theta`. Most of the times, uniform prior probabilities are used to constraint the parameter values to physical/likely regions of the parameter space. What is really important to remember is that the posterior probability {math}`p(\theta | data)` is proportional to the product of the likelihood {math}`p(data | \theta)` times the parameter prior probability {math}`p(\theta)`. Given that we'll actually maximize the logarithm of the posterior probability, the log posterior probability is computed as the sum of the logarithm of the prior probability and the logarithm of the likelihood. For an introduction to Bayesian inference, I recommend reading the following [article](https://towardsdatascience.com/a-gentle-introduction-to-bayesian-inference-6a7552e313cb).

#### Choosing an appropriate prior function

#### Choosing an appropriate likelihood function

### Setting up the posterior probability in pySODM

```
#########################
## Calibrate the model ##
#########################

if __name__ == '__main__':

    # The datasets, the model states to match, their weights, the log likelihood functions and log likelihood function arguments
    data=[d, ]
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha,]

    # Calibated parameters, their bounds and preferred labels
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]

    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model, pars, bounds, data, states,log_likelihood_fnc,
                                                   log_likelihood_fnc_args, weights, labels=labels)
```

#### Nelder-Mead optimization

```
if __name__ == '__main__':

    # Extract start- and enddate of dataset
    start_date = d.index.min()
    end_date = d.index.max()
    # Initial guess
    theta = np.array([0.50,])
    # Run Nelder-Mead optimisation
    theta = nelder_mead.optimize(objective_function, theta, [0.10,], processes=1, max_iter=10)[0]
```

```
if __name__ == '__main__':
    # Update beta with the calibrated value
    model.parameters.update({'beta': theta[0]})
    # Simulate the model
    out = model.sim([start_date, end_date])
    # Visualize result
    fig,ax=plt.subplots()
    ax.plot(out['date'], out['I'], color='red', label='Infectious')
    ax.scatter(d.index, d.values, color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2, label='data')
    ax.legend()
    plt.show()
    plt.close()
```

#### Bayesian optimization

#### Visualizing the result



### Simulating scenarios