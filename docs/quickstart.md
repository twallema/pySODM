## Quickstart

In this quickstart tutorial, we'll set up a simple (ODE) SIR disease model and calibrate its basic reproduction number to a synthetically generated dataset. We'll then asses what happens if we lower the pathogens infectivity. We'll use a simple model but cover the most important concepts and features of `pySODM`. Further, this quickstart tutorial serves as a template for a typical workflow:
1. Import dependencies
2. Load the dataset
3. Load/Define a model
4. Initialize the model
5. Perform a frequentist optimization
6. Perform a bayesian optimization
7. Visualize the result

The quickstart example can be reproduced using `~/tutorials/quickstart/quickstart.py`

### Import dependencies

We'll first import some of the most common dependencies and then import the pySODM code on the go. That way, the imports of the necessary pySODM code are located where they are required which is more illustrative than including them all here.

```
import numpy as np
import pandas as pd
from matplotlib.pyplot import plt
```

### Generating synthetic data

First, we'll generate a sythetic dataset of disease cases. We accomplish this by applying negative binomial noise on an exponentially growing curve with a doubling time of 10 days. We'll assume the first case was detected on December 1st, 2022 and data was collected on every weekday until December 21st, 2022. A dispersion factor `alpha=0.03` is used, which was representative for data collection during the COVID-19 pandemic in Belgium.

```
# Parameters
alpha = 0.03 # Overdispersion
t_d = 10 # Doubling time
# Sample data
dates = pd.date_range('2022-12-01','2023-01-21')
x = np.linspace(start=0, stop=len(dates)-1, num=len(dates))
y = np.random.negative_binomial(1/alpha, (1/alpha)/(np.exp(x*np.log(2)/td) + (1/alpha)))
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

We'll simulate an SIR model governed by the following set of [equations](https://medium.com/@shaliniharkar/sir-model-for-spread-of-disease-the-differential-equation-model-7e441e8636ab). The model has three states: 1) The number of individuals susceptible to the disease (S), 2) the number of infectious individuals (I), and 3) the number of removed individuals (R). The model has two parameters: 1) `beta`, the rate of transmission and, 2) `gamma`, the duration of infectiousness. 

To define the SIR model, first, the `ODEModel` class must be loaded from `~/src/models/base.by`. The `ODEModel` class is then passed on to our model class, which we conveniently name `ODE_SIR`. Inside our model class, we'll have to define the names of the model states and parameters, as well as an `integrate` function where the differentials are computed. There are some formatting requirements to the integrate function, which are checked upon model initialization,

1. The integrate function must be a static method (include `@staticmethod`)
2. The integrate function must have the timestep `t` as its first input
3. The timestep `t` is first followed by the states, then the parameters
4. The integrate function must return a differential for every model state

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

After defining our model, we'll initialize it by defining two dictionarys: 1) A dictionary containing the initial states of the model. 2) A dictionary containing the values of all model parameters. Note that we don't need to define the number of recovered individuals, undefined states are automatically filled with zeros.

```
model = ODE_SIR(states=init_states={'S': 1000, 'I': 1}, parameters={'beta': 0.35, 'gamma': 5})
```

Initializing the model starts a series of input checks on the initial states, parameters and the model class which should prevent the most common mistakes. We can then easily simulate the model using the `sim()` method of the `ODEModel` class. In the example below, dates are used as coordinates of the time axis.   

```
# Extract start- and enddate of dataset
start_date = d.index.min()
end_date = d.index.max()

# Simulate the model
out = model.sim([start_date, end_date+pd.Timedelta(days=100)])

# Visualize the result
fig,ax=plt.subplots(figsize=(12,3))
ax.plot(out['date'], out['S'], color='green', label='Susceptible')
ax.plot(out['date'], out['I'], color='red', label='Infectious')
ax.plot(out['date'], out['R'], color='black', label='Removed')
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.legend()
plt.show()
plt.close()
```

![SIR](/_static/figs/quickstart_SIR.png)

The `sim()` method can also be run with timesteps as coordinates of the time axis (int/float).
```
# Simulate from t=0 until t=100
out = model.sim([0, 100])
# Is equivalent to
out = model.sim(100)
```
By convention, the name of the time axis in the `xarray` output is equal to `date` when using dates and `time` when using timesteps. The user can acces the integration methods and relative tolerance of `scipy.solve_ivp()` by supplying the `method` and `rtol` arguments to the `sim()` function. The user can determine the output timestep frequency by changing the optional `output_timestep` argument.

### Calibrating the model

#### Setting up the objective function

Before commencing the optimization we must instruct our computer what deviations between the data and model prediction are allowed, and this is referred to as an objective function. We will attempt to maximize the posterior probability, which is the probability of our model's parameters in light of the data. It contrasts with the likelihood function, which is the probability of the data given the model's parameters. The two are related as follows by Bayes' theorem,

$$ p (\theta | data) = \frac{p(data | \theta) p(\theta)}{p(data)} $$

Here, p(data) is used for normalization and can be forgotten for all practical purposes. What is important is that the posterior probability $p(\theta | data)$ is proportional to the product of the Likelihood $p(data | \theta)$ times the parameter Prior probability $p(\theta)$. We'll actually attempt to maximize the logarithm of the posterior probability, which is equal to the sum of the logarithm of the prior probability and the logarithm of the likelihood. 

The choice of priors and likelihoods. 

To gain some further insights in bayesian inference with `emcee`, I highly recommend recommend checking out the following [emcee tutorial](https://emcee.readthedocs.io/en/stable/tutorials/line/). For an introduction to Bayesian inference, I recommend reading the following [article](https://towardsdatascience.com/a-gentle-introduction-to-bayesian-inference-6a7552e313cb).

The example below.

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