# pySODM
*Simulating and Optimising Dynamical Models in Python 3*

## Documentation website

https://twallema.github.io/pySODM/index.html

## Aim & Scope

The goal of pySODM is to simplify building a dynamic model with ordinary differential equations (ODE) or stochastic difference equations (SDE) and calibrating them to data using frequentist or bayesian methods, as these constitute a typical workflow in academic modeling & simulation projects. pySODM does not aim to provide novice modelers with a high-level interface for building systems of ODEs or SDEs, such as [pySD](https://pysd.readthedocs.io/en/master/) or [BPTK-Py](https://bptk.transentis.com/en/latest/). Rather, it aims to provide modelers with a set of DRY building blocks to build arbitrarily complex models. An example would be a compartmental epidemiological model for COVID-19 where the number of social contacts at each timestep *t* is computed using a neural network. In a typical modeling & simulation project, some parts of the code will be serve as generic building blocks while others will contain ad-hoc workflows (think datasets and their conversions, notebooks for exploration). The foundations of pySODM were implemented by Stijn Van Hoey and Joris Van Den Bossche in May 2020. Their code was used and modified by Tijs Alleman, Jenna Vergeynst and Michiel Rollier to build compartmental models for COVID-19 in Belgium. pySODM is the distillate of this collaboration.

## What can pySODM do?

1) Build a dynamic system model.
- Use ordinary differential equations (ODE) and solve them using `scipy.integrate.solve_ivp()`
- Use a stochastic difference equations (SDE) and sovle them using SSA or Tau-Leaping
- No high-level interface. The user is responsible for the contents of the integration function (timestep, states, parameters in; differentials out). `pySODM` does provide input checks on the sizes and data types of model states and parameters.
- *Stratify* or *vectorize* model states easily in *n* dimensions

2) Simulate the model.
- Model output is stored using `xarray`. Every *stratification* can be given a name and coordinates to ease handling of simulation output.
- Model parameters can be varied over time using arbitrarily complex functions
- Perform repeated simulations with sampling of model parameters and support for `multiprocessing`

3) Calibrate the model to data.
- Provides building blocks to construct a log likelihood function for optimization
- Supports matching model states to multiple datasets over stratifications and with mismatching timesteps
- Provides a pure-Python implementation of Particle Swarm Optimization and Nelder-Mead Optimization, modified to support `multiprocessing`
- Interfaces to `emcee` for Markov-Chain Monte-Carlo sampling

Additionally, we recommend the use of [SAlib](https://salib.readthedocs.io/en/latest/) to perform global sensitivity analysis.