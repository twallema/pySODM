# pySODM
*Simulating and Optimising Dynamical Models*

## Aim & Scope

The goal of pySODM is to simplify the time it takes to build a dynamic model with ordinary differential equations (ODE) or stochastic difference equations (SDE) and calibrate them to data using frequentist or bayesian methods, as we have found these to constitute a common workflow in academic modeling & simulation projects. pySODM does not aim to provide novice modelers with a high-level interface for building systems of ODEs or SDEs, examples are [pySD](https://pysd.readthedocs.io/en/master/) or [BPTK-Py](https://bptk.transentis.com/en/latest/). Rather, it aims to provide modelers with a set of DRY building blocks to build complex models. An example would be a compartmental epidemiological model where the number of contacts at each timestep *t* is computed using a neural network. In a typical modeling & simulation project, some parts of the code will be serve as generic building blocks while others will contain ad-hoc workflows (think datasets and their conversions, notebooks for exploration). The foundations of pySODM were implemented by Stijn Van Hoey and Joris Van Den Bossche in May 2020. Their code was used and modified by Tijs Alleman, Jenna Vergeynst and Michiel Rollier to build compartmental models for COVID-19 in Belgium. pySODM is the DRY distillate of this collaboration.

## What can pySODM do for you?

1) Build a dynamic system model.
- Use ordinary differential equations (ODE) solved using `scipy.integrate.solve_ivp()`
- Use a stochastic simulation algorithm (SSA or Tau-leaping)
- No high-level interface. The user is responsible for the contents of the integration function. `pySODM` provides input checks.
- *Stratify* or *vectorize* model states in *n* dimensions

2) Simulate the model.
- Model output is stored using `xarray`
- Model parameters can be varied over time using arbitrarily complex methods
- Repeated simulations with sampling of model parameters and `multiprocessing` support

3) Calibrate the model to data.
- Supports matching model states to multiple datasets
- Provides building blocks to construct a log likelihood function
- Automatically matches timesteps and stratifications
- Provides a pure-Python implementation of Particle Swarm Optimization and Nelder-Mead Optimization, modified to support `multiprocessing`
- Interfaces to `emcee` for Markov-Chain Monte-Carlo sampling

Additionally, we recommend the use of [SAlib](https://salib.readthedocs.io/en/latest/) to perform global sensitivity analysis.

## Documentation and tutorials

Insert link to documentation website.