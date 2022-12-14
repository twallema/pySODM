## pySODM: Simulating and Optimising Dynamical Models in Python 3

### Aim & Scope

Over the past four years I've been engaged in several academic modeling & simulation projects in the fields of chemical engineering and epidemiology. I've found these projects typically constitute the following steps:

1. Build a system dynamical model to describe some real-world phenomenon.
2. Calibrate the model to a set of experimental data.
3. Extract knowledge from the calibrated parameter values. Perform additional sensitivity analyses to finetune the model structure.
4. Use the model to make projections outside the calibrated range.

The goal of pySODM is to reduce the time it takes to step through this workflow by coupling the necessary low-level packages for solving sets of differential equations (`scipy.solve_ivp()`), and for performing optimizations (`emcee`). pySODM exploits `xarray` Datasets to function as the glue between the aforementioned packages by uniformly formatting model output. Additional time is saved because basic simulation workflows, such as varying model parameters during the simulation or sampling model parameters from distributions, and basic optimization workflows, such as constructing an objective function, were implemented in an general way. Such features may seem easy to implement, but their complexity often blows up in the face of academic realities, such as,
- I want my models to have n-dimensional states, where each axis in the output represents some property. For instance, having age groups or spatial patches in an SIR model.
- I want to calibrate my model against high-dimensional data. For instance, disease case data are available on each day and for each age group.
- I want a model parameter to be updated in accordance with some very large dataset on each simulation timestep. For instance, administering vaccines in an SIR model.
- I want to calibrate a mixture of model parameters, some being floats, others being vectors to a dataset.
- etc ...

We don't want to have to program these case-specific pipelines every time we build a model. pySODMs added value lies wholy in making the typical modeling & simulation workflow faster. pySODM does not contain implementations of novel solution techniques for ODEs or SDEs, novel optimization algorithms, or novel sensitivity analysis. pySODM does not aim to provide novice modelers with a high-level interface for building systems of ODEs or SDEs, such as [pySD](https://pysd.readthedocs.io/en/master/) or [BPTK-Py](https://bptk.transentis.com/en/latest/). pySODM also does not include the tools to perform sensitivity analysis (step 3 of our workflow), however, coupling to [SAlib](https://salib.readthedocs.io/en/latest/) is straightforward.

The foundations of pySODM were implemented by Stijn Van Hoey and Joris Van Den Bossche in May 2020. Their code was used and modified by Tijs Alleman, Jenna Vergeynst and Michiel Rollier to build compartmental models for COVID-19 in Belgium (checkout the [references](references.md)). pySODM is the distillate of this collaboration. My motivation for spending the extra time to fully develop pySODM was to make future tutoring of students easier. I hope others may find pySODM as usefull as I do.  

### Overview of features

| Workflow                     | Features                                                                                                                        |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Building a dynamic model     | ODEs and Gillespie (SSA, tau-leaping)                                                                                           |
|                              | Stratification in n-dimensional states                                                                                          |
|                              | Output handling in xarray (supports int/float as well as datetime timesteps, assignment of coordinates to stratifications)      |
|                              | 2D model states (experimental; ODE only)                                                                                        |
| Simulating a dynamic model   | Time-dependent variation of model parameters using complex functions and large datasets                                         |
|                              | Simulating models with parameters drawn from distributions for sensitivity/scenario analysis                                    |
|                              | Supports multiprocessing                                                                                                        |
| Calibrating a dynamic model  | Toolbox to set up log likelihood functions: correct alignment of data and model prediction, calibration of vector parameters, analysis of mean-variance realtionship, etc.    |
|                              | Nelder-Mead Optimization and Particle Swarm Optimization with multiprocessing support                                           |
|                              | Pipeline to `emcee` for Bayesian Inference of model parameters                                                                  |

### Roadmap

The following features will be implemented in future versions of pySODM,

- Coupling of ODE Models with different stratifications. The user will be able to define a model, consisting of two submodels with states of different sizes. These two models will share one `integrate` function to make coupling of the differentials possible. Output will be returned in seperate `xarray` Datasets. High priority.

- Parameter with a double stratification and the calibration of n-dimensional parameters by flattening. Low priority.