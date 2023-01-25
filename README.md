## pySODM

*Simulating and Optimising Dynamical Models in Python 3*

![build](https://github.com/twallema/pySODM/.github/workflows/tests.yml/badge.svg)
![docs](https://github.com/twallema/pySODM/.github/workflows/deploy.yml/badge.svg)
[![HitCount](https://hits.dwyl.com/twallema/pySODM.svg)](https://hits.dwyl.com/twallema/pySODM)

https://twallema.github.io/pySODM

### Aim & Scope

A modeling and simulation workflow will typically constitute the following steps,
1. Build a system dynamical model to describe some real-world phenomenon.
2. Calibrate the model to a set of experimental data.
3. Extract knowledge from the calibrated parameter values. Perform additional sensitivity analyses to finetune the model structure.
4. Use the model to make projections outside the calibrated range.

The goal of pySODM is to reduce the time it takes to step through this workflow. pySODM allows users to build models with n-dimensional labeled states and calibrate them to n-dimensional datasets by automatically aligning data. It is not a high-level simulation interface, it is a mid-level interface leaving its users the freedom to build arbitrarily complex models. 

### Overview of features

| Workflow                     | Features                                                                                                                        |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Building a model     | Solve coupled systems of equations deterministically (integration) or stochastically (Gillespie's SSA and Tau-Leaping)                  |
|                              | n-Dimensional model states with coordinates. Different states can have different sizes.                                         |
|                              | Easy indexing, manipulating, saving, and piping to third-party software of model output by formatting simulation output as `xarray.Dataset` |
| Simulating a model   | Vary model parameters during the simulation in accordance with an arbitrarily complex function containing any input                     |
|                              | Use *draw functions* to perform repeated simulations for sensitivity analysis. With multiprocessing support                     |
| Calibrating a model  | Calibration of n-dimensional model parameters                                                                                           |
|                              | Construct a posterior probability function                                                                                      |
|                              | Automatic alignment of data and model prediction over timesteps and dimensions                                                  |
|                              | Nelder-Mead Simplex and Particle Swarm Optimization for frequentist optimization                                                |
|                              | Pipeline to and backend for `emcee.EnsembleSampler` to perform Bayesian inference of model parameters                           |
|                              | Analysis of the mean-variance ratio in count-based datasets to aid in the choice of an appropriate likelihood function          |

### Getting started

The [quistart tutorial](quickstart.md) will teach you the basics of building and simulating models with n-dimensional labeled states in pySODM. It will demonstrate how to vary model parameters over the course of a simulation and how to perform repeated simulations with sampling of model parameters.

The [workflow](worfklow.md) tutorial provides a step-by-step introduction to a modeling and simulation workflow by inferring the distributions of the model parameters of a simple compartmental disease model using a synthetic dataset. 

The [enzyme kinetics](enzyme_kinetics.md) and [influenza 17-18](influenza_1718.md) case studies apply the modeling and simulation workflow to more advanced, real-world problems. In the enzyme kinetics case study, a 1D packed-bed reactor model is implemented in pySODM by reducing the two PDEs to a set of coupled ODEs by using the method-of-lines. In the Influenza 17-18 case study, a stochastic, age-structured model for influenza is developped and calibrated to the Influenza incidence data reported by the Belgian Federal Institute of Public Health. These case studies mainly serve to demonstrate pySODM's capabilities across scientific disciplines and highlight the arbitrarily complex nature of the models that can be built with pySODM. For an academic description of pySODM and on the Enzyme Kinetics and Influenza 17-18 case studies, checkout our manuscript (*coming soon*).

### Versions

- Version 0.2.0 (2023-01-19, PR #25)
    > Introduction of `state_dimensions` in model declaration, allowing the user to define models where states can have different sizes.
- Version 0.1 (2022-12-23, PR #14)
    > Application pySODM to three use cases. Documentation website. Unit tests for ODEModel, SDEModel and calibration. 
    - Version 0.1.1 (2023-01-09, PR #20)
        > Start of semantic versions: Major.Minor.Patch
    - Version 0.1.2 (2023-01-11, PR #23)
        > Calibration of 1-D model parameters generalized to n-dimensions.
        > Added 'aggregation functions' to the `log_posterior_probability` class to perform custom aggregations of model output before matching with data.
        > `xarray.DataArray`/`xarray.Dataset` can be used as datasets during calibration. Internally converted to `pd.DataFrame`.
- Version 0.0 (2022-11-14)
    - First pySODM version. Obtained by splitting the generally applicable parts from the ad-hoc parts in UGentBiomath/COVID19-Model. Without documentation website. 
- Pre-development (2020-05-01 - 2022-11-24)
    - Code developped to model the spread of SARS-CoV-2 in Belgium (UGentBiomath/COVID19-Model).