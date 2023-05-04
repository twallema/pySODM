## pySODM
*Simulating and Optimising Dynamical Models in Python 3*

![build](https://github.com/twallema/pySODM/actions/workflows/tests.yml/badge.svg) ![docs](https://github.com/twallema/pySODM/actions/workflows/deploy.yml/badge.svg) [![HitCount](https://hits.dwyl.com/twallema/pySODM.svg)](https://hits.dwyl.com/twallema/pySODM)

### Quick installation 
```
pip install pySODM
```
### Resources

Documentation: https://twallema.github.io/pySODM

Manuscript: https://arxiv.org/abs/2301.10664

pyPI: https://pypi.org/project/pySODM/ 

### Aim & Scope

A modeling and simulation workflow will typically constitute the following steps (see [Villaverde et. al](https://doi.org/10.1093/bib/bbab387)),
1. Build a system dynamical model to describe some real-world phenomenon. Assess structural identifiability.
2. Calibrate the model to a set of experimental data.
3. Extract knowledge from the calibrated parameter values (assess practical identifiability).
4. Use the model to make projections outside the calibrated range.

The aim of pySODM is to reduce the time it takes to step through this workflow. pySODM provides a *template* to construct, simulate and calibrate dynamical systems governed by differential equations. Models can have n-dimensional labeled states of different sizes and can be simulated deterministically and stochastically. Model parameters can be time-dependent by means of complex functions
with arbitrary inputs. The labeled n-dimensional model states can be aligned with n-dimensional
data to compute the posterior probability function, which can subsequently be optimised.

### Overview of features

| Workflow                     | Features                                                                                                                        |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Construct a dynamical model     | Implement coupled systems of differential equations            |
|                                 | States can be n-dimensional and of different sizes, allowing users to build models with subprocesses                                       |
|                                 | Allows n-dimensional model states to be labelled with coordinates and dimensions by using `xarray.Dataset}` to store simulation output |
|                                 | Easy indexing, manipulating, saving, and piping to third-party software of model output by formatting simulation output as `xarray.Dataset` |
| Simulating the model            | Continuous and discrete deterministic simulation or discrete stochastic simulation (Gillespie's Stochastic Simulation Algorithm or Tau-Leaping) |
|                                 | Vary model parameters during the simulation generically using a complex function |
|                                 | Use *draw functions* to perform repeated simulations for sensitivity analysis. With multiprocessing support |
| Calibrate the model             | Construct and maximize a posterior probability function  |
|                                 | Automatic alignment of data and model forecast over timesteps and coordinates  |
|                                 | Nelder-Mead Simplex and Particle Swarm Optimization for point estimation of model parameters |
|                                 | Pipeline to and backend for `emcee.EnsembleSampler` to perform Bayesian inference of model parameters                           |

### Getting started

The [quistart tutorial](quickstart.md) will teach you the basics of building and simulating models with n-dimensional labeled states in pySODM. It will demonstrate how to vary model parameters over the course of a simulation and how to perform repeated simulations with sampling of model parameters.

The [workflow](worfklow.md) tutorial provides a step-by-step introduction to a modeling and simulation workflow by inferring the distributions of the model parameters of a simple compartmental disease model using a synthetic dataset. 

The [enzyme kinetics](enzyme_kinetics.md) and [influenza 17-18](influenza_1718.md) case studies apply the modeling and simulation workflow to more advanced, real-world problems. In the enzyme kinetics case study, a 1D packed-bed reactor model is implemented in pySODM by reducing the two PDEs to a set of coupled ODEs by using the method-of-lines. In the Influenza 17-18 case study, a stochastic, age-structured model for influenza is developped and calibrated to the Influenza incidence data reported by the Belgian Federal Institute of Public Health. These case studies mainly serve to demonstrate pySODM's capabilities across scientific disciplines and highlight the arbitrarily complex nature of the models that can be built with pySODM. For an academic description of pySODM and on the Enzyme Kinetics and Influenza 17-18 case studies, checkout our [manuscript](https://arxiv.org/abs/2301.10664).

### Versions

- Version 0.2.0 (2023-01-19, PR #25)
    > Introduction of `state_dimensions` in model declaration, allowing the user to define models where states can have different sizes.
    - Version 0.2.1 (2023-01-27, PR #30)
        > Fixed bugs encountered when incorporating pySODM into the SARS-CoV-2 Dynamic Transmission Models. More thorough checks on `bounds` in the log posterior probability function. Finished manuscript.
    - Version 0.2.2 (2023-01-27, PR #31)
        > Published to pyPI.
    - Version 0.2.3 (2023-05-04, PR #46)
        > Fixed minor bugs encountered when using pySODM for a dynamic input-output model of the Belgian economy. Published to pyPI.
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