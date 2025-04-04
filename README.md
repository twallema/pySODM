## pySODM
*Simulating and Optimising Dynamical Models in Python 3*

Resources: [documentation](https://twallema.github.io/pySODM), [peer-reviewed paper](https://www.sciencedirect.com/science/article/pii/S1877750323002089), [pyPI](https://pypi.org/project/pySODM/)

![build](https://github.com/twallema/pySODM/actions/workflows/unit_tests.yml/badge.svg) ![docs](https://github.com/twallema/pySODM/actions/workflows/deploy_documentation.yml/badge.svg)

### Quick installation 
```
pip install pySODM
```

### Aim & Scope

All the simulation projects I've undertaken over the past six years required me to do most of the following,
1. Integrate a system dynamical model
2. Its states may be represented by n-dimensional numpy arrays, labeled using coordinates
3. Its parameters may have time-dependencies 
4. Its intial conditions may vary 
5. Its parameters may be stochastic
6. It may have to be calibrate to a dataset(s) by defining and optimising a cost function

all these features required me to wrap code around an ODE solver, typically `scipy.solve_ivp`, and I got tired of recycling the same code over and over again, so I packaged it into pySODM.

Does other simulation software exist in Python? Sure, but most rely on symbolic transitions, which places a limit on the attainable complexity of a model, making it unfit for academic research. I wanted a piece a software that nicely does all the nasty bookkeeping like keeping track of state sizes, time dependencies on parameters, aligning simulations with datasets etc. and does so in a **generically applicable** way so that I'd never hit a software wall mid-project.

### Overview of features

| Workflow                     | Features                                                                                                                        |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Construct a dynamical model     | Implement systems of coupled differential equations            |
|                                 | Labeled n-dimensional model states, states can have different sizes                                     |
|                                 | Leverages `xarray.Dataset` to store labeled n-dimensional simulation output |
| Simulating the model            | Deterministic (ODE) or stochastic simulation (Jump process) |
|                                 | *Time-dependent model parameter functions* to vary parameters during the course of a simulation |
|                                 | *Draw functions* to vary model parameters during consecutive simulations. |
|                                 | *Initial condition functions* to vary the initial condition during consecutive simulations |
| Calibrate the model             | Construct and maximize a posterior probability function  |
|                                 | Automatically aligns data and model forecast  |
|                                 | Nelder-Mead Simplex and Particle Swarm Optimization |
|                                 | Bayesian inference with `emcee.EnsembleSampler`                  |


### Getting started

- Detailed [installation instructions](installation.md).

- The [quistart tutorial](quickstart.md) teaches the basics of building and simulating models with n-dimensional labeled states in pySODM. It demonstrates the use of *time-dependent parameter functions* (TDPFs) to vary model parameters over the course of a simulation and *draw functions* to vary model parameters during consecutive simulations.

- The [workflow](worfklow.md) tutorial provides a step-by-step introduction to building a mathematical model and calibrating its parameters to a dataset. An SIR disease model is built and the basic reproduction number during an outbreak is determined by calibrating the model to the outbreak data. 

- The [enzyme kinetics](enzyme_kinetics.md) and [influenza 17-18](influenza_1718.md) case studies apply the [workflow](workflow.md) to more advanced, real-world problems. In the enzyme kinetics case study, a 1D packed-bed reactor model is implemented in pySODM by reducing the PDEs to a set of coupled ODEs by using the method-of-lines. In the Influenza 17-18 case study, a stochastic, age-structured model for influenza is developped and calibrated to the Influenza incidence data reported by the Belgian Federal Institute of Public Health. These case studies mainly serve to demonstrate pySODM's capabilities across scientific disciplines and highlight the arbitrarily complex nature of the models that can be built with pySODM. For an academic exposee of pySODM, the Enzyme Kinetics and Influenza 17-18 case studies, checkout our [peer-reviewed paper](https://www.sciencedirect.com/science/article/pii/S1877750323002089).

### Versions

- Version 0.2.0 (2023-01-19, PR #25)
    > Introduction of `dimensions_per_state` in model declaration, allowing the user to define models where states can have different sizes.
    - Version 0.2.1 (2023-01-27, PR #30)
        > Fixed bugs encountered when incorporating pySODM into the SARS-CoV-2 Dynamic Transmission Models. More thorough checks on `bounds` in the log posterior probability function. Finished manuscript.
    - Version 0.2.2 (2023-01-27, PR #31)
        > Published to pyPI.
    - Version 0.2.3 (2023-05-04, PR #46)
        > Fixed minor bugs encountered when using pySODM for a dynamic input-output model of the Belgian economy. Published to pyPI.
    - Version 0.2.4 (2023-12-04, PR #62)
        > Validated the use of Python 3.11. Efficiency gains in simulation of jump processes. Ommitted dependency on Numba. All changes related to publishing our software manuscript in Journal of Computational Science. Improved nomenclature in model defenition.
    - Version 0.2.5 (2024-10-08, PR #106)
        > Validated the use of Python 3.12. Validated pySODM on macOS Sonoma 14.5. 'draw functions' only have 'parameters' as mandatory input, followed by an arbitrary number of additional parameters (PR #75). Tutorial environment can now be found in `tutorial_env.yml` and was renamed `PYSODM-TUTORIALS` (PR #76). Users can choose when the simulation starts when calibrating a model (PR #92). Initial model states can now be a function returning a dictionary of states. This initial condition function can have arguments, which become part of the model's parameters, and can therefore be optimised (PR #99).  Deprecation of legacy 'warmup' parameter (PR #100). Change 'states' --> 'initial_states' as input needed to initialize a model (PR #102).
    - Version 0.2.6 (2025-04-04, PR #143)
        > Harmonize NM and PSO optimizer output (PR #115). Add regularisation weights and input checks to log prior functions (PR #119). Use of `emcee_to_samples_dictionary` deprecated in favor of `xarray.Dataset` to save samples long-term (PR #124). Deprecated `output_timestep` in pySODM model's `sim()` function, and added functionality to define the unit of time when using dates (PR #133). Validate the use of Python 3.13, and add a minimum required Python version (PR #138).
- Version 0.1 (2022-12-23, PR #14)
    > Application pySODM to three use cases. Documentation website. Unit tests for ODE, JumpProcess and calibration. 
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