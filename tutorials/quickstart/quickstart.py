"""
This script contains a calibration of an influenza model to 2017-2018 data.
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

from pyexpat import model
import sys,os
import emcee
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import nelder_mead
from pySODM.optimization.utils import add_negative_binomial_noise, assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, ll_negative_binomial

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

######################################################
## Generate a synthetic dataset with overdispersion ##
######################################################

dates = pd.date_range('2022-12-01','2023-01-21')
x = np.linspace(start=0, stop=len(dates)-1, num=len(dates))
alpha = 0.03
td = 10
y = np.random.negative_binomial(1/alpha, (1/alpha)/(np.exp(x*np.log(2)/td) + (1/alpha)))
d = pd.Series(index=dates, data=y, name='CASES')
# Index name must be data for calibration to work
d.index.name = 'date'
# Data collection only on weekdays
d = d[d.index.dayofweek < 5]

##################
## Define model ##
##################

# Import the ODEModel class
from pySODM.models.base import ODEModel, SDEModel

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

# Define parameters and initial condition
params={'beta':0.35, 'gamma':5}
init_states = {'S': 1000, 'I': 1}

# Initialize model
model = ODE_SIR(states=init_states, parameters=params)

#########################
## Calibrate the model ##
#########################

if __name__ == '__main__':

    #######################
    ## Variance analysis ##
    #######################

    results, ax = variance_analysis(d, resample_frequency='W')
    dispersion = results.loc['negative binomial', 'theta']
    print(results)
    plt.show()
    plt.close()

    ##############################
    ## Frequentist optimization ##
    ##############################

    # Define dataset
    data=[d, ]
    states = ["I",]
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha,]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model, pars, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args, labels=labels)
    # Extract start- and enddate
    start_date = d.index.min()
    end_date = d.index.max()
    # Initial guess
    theta = [0.50,]
    # Run Nelder-Mead optimisation
    theta = nelder_mead.optimize(objective_function, theta, [0.10,], processes=1, max_iter=30)[0]
    # Simulate the model
    model.parameters.update({'beta': theta[0]})
    out = model.sim([start_date, end_date])
    # Visualize result
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(out['date'], out['I'], color='red', label='Infectious')
    ax.scatter(d.index, d.values, color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2, label='data')
    ax.legend()
    plt.show()
    plt.close()

    ###########################
    ## Bayesian optimization ##
    ###########################

    # Variables
    n_mcmc = 200
    multiplier_mcmc = 9
    processes = 9
    print_n = 50
    discard = 50
    samples_path = 'sampler_output/'
    fig_path = 'sampler_output/'
    identifier = 'username'
    run_date = str(datetime.date.today())
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.10*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=bounds)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_date.strftime("%Y-%m-%d"), 'end_calibration': end_date.strftime("%Y-%m-%d"),
              'n_chains': nwalkers, 'starting_estimate': list(theta)}
    # Sample
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, (), {},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings)
    # Generate a sample dictionary and save it as .json for long-term storage
    samples_dict = emcee_sampler_to_dictionary(discard=discard, samples_path=samples_path, identifier=identifier)
    # Visualize the distribution of the basic reproduction number
    fig,ax=plt.subplots()
    ax.hist(np.array(samples_dict['beta'])*model.parameters['gamma'], density=True, color='black', alpha=0.6)
    ax.set_xlabel('$R_0$')
    plt.show()
    plt.close()

    # Define draw function
    def draw_fcn(param_dict, samples_dict):
        param_dict['beta'] = np.random.choice(np.array(samples_dict['beta']))
        return param_dict
    # Simulate model
    out = model.sim([start_date, end_date+pd.Timedelta(days=28)], N=30, samples=samples_dict, draw_fcn=draw_fcn, processes=processes)
    # Add negative binomial observation noise
    out = add_negative_binomial_noise(out, dispersion)
    # Visualize result
    fig,ax=plt.subplots(figsize=(12,3))
    for i in range(30):
        ax.plot(out['date'], out['I'].isel(draws=i), color='red', alpha=0.05)
    ax.plot(out['date'], out['I'].mean(dim='draws'), color='red', alpha=0.6)
    ax.scatter(d.index, d.values, color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.show()
    plt.close()

