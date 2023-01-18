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

##################
## Define model ##
##################

# Import the ODEModel class
from models import ODE_SIR_SI as SIR_SI
# Define parameters and initial condition
params={'alpha': 10*np.array([0.005, 0.01, 0.02, 0.015]), 'gamma': 5, 'beta': 7}
init_states = {'S': [606938, 1328733, 7352492, 2204478], 'S_v': 1e6, 'I_v': 2}
# Define model coordinates
age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)])
coordinates={'age_group': age_groups}
# Initialize model
model = SIR_SI(states=init_states, parameters=params, coordinates=coordinates)

################################
## Generate synthetic dataset ##
################################

# Simulate model
out = model.sim(120)

# Overdisperse data from humans
alpha=0.03
y = np.random.negative_binomial(1/alpha, (1/alpha)/(out['I'] + (1/alpha)))
out['I'] = (['age_group','time'], y)
data_humans = out['I'].to_series()

# Overdisperse data from mosquitos
y = np.random.negative_binomial(1/alpha, (1/alpha)/(out['I_v'] + (1/alpha)))
out['I_v'] = (['time'], y)
data_mosquitos = out['I_v'].to_series()

# Start and enddate
start_date = out['time'].isel(time=0).values
end_date = out['time'].isel(time=-1).values

#########################
## Calibrate the model ##
#########################

if __name__ == '__main__':

    ##############################
    ## Frequentist optimization ##
    ##############################

    # Define dataset
    data=[data_humans, data_mosquitos]
    states = ["I","I_v"]
    log_likelihood_fnc = [ll_negative_binomial,ll_negative_binomial]
    log_likelihood_fnc_args = [len(age_groups)*[alpha,], alpha]
    # Calibated parameters and bounds
    pars = ['alpha', 'beta']
    labels = ['$\\alpha$','$\\beta$']
    bounds = [(1e-6,1),(1e-6,1)]
    def aggfunc1(output):
        return output
    def aggfunc2(output):
        return output
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model, pars, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args, labels=labels)#,aggregation_function=[aggfunc1,aggfunc2])
    # Initial guess
    theta = [0.10, 0.10, 0.10, 0.10, 7]
    # Run Nelder-Mead optimisation
    theta = nelder_mead.optimize(objective_function, theta, 0.10*np.ones(len(theta)), processes=18, max_iter=30)[0]
    # Simulate the model
    model.parameters.update({'beta': theta[0]})
    out = model.sim([start_date, end_date])
    # Visualize result
    fig,ax=plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
    ax[0].plot(out['time'], out['I_v']/init_states['S_v']*100, color='red', label='Infected')
    ax[0].scatter(data_mosquitos.index.get_level_values('time').unique(), data_mosquitos, alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2, label='data')
    ax[0].set_ylabel('Health state vectors (%)')
    ax[0].legend()
    ax[0].set_title('Vector lifespan: ' + str(params['beta']) + ' days')
    colors=['black', 'red', 'green', 'blue']
    labels=['0-5','5-15','15-65','65-120']
    for i,age_group in enumerate(age_groups):
        ax[1].plot(out['time'], out['I'].sel(age_group=age_group)/init_states['S'][i]*100000, color=colors[i], label=labels[i])
    for age_group in age_groups:
        ax[1].scatter(data_humans.index.get_level_values('time').unique(), data_humans.loc[age_group, slice(None)], alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2, label='data')
    ax[1].set_ylabel('Infectious humans per 100K')
    ax[1].legend()
    ax[1].set_xlabel('time (days)')
    ax[1].set_title('Vector-to-human transfer rate: '+str(params['alpha']))
    plt.show()
    plt.close()

    ###########################
    ## Bayesian optimization ##
    ###########################

    # Variables
    n_mcmc = 400
    multiplier_mcmc = 9
    processes = 9
    print_n = 50
    discard = 50
    samples_path = 'sampler_output/'
    fig_path = 'sampler_output/'
    identifier = 'username'
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.10*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=bounds)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_date.strftime("%Y-%m-%d"), 'end_calibration': end_date.strftime("%Y-%m-%d"),
              'n_chains': nwalkers, 'starting_estimate': list(theta)}
    # Sample
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, 
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n,
                                    processes=processes, progress=True,settings_dict=settings)
    # Generate a sample dictionary and save it as .json for long-term storage
    samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, discard=discard)
    
    ########################################
    ## Results: Basic reproduction number ##
    ########################################
    
    # Visualize the distribution of the basic reproduction number
    fig,ax=plt.subplots(figsize=(12,4))
    ax.hist(np.array(samples_dict['beta'])*model.parameters['gamma'], density=True, color='black', alpha=0.6)
    ax.set_xlabel('$R_0$')
    plt.show()
    plt.close()

    ##############################
    ## Results: Goodness-of-fit ##
    ##############################

    # Define draw function
    def draw_fcn(param_dict, samples_dict):
        param_dict['beta'] = np.random.choice(samples_dict['beta'])
        return param_dict
    # Simulate model
    out = model.sim([start_date, end_date+pd.Timedelta(days=2*28)], N=100, samples=samples_dict, draw_function=draw_fcn, processes=processes)
    # Add negative binomial observation noise
    out = add_negative_binomial_noise(out, alpha)
    # Visualize result
    fig,ax=plt.subplots(figsize=(12,4))
    for i in range(100):
        ax.plot(out['date'], out['I'].isel(draws=i), color='red', alpha=0.05)
    ax.plot(out['date'], out['I'].mean(dim='draws'), color='red', alpha=0.6)
    ax.scatter(d.index, d.values, color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_ylabel('Number of infected')
    plt.show()
    plt.close()

    ########################
    ## Results: scenarios ##
    ########################

    # Define a time-dependent parameter function
    def lower_infectivity(t, states, param, start_measures):
        if pd.to_datetime(t) < start_measures:
            return param
        else:
            return param/2

    # Define draw function
    def draw_fcn(param_dict, samples_dict):
        param_dict['beta'] = np.random.choice(samples_dict['beta'])
        return param_dict

    # Attach its arguments to the parameter dictionary
    params.update({'start_measures': end_date})

    # Initialize the model with the time dependent parameter funtion
    model_with = ODE_SIR(states=init_states, parameters=params, time_dependent_parameters={'beta': lower_infectivity})

    # Simulate the model
    out_with = model_with.sim([start_date, end_date+pd.Timedelta(days=2*28)], N=100, samples=samples_dict, draw_function=draw_fcn, processes=processes)

    # Add negative binomial observation noise
    out_with = add_negative_binomial_noise(out_with, alpha)

    # Visualize result
    fig,ax=plt.subplots(figsize=(12,4))
    for i in range(100):
        ax.plot(out['date'], out['I'].isel(draws=i), color='red', alpha=0.05)
        ax.plot(out_with['date'], out_with['I'].isel(draws=i), color='blue', alpha=0.05)
    ax.plot(out['date'], out['I'].mean(dim='draws'), color='red', alpha=0.6)
    ax.plot(out_with['date'], out_with['I'].mean(dim='draws'), color='blue', alpha=0.6)
    ax.scatter(d.index, d.values, color='black', alpha=0.6, linestyle='None', facecolors='none', s=60, linewidth=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_ylabel('Number of infected')
    plt.show()
    plt.close()


