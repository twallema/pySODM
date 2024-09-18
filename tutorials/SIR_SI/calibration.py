"""
This script contains a calibration of a vector-born disease model to synthetic data
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2023 by T.W. Alleman, BIOSPACE, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

# General purpose packages
import corner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_negative_binomial_noise, assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, ll_negative_binomial, ll_poisson

##################
## Define model ##
##################

# Import the ODE class
from models import ODE_SIR_SI as SIR_SI
# Define parameters and initial condition
params={'alpha': np.array([0.05, 0.1, 0.2, 0.15]), 'gamma': 5, 'beta': 7}
init_states = {'S': [606938, 1328733, 7352492, 2204478], 'S_v': 1e6, 'I_v': 2}
# Define model coordinates
age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)])
coordinates={'age_group': age_groups}
# Initialize model
model = SIR_SI(init_states, params, coordinates=coordinates)

################################
## Generate synthetic dataset ##
################################

# Simulate model
out = model.sim(120)

# Overdisperse data from humans
alpha=0.03
y = np.random.negative_binomial(1/alpha, (1/alpha)/(out['I'] + (1/alpha)))
out['I'] = (['time', 'age_group'], y)
data_humans = out['I'].to_series()

# Overdisperse data from mosquitos
y = np.random.negative_binomial(1/alpha, (1/alpha)/(out['I_v'] + (1/alpha)))
out['I_v'] = (['time'], y)
data_mosquitos = out['I_v'].to_series()

# Start and enddate
start_date = float(out['time'].isel(time=0).values)
end_date = float(out['time'].isel(time=-1).values)

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
    log_likelihood_fnc_args = [len(age_groups)*[alpha,],alpha]
    # Calibated parameters and bounds
    pars = ['beta','alpha']
    labels = ['$\\beta$','$\\alpha$']
    bounds = [(1,21),(0.02,0.30)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model, pars, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args, labels=labels)
    # Initial guess --> pso
    theta = pso.optimize(objective_function, swarmsize=3*18, max_iter=30, processes=18, debug=True)[0]
    # Run Nelder-Mead optimisation
    theta = nelder_mead.optimize(objective_function, theta, 0.10*np.ones(len(theta)), processes=18, max_iter=30)[0]
    # Simulate the model
    model.parameters.update({'beta': theta[0], 'alpha': theta[1:]})
    out = model.sim([start_date, end_date])
    # Visualize result
    fig,ax=plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
    ax[0].plot(out['time'], out['I_v']/init_states['S_v']*100, color='red', label='Infected')
    ax[0].plot(data_mosquitos.index.get_level_values('time').unique(), data_mosquitos/init_states['S_v']*100, color='black', label='data')
    ax[0].set_ylabel('Health state vectors (%)')
    ax[0].legend()
    ax[0].set_title('Vector lifespan: ' + str(params['beta']) + ' days')
    colors=['black', 'red', 'green', 'blue']
    labels=['0-5','5-15','15-65','65-120']
    for i,age_group in enumerate(age_groups):
        ax[1].plot(out['time'], out['I'].sel(age_group=age_group)/init_states['S'][i]*100000, color=colors[i], label=labels[i])
        ax[1].plot(data_humans.index.get_level_values('time').unique(), data_humans.loc[slice(None), age_group]/init_states['S'][i]*100000, color='black', label='data', alpha=0.6)
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
    n_mcmc = 100
    multiplier_mcmc = 9
    processes = 9
    print_n = 50
    discard = 50
    samples_path = 'sampler_output/'
    fig_path = 'sampler_output/'
    identifier = 'username'
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.10*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=objective_function.expanded_bounds)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_date, 'end_calibration': end_date, 'n_chains': nwalkers, 'starting_estimate': list(theta)}
    # Sample
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, 
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n,
                                    processes=processes, progress=True,settings_dict=settings)
    # Generate a sample dictionary and save it as .json for long-term storage
    samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, discard=discard)
    # Look at the resulting distributions in a cornerplot
    CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E")
    fig = corner.corner(sampler.get_chain(discard=discard, thin=2, flat=True), labels=objective_function.expanded_labels, **CORNER_KWARGS)
    for idx,ax in enumerate(fig.get_axes()):
        ax.grid(False)
    plt.show()
    plt.close()

    ##############################
    ## Results: Goodness-of-fit ##
    ##############################

    # Define draw function
    import random
    def draw_fcn(parameters, initial_states, samples):
        idx, parameters['beta'] = random.choice(list(enumerate(samples['beta'])))
        parameters['alpha'] = np.array([slice[idx] for slice in samples['alpha']])
        return parameters, initial_states
    # Simulate model
    N = 100
    out = model.sim([start_date, end_date+60], N=N, draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict}, processes=processes)
    # Add negative binomial observation noise
    out = add_negative_binomial_noise(out, alpha)
    # Visualize result
    fig,ax=plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
    for i in range(N):
        ax[0].plot(out['time'], out['I_v'].isel(draws=i)/init_states['S_v']*100, color='red', linewidth=2, alpha=0.03)
    ax[0].plot(data_mosquitos.index.get_level_values('time').unique(), data_mosquitos/init_states['S_v']*100, color='black', label='data')
    ax[0].set_ylabel('Health state vectors (%)')
    ax[0].legend()
    beta = np.mean(samples_dict['beta'])
    ax[0].set_title(f'Vector lifespan: {beta:.1f} days')
    colors=['black', 'red', 'green', 'blue']
    labels=['0-5','5-15','15-65','65-120']
    for i,age_group in enumerate(age_groups):
        ax[1].plot(data_humans.index.get_level_values('time').unique(), data_humans.loc[slice(None), age_group]/init_states['S'][i]*100000, color=colors[i], label=labels[i])
        for j in range(N):
            ax[1].plot(out['time'], out['I'].sel(age_group=age_group).isel(draws=j)/init_states['S'][i]*100000, color=colors[i], linewidth=2, alpha=0.03)
    ax[1].set_ylabel('Infectious humans per 100K')
    alpha = [np.mean(samples_dict['alpha'][0]), np.mean(samples_dict['alpha'][1]), np.mean(samples_dict['alpha'][2]), np.mean(samples_dict['alpha'][3])]
    alpha = [round(i, 2) for i in alpha]
    ax[1].set_title(f'Vector-to-human transfer rate: {alpha}')
    ax[1].legend()
    ax[1].set_xlabel('time (days)')
    plt.show()
    plt.close()
