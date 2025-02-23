"""
This script contains the calibration of an influenza model to the 2017-2018 surveillance data
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, BIOSPACE, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

# General purpose packages
import os
import emcee
import random
import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_negative_binomial_noise, assign_theta
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler
from pySODM.optimization.objective_functions import log_posterior_probability, ll_negative_binomial
# pySODM dependecies
import corner

##############
## Settings ##
##############

tau = 0.50                                      # Timestep of Tau-Leaping algorithm
alpha = 0.03                                    # Overdispersion factor (based on COVID-19)
end_calibration = '2018-03-01'                  # Enddate of calibration
identifier = 'twallema_2018-03-01'              # Give any output of this script an ID
n_pso = 30                                      # Number of PSO iterations
multiplier_pso = 10                             # PSO swarm size
n_mcmc = 500                                    # Number of MCMC iterations
multiplier_mcmc = 10                            # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 100                                   # Print diagnostics every print_n iterations
discard = 50                                    # Discard first `discard` iterations as burn-in
thin = 10                                       # Thinning factor emcee chains
n = 100                                         # Repeated simulations used in visualisations
processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())) # Retrieve CPU count

###############
## Load data ##
###############

# Load case data
data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/interim/ILI_weekly_ABS.csv'), index_col=[0,1], parse_dates=True, date_format='%Y-%m-%d')
data = data.squeeze()
# Load case data per 100K
data_100K = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/interim/ILI_weekly_100K.csv'), index_col=[0,1], parse_dates=True, date_format='%Y-%m-%d')
data_100K = data_100K.squeeze()
# Re-insert pd.IntervalIndex (pd.IntervalIndex is always loaded as a string..)
age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)], closed='left')
iterables = [data.index.get_level_values('DATE').unique(), age_groups]
names = ['date', 'age_group']
index = pd.MultiIndex.from_product(iterables, names=names)
df_influenza = pd.Series(index=index, name='CASES', data=data.values)/7 # convert weekly cumulative to daily week midpoint 
df_influenza_100K = pd.Series(index=index, name='CASES', data=data_100K.values)/7 # convert weekly cumulative to daily week midpoint 
# Extract start and enddate
start_visualisation = df_influenza.index.get_level_values('date').unique()[8].strftime("%Y-%m-%d")
end_visualisation = df_influenza.index.get_level_values('date').unique()[-1].strftime("%Y-%m-%d")
start_calibration = start_visualisation
# Hardcode Belgian demographics (Jan 1, 2018)
initN = pd.Series(index=age_groups, data=[620914, 1306826, 7317774, 2130556])

################
## Load model ##
################

from models import JumpProcess_influenza_model as influenza_model

#####################################
## Define a social policy function ##
#####################################

# Contact matrices integrated with contact duration (physical contact only)
N_noholiday_week = np.transpose(np.array([[10.47, 3.16, 10.65, 0.55],
                                         [1.65, 19.41, 11.79, 0.70],
                                         [0.93, 1.98, 14.09, 0.96],
                                         [0.18, 0.45, 3.67, 3.09]]))

N_noholiday_weekend = np.transpose(np.array([[1.81,	2.46, 8.92,	1.52],
                                            [1.28, 7.07, 14.76, 1.64],
                                            [0.78, 2.49, 15.26, 1.17],
                                            [0.51, 1.05, 4.45, 2.00]]))

N_holiday_week = np.transpose(np.array([[4.54, 2.47, 10.86, 0.35],
                                        [1.29, 5.76, 9.64, 1.47],
                                        [0.96, 1.63, 13.08, 1.38],
                                        [0.12, 0.94, 5.23, 5.10]]))

N = {
    'holiday': {'week': N_holiday_week, 'weekend': N_noholiday_weekend},
    'no_holiday': {'week': N_noholiday_week, 'weekend': N_noholiday_weekend}
}

from models import make_contact_matrix_function
contact_function = make_contact_matrix_function(N).contact_function

#################
## Setup model ##
#################

# Define model parameters
params={'alpha': 1, 'beta': 0.0174, 'gamma': 1, 'delta': 3,'f_ud': np.array([0.01, 0.64, 0.905, 0.60]), 'N': N['holiday']['week']}
# Define initial condition
init_states = {'S': list(initN.values),
              'E': list(np.rint((1/(1-params['f_ud']))*df_influenza.loc[start_calibration, slice(None)])),
              'Ip': list(np.rint((1/(1-params['f_ud']))*df_influenza.loc[start_calibration, slice(None)])),
              'Iud': list(np.rint((params['f_ud']/(1-params['f_ud']))*df_influenza.loc[start_calibration, slice(None)])),
              'Id': list(np.rint(df_influenza.loc[start_calibration, slice(None)])),
              'Im_inc': list(np.rint(df_influenza.loc[start_calibration, slice(None)]))}

# Define model coordinates
coordinates={'age_group': age_groups}
# Initialize model
model = influenza_model(initial_states=init_states, parameters=params, coordinates=coordinates,
                            time_dependent_parameters={'N': contact_function})

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # Define dataset
    data=[df_influenza[start_calibration:end_calibration], ]
    states = ["Im_inc",]
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [len(age_groups)*[alpha,],]
    # Calibated parameters and bounds
    pars = ['beta', 'f_ud']
    labels = ['$\\beta$', '$f_{ud}$']
    bounds = [(1e-6,0.08), (1e-3,1-1e-3)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,labels=labels,
                                                   simulation_kwargs={'tau': tau})
    # Extract expanded bounds and labels
    expanded_labels = objective_function.expanded_labels 
    expanded_bounds = objective_function.expanded_bounds                                   
    # PSO
    theta, _ = pso.optimize(objective_function,
                        swarmsize=multiplier_pso*processes, max_iter=n_pso, processes=processes, debug=True)
    # Nelder-mead
    #step = len(expanded_bounds)*[0.10,]
    #theta, _ = nelder_mead.optimize(objective_function, np.array(theta), step, processes=processes,
    # max_iter=n_pso)

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_calibration, end_visualisation], N=n, tau=tau)
    # Add poisson obervational noise
    out = add_negative_binomial_noise(out, alpha)
    # Visualize
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8,6))
    axs = axs.reshape(-1)
    for id, age_class in enumerate(df_influenza.index.get_level_values('age_group').unique()):
        # Data
        axs[id].scatter(df_influenza[start_calibration:end_calibration].index.get_level_values('date').unique(), df_influenza.loc[slice(start_calibration,end_calibration), age_class], color='black', alpha=0.3, linestyle='None', facecolors='None', s=60, linewidth=2, label='observed')
        axs[id].scatter(df_influenza[end_calibration:end_visualisation].index.get_level_values('date').unique(), df_influenza.loc[slice(end_calibration,end_visualisation), age_class], color='red', alpha=0.3, linestyle='None', facecolors='None', s=60, linewidth=2, label='unobserved')
        # Model trajectories
        for i in range(n):
            axs[id].plot(out['date'],out['Im_inc'].sel(age_group=age_class).isel(draws=i), color='blue', alpha=0.05, linewidth=1)
        # Format
        axs[id].set_title(age_class)
        axs[id].legend()
        axs[id].xaxis.set_major_locator(plt.MaxNLocator(5))
        for tick in axs[id].get_xticklabels():
            tick.set_rotation(30)
        axs[id].grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()

    ##########
    ## MCMC ##
    ##########

    # Variables
    samples_path='sampler_output/'
    fig_path='sampler_output/'
    run_date = str(datetime.date.today())
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.30*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=expanded_bounds)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_calibration, 'end_calibration': end_calibration,
              'starting_estimate': list(theta), 'tau': tau}
    # Sample n_mcmc iterations
    sampler, samples_xr = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function,
                                    discard=discard, thin=thin, fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes,
                                    progress=True, settings_dict=settings)                                                                               
    # Look at the resulting distributions in a cornerplot
    CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E")
    fig = corner.corner(sampler.get_chain(discard=discard, thin=2, flat=True), labels=expanded_labels, **CORNER_KWARGS)
    for idx,ax in enumerate(fig.get_axes()):
        ax.grid(False)
    plt.show()
    plt.close()

    ######################
    ## Visualize result ##
    ######################
 
    # Define draw function
    def draw_fcn(parameters, samples_xr):
        # get a random iteration and markov chain
        i = random.randint(0, len(samples_xr.coords['iteration'])-1)
        j = random.randint(0, len(samples_xr.coords['chain'])-1)
        # assign parameters
        parameters['beta'] = samples_xr['beta'].sel({'iteration': i, 'chain': j}).values
        parameters['f_ud'] = samples_xr['f_ud'].sel({'iteration': i, 'chain': j}).values
        return parameters
    # Simulate model
    out = model.sim([start_visualisation, end_visualisation], N=n, tau=tau, output_timestep=1,
                    draw_function=draw_fcn, draw_function_kwargs={'samples_xr': samples_xr}, processes=processes)
    # Add negative binomial observation noise
    out_noise = add_negative_binomial_noise(out, alpha)

    # Visualize
    markers=['^', 's', 'o', 'x']
    colors=['black', 'green', 'red', 'blue']
    labels=['[0,5(','[5,15(','[15,65(','[65,120(']
    fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=(8,6))
    axs = axs.reshape(-1)
    
    for id, age_class in enumerate(df_influenza_100K.index.get_level_values('age_group').unique()):
        # Data
        axs[id].plot(df_influenza_100K[start_calibration:end_calibration].index.get_level_values('date').unique(), df_influenza_100K.loc[slice(start_calibration,end_calibration), age_class]*7, color='black', marker='o', label='Observed data')
        axs[id].plot(df_influenza_100K[pd.Timestamp(end_calibration)-pd.Timedelta(days=7):end_visualisation].index.get_level_values('date').unique(), df_influenza_100K.loc[slice(pd.Timestamp(end_calibration)-pd.Timedelta(days=7),end_visualisation), age_class]*7, color='red', marker='o', label='Unobserved data')
        # Model trajectories
        axs[id].plot(out['date'],out['Im_inc'].sel(age_group=age_class).mean(dim='draws')/initN.loc[age_class]*100000*7, color='black', linestyle='--', alpha=0.7, linewidth=1, label='Model mean')
        axs[id].fill_between(out_noise['date'].values,out_noise['Im_inc'].sel(age_group=age_class).quantile(dim='draws', q=0.025)/initN.loc[age_class]*100000*7, out_noise['Im_inc'].sel(age_group=age_class).quantile(dim='draws', q=0.975)/initN.loc[age_class]*100000*7, color='black', alpha=0.15, label='Model 95% CI')
        #axs[id].fill_between(out['date'].values,out['Im_inc'].sel(age_group=age_class).quantile(dim='draws', q=0.025)/initN.loc[age_class]*100000*7, out['Im_inc'].sel(age_group=age_class).quantile(dim='draws', q=0.975)/initN.loc[age_class]*100000*7, color='black', alpha=0.20, label='Model 95% CI')
        # Format figure
        if id==3:
            axs[id].legend()      
        axs[id].set_title(age_class)
        axs[id].xaxis.set_major_locator(plt.MaxNLocator(5))
        for tick in axs[id].get_xticklabels():
            tick.set_rotation(30)
        axs[id].grid(False)
        if ((id == 0) | (id == 2)):
            axs[id].set_ylabel('GP consultations (-)')

    plt.tight_layout()
    plt.show()
    plt.close()
