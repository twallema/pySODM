"""
This script contains a calibration of an influenza model to 2017-2018 data.
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."


############################
## Load required packages ##
############################

import sys,os
import random
import emcee
import datetime
import json
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, log_prior_uniform, ll_poisson
# pySODM dependecies
import corner

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

###############
## Load data ##
###############

# Load data
data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/interim/data_influenza_1718_format.csv'), index_col=[0,1], parse_dates=True)
data = data.squeeze()
# Re-insert pd.IntervalIndex (pd.IntervalIndex is always loaded as a string..)
age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)])
iterables = [data.index.get_level_values('DATE').unique(), age_groups]
names = ['date', 'Nc']
index = pd.MultiIndex.from_product(iterables, names=names)
df_influenza = pd.Series(index=index, name='CASES', data=data.values)
# Hardcode Belgian demographics
initN = pd.Series(index=age_groups, data=[606938, 1328733, 7352492, 2204478])

################
## Load model ##
################

from models import ODE_influenza_model as influenza_model

#################
## Setup model ##
#################

# Number of repeated experiments
N=20
# Set start date and warmup
warmup=25
start_idx=0
start_date = df_influenza.index.get_level_values('date').unique()[start_idx]
end_date = df_influenza.index.get_level_values('date').unique()[-1] 
sim_len = (end_date - start_date)/pd.Timedelta(days=1)+warmup
# Get initial condition
I_init = df_influenza.loc[start_date]
# Define contact matrix (PolyMod study)
Nc = np.array([[1.3648649, 1.1621622, 5.459459, 0.3918919],
             [0.5524476, 5.1328671,  6.265734, 0.4055944],
             [0.3842975, 0.8409091, 10.520661, 0.9008264],
             [0.2040816, 0.5918367,  4.612245, 2.1428571]])
# Define model parameters
params={'beta':0.10,'sigma':1,'f_a':0.75,'gamma':5,'Nc':np.transpose(Nc)}
# Define initial condition
init_states = {'S': initN.values,'E': np.rint(initN.values/initN.values[0])}
# Define model coordinates
coordinates=[age_groups,]
# Initialize model
model = influenza_model(init_states,params,coordinates)

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # Maximum number of PSO iterations
    n_pso = 10
    # Maximum number of MCMC iterations
    n_mcmc = 100
    # PSO settings
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    multiplier_pso = 30
    maxiter = n_pso
    popsize = multiplier_pso*processes
    # MCMC settings
    multiplier_mcmc = 90
    max_n = n_mcmc
    print_n = 5
    # Define dataset
    data=[df_influenza[start_date:end_date], ]
    states = ["Im_inc",]
    weights = np.array([1,]) # Scores of individual contributions: Dataset: 0, total ll: -4590, Dataset: 1, total ll: -4694, Dataset: 2, total ll: -4984
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [[],]
    # Calibated parameters and bounds
    pars = ['beta', 'f_a']
    bounds = [(0.001,0.10), (0,1)]
    # Setup prior functions and arguments
    log_prior_fnc = len(bounds)*[log_prior_uniform,]
    log_prior_fnc_args = bounds
    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability([],[],model,pars,data,states,
                                               log_likelihood_fnc,log_likelihood_fnc_args,-weights)
    # PSO
    theta = pso.optimize(objective_function, bounds, kwargs={'simulation_kwargs':{'warmup': warmup}},
                       swarmsize=multiplier_pso*processes, maxiter=n_pso, processes=processes, debug=True)[0]    
    # Nelder-mead
    step = len(bounds)*[0.05,]
    theta = nelder_mead.optimize(objective_function, np.array(theta), bounds, step, kwargs={'simulation_kwargs':{'warmup': warmup}}, processes=processes, max_iter=n_pso)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters.update({'beta': theta[0],})
    out = model.sim(end_date, start_date=start_date, warmup=warmup, samples={}, N=N)

    # Visualize
    fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=(8,6))
    axs = axs.reshape(-1)
    for id, age_class in enumerate(df_influenza.index.get_level_values('Nc').unique()):
        # Data
        axs[id].plot(df_influenza.index.get_level_values('date').unique(),df_influenza.loc[slice(None),age_class], color='orange')
        # Model trajectories
        for i in range(N):
            axs[id].plot(out['time'],out['Im_inc'].sel(Nc=age_class).isel(draws=i), color='black', alpha=0.1, linewidth=1)
        # Format
        axs[id].set_title(age_class)
        axs[id].legend(['$I_{m, inc}$','data'])
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

    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=[0.10, 0.10], multiplier=multiplier_mcmc, bounds=log_prior_fnc_args)
    # Labels for traceplots
    labels = ['$\\beta$', '$f_a$']
    pars_postprocessing = ['beta', 'f_a']
    # Variables
    samples_path='sampler_output/'
    fig_path='sampler_output/'
    identifier = 'username'
    run_date = str(datetime.date.today())
    # initialize objective function
    objective_function = log_posterior_probability(log_prior_fnc,log_prior_fnc_args,model,pars,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_date.strftime("%Y-%m-%d"), 'end_calibration': end_date.strftime("%Y-%m-%d"), 'n_chains': nwalkers,
                'warmup': warmup, 'labels': labels, 'parameters': pars_postprocessing, 'starting_estimate': list(theta)}
    # Sample 100 iterations
    sampler = run_EnsembleSampler(pos, max_n, identifier, objective_function, (), {'simulation_kwargs': {'warmup': warmup}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, labels=labels, backend=None, processes=processes, progress=True,
                                    settings_dict=settings)
    backend = emcee.backends.HDFBackend(os.path.join(os.getcwd(),samples_path+'username_BACKEND_'+run_date+'.h5'))
    # Sample 100 more
    sampler = run_EnsembleSampler(pos, max_n, identifier, objective_function, (), {'simulation_kwargs': {'warmup': warmup}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, labels=labels, backend=backend, processes=processes, progress=True,
                                    settings_dict=settings)
    # Generate a sample dictionary
    # Have a look at the script `emcee_sampler_to_dictionary.py`, which does the same thing as the function below but can be used while your MCMC is running.
    samples_dict = emcee_sampler_to_dictionary(sampler, pars_postprocessing, discard=50, settings=settings)
    # Save samples dictionary to json for long-term storage: _SETTINGS_ and _BACKEND_ can be removed at this point
    with open(str(identifier)+'_SAMPLES_'+run_date+'.json', 'w') as fp:
        json.dump(samples_dict, fp)
    # Look at the resulting distributions in a cornerplot
    CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E")
    fig = corner.corner(sampler.get_chain(discard=50, thin=2, flat=True), labels=labels, **CORNER_KWARGS)
    for idx,ax in enumerate(fig.get_axes()):
        ax.grid(False)
    plt.show()
    plt.close()

    ######################
    ## Visualize result ##
    ######################

    # TODO: Observational noise!

    # Define draw function
    def draw_fcn(param_dict, samples_dict):
        idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))  
        param_dict['f_a'] = samples_dict['f_a'][idx]
        return param_dict
    # Simulate model
    out = model.sim(end_date, start_date=start_date, warmup=warmup, N=N, samples=samples_dict, draw_fcn=draw_fcn, processes=processes)
    # Visualize
    fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=(8,6))
    axs = axs.reshape(-1)
    for id, age_class in enumerate(df_influenza.index.get_level_values('Nc').unique()):
        # Data
        axs[id].plot(df_influenza.index.get_level_values('date').unique(),df_influenza.loc[slice(None),age_class], color='orange')
        # Model trajectories
        for i in range(N):
            axs[id].plot(out['time'],out['Im_inc'].sel(Nc=age_class).isel(draws=i), color='black', alpha=0.1, linewidth=1)
        #axs[id].fill_between(out['time'].values,out['Im_inc'].sel(Nc=age_class).quantile(dim='draws', q=0.025),
        #                     out['Im_inc'].sel(Nc=age_class).quantile(dim='draws', q=0.975), color='black', alpha=0.1)
        axs[id].set_title(age_class)
        axs[id].legend(['$I_{m, inc}$','data'])
        axs[id].xaxis.set_major_locator(plt.MaxNLocator(5))
        for tick in axs[id].get_xticklabels():
            tick.set_rotation(30)
        axs[id].grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()