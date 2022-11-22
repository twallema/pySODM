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
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_poisson_noise, assign_theta
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
names = ['date', 'age_group']
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
params={'beta':0.10,'sigma':1,'f_a':0.75*np.ones(4),'gamma':5,'Nc':np.transpose(Nc)}
# Define initial condition
init_states = {'S': initN.values,'E': np.rint(initN.values/initN.values[0])}
# Define model coordinates
coordinates={'age_group': age_groups}
# Initialize model
model = influenza_model(init_states,params,coordinates)

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # Variables
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    n_pso = 5
    multiplier_pso = 30

    # Define dataset
    data=[df_influenza[start_date:end_date], ]
    states = ["Im_inc",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [[],]
    # Calibated parameters and bounds
    pars = ['beta', 'f_a']
    labels = ['$\\beta$', '$f_a$']
    bounds = [(1e-6,0.20), (0,1)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)
    # Extract expanded bounds and labels
    expanded_labels = objective_function.expanded_labels 
    expanded_bounds = objective_function.expanded_bounds                                   
    # PSO
    theta = pso.optimize(objective_function, kwargs={'simulation_kwargs':{'warmup': warmup}},
                       swarmsize=multiplier_pso*processes, maxiter=n_pso, processes=processes, debug=True)[0]    
    # Nelder-mead
    step = len(expanded_bounds)*[0.05,]
    theta = nelder_mead.optimize(objective_function, np.array(theta), step, kwargs={'simulation_kwargs':{'warmup': warmup}}, processes=processes, max_iter=n_pso)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_date, end_date], warmup=warmup, samples={}, N=N)
    # Add poisson obervational noise
    out = add_poisson_noise(out)
    # Visualize
    fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=(8,6))
    axs = axs.reshape(-1)
    for id, age_class in enumerate(df_influenza.index.get_level_values('age_group').unique()):
        # Data
        axs[id].plot(df_influenza.index.get_level_values('date').unique(),df_influenza.loc[slice(None),age_class], color='orange')
        # Model trajectories
        for i in range(N):
            axs[id].plot(out['date'],out['Im_inc'].sel(age_group=age_class).isel(draws=i), color='black', alpha=0.1, linewidth=1)
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

    # Variables
    n_mcmc = 10
    multiplier_mcmc = 9
    print_n = 5
    discard=5
    samples_path='sampler_output/'
    fig_path='sampler_output/'
    identifier = 'username'
    run_date = str(datetime.date.today())
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.10*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=expanded_bounds)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_date.strftime("%Y-%m-%d"), 'end_calibration': end_date.strftime("%Y-%m-%d"),
              'n_chains': nwalkers, 'warmup': warmup, 'starting_estimate': list(theta), 'labels': expanded_labels}
    # Sample 100 iterations
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, (), {'simulation_kwargs': {'warmup': warmup}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings)
    # Sample 100 more
    backend = emcee.backends.HDFBackend(os.path.join(os.getcwd(),samples_path+'username_BACKEND_'+run_date+'.hdf5'))
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, (), {'simulation_kwargs': {'warmup': warmup}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=backend, processes=processes, progress=True,
                                    settings_dict=settings)
    # Generate a sample dictionary and save it as .json for long-term storage
    # Have a look at the script `emcee_sampler_to_dictionary.py`, which does the same thing as the function below but can be used while your MCMC is running.
    samples_dict = emcee_sampler_to_dictionary(discard=discard, samples_path=samples_path, identifier=identifier)
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
    def draw_fcn(param_dict, samples_dict):
        idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
        param_dict['f_a'] = np.array([slice[idx] for slice in samples_dict['f_a']])
        return param_dict
    # Simulate model
    out = model.sim([start_date, end_date], warmup=warmup, N=N, samples=samples_dict, draw_fcn=draw_fcn, processes=processes)
    # Add poisson observation noise
    out = add_poisson_noise(out)
    # Visualize
    fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=(8,6))
    axs = axs.reshape(-1)
    for id, age_class in enumerate(df_influenza.index.get_level_values('age_group').unique()):
        # Data
        axs[id].plot(df_influenza.index.get_level_values('date').unique(),df_influenza.loc[slice(None),age_class], color='orange')
        # Model trajectories
        for i in range(N):
            axs[id].plot(out['date'],out['Im_inc'].sel(age_group=age_class).isel(draws=i), color='black', alpha=0.1, linewidth=1)
        axs[id].set_title(age_class)
        axs[id].legend(['$I_{m, inc}$','data'])
        axs[id].xaxis.set_major_locator(plt.MaxNLocator(5))
        for tick in axs[id].get_xticklabels():
            tick.set_rotation(30)
        axs[id].grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()
