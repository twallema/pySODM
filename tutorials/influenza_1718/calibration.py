"""
This script contains a calibration of an influenza model to 2017-2018 data.
"""

__author__      = "Tijs Alleman & Wolf Demunyck"
__copyright__   = "Copyright (c) 2022 by T.W. Alleman, BIOMATH, Ghent University. All Rights Reserved."

############################
## Load required packages ##
############################

import sys,os
import emcee
import random
import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_poisson_noise, add_negative_binomial_noise, assign_theta, variance_analysis
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson, ll_negative_binomial
# pySODM dependecies
import corner

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

###############
## Load data ##
###############

# Load case data
data = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/interim/data_influenza_1718_format.csv'), index_col=[0,1], parse_dates=True)
data = data.squeeze()
# Load case data per 100K
data_100K = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/interim/data_influenza_1718_format_100K.csv'), index_col=[0,1], parse_dates=True)
data_100K = data_100K.squeeze()
# Re-insert pd.IntervalIndex (pd.IntervalIndex is always loaded as a string..)
age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)], closed='left')
iterables = [data.index.get_level_values('DATE').unique(), age_groups]
names = ['date', 'age_group']
index = pd.MultiIndex.from_product(iterables, names=names)
df_influenza = pd.Series(index=index, name='CASES', data=data.values)
df_influenza_100K = pd.Series(index=index, name='CASES', data=data_100K.values)
# Extract start and enddate
start_date = df_influenza.index.get_level_values('date').unique()[0]
end_date = df_influenza.index.get_level_values('date').unique()[-1] 
# Hardcode Belgian demographics
initN = pd.Series(index=age_groups, data=np.array([606938, 1328733, 7352492, 2204478]))  

##############
## Settings ##
##############

alpha = 0.03 # Overdispersion of data
N = 10 # Repeated simulations
start_calibration = start_date 
end_calibration = pd.Timestamp('2018-03-01')
identifier = 'twallema_2018-03-01'

################
## Load model ##
################

from models import SDE_influenza_model as influenza_model

#####################################
## Define a social policy function ##
#####################################

from models import make_contact_matrix_function

# Physical contacts > 15 min
N_noholiday = np.transpose(np.array([[2.60, 0.96, 3.73, 0.47],
                                     [0.50, 5.87, 4.52, 0.35],
                                     [0.33, 0.76, 5.65, 0.48],
                                     [0.16, 0.22, 1.81, 1.09]]))

N_holiday = np.transpose(np.array([[1.15, 0.74, 3.28, 0.26],
                                   [0.38, 1.85, 3.58, 0.56],
                                   [0.29, 0.60, 5.22, 0.60],
                                   [0.09, 0.36, 2.27, 1.50]]))

# Initialize contact function
contact_function = make_contact_matrix_function(N_noholiday, N_holiday).contact_function

#################
## Setup model ##
#################

# Define model parameters
params={'beta': 0.034, 'sigma': 1, 'f_a': np.array([0.02, 0.60, 0.87, 0.71]), 'gamma': 5, 'N': N_noholiday, 'ramp_time': 0}
# Define initial condition
init_states = {'S': list(initN.values),
               'E': list(np.rint((1/(1-params['f_a']))*df_influenza.loc[start_calibration, slice(None)])),
               'Ia': list(np.rint((params['f_a']/(1-params['f_a']))*params['gamma']*df_influenza.loc[start_calibration, slice(None)])),
               'Im': list(np.rint(params['gamma']*df_influenza.loc[start_calibration, slice(None)])),
               'Im_inc': list(np.rint(df_influenza.loc[start_calibration, slice(None)]))}
# Define model coordinates
coordinates={'age_group': age_groups}
# Initialize model
model = influenza_model(init_states,params,coordinates,time_dependent_parameters={'N': contact_function})

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # Variables
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    n_pso = 20
    multiplier_pso = 20
    # Define dataset
    data=[df_influenza[start_calibration:end_calibration], ]
    states = ["Im_inc",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [4*[alpha,],]
    # Calibated parameters and bounds
    pars = ['beta', 'f_a']
    labels = ['$\\beta$', '$f_a$']
    bounds = [(1e-6,0.06), (0,0.99)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)
    # Extract expanded bounds and labels
    expanded_labels = objective_function.expanded_labels 
    expanded_bounds = objective_function.expanded_bounds                                   
    # PSO
    theta = pso.optimize(objective_function,
                        swarmsize=multiplier_pso*processes, max_iter=n_pso, processes=processes, debug=True)[0]
    #theta = [0.0411, 0.018, 0.60, 0.87, 0.71]
    # Nelder-mead
    step = len(expanded_bounds)*[0.10,]
    theta = nelder_mead.optimize(objective_function, np.array(theta), step, processes=processes, max_iter=n_pso)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_calibration, end_date], samples={}, N=N, output_timestep=0.1)
    # Add poisson obervational noise
    out = add_negative_binomial_noise(out, alpha)
    # Visualize
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8,6))
    axs = axs.reshape(-1)
    for id, age_class in enumerate(df_influenza.index.get_level_values('age_group').unique()):
        # Data
        axs[id].scatter(df_influenza[start_calibration:end_calibration].index.get_level_values('date').unique(), df_influenza.loc[slice(start_calibration,end_calibration), age_class], color='black', alpha=0.3, linestyle='None', facecolors='None', s=60, linewidth=2, label='observed')
        axs[id].scatter(df_influenza[end_calibration:end_date].index.get_level_values('date').unique(), df_influenza.loc[slice(end_calibration,end_date), age_class], color='red', alpha=0.3, linestyle='None', facecolors='None', s=60, linewidth=2, label='unobserved')
        # Model trajectories
        for i in range(N):
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
    n_mcmc = 30
    multiplier_mcmc = 9
    print_n = 50
    discard = 30
    samples_path='sampler_output/'
    fig_path='sampler_output/'
    run_date = str(datetime.date.today())
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.20*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=expanded_bounds)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_calibration.strftime("%Y-%m-%d"), 'end_calibration': end_calibration.strftime("%Y-%m-%d"),
              'n_chains': nwalkers, 'starting_estimate': list(theta), 'labels': expanded_labels}
    # Sample n_mcmc iterations
    sampler = run_EnsembleSampler(pos, 50, identifier, objective_function,
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings)                           
    # Sample 5*n_mcmc more
    for i in range(20):
        backend = emcee.backends.HDFBackend(os.path.join(os.getcwd(),samples_path+identifier+'_BACKEND_'+run_date+'.hdf5'))
        sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function,
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=backend, processes=processes, progress=True,
                                    settings_dict=settings)                                                         
    # Generate a sample dictionary and save it as .json for long-term storage
    # Have a look at the script `emcee_sampler_to_dictionary.py`, which does the same thing as the function below but can be used while your MCMC is running.
    samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, discard=discard)
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
        # Sample model parameters
        idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
        param_dict['f_a'] = np.array([slice[idx] for slice in samples_dict['f_a']])
        param_dict['ramp_time'] = np.random.normal(0,2)
        return param_dict
    # Simulate model
    out = model.sim([start_date, end_date], N=N, output_timestep=0.5, samples=samples_dict, draw_function=draw_fcn, processes=processes)
    # Add negative binomial observation noise
    out = add_negative_binomial_noise(out, alpha)

    # Visualize
    markers=['^', 's', 'o', 'x']
    colors=['black', 'green', 'red', 'blue']
    labels=['[0,5(','[5,15(','[15,65(','[65,120(']
    fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=(8,6))
    axs = axs.reshape(-1)
    for id, age_class in enumerate(df_influenza_100K.index.get_level_values('age_group').unique()):
        # Data
        axs[id].plot(df_influenza_100K[start_date:end_calibration].index.get_level_values('date').unique(), df_influenza_100K.loc[slice(start_date,end_calibration), age_class]*7, color='black', marker='o', label='Observed data')
        axs[id].plot(df_influenza_100K[end_calibration-pd.Timedelta(days=7):end_date].index.get_level_values('date').unique(), df_influenza_100K.loc[slice(end_calibration-pd.Timedelta(days=7),end_date), age_class]*7, color='red', marker='o', label='Unobserved data')
        # Model trajectories
        axs[id].plot(out['date'],out['Im_inc'].sel(age_group=age_class).mean(dim='draws')/initN.loc[age_class]*100000*7, color='black', linestyle='--', alpha=0.7, linewidth=1, label='Model mean')
        axs[id].fill_between(out['date'],out['Im_inc'].sel(age_group=age_class).quantile(dim='draws', q=0.025)/initN.loc[age_class]*100000*7, out['Im_inc'].sel(age_group=age_class).quantile(dim='draws', q=0.975)/initN.loc[age_class]*100000*7, color='black', alpha=0.15, label='Model 95% CI')
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
