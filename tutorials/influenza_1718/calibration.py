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
from functools import lru_cache
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
initN = pd.Series(index=age_groups, data=np.array([606938, 1328733, 7352492, 2204478]))

################
## Load model ##
################

from models import SDE_influenza_model as influenza_model

#####################################
## Define a social policy function ##
#####################################

class make_contact_matrix_function():

    # Initialize class with contact matrices
    def __init__(self, Nc_work, Nc_school, Nc_except_workschool):
        self.Nc_work = Nc_work
        self.Nc_school = Nc_school
        self.Nc_except_workschool = Nc_except_workschool
    
    # Define a call function to return the right contact matrix
    @lru_cache()
    def __call__(self, t, work=1, school=1):
        return self.Nc_except_workschool + work*self.Nc_work + school*self.Nc_school
    
    # Define a pySODM compatible wrapper with the social policies
    def contact_function(self, t, states, param):
        if t <= pd.Timestamp('2017-12-20'):
            return self.__call__(t)
        # Christmass holiday
        elif pd.Timestamp('2017-12-20') < t <= pd.Timestamp('2018-01-05'):
            return self.__call__(t, work=0.60, school=0)
        # Christmass holiday --> Winter holiday
        elif pd.Timestamp('2017-01-05') < t <= pd.Timestamp('2018-02-09'):
            return self.__call__(t)
        # Winter holiday
        elif pd.Timestamp('2018-02-09') < t <= pd.Timestamp('2018-02-16'):
            return self.__call__(t, work=0.80, school=0)
        # Winter holiday --> Easter holiday
        elif pd.Timestamp('2018-02-16') < t <= pd.Timestamp('2018-03-28'):
            return self.__call__(t)
        # Easter holiday
        elif pd.Timestamp('2018-03-28') < t <= pd.Timestamp('2018-04-16'):
            return self.__call__(t, work=0.60, school=0)
        else:
            return self.__call__(t)

# Hardcode the contact matrices
Nc_except_workschool = np.transpose(np.array([[0.68,0.78,3.27,0.45],
                                              [0.41,2.15,3.52,0.46],
                                              [0.29,0.59,3.50,0.49],
                                              [0.15,0.30,1.85,1.45]]))

Nc_school = np.transpose(np.array([[2.01*(2/5),0.27*(2/5),0.40*(2/5),0.00*(2/5)],
                                   [0.14*(2/5),3.21,0.27,0.00],
                                   [0.04*(2/5),0.05,0.27,0.00],
                                   [0.00*(2/5),0.00,0.00,0.00]]))

Nc_work = np.transpose(np.array([[0.00,0.00,0.47,0.05],
                                 [0.00,0.00,0.91,0.00],
                                 [0.04,0.15,3.12,0.16],
                                 [0.02,0.00,0.61,0.04]]))

# Initialize contact function
contact_function = make_contact_matrix_function(Nc_work, Nc_school, Nc_except_workschool).contact_function

#################
## Setup model ##
#################

# Number of repeated experiments
N=20
# Set start date and warmup
warmup=0
start_idx=0
start_date = df_influenza.index.get_level_values('date').unique()[start_idx]
end_date = df_influenza.index.get_level_values('date').unique()[-1] 
sim_len = (end_date - start_date)/pd.Timedelta(days=1)+warmup
# Get initial condition
I_init = df_influenza.loc[start_date]
# Define model parameters
f_a = np.array([0.02, 0.61, 0.88, 0.75])
gamma = 5
params={'beta':0.10, 'sigma':1, 'f_a':f_a, 'gamma':5, 'Nc':Nc_except_workschool+Nc_school+Nc_work}
# Define initial condition
init_states = {'S':initN.values ,'E': (1/(1-f_a))*df_influenza.loc[start_date, slice(None)],
                                 'Ia': (f_a/(1-f_a))*gamma*df_influenza.loc[start_date, slice(None)],
                                 'Im': gamma*df_influenza.loc[start_date, slice(None)],
                                 'Im_inc': df_influenza.loc[start_date, slice(None)]}
# Define model coordinates
coordinates={'age_group': age_groups}
# Initialize model
model = influenza_model(init_states,params,coordinates,time_dependent_parameters={'Nc': contact_function})

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #######################
    ## Variance analysis ##
    #######################

    results, ax = variance_analysis(df_influenza, resample_frequency='5D')
    alpha = results.loc[(slice(None),'negative binomial'), 'theta'].values
    #plt.show()
    plt.close()

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # Variables
    processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()/2))
    n_pso = 50
    multiplier_pso = 20
    # Define dataset
    data=[df_influenza[start_date:end_date], ]
    states = ["Im_inc",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha,]
    # Calibated parameters and bounds
    pars = ['beta', 'f_a']
    labels = ['$\\beta$', '$f_a$']
    bounds = [(1e-6,0.05), (0,0.99)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)
    # Extract expanded bounds and labels
    expanded_labels = objective_function.expanded_labels 
    expanded_bounds = objective_function.expanded_bounds                                   
    # PSO
    #theta = pso.optimize(objective_function, kwargs={'simulation_kwargs':{'warmup': warmup}},
    #                   swarmsize=multiplier_pso*processes, max_iter=n_pso, processes=processes, debug=True)[0]
    theta = [0.0345, 0.02, 0.61, 0.88, 0.75]

    # Nelder-mead
    #step = len(expanded_bounds)*[0.01,]
    #theta = nelder_mead.optimize(objective_function, np.array(theta), step, kwargs={'simulation_kwargs':{'warmup': warmup}}, processes=processes, max_iter=n_pso)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_date, end_date], warmup=warmup, samples={}, N=N)
    # Add poisson obervational noise
    out = add_negative_binomial_noise(out, np.mean(alpha))
    # Visualize
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8,6))
    axs = axs.reshape(-1)
    for id, age_class in enumerate(df_influenza.index.get_level_values('age_group').unique()):
        # Data
        axs[id].plot(df_influenza.index.get_level_values('date').unique(),df_influenza.loc[slice(None),age_class], color='orange')
        #axs[id].scatter(df_influenza.index.get_level_values('date').unique(), df_influenza.loc[slice(None),age_class], color='black', alpha=0.4, linestyle='None', facecolors='none', s=60, linewidth=2, label='data')
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
    n_mcmc = 100
    multiplier_mcmc = 9
    print_n = 5
    discard=50
    samples_path='sampler_output/'
    fig_path='sampler_output/'
    identifier = 'username'
    run_date = str(datetime.date.today())
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.01*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=expanded_bounds)
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
        idx, param_dict['beta'] = random.choice(list(enumerate(samples_dict['beta'])))
        param_dict['f_a'] = np.array([slice[idx] for slice in samples_dict['f_a']])
        return param_dict
    # Simulate model
    out = model.sim([start_date, end_date], warmup=warmup, N=N, samples=samples_dict, draw_function=draw_fcn, processes=processes)
    # Add poisson observation noise
    out = add_negative_binomial_noise(out, alpha)
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
