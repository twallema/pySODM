import pytest
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pySODM.models.base import ODEModel
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_negative_binomial_noise
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson, ll_negative_binomial
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary

# Only tested for ODEModel but the model output is identical so this shouldn't matter

##############
## Settings ##
##############

# Data generation
starttime = 50
endtime = 60
warmup = starttime
n_datapoints = 20
alpha = 0.05

# Visualisations?
plot = False

###################################################################################
## Dummy dataset consisting of two stratifications: age groups and spatial units ##
###################################################################################

# Generate a multiindex dataframe with three index levels: time, age groups, spatial componenet
time = np.linspace(starttime, endtime, num=20)
age_groups = ['0-20', '20-120']
spatial_units = [0, 1, 2]
index = pd.MultiIndex.from_product([time, age_groups, spatial_units], names=['time', 'age_groups', 'spatial_units'])
df = pd.DataFrame(index=index, columns=['cases'], dtype=float)
# Generate overdispersed data
cases = np.expand_dims(np.random.negative_binomial(1/alpha, (1/alpha)/(np.exp((1/7)*time) + (1/alpha))), axis=1)
# Fill all columns of the dataframe with the same data
for age_group in age_groups:
    for spatial_unit in spatial_units:
        df.loc[slice(None), age_group, spatial_unit] = cases

##################################
## Model without stratification ##
##################################

class SIR(ODEModel):

    # state variables and parameters
    state_names = ['S', 'I', 'R']
    parameter_names = ['beta', 'gamma']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma):
        """Basic SIR model"""
        N = S + I + R
        dS = -beta*I*S/N
        dI = beta*I*S/N - gamma*I
        dR = gamma*I
        return dS, dI, dR

# ~~~~~~~~~~~~~~~~
# Correct approach
# ~~~~~~~~~~~~~~~~

def test_correct_approach_wo_stratification():

    # Define parameters and initial states
    parameters = {"beta": 0.1, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 1], "I": [1], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)
    # Define dataset
    data=[df.groupby(by=['time']).sum(),]
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha,]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability(model,pars,bounds,data,states,
                                                    log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)

    # PSO
    theta = pso.optimize(objective_function, kwargs={'simulation_kwargs':{'warmup': warmup}},
                        swarmsize=10, max_iter=10, processes=1, debug=True)[0]
    # Nelder-Mead
    theta = nelder_mead.optimize(objective_function, np.array(theta), [0.05,], 
                        kwargs={'simulation_kwargs':{'warmup': warmup}}, processes=1, max_iter=10)[0]
    if plot:
        # Visualize results
        model.parameters['beta'] = theta[0]
        out = model.sim([0,endtime])
        fig,ax=plt.subplots()
        ax.scatter(df.index.get_level_values('time').unique(), df.groupby(by=['time']).sum())
        ax.plot(out['time'], out['I'])
        plt.show()
        plt.close()

    # Variables
    n_mcmc = 10
    multiplier_mcmc = 5
    print_n = 5
    discard = 5
    samples_path='sampler_output/'
    fig_path='sampler_output/'
    identifier = 'username'
    # initialize objective function
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.05*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=objective_function.expanded_bounds)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': 0, 'end_calibration': 50, 'n_chains': nwalkers,
                'warmup': 0, 'labels': labels, 'parameters': pars, 'starting_estimate': list(theta)}
    # Sample 100 iterations
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, (), {'simulation_kwargs': {'warmup': warmup}},
                                   fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=1, progress=True,
                                   settings_dict=settings)
    #Generate samples dict
    samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, discard=discard)

    #Visualize result
    if plot:
        # Define draw function
        def draw_fcn(param_dict, samples_dict):
            param_dict['beta'] = np.random.choice(samples_dict['beta'])
            return param_dict
        # Simulate model
        out = model.sim([0,endtime+30], N=30, samples=samples_dict, draw_fcn=draw_fcn)
        # Add poisson observation noise
        out = add_negative_binomial_noise(out, alpha)
        # Visualize
        fig,ax=plt.subplots()
        ax.scatter(df.index.get_level_values('time').unique(), df.groupby(by=['time']).sum())
        for i in range(30):
            ax.plot(out['time'], out['I'].isel(draws=i), linewidth=1, alpha=0.1, color='blue')
        plt.show()
        plt.close()

def test_xarray_datasets():
    """ Test the use of an xarray.DataArray as dataset
    """
    # Define parameters and initial states
    parameters = {"beta": 0.1, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 1], "I": [1], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)
    # Define dataset
    data=[df.groupby(by=['time']).sum().to_xarray(),]
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha,]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability(model,pars,bounds,data,states,
                                                    log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)

test_xarray_datasets()

class SIR_nd_beta(ODEModel):

    # state variables and parameters
    state_names = ['S', 'I', 'R']
    parameter_names = ['beta', 'gamma']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma):
        """Basic SIR model"""
        beta = beta[0,0,0]
        N = S + I + R
        dS = -beta*I*S/N
        dI = beta*I*S/N - gamma*I
        dR = gamma*I
        return dS, dI, dR

def test_calibration_nd_parameter():
    """ Test the calibration of an n-dimensional parameter
    """
    # Define parameters and initial states
    parameters = {"beta": 0.1*np.ones([2,2,2]), "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 1], "I": [1], "R": [0]}
    # Build model
    model = SIR_nd_beta(initial_states, parameters)
    # Define dataset
    data=[df.groupby(by=['time']).sum(),]
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha,]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability(model,pars,bounds,data,states,
                                                    log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)
    # PSO
    theta = pso.optimize(objective_function,
                        swarmsize=10, max_iter=20, processes=1, debug=True)[0]
    # Nelder-Mead
    theta = nelder_mead.optimize(objective_function, np.array(theta), 8*[0.05,],
                        processes=1, max_iter=20)[0]

test_calibration_nd_parameter()

def break_stuff_wo_stratification():

    # Define parameters and initial states
    parameters = {"beta": 0.1, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 1], "I": [1], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)

    # Non-existant parameter
    # ~~~~~~~~~~~~~~~~~~~~~~
    # Define dataset
    data=[df.groupby(by=['time']).sum(),]
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [[],]
    # Calibated parameters and bounds
    pars = ['dummy',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(Exception, match="is not a valid model parameter!"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # Axes in data not present in model
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Define dataset
    data=[df.groupby(by=['time','age_groups']).sum(),]
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [[],]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(Exception, match="Your model has no stratifications."):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)
    
    # Wrong type of dataset
    # ~~~~~~~~~~~~~~~~~~~~~

    # Define dataset
    data=[np.array([0,1,2,3,4,5]),]
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [[],]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(Exception, match="expected pd.Series, pd.DataFrame"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # pd.DataFrame with more than one column
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Add extra column to dataset
    d = df.groupby(by=['time']).sum()
    d['bluh'] = df.groupby(by=['time']).sum().values
    # Define dataset
    data=[d,]
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [[],]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(Exception, match="expected one column."):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

# For some weird ass reason this doesn't run if I don't call it..
break_stuff_wo_stratification()

def break_log_likelihood_functions_wo_stratification():

    # Define parameters and initial states
    parameters = {"beta": 0.1, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 1], "I": [1], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)
    # Define dataset
    data=[df.groupby(by=['time']).sum(),]
    states = ["I",]
    weights = np.array([1,])
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]

    # Poisson log likelihood
    # ~~~~~~~~~~~~~~~~~~~~~~

    # no arguments
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = []
    # Setup objective function without priors and with negative weights 
    with pytest.raises(ValueError, match="The number of datasets"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # wrong type: float
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [alpha,]
    with pytest.raises(ValueError, match="dataset has no extra arguments. Expected an empty list"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # wrong type: list
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [[alpha,],]
    with pytest.raises(ValueError, match="dataset has no extra arguments. Expected an empty list"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # wrong type: np.array
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [np.array([alpha,]),]
    with pytest.raises(ValueError, match="dataset has no extra arguments. Expected an empty list"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # Negative binomial log likelihood
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # no arguments
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = []
    with pytest.raises(ValueError, match="The number of datasets"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # wrong type: list
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [[alpha,]]
    with pytest.raises(ValueError, match="accepted types are int, float, np.ndarray and pd.Series"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

break_log_likelihood_functions_wo_stratification()

###################################
## Model with one stratification ##
###################################

class SIRstratified(ODEModel):

    # state variables and parameters
    state_names = ['S', 'I', 'R']
    parameter_names = ['gamma']
    parameter_stratified_names = ['beta']
    stratification_names = ['age_groups']

    @staticmethod
    def integrate(t, S, I, R, gamma, beta):
        """Basic SIR model"""
        # Model equations
        N = S + I + R
        dS = -beta*S*I/N
        dI = beta*S*I/N - gamma*I
        dR = gamma*I
        return dS, dI, dR

# Calibration of the two elements [beta_0, beta_1] of beta, to the respective timeseries per age group
def test_correct_approach_with_one_stratification_0():

    # Setup model
    parameters = {"gamma": 0.2, "beta": np.array([0.1, 0.1])}
    initial_states = {"S": [500_000 - 1, 500_000 - 1], "I": [1, 1], "R": [0, 0]}
    coordinates = {'age_groups': ['0-20','20-120']}
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    # Variables that don't really change
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [[],]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Define dataset with a coordinate not in the model
    data=[df.groupby(by=['time','age_groups']).sum(),]
    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability(model,pars,bounds,data,states,
                                                    log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)
    # Extract formatted parameter_names, bounds and labels
    labels = objective_function.expanded_labels 
    bounds = objective_function.expanded_bounds

    # PSO
    theta = pso.optimize(objective_function,
                        swarmsize=10, max_iter=20, processes=1, debug=True)[0]
    # Nelder-Mead
    theta = nelder_mead.optimize(objective_function, np.array(theta), [0.05, 0.05],
                        processes=1, max_iter=20)[0]

    # Assert equality of betas!
    assert np.isclose(theta[0], theta[1], rtol=1e-01)

test_correct_approach_with_one_stratification_0()

def break_stuff_with_one_stratification():

    # Axes in data not present in model
    # Setup model
    parameters = {"gamma": 0.2, "beta": np.array([0.1, 0.1])}
    initial_states = {"S": [500_000 - 1, 500_000 - 1], "I": [1, 1], "R": [0, 0]}
    coordinates = {'age_groups': ['0-20','20-120']}
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    # Variables that don't really change
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_poisson,]
    log_likelihood_fnc_args = [[],]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Define dataset with a coordinate not in the model
    data=[df,]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(Exception, match="0th dataset coordinate 'spatial_units' is"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)
    
    # Coordinate in dataset not found in the model
    # Setup model
    parameters = {"gamma": 0.2, "beta": np.array([0.1, 0.1])}
    initial_states = {"S": [500_000 - 1, 500_000 - 1], "I": [1, 1], "R": [0, 0]}
    coordinates = {'age_groups': ['020','20-120']}
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    # Define dataset with a coordinate not in the model
    data=[df,]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(Exception, match="coordinate '0-20' of stratification 'age_groups' in the 0th"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

break_stuff_with_one_stratification()

def break_log_likelihood_functions_with_one_stratification():

    # Define parameters and initial states
    parameters = {"gamma": 0.2, "beta": np.array([0.1, 0.1])}
    initial_states = {"S": [500_000 - 1, 500_000 - 1], "I": [1, 1], "R": [0, 0]}
    coordinates = {'age_groups': ['0-20','20-120']}
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    # Define dataset
    data=[df.groupby(by=['time','age_groups']).sum(),]
    states = ["I",]
    weights = np.array([1,])
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]

    # Poisson log likelihood equals no stratification case (verified)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Negative binomial log likelihood
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # no arguments
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = []
    with pytest.raises(ValueError, match="The number of datasets"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # wrong type: list of incorrect length
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [[alpha,]]
    with pytest.raises(ValueError, match="length of list/1D np.array containing arguments of the log likelihood"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # wrong type: np.array of wrong dimensionality
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha*np.ones([5,5]), ]
    with pytest.raises(ValueError, match="np.ndarray containing arguments of the log likelihood function"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)    

    # correct type: np.array of right size
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha*np.ones([2]), ]
    log_posterior_probability(model,pars,bounds,data,states,
                                log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)        

break_log_likelihood_functions_with_one_stratification()

####################################
## Model with two stratifications ##
####################################

class SIRdoublestratified(ODEModel):

    # state variables and parameters
    state_names = ['S', 'I', 'R']
    parameter_names = ['gamma']
    parameter_stratified_names = [['beta'],[]]
    stratification_names = ['age_groups', 'spatial_units']

    @staticmethod
    def integrate(t, S, I, R, gamma, beta):
        """Basic SIR model"""
        # Model equations
        beta = np.expand_dims(beta,axis=1)
        N = S + I + R
        dS = -beta*S*I/N
        dI = beta*S*I/N - gamma*I
        dR = gamma*I
        return dS, dI, dR

def test_correct_approach_with_two_stratifications():
    # Setup model
    parameters = {"gamma": 0.2, "beta": np.array([0.1, 0.1])}
    initial_states = {"S": [[500_000 - 1, 500_000 - 1, 500_000 - 1],[500_000 - 1, 500_000 - 1, 500_000 - 1]], "I": [[1,1,1],[1,1,1]]}
    coordinates = {'age_groups': ['0-20','20-120'], 'spatial_units': [0,1,2]}
    model = SIRdoublestratified(initial_states, parameters, coordinates=coordinates)
    # Variables that don't really change
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha*np.ones([2,3]),]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Define dataset with a coordinate not in the model
    data=[df,]
    # Setup objective function without priors and with negative weights 
    objective_function = log_posterior_probability(model,pars,bounds,data,states,
                                                    log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels)
    # Extract formatted parameter_names, bounds and labels
    pars_postprocessing = objective_function.parameter_names_postprocessing
    labels = objective_function.expanded_labels 
    bounds = objective_function.expanded_bounds
    # PSO
    theta = pso.optimize(objective_function,
                        swarmsize=10, max_iter=30, processes=1, debug=True)[0]
    # Nelder-Mead
    theta = nelder_mead.optimize(objective_function, np.array(theta), [0.05, 0.05],
                        processes=1, max_iter=30)[0]

    # Assert equality of betas!
    assert np.isclose(theta[0], theta[1], rtol=1e-01)

test_correct_approach_with_two_stratifications()

def test_aggregation_function():
    # Setup model
    parameters = {"gamma": 0.2, "beta": np.array([0.1, 0.1])}
    initial_states = {"S": [[500_000 - 1, 500_000 - 1, 500_000 - 1],[500_000 - 1, 500_000 - 1, 500_000 - 1]], "I": [[1,1,1],[1,1,1]]}
    coordinates = {'age_groups': ['0-20','20-120'], 'spatial_units': [0,1,2]}
    model = SIRdoublestratified(initial_states, parameters, coordinates=coordinates)
    # Variables that don't really change
    states = ["I",]
    weights = np.array([1,])
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha*np.ones([2]),]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Define dataset with a coordinate not in the model
    data=[df.groupby(by=['time','age_groups']).sum(),]
    # Define an aggregation function
    def aggregation_function(output):
        return output.sum(dim='spatial_units')
    # Correct use
    objective_function = log_posterior_probability(model,pars,bounds,data,states,
                                                    log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels,aggregation_function=aggregation_function)
    # Correct use
    objective_function = log_posterior_probability(model,pars,bounds,data,states,
                                                    log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels,aggregation_function=[aggregation_function,])
    # Misuse
    with pytest.raises(ValueError, match="number of aggregation functions must be equal to one or"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels,aggregation_function=[aggregation_function,aggregation_function])                                                    

test_aggregation_function()

def break_log_likelihood_functions_with_two_stratifications():

    # Setup model
    parameters = {"gamma": 0.2, "beta": np.array([0.1, 0.1])}
    initial_states = {"S": [[500_000 - 1, 500_000 - 1, 500_000 - 1],[500_000 - 1, 500_000 - 1, 500_000 - 1]], "I": [[1,1,1],[1,1,1]]}
    coordinates = {'age_groups': ['0-20','20-120'], 'spatial_units': [0,1,2]}
    model = SIRdoublestratified(initial_states, parameters, coordinates=coordinates)
    # Variables that don't really change
    states = ["I",]
    weights = np.array([1,])
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]
    # Define dataset with a coordinate not in the model
    data=[df,]

    # np.array of wrong size
    # ~~~~~~~~~~~~~~~~~~~~~~

    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha*np.ones([1,3]),]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(ValueError, match="Shape of np.array containing arguments of the log likelihood function"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)
    
    # np.array with too many dimensions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha*np.ones([1,3,1]),]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(ValueError, match="Shape of np.array containing arguments of the log likelihood function"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # np.array with too little dimensions
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha*np.ones(1),]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(ValueError, match="Shape of np.array containing arguments of the log likelihood function"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

    # float
    # ~~~~~

    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [1,]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(TypeError, match="accepted types are np.ndarray and pd.Series"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)    

    # float in a list
    # ~~~~~~~~~~~~~~~

    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [[1,],]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(TypeError, match="accepted types are np.ndarray and pd.Series"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)    

    # list
    # ~~~~

    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [[0,1,2],]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(TypeError, match="accepted types are np.ndarray and pd.Series"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)    

    # np.array placed inside too many lists
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [[alpha*np.ones([1,3,1])],]
    # Setup objective function without priors and with negative weights 
    with pytest.raises(TypeError, match="accepted types are np.ndarray and pd.Series"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

break_log_likelihood_functions_with_two_stratifications()