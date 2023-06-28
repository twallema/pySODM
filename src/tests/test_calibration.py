import pytest
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pySODM.models.base import ODE
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_negative_binomial_noise
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson, ll_negative_binomial
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary

# Only tested for ODE but the model output is identical so this shouldn't matter

##############
## Settings ##
##############

# Data generation
starttime = 50
endtime = 60
warmup = starttime
n_datapoints = 20
alpha = 0.05

###################################################################################
## Dummy dataset consisting of two dimensions: age groups and spatial units ##
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

#############################
## Model without dimension ##
#############################

class SIR(ODE):

    # state variables and parameters
    states = ['S', 'I', 'R']
    parameters = ['beta', 'gamma']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma):
        """Basic SIR model"""
        N = S + I + R
        dS = -beta*I*S/N
        dI = beta*I*S/N - gamma*I
        dR = gamma*I
        return dS, dI, dR

# Do the same for every input argument of the log_posterior_probability
def test_weights():
    # Define parameters and initial states
    parameters = {"beta": 0.1, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 1], "I": [1], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)
    # Define dataset
    data=[df.groupby(by=['time']).sum(),]
    states = ["I",]
    log_likelihood_fnc = [ll_negative_binomial,]
    log_likelihood_fnc_args = [alpha,]
    # Calibated parameters and bounds
    pars = ['beta',]
    labels = ['$\\beta$',]
    bounds = [(1e-6,1),]

    # Correct: list
    log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,[1,],labels=labels)
    # Correct: np.array
    log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,np.array([1,]),labels=labels)
    # Incorrect: list of wrong length
    with pytest.raises(ValueError, match="the extra arguments of the log likelihood function"):
        log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,[1,1],labels=labels)
    # Incorrect: numpy array with more than one dimension
    with pytest.raises(TypeError, match="Expected a 1D np.array for input argument"):
        log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,np.ones([3,3]),labels=labels)
    # Incorrect: datatype string
    with pytest.raises(TypeError, match="Expected a list/1D np.array for input argument"):
        log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,'hey',labels=labels)

def test_correct_approach_wo_dimension():

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

    if __name__ == "__main__":
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
        ndim, nwalkers, pos = perturbate_theta(theta, pert=0.05*np.ones(len(theta)), bounds=objective_function.expanded_bounds, multiplier=multiplier_mcmc)
        # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
        settings={'start_calibration': 0, 'end_calibration': 50, 'n_chains': nwalkers,
                    'warmup': 0, 'labels': labels, 'parameters': pars, 'starting_estimate': list(theta)}
        # Sample 100 iterations
        sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, (), {'simulation_kwargs': {'warmup': warmup}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=1, progress=True,
                                    settings_dict=settings)
        #Generate samples dict
        samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, discard=discard)

def break_stuff_wo_dimension():

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
    with pytest.raises(Exception, match="Your model has no dimensions."):
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

def break_log_likelihood_functions_wo_dimension():

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

class SIR_nd_beta(ODE):

    # state variables and parameters
    states = ['S', 'I', 'R']
    parameters = ['beta', 'gamma']

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

###################################
## Model with one dimension ##
###################################

class SIRstratified(ODE):

    # state variables and parameters
    states = ['S', 'I', 'R']
    parameters = ['gamma']
    stratified_parameters = ['beta']
    dimensions = ['age_groups']

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
def test_correct_approach_with_one_dimension_0():

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

def break_stuff_with_one_dimension():

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
    with pytest.raises(Exception, match="coordinate '0-20' of dimension 'age_groups' in the 0th"):
        log_posterior_probability(model,pars,bounds,data,states,
                                    log_likelihood_fnc,log_likelihood_fnc_args,-weights,labels=labels)

def break_log_likelihood_functions_with_one_dimension():

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

    # Poisson log likelihood equals no dimension case (verified)
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

####################################
## Model with two dimensions ##
####################################

class SIRdoublestratified(ODE):

    # state variables and parameters
    states = ['S', 'I', 'R']
    parameters = ['gamma']
    stratified_parameters = [['beta'],[]]
    dimensions = ['age_groups', 'spatial_units']

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

def test_correct_approach_with_two_dimensions():
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
    pars_postprocessing = objective_function.parameters_names_postprocessing
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

def break_log_likelihood_functions_with_two_dimensions():

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

##################################################
## Model where states have different dimensions ##
##################################################

class ODE_SIR_SI(ODE):
    """
    An age-stratified SIR model for humans, an unstratified SI model for a disease vector (f.i. mosquito)
    """

    states = ['S', 'I', 'R', 'S_v', 'I_v']
    parameters = ['beta', 'gamma']
    stratified_parameters = ['alpha']
    dimensions = ['age_groups']
    dimensions_per_state = [['age_groups'],['age_groups'],['age_groups'],[],[]]

    @staticmethod
    def integrate(t, S, I, R, S_v, I_v, alpha, beta, gamma):

        # Calculate total mosquito population
        N = S + I + R
        N_v = S_v + I_v
        # Calculate human differentials
        dS = -alpha*(I_v/N_v)*S
        dI = alpha*(I_v/N_v)*S - 1/gamma*I
        dR = 1/gamma*I
        # Calculate mosquito differentials
        dS_v = -np.sum(alpha*(I/N)*S_v) + (1/beta)*N_v - (1/beta)*S_v
        dI_v = np.sum(alpha*(I/N)*S_v) - (1/beta)*I_v

        return dS, dI, dR, dS_v, dI_v

def test_SIR_SI():

    #################
    ## Setup model ##
    #################

    # Define parameters and initial condition
    params={'alpha': np.array([0.05, 0.1]), 'gamma': 5, 'beta': 7}
    init_states = {'S': [606938, 1328733], 'S_v': 1e6, 'I_v': 2}
    # Define model coordinates
    age_groups = ['0-20','20-120']
    coordinates={'age_groups': age_groups}
    # Initialize model
    model = ODE_SIR_SI(init_states, params, coordinates=coordinates)

    # Variables that don't really change
    states = ["I","I_v"]
    weights = np.array([1,1])
    log_likelihood_fnc = [ll_negative_binomial,ll_negative_binomial]
    log_likelihood_fnc_args = [len(age_groups)*[alpha,], alpha]
    # Calibated parameters and bounds
    pars = ['alpha','beta']
    labels = ['$\\alpha$','$\\beta$']
    bounds = [(1e-4,1),(1,21)]
    # Human population --> calibrate to age stratified data
    # Vector population --> calibrate to unstratified data
    data=[df.groupby(by=['time','age_groups']).sum(), df.groupby(by=['time']).sum()]
    # Correct use
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,weights,labels=labels,)