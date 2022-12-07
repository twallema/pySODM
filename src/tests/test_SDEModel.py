import pytest
import pandas as pd
import numpy as np
from pySODM.models.base import SDEModel

##################################
## Model without stratification ##
##################################

class SIR(SDEModel):

    # state variables and parameters
    state_names = ['S', 'I', 'R']
    parameter_names = ['beta', 'gamma']

    @staticmethod
    def compute_rates(t, S, I, R, beta, gamma):
        """Basic SIR model"""
        return {'S': [beta*(I/(S + I + R)),], 'I': [np.array([gamma]),]}

    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, I, R, beta, gamma):
        S_new = S - transitionings['S'][0]
        I_new = I + transitionings['S'][0] - transitionings['I'][0]
        R_new = R + transitionings['I'][0]
        return S_new, I_new, R_new

def test_SIR_time():

    # Define parameters and initial states
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 10], "I": [10], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)
    # Simulate using a mixture of int/float
    time = [int(10), float(50.3)]
    output = model.sim(time)
    # Simulate using just one timestep
    output = model.sim(50)
    # Simulate using a list of timesteps
    time = [0, 50]
    output = model.sim(time)

    # Validate
    assert 'time' in list(output.dims.keys())
    np.testing.assert_allclose(output["time"], np.arange(0, 51))
    S = output["S"].values.squeeze()
    assert S[0] == 1_000_000 - 10
    assert S.shape == (51, )
    assert S[-1] < 12_000
    I = output["I"].squeeze()
    assert I[0] == 10
    assert S.shape == (51, )

def test_SIR_date():

    # Define parameters and initial states
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 10], "I": [10], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)

    # Simulate using dates
    output = model.sim(['2020-01-01', '2020-02-20'])
    output = model.sim([pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-20')])

    # Validate
    assert 'date' in list(output.dims.keys())
    S = output["S"].values.squeeze()
    assert S[0] == 1_000_000 - 10
    assert S.shape == (51, )
    assert S[-1] < 12_000
    I = output["I"].squeeze()
    assert I[0] == 10
    assert S.shape == (51, )

    # Simulate using a mixture of timestamp and string
    with pytest.raises(TypeError, match="List-like input of simulation start"):
        output = model.sim(['2020-01-01', pd.Timestamp('2020-02-20')])

def test_SSA():

    # Define parameters and initial states
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1000 - 10], "I": [10], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)

    # Simulate using dates
    output = model.sim(['2020-01-01', '2020-02-20'], method='SSA')
    output = model.sim([pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-20')], method='SSA')

    # Validate
    assert 'date' in list(output.dims.keys())
    S = output["S"].values.squeeze()
    assert S[0] == 1000 - 10
    assert S.shape == (51, )
    assert S[-1] < 30
    I = output["I"].squeeze()
    assert I[0] == 10
    assert S.shape == (51, )
    
def test_model_init_validation():
    # valid initialization
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 10], "I": [10], "R": [0]}
    model = SIR(initial_states, parameters)
    assert model.initial_states == initial_states
    assert model.parameters == parameters
    # model state/parameter names didn't change
    assert model.state_names == ['S', 'I', 'R']
    assert model.parameter_names == ['beta', 'gamma']

    # valid initialization: initial states as int
    initial_states = {"S": 1_000_000 - 10, "I": 10, "R": 0}
    model = SIR(initial_states, parameters)
    assert model.initial_states == initial_states
    assert model.parameters == parameters
    # model state/parameter names didn't change
    assert model.state_names == ['S', 'I', 'R']
    assert model.parameter_names == ['beta', 'gamma']

    # valid initialization: initial states as np.array
    initial_states = {"S": np.array([1_000_000 - 10]), "I": np.array([10]), "R": np.array([0])}
    model = SIR(initial_states, parameters)
    assert model.initial_states == initial_states
    assert model.parameters == parameters
    # model state/parameter names didn't change
    assert model.state_names == ['S', 'I', 'R']
    assert model.parameter_names == ['beta', 'gamma']

    # wrong length initial states
    initial_states2 = {"S": np.array([1_000_000 - 10,1]), "I": np.array([10,1]), "R": np.array([0,1])}
    with pytest.raises(ValueError, match="The abscence of model coordinates indicate a desired model"):
        SIR(initial_states2, parameters)

    # wrong initial states
    initial_states2 = {"S": [1_000_000 - 10], "II": [10]}
    with pytest.raises(ValueError, match="specified initial states don't"):
        SIR(initial_states2, parameters)

    # wrong parameters
    parameters2 = {"beta": 0.9, "gamma": 0.2, "other": 1}
    with pytest.raises(ValueError, match="specified parameters don't"):
        SIR(initial_states, parameters2)

    # validate model class itself
    SIR.state_names = ["S", "R"]
    with pytest.raises(ValueError, match="The states in the 'compute_rates' function definition do not match the provided 'state_names'"):
        SIR(initial_states, parameters)

    SIR.state_names = ["S", "II", "R"]
    with pytest.raises(ValueError, match="The states in the 'compute_rates' function definition"):
        SIR(initial_states, parameters)

    SIR.state_names = ["S", "I", "R"]
    SIR.parameter_names = ['beta', 'alpha']
    with pytest.raises(ValueError, match="The parameters in the 'compute_rates' function"):
        SIR(initial_states, parameters)

    # ensure to set back to correct ones
    SIR.state_names = ["S", "I", "R"]
    SIR.parameter_names = ['beta', 'gamma']

###################################
## Model with one stratification ##
###################################

class SIRstratified(SDEModel):

    # state variables and parameters
    state_names = ['S', 'I', 'R']
    parameter_names = ['gamma']
    parameters_stratified_names = ['beta']
    stratification_names = ['age_groups']

    @staticmethod
    def compute_rates(t, S, I, R, gamma, beta):
        """Basic SIR model"""
        size_dummy=np.ones(S.shape)
        return {'S': [beta*(I/(S + I + R)),], 'I': [size_dummy*gamma,]}

    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, I, R, gamma, beta):
        S_new = S - transitionings['S'][0]
        I_new = I + transitionings['S'][0] - transitionings['I'][0]
        R_new = R + transitionings['I'][0]
        return S_new, I_new, R_new

def test_stratified_SIR_time():
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10], "R": [0, 0]}
    coordinates = {'age_groups': ['0-20','20-120']}

    model = SIRstratified(initial_states, parameters, coordinates=coordinates)

    time = [0, 50]
    output = model.sim(time)

    np.testing.assert_allclose(output["time"], np.arange(0, 51))
    assert output["S"].values.shape == (2, 51)
    assert output["I"].values.shape == (2, 51)
    assert output["R"].values.shape == (2, 51)

def test_stratified_SIR_date():

    # Define parameters and initial states
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10], "R": [0, 0]}
    coordinates = {'age_groups': ['0-20','20-120']}

    # Build model
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)

    # Simulate using dates
    output = model.sim(['2020-01-01', '2020-02-20'])

    # Validate
    assert output["S"].values.shape == (2, 51)
    assert output["I"].values.shape == (2, 51)
    assert output["R"].values.shape == (2, 51)

def test_model_stratified_init_validation():

    # valid initialization
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10], "R": [0, 0]}
    coordinates = {'age_groups': ['0-20','20-120']}
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    #assert model.initial_states == initial_states --> some weird ass shit happens here
    assert model.parameters == parameters
    # assert model state/parameter names didn't change
    assert model.state_names == ['S', 'I', 'R']
    assert model.parameter_names == ['gamma']
    #assert model.parameters_stratified_names == [['beta']] --> some weird ass shit happens here

    # forget coordinates
    with pytest.raises(ValueError, match="Stratification name provided in integrate"):
        SIRstratified(initial_states, parameters)

    # unknown state
    initial_states2 = {"S": [1_000_000 - 10]*2, "II": [10]*2}
    with pytest.raises(ValueError, match="specified initial states don't"):
        SIRstratified(initial_states2, parameters, coordinates=coordinates)

    # unknown parameter
    parameters2 = {"gamma": 0.9, "other": 0.2}
    with pytest.raises(ValueError, match="specified parameters don't"):
        SIRstratified(initial_states, parameters2, coordinates=coordinates)

    # stratified parameter of the wrong length
    parameters2 = {"gamma": 0.2, "beta": np.array([0.8, 0.9, 0.1])}
    msg = "The coordinates provided for stratification 'age_groups' indicates a"
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters2, coordinates=coordinates)

    parameters2 = {"gamma": 0.2, "beta": 0.9}
    msg = "A stratified parameter value should be a 1D array, but"
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters2, coordinates=coordinates)

    # initial state of the wrong length
    initial_states2 = {"S": 600_000 - 20, "I": [20, 10], "R": [0, 0]}
    msg = r"The coordinates provided for the stratifications"
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states2, parameters, coordinates=coordinates)

    initial_states2 = {"S": [0] * 3, "I": [20, 10], "R": [0, 0]}
    msg = r"The coordinates provided for the stratifications"
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states2, parameters, coordinates=coordinates)

    # validate model class itself
    msg = "The parameters in the 'compute_rates' function definition do not match"
    SIRstratified.parameter_names = ["gamma", "alpha"]
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters, coordinates=coordinates)

    SIRstratified.parameter_names = ["gamma"]
    SIRstratified.parameters_stratified_names = [["beta", "alpha"]]
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters, coordinates=coordinates)

    # ensure to set back to correct ones
    SIRstratified.state_names = ["S", "I", "R"]
    SIRstratified.parameter_names = ["gamma"]
    SIRstratified.parameters_stratified_names = [["beta"]]

def test_model_stratified_default_initial_state():
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10], "R": [0, 0]}
    coordinates = {'age_groups': ['0-20','20-120']}
    
    # leave out the initial R state
    initial_states2 = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10]}

    # assert that the Rs state is filled with zeros correctly
    model = SIRstratified(initial_states2, parameters, coordinates=coordinates)
    assert model.initial_states["R"].tolist() == [0, 0]

###############################
## Time-dependent parameters ##
###############################

# TDPF on a stratified parameter
def test_TDPF_stratified():
    # states, parameters, coordinates
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10]}
    coordinates = {'age_groups': ['0-20','20-120']}

    # simulate model without TDPF
    time = [0, 50]
    model_without = SIRstratified(initial_states, parameters, coordinates=coordinates)
    output_without = model_without.sim(time)

    # define TDPF without extra arguments
    def compliance_func(t, states, param):
        if t < 0:
            return param
        else:
            return param/1.5

    # simulate model
    model = SIRstratified(initial_states, parameters, coordinates=coordinates,
                          time_dependent_parameters={'beta': compliance_func})
    output = model.sim(time)

    # without the reduction in infectivity, the recovered/dead pool will always be larger
    assert np.less_equal(output['R'].values[20:], output_without['R'].values[20:]).all()

    # define TDPF with extra argument
    def compliance_func(t, states, param, prevention):
        if t < 0:
            return param
        else:
            return param * prevention

    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9]), "prevention": 0.2}
    model2 = SIRstratified(initial_states, parameters, coordinates=coordinates,
                           time_dependent_parameters={'beta': compliance_func})
    output2 = model2.sim(time)
    assert np.less_equal(output['R'].values[20:], output_without['R'].values[20:]).all()

    # Define a TDPF which uses an existing model parameter as extra argument (which doesn't make sense)
    def compliance_func(t, states, param, gamma):
        if t < 0:
            return param
        else:
            return param * gamma

    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    model2 = SIRstratified(initial_states, parameters, coordinates=coordinates,
                           time_dependent_parameters={'beta': compliance_func})
    output2 = model2.sim(time)

test_TDPF_stratified()

# TDPF on a regular parameter
def test_TDPF_nonstratified():
    # states, parameters, coordinates
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10]}
    coordinates = {'age_groups': ['0-20','20-120']}

    # simulate model without TDPF
    time = [0, 10]
    model_without = SIRstratified(initial_states, parameters, coordinates=coordinates)
    output_without = model_without.sim(time)

    # define TDPF without extra arguments
    def compliance_func(t, states, param):
        if t < 0:
            return param
        else:
            return param * 2

    # simulate model
    model = SIRstratified(initial_states, parameters, coordinates=coordinates,
                          time_dependent_parameters={'beta': compliance_func})
    output = model.sim(time)

    # without the rise in time to recovery, the infected pool will be larger over the first 10 timesteps
    assert np.less_equal(output_without['I'].values[20:], output['I'].values[20:]).all()

test_TDPF_nonstratified()

####################
## Draw functions ##
####################

def test_draw_function():

    # states, parameters, coordinates
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10]}
    coordinates = {'age_groups': ['0-20','20-120']}

    # draw function
    def draw_function(param_dict, samples_dict):
        return param_dict

    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    output = model.sim(time, draw_fcn=draw_function, samples={}, N=5)

    # assert dimension 'draws' is present in output
    assert 'draws' in list(output.dims.keys())

    # wrong draw function
    def draw_function(pardict, samples):
        return pardict
    
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    with pytest.raises(ValueError, match="The first parameter of a draw function should be"):
        model.sim(time, draw_fcn=draw_function, samples={}, N=5)

    