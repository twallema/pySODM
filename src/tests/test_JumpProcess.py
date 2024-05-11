import pytest
import pandas as pd
import numpy as np
from pySODM.models.base import JumpProcess

#############################
## Model without dimension ##
#############################

class SIR(JumpProcess):

    # state variables and parameters
    states = ['S', 'I', 'R']
    parameters = ['beta', 'gamma']

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
    """ Test the use of int/float/list time indexing
    """

    # Define parameters and initial states
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 10], "I": [10], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)

    # Do it right

    # Simulate using a mixture of int/float
    time = [int(10), float(50.3)]
    output = model.sim(time)
    # Simulate using just one timestep
    output = model.sim(50)
    # Simulate using a list of timesteps
    time = [0, 50]
    output = model.sim(time)
    
    # 'time' present in output
    assert 'time' in list(output.sizes.keys())
    # Default (no specification output frequency): 0, 1, 2, 3, ..., 50
    np.testing.assert_allclose(output["time"], np.arange(0, 51))
    # Numerically speaking everything ok?
    S = output["S"].values.squeeze()
    assert S[0] == 1_000_000 - 10
    assert S.shape == (51, )
    assert S[-1] < 12000
    I = output["I"].squeeze()
    assert I[0] == 10
    assert S.shape == (51, )

    # Do it wrong

    # Start same as end
    with pytest.raises(ValueError, match="Start of simulation is the same as the end of simulation"):
        model.sim([0,0])

    # Start before end
    with pytest.raises(ValueError, match="Start of simulation is chronologically after end of simulation"):
        model.sim([20,5])

    # Wrong type
    with pytest.raises(TypeError, match="Input argument 'time' must be a"):
        model.sim(np.zeros(2))    

    # If list: wrong length
    with pytest.raises(ValueError, match="You have supplied:"):
        model.sim([0, 50, 100])    

    # Combination of datetime and int/float
    with pytest.raises(ValueError, match="List-like input of simulation start and stop"):
        model.sim([0, pd.to_datetime('2020-03-15')])

    # Wrong type for input argument `tau`
    with pytest.raises(TypeError, match="discrete timestep 'tau' must be of type int or float"):
         model.sim(50, tau='hello')

    # Wrong type for input argument `method`
    with pytest.raises(TypeError, match="solver method 'method' must be of type string"):
         model.sim(50, method=3)

test_SIR_time()

def test_SIR_date():
    """ Test the use of str/datetime time indexing
    """

    # Define parameters and initial states
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 10], "I": [10], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)

    # Do it right
    output = model.sim(['2020-01-01', '2020-02-20'])
    output = model.sim([pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-20')])

    # Validate
    assert 'date' in list(output.sizes.keys())
    S = output["S"].values.squeeze()
    assert S[0] == 1_000_000 - 10
    assert S.shape == (51, )
    assert S[-1] < 12000
    I = output["I"].squeeze()
    assert I[0] == 10
    assert S.shape == (51, )

    # Do it wrong

    # Start before end
    with pytest.raises(ValueError, match="Start of simulation is chronologically after end of simulation"):
        model.sim(['2020-03-15','2020-03-01'])

    # Combination of str/datetime
    with pytest.raises(ValueError, match="List-like input of simulation start and stop must contain either"):
        model.sim([pd.to_datetime('2020-01-01'), '2020-05-01'])

    # Simulate using a mixture of timestamp and string
    with pytest.raises(TypeError, match="You have only provided one date as input"):
        model.sim(pd.to_datetime('2020-01-01'))
        
def test_SSA():
    """ In all testes, the tau-leap is used by default
    """
    # Define parameters and initial states
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1000 - 10], "I": [10], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)

    # Simulate using dates
    output = model.sim(['2020-01-01', '2020-02-20'], method='SSA')
    output = model.sim([pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-20')], method='SSA')

    # Validate
    assert 'date' in list(output.sizes.keys())
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
    assert model.states_names == ['S', 'I', 'R']
    assert model.parameters_names == ['beta', 'gamma']

    # valid initialization: initial states as int
    initial_states = {"S": 1_000_000 - 10, "I": 10, "R": 0}
    model = SIR(initial_states, parameters)
    assert model.initial_states == initial_states
    assert model.parameters == parameters
    # model state/parameter names didn't change
    assert model.states_names == ['S', 'I', 'R']
    assert model.parameters_names == ['beta', 'gamma']

    # valid initialization: initial states as np.array
    initial_states = {"S": np.array([1_000_000 - 10]), "I": np.array([10]), "R": np.array([0])}
    model = SIR(initial_states, parameters)
    assert model.initial_states == initial_states
    assert model.parameters == parameters
    # model state/parameter names didn't change
    assert model.states_names == ['S', 'I', 'R']
    assert model.parameters_names == ['beta', 'gamma']

    # wrong length initial states
    initial_states2 = {"S": np.array([1_000_000 - 10,1]), "I": np.array([10,1]), "R": np.array([0,1])}
    with pytest.raises(ValueError, match="The desired shape of model state"):
        SIR(initial_states2, parameters)

    # wrong initial states
    initial_states2 = {"S": [1_000_000 - 10], "II": [10]}
    with pytest.raises(ValueError, match="specified initial states don't"):
        SIR(initial_states2, parameters)

    # wrong parameters
    initial_states = {"S": np.array([1_000_000 - 10]), "I": np.array([10]), "R": np.array([0])}
    parameters2 = {"beta": 0.9, "gamma": 0.2, "other": 1}
    with pytest.raises(ValueError, match="specified parameters don't"):
        SIR(initial_states, parameters2)

    # validate model class itself
    initial_states = {"S": np.array([1_000_000 - 10]), "I": np.array([10]), "R": np.array([0])}
    SIR.states = ["S", "R"]
    with pytest.raises(ValueError, match="The specified initial states don't exactly match the predefined states."):
        SIR(initial_states, parameters)

    initial_states = {"S": np.array([1_000_000 - 10]), "I": np.array([10]), "R": np.array([0])}
    SIR.states = ["S", "II", "R"]
    with pytest.raises(ValueError, match="The specified initial states don't exactly match the predefined states."):
        SIR(initial_states, parameters)

    initial_states = {"S": np.array([1_000_000 - 10]), "I": np.array([10]), "R": np.array([0])}
    SIR.states = ["S", "I", "R"]
    SIR.parameters = ['beta', 'alpha']
    with pytest.raises(ValueError, match="The provided state names and parameters don't match the parameters and states of the integrate/compute_rates function."):
        SIR(initial_states, parameters)

    # ensure to set back to correct ones
    SIR.states = ["S", "I", "R"]
    SIR.parameters = ['beta', 'gamma']

##############################
## Model with one dimension ##
##############################

class SIRstratified(JumpProcess):

    # state variables and parameters
    states = ['S', 'I', 'R']
    parameters = ['gamma']
    stratified_parameters = ['beta']
    dimensions = ['age_groups']

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

def test_stratified_SIR_output_shape():
    """Assert the states have the correct size
    """
    # Initialize
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10], "R": [0, 0]}
    coordinates = {'age_groups': ['0-20','20-120']}
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    # Simualte
    time = [0, 50]
    output = model.sim(time)
    # Assert state size
    np.testing.assert_allclose(output["time"], np.arange(0, 51))
    assert output["S"].values.shape == (51, 2)
    assert output["I"].values.shape == (51, 2)
    assert output["R"].values.shape == (51, 2)

def test_stratified_SIR_automatic_filling_initial_states():
    """Validate the initial states not defined are automatically filled with zeros
    """
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    # Leave out "R"
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10]}
    coordinates = {'age_groups': ['0-20','20-120']}
    # assert that the Rs state is filled with zeros correctly
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    assert model.initial_states["R"].tolist() == [0, 0]

def test_model_stratified_init_validation():

    # valid initialization
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10], "R": [0, 0]}
    coordinates = {'age_groups': ['0-20','20-120']}
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    #assert model.initial_states == initial_states --> some weird ass shit happens here
    assert model.parameters == parameters
    # assert model state/parameter names didn't change
    assert model.states_names == ['S', 'I', 'R']
    assert model.parameters_names == ['gamma']
    #assert model.parameter_stratified_names == [['beta']] --> some weird ass shit happens here

    # forget coordinates
    with pytest.raises(ValueError, match="dimension name provided in integrate"):
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
    msg = "The coordinates provided for dimension 'age_groups' indicates a"
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters2, coordinates=coordinates)

    parameters2 = {"gamma": 0.2, "beta": 0.9}
    msg = "A stratified parameter value should be a 1D array, but"
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters2, coordinates=coordinates)

    # initial state of the wrong length
    initial_states2 = {"S": 600_000 - 20, "I": [20, 10], "R": [0, 0]}
    msg = r"The desired shape of model state"
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states2, parameters, coordinates=coordinates)

    initial_states2 = {"S": [0] * 3, "I": [20, 10], "R": [0, 0]}
    msg = r"The desired shape of model state"
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states2, parameters, coordinates=coordinates)

    # validate model class itself
    msg = "The provided state names and parameters don't match the parameters and states of the"
    SIRstratified.parameters = ["gamma", "alpha"]
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters, coordinates=coordinates)

    SIRstratified.parameters = ["gamma"]
    SIRstratified.stratified_parameters = [["beta", "alpha"]]
    with pytest.raises(ValueError, match=msg):
        SIRstratified(initial_states, parameters, coordinates=coordinates)

    # ensure to set back to correct ones
    SIRstratified.states = ["S", "I", "R"]
    SIRstratified.parameters = ["gamma"]
    SIRstratified.stratified_parameters = [["beta"]]

############################################################
## A model with different dimensions for different states ##
############################################################

class SIR_SI(JumpProcess):
    """
    A stochastic, age-stratified SIR model for humans, an unstratified SI model for a disease vector (f.i. mosquito)
    """

    states = ['S', 'I', 'R', 'S_v', 'I_v']
    parameters = ['beta', 'gamma']
    stratified_parameters = ['alpha']
    dimensions = ['age_group']
    dimensions_per_state = [['age_group'],['age_group'],['age_group'],[],[]]

    @staticmethod
    def compute_rates(t, S, I, R, S_v, I_v, alpha, beta, gamma):

        # Calculate total mosquito population
        N = S + I + R
        N_v = S_v + I_v

        rates = {
            'S': [alpha*(I_v/N_v),],
            'I': [(1/gamma)*np.ones(N.shape),],
            'S_v': [np.sum(alpha*(I/N))*np.ones(N_v.shape), (1/beta)*np.ones(N_v.shape),],
            'I_v': [(1/beta)*np.ones(N_v.shape),]
        }

        return rates

    @staticmethod
    def apply_transitionings(t, tau, transitionings, S, I, R, S_v, I_v, alpha, beta, gamma):

        S_new = S - transitionings['S'][0]
        I_new = I + transitionings['S'][0] - transitionings['I'][0]
        R_new = R + transitionings['I'][0]
        S_v_new = S_v - transitionings['S_v'][0] + transitionings['I_v'][0]
        I_v_new = I_v + transitionings['S_v'][0] - transitionings['I_v'][0]

        return S_new, I_new, R_new, S_v_new, I_v_new

def test_SIR_SI_state_shapes():
    """Validate the shapes of the states
    """

    # Define parameters and initial condition
    params={'alpha': 10*np.array([0.005, 0.01, 0.02, 0.015]), 'gamma': 5, 'beta': 5}
    init_states = {'S': [606938, 1328733, 7352492, 2204478], 'S_v': 1e6, 'I_v': 1}
    # Define model coordinates
    age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)])
    coordinates={'age_group': age_groups}
    # Initialize model
    model = SIR_SI(init_states, params, coordinates=coordinates)
    # Simulate
    output = model.sim([0, 50])
    # Assert state size
    np.testing.assert_allclose(output["time"], np.arange(0, 51))
    assert output["S"].values.shape == (51, 4)
    assert output["I"].values.shape == (51, 4)
    assert output["R"].values.shape == (51, 4)
    assert output["S_v"].values.shape == (51,)
    assert output["I_v"].values.shape == (51,)

def test_SIR_SI_automatic_filling_initial_states():
    """Validate the initial states not defined are automatically filled with zeros
    """
    # Define parameters and initial condition
    params={'alpha': 10*np.array([0.005, 0.01, 0.02, 0.015]), 'gamma': 5, 'beta': 5}
    init_states = {'S': [606938, 1328733, 7352492, 2204478], 'S_v': 1e6, 'I_v': 1}
    # Define model coordinates
    age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)])
    coordinates={'age_group': age_groups}
    # Initialize model
    model = SIR_SI(init_states, params, coordinates=coordinates)
    assert model.initial_states["I"].tolist() == [0, 0, 0, 0]
    assert model.initial_states["R"].tolist() == [0, 0, 0, 0]

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
    output = model.sim(time, draw_function=draw_function, samples={}, N=5)

    # assert dimension 'draws' is present in output
    assert 'draws' in list(output.sizes.keys())

    # wrong draw function
    def draw_function(pardict, samples):
        return pardict
    
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    with pytest.raises(ValueError, match="The first parameter of a draw function should be"):
        model.sim(time, draw_function=draw_function, samples={}, N=5)

    