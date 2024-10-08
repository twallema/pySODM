import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pySODM.models.base import ODE

#############################
## Model without dimension ##
#############################

class SIR(ODE):
    """A Simple SIR model without dimension
    """

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
    # dates and times
    ## Start before end
    with pytest.raises(ValueError, match="start of simulation is chronologically after end of simulation"):
        model.sim([20,5])

    ## Start same as end
    with pytest.raises(ValueError, match="start of simulation is the same as the end of simulation"):
        model.sim([0,0])

    ## Wrong type
    with pytest.raises(TypeError, match="a single int/float representing the end of the simulation"):
        model.sim(np.zeros(2))    

    ## If list: wrong length
    with pytest.raises(ValueError, match="wrong length of list-like simulation start and stop"):
        model.sim([0, 50, 100])    

    ## Combination of datetime and int/float
    with pytest.raises(ValueError, match="simulation start and stop must have the format:"):
        model.sim([0, datetime(2020,3,15)])

    # simulation method
    ## Wrong type for input argument `rtol`
    with pytest.raises(TypeError, match="relative solver tolerance 'rtol' must be of type float"):
         model.sim(50, rtol='hello')

    ## Wrong type for input argument `method`
    with pytest.raises(TypeError, match="solver method 'method' must be of type string"):
         model.sim(50, method=3)

    ## Right type for input argument 'method' but non-existing method
    with pytest.raises(ValueError, match="invalid solution method 'bliblablu'. valid methods:"):
        model.sim(50, method='bliblablu')

    ## Wrong type for discrete timestep 'tau'
    with pytest.raises(TypeError, match="discrete timestep 'tau' must be of type int or float"):
        model.sim(50, tau='bliblablu')

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
    output = model.sim([datetime(2020,1,1), datetime(2020,2,20)])

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
    # dates and times
    ## Start before end
    with pytest.raises(ValueError, match="start of simulation is chronologically after end of simulation"):
        model.sim(['2020-03-15','2020-03-01'])

    ## Start same as end
    with pytest.raises(ValueError, match="start of simulation is the same as the end of simulation"):
        model.sim([datetime(2020,1,1),datetime(2020,1,1)])

    ## Combination of str/datetime
    with pytest.raises(ValueError, match="simulation start and stop must have the format:"):
        model.sim([datetime(2020,1,1), '2020-05-01'])

    ## string not formatted 'yyyy-mm-dd'
    with pytest.raises(ValueError, match="time data '2020/03/01' does not match format"):
        model.sim(['2020/03/01', '2020-05-01'])

    ## Only providing one date
    with pytest.raises(TypeError, match="a single int/float representing the end of the simulation"):
        model.sim(datetime(2020,1,1))
 
def test_SIR_discrete_stepper():
    """ Test the use of the discrete timestepper, `_solve_discrete()`
    """

    # Define parameters and initial states
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = {"S": [1_000_000 - 10], "I": [10], "R": [0]}
    # Build model
    model = SIR(initial_states, parameters)

    # Do it right

    # Simulate model
    output = model.sim(50, tau=1)
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
    with pytest.raises(TypeError, match="discrete timestep 'tau' must be of type int or float"):
         model.sim(50, tau='hello')

def test_model_init_validation():
    # valid initialization: initial states as lists
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

    # wrong datatype initial state
    initial_states2 = [1_000_000 - 10, 10, 0]
    with pytest.raises(TypeError, match="a callable function generating a dictionary."):
        SIR(initial_states2, parameters)

    # wrong length initial states
    initial_states2 = {"S": np.array([1_000_000 - 10,1]), "I": np.array([10,1]), "R": np.array([0,1])}
    with pytest.raises(ValueError, match="The desired shape of model state"):
        SIR(initial_states2, parameters)

    # wrong initial states
    initial_states2 = {"S": [1_000_000 - 10], "II": [10]}
    with pytest.raises(ValueError, match="specified initial states don't"):
        SIR(initial_states2, parameters)

    # wrong type on parameters
    parameters2 = [0.9, 0.2, 1]
    with pytest.raises(TypeError, match="'parameters' should be a dictionary."):
        SIR(initial_states, parameters2)

    # wrong parameters
    parameters2 = {"beta": 0.9, "gamma": 0.2, "other": 1}
    with pytest.raises(ValueError, match="specified parameters don't"):
        SIR(initial_states, parameters2)

    # validate model class itself
    SIR.states = ["S", "R"]
    with pytest.raises(ValueError):
        SIR(initial_states, parameters)

    SIR.states = ["S", "II", "R"]
    with pytest.raises(ValueError):
        SIR(initial_states, parameters)

    SIR.states = ["S", "I", "R"]
    SIR.parameters = ['beta', 'alpha']
    with pytest.raises(ValueError):
        SIR(initial_states, parameters)

    # ensure to set back to correct ones
    SIR.states = ["S", "I", "R"]
    SIR.parameters = ['beta', 'gamma']

##############################
## Model with one dimension ##
##############################

class SIRstratified(ODE):
    """An SIR Model with one dimension and the same state size
    """
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

def test_stratified_SIR_init_validation():

    # valid initialization
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10], "R": [0, 0]}
    coordinates = {'age_groups': ['0-20','20-120']}
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)

    assert model.initial_states == initial_states
    assert model.parameters == parameters
    assert model.states_names == ['S', 'I', 'R']
    assert model.parameters_names == ['gamma']

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
    msg = "The input arguments of the integrate/compute_rates function should be 't', followed by the model's 'states' and 'parameters'."
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

class SIR_SI(ODE):
    """
    An age-stratified SIR model for humans, an unstratified SI model for a disease vector (f.i. mosquito) with a twist
    The S states is higher-dimensional to test the possiblities
    """

    states = ['S', 'I', 'R', 'S_v', 'I_v']
    parameters = ['beta', 'gamma']
    stratified_parameters = ['alpha']
    dimensions = ['age_group']
    dimensions_per_state = [['age_group','age_group'],['age_group'],['age_group'],[],[]]

    @staticmethod
    def integrate(t, S, I, R, S_v, I_v, alpha, beta, gamma):
        # Make S state 1D again
        S = np.mean(S,axis=1)
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
        # Explode S state to 2D
        dS_new = np.zeros([len(dS), len(dS)])
        for i in range(len(dS)):
            dS_new[i,:] = dS 
        return dS_new, dI, dR, dS_v, dI_v

def test_SIR_SI_state_shapes():
    """Validate the shapes of the states
    """

    # Define parameters and initial condition
    params={'alpha': 10*np.array([0.005, 0.01, 0.02, 0.015]), 'gamma': 5, 'beta': 5}
    init_states = {'S': np.array([[606938, 1328733, 7352492, 2204478],[606938, 1328733, 7352492, 2204478],[606938, 1328733, 7352492, 2204478],[606938, 1328733, 7352492, 2204478]]), 'S_v': 1e6, 'I_v': 1}
    # Define model coordinates
    age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,15),(15,65),(65,120)])
    coordinates={'age_group': age_groups}
    # Initialize model
    model = SIR_SI(init_states, params, coordinates=coordinates)
    # Simulate
    output = model.sim([0, 50])
    # Assert state size
    np.testing.assert_allclose(output["time"], np.arange(0, 51))
    assert output["S"].values.shape == (51, 4, 4)
    assert output["I"].values.shape == (51, 4)
    assert output["R"].values.shape == (51, 4)
    assert output["S_v"].values.shape == (51,)
    assert output["I_v"].values.shape == (51,)

def test_SIR_SI_automatic_filling_initial_states():
    """Validate the initial states not defined are automatically filled with zeros
    """
    # Define parameters and initial condition
    params={'alpha': 10*np.array([0.005, 0.01, 0.02, 0.015]), 'gamma': 5, 'beta': 5}
    init_states = {'S': np.array([[606938, 1328733, 7352492, 2204478],[606938, 1328733, 7352492, 2204478],[606938, 1328733, 7352492, 2204478],[606938, 1328733, 7352492, 2204478]]), 'S_v': 1e6, 'I_v': 1}
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
            return param / 1.5

    # simulate model
    model = SIRstratified(initial_states, parameters, coordinates=coordinates,
                          time_dependent_parameters={'beta': compliance_func})
    output = model.sim(time)

    # without the reduction in infectivity, the recovered/dead pool will always be larger
    assert np.less_equal(output['R'].values, output_without['R'].values).all()

    # define TDPF with extra argument
    def compliance_func(t, states, param, prevention):
        if t < 0:
            return param
        else:
            return param * prevention

    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9]), "prevention": 0.2}
    model2 = SIRstratified(initial_states, parameters, coordinates=coordinates,
                           time_dependent_parameters={'beta': compliance_func})
    assert np.less_equal(output['R'].values, output_without['R'].values).all()
    assert model2.parameters == parameters

    # Define a TDPF which uses an existing model parameter as extra argument
    def compliance_func(t, states, param, gamma):
        if t < 0:
            return param
        else:
            return param * gamma

    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    model2 = SIRstratified(initial_states, parameters, coordinates=coordinates,
                           time_dependent_parameters={'beta': compliance_func})

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
    assert np.less_equal(output_without['I'].values, output['I'].values).all()

#################################
## Initial condition functions ##
#################################

def test_ICF():
    """ Test an ICF without parameters
    """

    # define initial condition function without arguments
    def set_initial_condition():
        return {"S": 1_000_000 - 10, "I": 10}
    
    # define parameters and initial states
    parameters = {"beta": 0.9, "gamma": 0.2}
    initial_states = set_initial_condition
    model = SIR(initial_states, parameters)

    # simulate the model
    output = model.sim([0, 50])

    # assert initial recovered were set to zero
    assert model.initial_states['R'] == 0, "initial state 'R' was not automatically set to zero"

    # assert no parameters have been added to the parameters dictionary
    assert model.parameters == parameters, "initial condition function without arguments has altered the model.parameters dictionary"

    # assert numerically everything is ok
    S = output["S"].values.squeeze()
    assert S[0] == 1_000_000 - 10, "initial condition functions alter simple SIR simulation"
    assert S.shape == (51, ), "initial condition functions alter simple SIR simulation"
    assert S[-1] < 12000, "initial condition functions alter simple SIR simulation"
    I = output["I"].squeeze()
    assert I[0] == 10, "initial condition functions alter simple SIR simulation"
    assert S.shape == (51, ), "initial condition functions alter simple SIR simulation"

def test_ICF_args():
    """ Test an ICF with parameters
    """

    # define initial condition function with arguments
    def set_initial_condition(S0, I0):
        return {"S": S0 - I0, "I": I0}
    
    # define parameters and initial states
    parameters = {"beta": 0.9, "gamma": 0.2, "S0": 1_000_000, "I0": 10}
    initial_states = set_initial_condition
    model = SIR(initial_states, parameters)

    # simulate the model
    output = model.sim([0, 50])

    # assert initial recovered were set to zero
    assert model.initial_states['R'] == 0, "initial state 'R' was not automatically set to zero"

    # assert no parameters have been added to the parameters dictionary
    assert model.parameters == parameters, "initial condition function without arguments has altered the model.parameters dictionary"

    # assert numerically everything is ok
    S = output["S"].values.squeeze()
    assert S[0] == 1_000_000 - 10, "initial condition functions alter simple SIR simulation"
    assert S.shape == (51, ), "initial condition functions alter simple SIR simulation"
    assert S[-1] < 12000, "initial condition functions alter simple SIR simulation"
    I = output["I"].squeeze()
    assert I[0] == 10, "initial condition functions alter simple SIR simulation"
    assert S.shape == (51, ), "initial condition functions alter simple SIR simulation"

def test_ICF_TDPF():
    """ Mix it up: ICF, TDPF and model have overlapping parameters
    """

    # define ICF with arguments
    def set_initial_condition(S0, I0, beta, overlapping_par):
        return {"S": S0 - I0, "I": I0}
    
    # define TDPF with arguments
    def compliance_func(t, states, param, beta, overlapping_par):
        return param
    
    # define parameters and initial states
    parameters = {"beta": 0.9, "gamma": 0.2, "S0": 1_000_000, "I0": 10, "overlapping_par": 0}
    initial_states = set_initial_condition
    model = SIR(initial_states, parameters, time_dependent_parameters={'beta': compliance_func})

    # simulate the model
    output = model.sim([0, 50])

    # assert numerically everything is ok
    S = output["S"].values.squeeze()
    assert S[0] == 1_000_000 - 10, "initial condition functions alter simple SIR simulation"
    assert S.shape == (51, ), "initial condition functions alter simple SIR simulation"
    assert S[-1] < 12000, "initial condition functions alter simple SIR simulation"
    I = output["I"].squeeze()
    assert I[0] == 10, "initial condition functions alter simple SIR simulation"
    assert S.shape == (51, ), "initial condition functions alter simple SIR simulation"

def break_ICF():
    """ Try out some abuses of ICFs
    """

    # forget parameters
    def set_initial_condition(S0, I0):
        return {"S": S0 - I0, "I": I0}
    parameters = {"beta": 0.9, "gamma": 0.2}
    with pytest.raises(ValueError, match="The specified parameters don't exactly match the predefined parameters."):
        model = SIR(set_initial_condition, parameters)
    
    # sets a non-existant state
    def set_initial_condition(S0, I0):
        return {"S": S0 - I0, "I": I0, "K": 10}
    parameters = {"beta": 0.9, "gamma": 0.2, "S0": 1_000_000, "I0": 10}
    with pytest.raises(ValueError, match="The specified initial states don't exactly match the predefined states."):
        model = SIR(set_initial_condition, parameters)

    # wrong state size
    def set_initial_condition(S0, I0):
        return {"S": S0 - I0, "I": np.array([I0, I0])}
    parameters = {"beta": 0.9, "gamma": 0.2, "S0": 1_000_000, "I0": 10}
    with pytest.raises(ValueError, match="The desired shape of model state"):
        model = SIR(set_initial_condition, parameters)

####################
## Draw functions ##
####################

def test_draw_function():

    # states, parameters, coordinates
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10]}
    coordinates = {'age_groups': ['0-20','20-120']}

    # correct draw function without additional arguments
    def draw_function(parameters):
        return parameters
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    output = model.sim(time, draw_function=draw_function, N=5)
    # assert dimension 'draws' is present in output
    assert 'draws' in list(output.sizes.keys())

    # correct draw function with additional arguments
    def draw_function(parameters, par_1, par_2):
        return parameters
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    output = model.sim(time, draw_function=draw_function, draw_function_kwargs={'par_1': 0, 'par_2': 0}, N=5)
    # assert dimension 'draws' is present in output
    assert 'draws' in list(output.sizes.keys())

def break_draw_function():

    # states, parameters, coordinates
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9])}
    initial_states = {"S": [600_000 - 20, 400_000 - 10], "I": [20, 10]}
    coordinates = {'age_groups': ['0-20','20-120']}

    # correct draw function but too few arguments provided in draw_function_kwargs
    def draw_function(parameters, par_1, par_2):
        return parameters
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    with pytest.raises(ValueError, match="incorrect arguments passed to draw function"):
        model.sim(time, draw_function=draw_function, draw_function_kwargs={'par_1': 0}, N=5)

    # correct draw function but too much arguments in draw_function_kwargs
    def draw_function(parameters):
        return parameters
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    with pytest.raises(ValueError, match="incorrect arguments passed to draw function"):
        model.sim(time, draw_function=draw_function, draw_function_kwargs={'par_1': 0}, N=5)

    # correct draw function but draw_function_kwargs is not a dict
    def draw_function(parameters):
        return parameters
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    with pytest.raises(TypeError, match="your `draw_function_kwargs` must be of type 'dict' but are of type"):
        model.sim(time, draw_function=draw_function, draw_function_kwargs='bliblablu', N=5)

    # correct draw function with extra args but user forgets to provide draw_function_kwargs to sim()
    def draw_function(parameters, par_1, par_2):
        return parameters
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    with pytest.raises(ValueError, match="the draw function 'draw_function' has 2 arguments in addition to the mandatory 'parameters' argument"):
        model.sim(time, draw_function=draw_function, N=5)

    # wrong draw function: not a function
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    with pytest.raises(TypeError, match="a 'draw function' must be callable"):
        model.sim(time, draw_function='bliblablu', N=5)

    # wrong draw function: first input argument is not 'parameters'
    def draw_function(par_1, parameters):
        return parameters
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    with pytest.raises(ValueError, match="must have 'parameters' as its first input. Its current inputs are"):
        model.sim(time, draw_function=draw_function, draw_function_kwargs={'par_1': 0}, N=5)

    # wrong draw function: output is not a dict
    def draw_function(parameters):
        return 5
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    with pytest.raises(TypeError, match="a draw function must return one dictionary containing the model's parameters."):
        model.sim(time, draw_function=draw_function, N=5)

    # wrong draw function: put a new key in model parameters dictionary that doesn't represent a model parameter
    def draw_function(parameters):
        parameters['bliblublo'] = 5
        return parameters
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_states, parameters, coordinates=coordinates)
    with pytest.raises(ValueError, match="must match the keys in input dictionary 'parameters'"):
        model.sim(time, draw_function=draw_function, N=5)

    # wrong draw function: draw function parameter alters initial condition thru IC function but resulting IC is wrong
    def initial_condition_function(S0):
        return {"S": S0, "I": [20, 10]}
    def draw_function(parameters):
        parameters['S0'] = np.array([100, 100, 100]) # wrong shape, must be length 2
        return parameters
    parameters = {"gamma": 0.2, "beta": np.array([0.8, 0.9]), "S0": [600_000 - 20, 400_000 - 10]}
    # simulate model
    time = [0, 10]
    model = SIRstratified(initial_condition_function, parameters, coordinates=coordinates)
    with pytest.raises(RuntimeError, match="draw function --> drawn parameter dictionary --> initial condition function --> invalid initial state."):
        model.sim(time, draw_function=draw_function, N=5)

    # user provides N but no draw function
    with pytest.raises(ValueError, match="attempting to perform N=100 repeated simulations without using a draw function"):
        model.sim(time, N=100)

    # or
    with pytest.raises(ValueError, match="attempting to perform N=100 repeated simulations without using a draw function"):
        model.sim(time, draw_function_kwargs={'arg_1': 0}, N=100)

########################
## Call all functions ##
########################

test_SIR_time()
test_SIR_date()
test_SIR_discrete_stepper()
test_model_init_validation()
test_stratified_SIR_output_shape()
test_stratified_SIR_automatic_filling_initial_states()
test_stratified_SIR_init_validation()
test_SIR_SI_state_shapes()
test_SIR_SI_automatic_filling_initial_states()
test_TDPF_stratified()
test_TDPF_nonstratified()
test_ICF()
test_ICF_args()
test_ICF_TDPF()
break_ICF()
test_draw_function()
break_draw_function()