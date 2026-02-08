import numpy as np
from conftest import (
    get_dynamics_model,
    get_grav_properties,
    get_atmospheric_model_no_aero,
    get_vehicle_model,
)


## -- DYNAMICS MODEL TESTS -- ##
def test_state_variable_and_state_vector_lengths_match():
    dynamics_model = get_dynamics_model()
    state_vars = dynamics_model.state_variables
    state_vec = dynamics_model.initial_state
    pass_condition: bool = len(state_vars) == len(state_vec)
    assert pass_condition


def test_kinematics_state_derivative_returns_correct_length():
    dynamics_model = get_dynamics_model()
    vehicle_model = get_vehicle_model()
    atmospheric_model = get_atmospheric_model_no_aero()
    state_vec = dynamics_model.initial_state
    kinematics_state_derivative = dynamics_model.get_kinematics_state_derivative(
        0.0, state_vec, vehicle_model, atmospheric_model
    )
    pass_condition: bool = len(kinematics_state_derivative) == len(state_vec)
    assert pass_condition


## -- NAVIGATION EQN TESTS -- ##
def test_navigation_eqn_returns_correct_length():
    dynamics_model = get_dynamics_model()
    intial_state_vec = dynamics_model.initial_state
    navigation_eqn_return = dynamics_model.navigation_equation(intial_state_vec)
    pass_condition: bool = len(navigation_eqn_return) == 3
    assert pass_condition


## -- ROTATION EQNS TESTS -- ##
def test_rotation_eqn_returns_correct_length():
    dynamics_model = get_dynamics_model()
    intial_state_vec = dynamics_model.initial_state
    rotation_eqn_return = dynamics_model.rotation_kinematic_equation(intial_state_vec)
    pass_condition: bool = len(rotation_eqn_return) == 3
    assert pass_condition


## -- FORCE BALANCE EQN TESTS -- ##
def test_force_eqn_returns_correct_length():
    dynamics_model = get_dynamics_model()
    intial_state_vec = dynamics_model.initial_state

    body_net_force = np.zeros(3)
    body_gravity = np.zeros(3)
    grav_properties = get_grav_properties()

    force_eqn_return = dynamics_model.force_balance_equation(
        intial_state_vec, body_net_force, body_gravity, grav_properties
    )
    pass_condition: bool = len(force_eqn_return) == 3
    assert pass_condition


## -- MOMENT BALANCE EQN TESTS -- ##
def test_moment_eqn_returns_correct_length():
    dynamics_model = get_dynamics_model()
    intial_state_vec = dynamics_model.initial_state

    body_net_moment = np.zeros(3)
    grav_properties = get_grav_properties()

    moment_eqn_return = dynamics_model.moment_balance_equation(
        intial_state_vec, body_net_moment, grav_properties
    )
    pass_condition: bool = len(moment_eqn_return) == 3
    assert pass_condition
