import math
import pytest
import numpy as np
from conftest import get_aircraft_simulation_no_aero, integrator


def is_equal_values(a, b, max_rel_error=10**-9) -> bool:
    rel_error = abs(a - b) / min(abs(a), abs(b))
    return rel_error <= max_rel_error


def is_equal_vectors(a, b, max_rms_error=10**-9) -> bool:
    mag = min(np.linalg.norm(a), np.linalg.norm(b))
    diff = np.linalg.norm(a - b)
    rms_error = float(diff / mag)
    return rms_error <= max_rms_error


## -- NO AERO TESTS -- #
@pytest.mark.parametrize(
    "speed, end_time",
    [
        (0, 5),
        (0, 100),
        (1, 5),
        (1, 100),
        (5, 5),
        (5, 100),
        (10, 5),
        (10, 100),
        (100, 5),
        (100, 100),
    ],
)
def test_inertial_position_from_u_freefall_no_aero(speed, end_time):
    aircraft_simulation = get_aircraft_simulation_no_aero()
    aircraft_simulation.dynamics_model.initial_state[0] = speed
    final_state = aircraft_simulation.get_final_time_state_vector(integrator, end_time)
    final_inertial_position = aircraft_simulation.dynamics_model.get_inertial_position(
        final_state
    )

    analytical_solution = np.array([speed * end_time, 0.0, 0.5 * 9.81 * end_time**2])

    pass_condition: bool = is_equal_vectors(
        final_inertial_position, analytical_solution
    )
    print(pass_condition)
    assert pass_condition


@pytest.mark.parametrize(
    "speed, end_time",
    [
        (0, 5),
        (0, 100),
        (1, 5),
        (1, 100),
        (5, 5),
        (5, 100),
        (10, 5),
        (10, 100),
        (100, 5),
        (100, 100),
    ],
)
def test_inertial_position_from_v_freefall_no_aero(speed, end_time):
    aircraft_simulation = get_aircraft_simulation_no_aero()
    aircraft_simulation.dynamics_model.initial_state[1] = speed
    final_state = aircraft_simulation.get_final_time_state_vector(integrator, end_time)
    final_inertial_position = aircraft_simulation.dynamics_model.get_inertial_position(
        final_state
    )

    analytical_solution = np.array([0.0, speed * end_time, 0.5 * 9.81 * end_time**2])

    pass_condition: bool = is_equal_vectors(
        final_inertial_position, analytical_solution
    )
    print(pass_condition)
    assert pass_condition


@pytest.mark.parametrize(
    "speed, end_time",
    [
        (0, 5),
        (0, 100),
        (1, 5),
        (1, 100),
        (5, 5),
        (5, 100),
        (10, 5),
        (10, 100),
        (100, 5),
        (100, 100),
    ],
)
def test_inertial_position_from_w_freefall_no_aero(speed, end_time):
    aircraft_simulation = get_aircraft_simulation_no_aero()
    aircraft_simulation.dynamics_model.initial_state[2] = speed
    final_state = aircraft_simulation.get_final_time_state_vector(integrator, end_time)
    final_inertial_position = aircraft_simulation.dynamics_model.get_inertial_position(
        final_state
    )

    analytical_solution = np.array(
        [0.0, 0.0, speed * end_time + 0.5 * 9.81 * end_time**2]
    )

    pass_condition: bool = is_equal_vectors(
        final_inertial_position, analytical_solution
    )
    print(pass_condition)
    assert pass_condition


@pytest.mark.parametrize(
    "speed, end_time",
    [
        (0, 5),
        (0, 100),
        (0.1, 5),
        (1, 100),
        (5, 5),
        (5, 100),
    ],
)
def test_inertial_position_from_p_freefall_no_aero(speed, end_time):
    aircraft_simulation = get_aircraft_simulation_no_aero()
    aircraft_simulation.dynamics_model.initial_state[3] = speed
    final_state = aircraft_simulation.get_final_time_state_vector(integrator, end_time)
    final_inertial_position = aircraft_simulation.dynamics_model.get_inertial_position(
        final_state
    )

    analytical_solution = np.array([0.0, 0.0, 0.5 * 9.81 * end_time**2])

    pass_condition: bool = is_equal_vectors(
        final_inertial_position, analytical_solution
    )
    assert pass_condition


@pytest.mark.parametrize(
    "speed, end_time",
    [
        (0, 5),
        (0, 100),
        (1, 5),
        (1, 100),
        (5, 5),
        (5, 100),
    ],
)
def test_inertial_position_from_q_freefall_no_aero(speed, end_time):
    aircraft_simulation = get_aircraft_simulation_no_aero()
    aircraft_simulation.dynamics_model.initial_state[4] = speed
    final_state = aircraft_simulation.get_final_time_state_vector(integrator, end_time)
    final_inertial_position = aircraft_simulation.dynamics_model.get_inertial_position(
        final_state
    )

    analytical_solution = np.array([0.0, 0.0, 0.5 * 9.81 * end_time**2])

    pass_condition: bool = is_equal_vectors(
        final_inertial_position, analytical_solution
    )
    assert pass_condition


@pytest.mark.parametrize(
    "speed, end_time",
    [
        (0, 5),
        (0, 100),
        (1, 5),
        (1, 100),
        (5, 5),
        (5, 100),
    ],
)
def test_inertial_position_from_r_freefall_no_aero(speed, end_time):
    aircraft_simulation = get_aircraft_simulation_no_aero()
    aircraft_simulation.dynamics_model.initial_state[5] = speed
    final_state = aircraft_simulation.get_final_time_state_vector(integrator, end_time)
    final_inertial_position = aircraft_simulation.dynamics_model.get_inertial_position(
        final_state
    )

    analytical_solution = np.array([0.0, 0.0, 0.5 * 9.81 * end_time**2])

    pass_condition: bool = is_equal_vectors(
        final_inertial_position, analytical_solution
    )
    assert pass_condition
