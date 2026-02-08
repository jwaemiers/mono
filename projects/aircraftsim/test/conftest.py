from typing import TypeAlias
import numpy as np
from scipy.integrate import solve_ivp
from aircraftsim import (
    VehicleModel,
    GravProperties,
    InertiaTensor,
    AeroProperties,
    DynamicsModel,
    AtmosphericModel,
    AircraftSim,
    PostProcessor,
)

NDArray1D: TypeAlias = np.ndarray


def get_inertia_tensor() -> InertiaTensor:
    return InertiaTensor(1, 2, 3, 4, 5, 6)


def get_grav_properties() -> GravProperties:
    return GravProperties(1, get_inertia_tensor())


def get_aero_properties() -> AeroProperties:
    return AeroProperties(0.1, 0.1)


def get_state_variables() -> dict[str, float]:
    variables = [
        "u",
        "v",
        "w",
        "p",
        "q",
        "r",
        "north",
        "east",
        "down",
        "roll",
        "pitch",
        "yaw",
    ]
    state = np.zeros(12)
    return dict(zip(variables, state))


def get_vehicle_state() -> NDArray1D:
    state_variables = get_state_variables()
    state = np.asarray(state_variables.values(), dtype=np.float128)
    return state


def get_dynamics_model() -> DynamicsModel:
    state_variables = get_state_variables()
    dynamics_model = DynamicsModel(state_variables)
    return dynamics_model


def get_vehicle_model() -> VehicleModel:
    gravs = get_grav_properties()
    aero = get_aero_properties()
    vehicle_model = VehicleModel(gravs, aero)
    return vehicle_model


def get_atmospheric_model_no_aero() -> AtmosphericModel:
    atmospheric_model = AtmosphericModel()
    return atmospheric_model


def get_post_processor() -> PostProcessor:
    post_processor = PostProcessor()
    return post_processor


def get_aircraft_simulation_no_aero() -> AircraftSim:
    aircraft_sim = AircraftSim(
        get_vehicle_model(),
        get_dynamics_model(),
        get_atmospheric_model_no_aero(),
        get_post_processor(),
    )
    return aircraft_sim


def integrator(func, initial_condition, t_span, ts=None):
    sol = solve_ivp(func, t_span, initial_condition, t_eval=ts)
    return sol.t, sol.y
