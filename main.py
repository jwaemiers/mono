import numpy as np
from scipy.integrate import solve_ivp
from aircraftsim import (
    VehicleModel,
    AeroProperties,
    GravProperties,
    InertiaTensor,
    DynamicsModel,
    AtmosphericModel,
    AircraftSim,
    PostProcessor,
)


def generate_vehicle_model() -> VehicleModel:
    inertia_tensor = InertiaTensor(31183, 20513, 23035, 0, 3931, 0)
    vehicle_gravs = GravProperties(10.0, inertia_tensor)
    vehicle_aero = AeroProperties(cd=0.5, cl=0.0)
    vehicle_model = VehicleModel(vehicle_gravs, vehicle_aero)
    return vehicle_model


def generate_dynamics_model() -> DynamicsModel:
    state_variables: dict[str, float] = {
        "u": 0.0,
        "v": 0.0,
        "w": 0.0,
        "p": 0.0,
        "q": 0.0,
        "r": 0.0,
        "north": 0.0,
        "east": 0.0,
        "down": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }

    dynamics_model = DynamicsModel(state_variables)
    return dynamics_model


def generate_atmospheric_model() -> AtmosphericModel:
    atmospheric_model = AtmosphericModel()
    return atmospheric_model


def generate_post_processor() -> PostProcessor:
    post_processor = PostProcessor()
    return post_processor


def generate_aircraft_simulation() -> AircraftSim:
    aircraft_simulation = AircraftSim(
        generate_vehicle_model(),
        generate_dynamics_model(),
        generate_atmospheric_model(),
        generate_post_processor(),
    )
    return aircraft_simulation


def integrator(func, initial_condition, t_span, ts=None):
    sol = solve_ivp(func, t_span, initial_condition, t_eval=ts)
    return sol.t, sol.y


def main():
    aircraft_simulation = generate_aircraft_simulation()

    t_span = (0, 10)
    ts = np.linspace(*t_span, num=5)

    aircraft_simulation.integrate_vehicle_state(integrator, t_span, ts)
    aircraft_simulation.post_processor.standard_plots()


if __name__ == "__main__":
    main()
