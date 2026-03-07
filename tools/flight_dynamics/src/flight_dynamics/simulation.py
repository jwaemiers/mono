import numpy as np
from control_input import ControlInput
from dynamics_model import DynamicsModel, DynamicsModelPointMass, DynamicsModel6DOF
from environment_model import (
    EnvironmentModel,
    NoWindConstantGravSeaLevelEnvironment,
    NoAeroConstantGravEnvironment,
)
from integrator import Integrator
from reference_frame_model import FrameModel, FrameModelFlatEarthNED
from vehicle_model import VehicleModel
from plotting_utils import Plotter


class Simulation:
    def __init__(self, environment, vehicle, frame, dynamics, integrator) -> None:
        self.environment_model: EnvironmentModel = environment
        self.vehicle_model: VehicleModel = vehicle
        self.frame_model: FrameModel = frame
        self.dynamics_model: DynamicsModel = dynamics
        self.integrator: Integrator = integrator

    def solve_dynamics(self, intial_state_vector, t_span) -> tuple:
        def dynamics_wrapper(time, state):
            control = ControlInput()
            control.throttle_percent = 0.0
            return self.dynamics_model.state_derivative(time, state, control)

        t_eval = np.linspace(t_span[0], t_span[1], 50)

        return self.integrator.integrate(
            dynamics_wrapper, t_span, intial_state_vector, t_eval
        )


def point_mass_3dof_model():

    no_wind_environment = NoWindConstantGravSeaLevelEnvironment()

    vehicle_model = VehicleModel()
    vehicle_model.mass = 1.0
    vehicle_model.cl = 0.1
    vehicle_model.cd = 0.5
    vehicle_model.reference_area = 1.0
    vehicle_model.max_thrust = 1.0

    flat_earth_ned_frame = FrameModelFlatEarthNED()

    dynamics_model = DynamicsModelPointMass(
        vehicle_model, no_wind_environment, flat_earth_ned_frame
    )

    integrator = Integrator()

    simulation = Simulation(
        no_wind_environment,
        vehicle_model,
        flat_earth_ned_frame,
        dynamics_model,
        integrator,
    )

    initial_state_vector = np.array([0, 0, 1, 1, 0, 0])
    solution = simulation.solve_dynamics(initial_state_vector, t_span=[0, 10])

    print(solution)


def aircraft_6dof_model(position, velocity, quaternion, rates, wind_model: str):

    if wind_model == "No Aero":
        environment = NoAeroConstantGravEnvironment()
    elif wind_model == "No Wind":
        environment = NoWindConstantGravSeaLevelEnvironment()
    else:
        raise ValueError(f"Unkown wind model: {wind_model}")

    no_thrust_control = ControlInput()
    no_thrust_control.throttle_percent = 0.0

    vehicle_model = VehicleModel()
    vehicle_model.mass = 1.0
    vehicle_model.inertias = np.array([1, 1, 1, 0, 0, 0])
    vehicle_model.cog = np.array([0, 0, 0])
    vehicle_model.cl = 0.1
    vehicle_model.cd = 0.5
    vehicle_model.reference_area = 1.0
    vehicle_model.max_thrust = 1.0

    flat_earth_ned_frame = FrameModelFlatEarthNED()

    initial_state_vector = np.array([*position, *velocity, *quaternion, *rates])

    dynamics_model = DynamicsModel6DOF(vehicle_model, environment, flat_earth_ned_frame)

    sim = Simulation(
        environment,
        vehicle_model,
        flat_earth_ned_frame,
        dynamics_model,
        Integrator(),
    )
    return sim.solve_dynamics(initial_state_vector, t_span=[0, 5])


if __name__ == "__main__":
    # point_mass_3dof_model()

    position = [0, 0, 0]
    velocity = [1, 0, -10]
    quaternion = [1, 0, 0, 0]
    rates = [0, 0, 0]

    t, y = aircraft_6dof_model(position, velocity, quaternion, rates, "No Aero")
    plotter = Plotter(t, y)
    plotter.plot_state_vs_time(dofs_to_plot=[3, 4, 5])
    plotter.show()
