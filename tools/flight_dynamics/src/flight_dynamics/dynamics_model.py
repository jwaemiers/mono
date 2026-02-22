from abc import ABC, abstractmethod
import numpy as np

from control_input import ControlInput
from environment_model import EnvironmentModel

# from reference_frame_model import FrameModel, FrameModelFlatEarthNED
from vehicle_model import VehicleModel
from vehicle_state import PointMassState
from flight_sim_typing import Array1D


class DynamicsModel(ABC):
    @abstractmethod
    def state_derivative(
        self,
        time: float,
        vehicle_state: Array1D,
        vehicle: VehicleModel,
        environment: EnvironmentModel,
        control: ControlInput,
    ) -> Array1D:
        raise NotImplementedError


class DynamicsModelPointMass(DynamicsModel):
    def state_derivative(
        self,
        time: float,
        vehicle_state: Array1D,
        vehicle: VehicleModel,
        environment: EnvironmentModel,
        control: ControlInput,
    ) -> Array1D:

        state: PointMassState = PointMassState.unpack(vehicle_state)

        vehicle_position = state.position_ned
        vehicle_velocity = state.velocity_ned

        # --- Environment --- #
        air_density = environment.get_air_density(vehicle_position)
        gravity = environment.get_gravity(vehicle_position)
        wind_velocity = environment.get_wind_velocity(vehicle_position, time)

        # --- Relative Wind --- #
        relative_wind_velocity = vehicle_velocity - wind_velocity
        air_speed = np.linalg.norm(relative_wind_velocity)

        wind_direction = (
            np.zeros(3) if air_speed < 1e-6 else relative_wind_velocity / air_speed
        )

        # --- Vehicle Forces --- #
        L, D, T = vehicle.aero_forces(air_speed, air_density, control)  # type: ignore

        # Drag
        F_drag = -D * wind_direction

        # Lift
        gravity_direction = gravity / np.linalg.norm(gravity)
        if air_speed < 1e-6:
            lift_direction = np.zeros(3)
        else:
            lift_direction = np.cross(
                wind_direction, np.cross(wind_direction, gravity_direction)
            )
            lift_direction /= np.linalg.norm(lift_direction)
        F_lift = L * lift_direction

        # Thrust
        F_thrust = T * wind_direction

        # Total
        F_external = F_lift + F_drag + F_thrust

        # --- Vehicle Acceleration --- #
        acc = F_external / vehicle.mass + gravity

        x_dot = np.concatenate([vehicle_velocity, acc])
        return x_dot
