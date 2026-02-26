from abc import ABC, abstractmethod
import numpy as np

from control_input import ControlInput
from environment_model import EnvironmentModel

from reference_frame_model import FrameModel
from vehicle_model import VehicleModel
from vehicle_state import VehicleState, PointMassState, Basic6DOFState
from flight_sim_typing import Array1D


class DynamicsModel(ABC):
    def __init__(
        self,
        vehicle: VehicleModel,
        environment: EnvironmentModel,
        frame: FrameModel,
    ) -> None:
        self.vehicle: VehicleModel = vehicle
        self.environment: EnvironmentModel = environment
        self.frame: FrameModel = frame

    @abstractmethod
    def state_derivative(
        self,
        time: float,
        vehicle_state_vector: Array1D,
        control: ControlInput,
    ) -> Array1D:
        raise NotImplementedError


class DynamicsModelPointMass(DynamicsModel):
    def state_derivative(
        self,
        time: float,
        vehicle_state_vector: Array1D,
        control: ControlInput,
    ) -> Array1D:

        state: PointMassState = PointMassState.unpack(vehicle_state_vector)
        vehicle_position = state.position_ned
        vehicle_velocity = state.velocity_ned

        # --- Environment --- #
        air_density = self.environment.air_density(vehicle_position)

        wind_velocity = self.environment.wind_velocity(vehicle_position, time)

        gravity_mag = self.environment.gravity(vehicle_position)
        gravity_direction = self.frame.gravity_vector(vehicle_position)
        gravity = gravity_mag * gravity_direction

        # --- Relative Wind --- #
        relative_wind_velocity = self.frame.relative_wind(
            vehicle_position, vehicle_velocity, wind_velocity
        )
        air_speed = np.linalg.norm(relative_wind_velocity)

        wind_direction = (
            np.zeros(3) if air_speed < 1e-6 else relative_wind_velocity / air_speed
        )

        # --- Vehicle Forces --- #
        L, D, T = self.vehicle.aero_forces(air_speed, air_density, control)  # type: ignore

        # Drag
        F_drag = -D * wind_direction

        # Lift
        lift_direction = self.frame.lift_direction(wind_direction, gravity_direction)
        F_lift = L * lift_direction

        # Thrust
        F_thrust = T * wind_direction

        # Total
        F_external = F_lift + F_drag + F_thrust

        # --- Vehicle Acceleration --- #
        acc = F_external / self.vehicle.mass + gravity

        x_dot = np.concatenate([vehicle_velocity, acc])
        return x_dot


class DynamicsModel6DOF(DynamicsModel):
    def state_derivative(
        self, time: float, vehicle_state_vector: Array1D, control: ControlInput
    ) -> Array1D:

        # Unpack Current State
        vehicle_state: Basic6DOFState = Basic6DOFState.unpack(vehicle_state_vector)
        translational_position = vehicle_state.translational_position
        translational_velocity = vehicle_state.translational_velocity
        orientation_quaternion = vehicle_state.orientation_quaternion
        rotational_velocity = vehicle_state.rotational_velocity

        # --- Frame Interface --- #
        gravity_direction = self.frame.gravity_vector(translational_position)
        non_inertial_acceleration = self.frame.non_inertial_acceleration(
            translational_position, translational_velocity, time
        )
        body_to_frame_quaternion = self.frame.body_to_frame(orientation_quaternion)
        frame_to_body_quaternion = quaternion_conjugate(body_to_frame_quaternion)

        # --- Environment Interface --- #
        wind_velocity = self.environment.wind_velocity(translational_position, time)
        air_density = self.environment.air_density(translational_position)
        gravity_magnitude = self.environment.gravity(translational_position)

        # --- Relative Wind --- #
        relative_wind_frame = self.frame.relative_wind(
            translational_position, translational_velocity, wind_velocity
        )
        relative_wind_body = frame_to_body_quaternion @ relative_wind_frame

        # --- Vehicle Interface --- #
        mass_inertia_matrix = self.vehicle.mass_inertia_matrix(vehicle_state)
        mass = mass_inertia_matrix[0, 0]
        inertia_tensor = mass_inertia_matrix[3:, 3:]

        body_forces, body_moments = self.vehicle.force_and_moments(
            vehicle_state, relative_wind_body, air_density, control
        )

        # --- Translational Dynamics --- #
        frame_forces = body_to_frame_quaternion @ body_forces
        gravity = gravity_direction * gravity_magnitude
        translational_acceleration = (
            frame_forces / mass + gravity + non_inertial_acceleration
        )

        # --- Rotational Dynamics --- #
        rotational_acceleration = np.linalg.solve(
            inertia_tensor,
            body_moments
            - np.cross(rotational_velocity, inertia_tensor @ rotational_velocity),
        )

        # --- Quaternion Kinematics --- #
        quaternion_dot = quaternion_derivative(
            orientation_quaternion, rotational_velocity, "fixed"
        )

        x_dot = np.concatenate(
            [
                translational_velocity,
                translational_acceleration,
                quaternion_dot,
                rotational_acceleration,
            ]
        )
        return x_dot


def quaternion_derivative(q: Array1D, omega: Array1D, reference_frame: str) -> Array1D:
    if reference_frame == "fixed":
        return 0.5 * quaternion_product(omega, q)
    elif reference_frame == "body":
        return 0.5 * quaternion_product(q, omega)
    raise ValueError


def quaternion_product(a: Array1D, b: Array1D) -> Array1D:
    a_s, a_v = a[0], a[1:]
    b_s, b_v = b[0], b[1:]

    c_s = a_s * b_s - np.dot(a_v, b_v)
    c_v = a_s * b_v + a_v + b_s + np.cross(a_v, b_v)

    c = np.zeros(4)
    c[0] = c_s
    c[1:] = c_v

    return c / np.linalg.norm(c)


def quaternion_conjugate(q: Array1D) -> Array1D:
    q_s, q_v = q[0], q[1:]
    q_bar = np.zeros(4)
    q_bar[0] = q_s
    q_bar[1:] = q_v
    return q_bar


def rotate_by_quaternion(v: Array1D, q: Array1D) -> Array1D:
    v_q = np.zeros(4)
    v_q[1:] = v

    q_conj = quaternion_conjugate(q)

    q_inter = quaternion_product(q_conj, v_q)

    v_rotated = quaternion_product(q_inter, q)

    return v_rotated[1:]
