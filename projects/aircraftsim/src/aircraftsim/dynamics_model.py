import numpy as np
from scipy.spatial.transform import Rotation as R
from aircraftsim.vehicle_model import VehicleModel, GravProperties

NDArray1D = np.ndarray
NDArray2D = np.ndarray


class DynamicsModel:
    def __init__(self, state_variables: dict[str, float]) -> None:
        self.state_variables: list[str] = list(state_variables.keys())
        self.initial_state: NDArray1D = np.array(list(state_variables.values()))
        pass

    def navigation_equation(self, vehicle_state: NDArray1D) -> NDArray1D:
        inertial_attitude = self.get_inertial_attitude(vehicle_state)
        dcm = self.get_direction_cosine_matrix(inertial_attitude)
        body_velocity = self.get_body_velocity(vehicle_state)
        inertial_velocity = dcm.T @ body_velocity
        return inertial_velocity

    def rotation_kinematic_equation(self, vehicle_state: NDArray1D) -> NDArray1D:
        inertial_attitude = self.get_inertial_attitude(vehicle_state)
        euler_to_rates = self.euler_to_rates_angle_transformation(inertial_attitude)
        body_rates = self.get_body_rates(vehicle_state)
        rate_to_euler = np.linalg.inv(euler_to_rates)
        inertial_angular_velocity = rate_to_euler @ body_rates
        return inertial_angular_velocity

    def force_balance_equation(
        self,
        vehicle_state: NDArray1D,
        body_net_force: NDArray1D,
        body_gravity: NDArray1D,
        grav_properties: GravProperties,
    ) -> NDArray1D:

        body_velocity: NDArray1D = self.get_body_velocity(vehicle_state)
        body_angular_velocity: NDArray1D = self.get_body_rates(vehicle_state)
        mass = grav_properties.mass

        return (
            body_net_force / mass
            + body_gravity
            - np.linalg.cross(body_angular_velocity, body_velocity)
        )

    def moment_balance_equation(
        self,
        vehicle_state: NDArray1D,
        body_net_moment: NDArray1D,
        grav_properties: GravProperties,
    ) -> NDArray1D:

        body_velocity: NDArray1D = self.get_body_velocity(vehicle_state)
        body_angular_velocity: NDArray1D = self.get_body_rates(vehicle_state)
        inertia_tensor = grav_properties.inertia_tensor.to_matrix()
        inv_inertia_tensor = np.linalg.inv(inertia_tensor)

        return inv_inertia_tensor @ (
            body_net_moment
            - np.linalg.cross(body_angular_velocity, body_velocity)
            @ inertia_tensor
            @ body_angular_velocity
        )

    def get_kinematics_state_derivative(
        self,
        time,
        vehicle_state: NDArray1D,
        vehicle_model: VehicleModel,
        atmospheric_model,
    ):

        grav_properties = vehicle_model.gravs
        body_net_force = vehicle_model.get_body_net_force(
            time, vehicle_state, atmospheric_model
        )
        body_net_moment = vehicle_model.get_body_net_moment(
            time, vehicle_state, atmospheric_model
        )
        body_gravity = self.get_body_gravity_acceleration(vehicle_state)

        navigation_derivative: NDArray1D = self.navigation_equation(vehicle_state)

        rotation_kinematic_derivative: NDArray1D = self.rotation_kinematic_equation(
            vehicle_state
        )

        force_balance_derivative: NDArray1D = self.force_balance_equation(
            vehicle_state,
            body_net_force,
            body_gravity,
            grav_properties,
        )

        moment_balance_derivative: NDArray1D = self.moment_balance_equation(
            vehicle_state,
            body_net_moment,
            grav_properties,
        )

        return np.hstack(
            (
                force_balance_derivative,
                moment_balance_derivative,
                navigation_derivative,
                rotation_kinematic_derivative,
            )
        )

    def get_body_gravity_acceleration(self, vehicle_state: NDArray1D) -> NDArray1D:
        inertial_position = self.get_inertial_position(vehicle_state)
        inertial_attitude = self.get_inertial_attitude(vehicle_state)

        inertial_gravity = self.get_gravity_magnitude(inertial_position)
        direction_cosine_matrix = self.get_direction_cosine_matrix(inertial_attitude)

        return direction_cosine_matrix @ inertial_gravity

    def get_gravity_magnitude(self, inertial_position) -> NDArray1D:
        return np.array([0, 0, 9.81])

    def get_direction_cosine_matrix(self, inertial_attitude) -> NDArray2D:
        dcm = R.from_euler("xyz", -inertial_attitude).as_matrix()
        return dcm

    def euler_to_rates_angle_transformation(self, inertial_attitude) -> NDArray2D:
        r, p, y = inertial_attitude
        sr, cr = np.sin(r), np.cos(r)
        sp, cp = np.sin(p), np.cos(p)
        etr: NDArray2D = np.array([[1, 0, -sp], [0, cr, sr * cp], [0, -sr, cr * cp]])
        return etr

    def get_body_velocity(self, vehicle_state: NDArray1D) -> NDArray1D:
        return vehicle_state[0:3]

    def get_body_rates(self, vehicle_state: NDArray1D) -> NDArray1D:
        return vehicle_state[3:6]

    def get_inertial_position(self, vehicle_state: NDArray1D) -> NDArray1D:
        return vehicle_state[6:9]

    def get_inertial_attitude(self, vehicle_state: NDArray1D) -> NDArray1D:
        return vehicle_state[9:12]

    def get_state_variable_time_series_dict(
        self, state_variable_time_series: NDArray2D
    ) -> dict[str, NDArray1D]:
        return dict(zip(self.state_variables, state_variable_time_series))
