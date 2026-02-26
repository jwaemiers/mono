import numpy as np
from control_input import ControlInput
from vehicle_state import VehicleState
from flight_sim_typing import Array1D


class VehicleModel:
    def __init__(self) -> None:
        self.mass: float
        self.inertias: Array1D
        self.cog: Array1D
        self.cd: float
        self.cl: float
        self.reference_area: float
        self.max_thrust: float

    def aero_forces(
        self, air_speed: float, air_density: float, control: ControlInput
    ) -> tuple[float, float, float]:  # Lift ,Drag, Thrust

        dynamic_pressure = 0.5 * air_density * air_speed**2

        lift_force_mag = dynamic_pressure * self.cl * self.reference_area

        drag_force_mag = dynamic_pressure * self.cd * self.reference_area

        thurst_force_mag = control.throttle_percent * self.max_thrust

        return lift_force_mag, drag_force_mag, thurst_force_mag

    def mass_inertia_matrix(self, vehicle_state: VehicleState) -> Array1D:
        mass_component = self.mass * np.identity(3)
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = self.inertias
        inertia_component = np.array(
            [[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]
        )
        cgx, cgy, cgz = self.cog
        cross_coupling_term = np.array([[0, -cgz, cgy], [cgz, 0, -cgx], [-cgy, cgx, 0]])

        mass_inertia_matrix = np.zeros((6, 6))
        mass_inertia_matrix[:3, :3] = mass_component
        mass_inertia_matrix[3:, 3:] = inertia_component
        mass_inertia_matrix[:3, 3:] = cross_coupling_term
        mass_inertia_matrix[3:, :3] = -cross_coupling_term

        return mass_inertia_matrix

    def force_and_moments(
        self, vehicle_state, relative_wind_body, air_density, control: ControlInput
    ) -> tuple[Array1D, Array1D]:

        thrust = self.max_thrust * control.throttle_percent * np.array([1, 0, 0])

        dynamic_pressure = 0.5 * air_density * np.linalg.norm(relative_wind_body) ** 2
        drag = (
            -self.cd
            * dynamic_pressure
            * self.reference_area
            * relative_wind_body
            / np.linalg.norm(relative_wind_body)
        )
        lift = self.cl * dynamic_pressure * self.reference_area * np.array([0, 0, 1])
        force = thrust + drag + lift

        moment = np.array([0, 0, 0])

        return force, moment
