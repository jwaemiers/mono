from abc import ABC
import numpy as np
from flight_sim_typing import Array1D


class FrameModel(ABC):
    def non_inertial_acceleration(
        self, position: Array1D, velocity: Array1D, time: float
    ) -> Array1D:
        raise NotImplementedError

    def body_to_frame(self, orientation_quaternion: Array1D) -> Array1D:
        raise NotImplementedError

    def gravity_vector(self, position: Array1D) -> Array1D:
        raise NotImplementedError

    def relative_wind(
        self, position: Array1D, velocity: Array1D, wind_velocity: Array1D
    ) -> Array1D:
        raise NotImplementedError

    def lift_direction(
        self, relative_wind: Array1D, gravity_direction: Array1D
    ) -> Array1D:
        raise NotImplementedError


class FrameModelFlatEarthNED(FrameModel):
    def lift_direction(
        self, relative_wind: Array1D, gravity_direction: Array1D
    ) -> Array1D:
        if np.linalg.norm(relative_wind) < 1e-6:
            return np.zeros(3)
        lift_dir = np.cross(relative_wind, np.cross(relative_wind, gravity_direction))
        return lift_dir / np.linalg.norm(lift_dir)

    def non_inertial_acceleration(
        self, position: Array1D, velocity: Array1D, time: float
    ) -> Array1D:
        return np.zeros(3)

    def body_to_frame(self, orientation_quaternion: Array1D) -> Array1D:
        return orientation_quaternion

    def gravity_vector(self, position: Array1D) -> Array1D:
        return np.array([0, 0, 1])

    def relative_wind(
        self, position: Array1D, velocity: Array1D, wind_velocity: Array1D
    ) -> Array1D:
        return velocity - wind_velocity
