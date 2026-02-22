from typing import TypeAlias
import numpy as np


NDArray1D: TypeAlias = np.ndarray
NDArray2D: TypeAlias = np.ndarray


class DynamicsModel:
    def __init__(self, intial_state) -> None:
        self.initial_state = intial_state

    def non_linear_dynamics(self, time: float, vehicle_state: NDArray1D) -> NDArray1D:
        raise NotImplementedError

    def _linear_acceleration(self, vehicle_state: NDArray1D) -> NDArray1D:
        raise NotImplementedError

    def _angular_acceleration(self, vehicle_state: NDArray1D) -> NDArray1D:
        raise NotImplementedError

    def _inertial_velocity(self, vehicle_state: NDArray1D) -> NDArray1D:
        raise NotImplementedError

    def _inertial_euler_rates(self, vehicle_state: NDArray1D) -> NDArray1D:
        raise NotImplementedError


class FlatEarthDynamicsModel(DynamicsModel):
    def __init__(self, intial_state) -> None:
        super().__init__(intial_state)

    def _linear_acceleration(self) -> NDArray1D:
        body_force: NDArray1D = ...
        mass: float = ...
        body_gravity: NDArray1D = ...
        body_rates: NDArray1D = ...
        body_velocity: NDArray1D = ...
        linear_accleration = (
            body_force / mass
            + body_gravity
            - np.linalg.cross(body_rates, body_velocity)
        )
        return linear_accleration
