from typing import Protocol
import numpy as np
from flight_sim_typing import Array1D


class EnvironmentModel(Protocol):
    def get_gravity(self, position: Array1D) -> Array1D:
        raise NotImplementedError

    def get_air_density(self, position: Array1D) -> float:
        raise NotImplementedError

    def get_wind_velocity(self, postiion: Array1D, time: float) -> Array1D:
        raise NotImplementedError


class NoAeroConstantGravEnvironment:
    def get_gravity(self, position: Array1D) -> Array1D:
        return np.array([0, 0, 9.81])

    def get_air_density(self, position: Array1D) -> float:
        return 0.0

    def get_wind_velocity(self, postiion: Array1D, time: float) -> Array1D:
        return np.array([0, 0, 0])


class NoWindConstantGravSeaLevelEnvironment:
    def get_gravity(self, position: Array1D) -> Array1D:
        return np.array([0, 0, 9.81])

    def get_air_density(self, position: Array1D) -> float:
        return 1.125

    def get_wind_velocity(self, postiion: Array1D, time: float) -> Array1D:
        return np.array([0, 0, 0])
