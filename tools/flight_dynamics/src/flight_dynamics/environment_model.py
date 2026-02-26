from typing import Protocol
import numpy as np
from flight_sim_typing import Array1D


class EnvironmentModel(Protocol):
    def gravity(self, position: Array1D) -> float:
        raise NotImplementedError

    def air_density(self, position: Array1D) -> float:
        raise NotImplementedError

    def wind_velocity(self, postiion: Array1D, time: float) -> Array1D:
        raise NotImplementedError


class NoAeroConstantGravEnvironment:
    def gravity(self, position: Array1D) -> float:
        return 9.81

    def air_density(self, position: Array1D) -> float:
        return 0.0

    def wind_velocity(self, postiion: Array1D, time: float) -> Array1D:
        return np.array([0, 0, 0])


class NoWindConstantGravSeaLevelEnvironment:
    def gravity(self, position: Array1D) -> float:
        return 9.81

    def air_density(self, position: Array1D) -> float:
        return 1.125

    def wind_velocity(self, postiion: Array1D, time: float) -> Array1D:
        return np.array([0, 0, 0])
