from abc import ABC
import numpy as np
from flight_sim_typing import Array1D


class FrameModel(ABC):
    def earth_rotation_rate(self, time: float) -> Array1D:
        raise NotImplementedError

    def transport_rate(self, position: Array1D, velocity: Array1D) -> Array1D:
        raise NotImplementedError

    def body_to_inertial(self, attitude: Array1D) -> Array1D:
        raise NotImplementedError

    def position_to_geodetic(self, position: Array1D) -> Array1D:
        raise NotImplementedError


class FrameModelFlatEarthNED(FrameModel):
    def earth_rotation_rate(self, time: float) -> Array1D:
        return np.zeros(3)

    def transport_rate(self, position: Array1D, velocity: Array1D) -> Array1D:
        return np.zeros(3)
