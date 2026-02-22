from typing import Protocol
from dataclasses import dataclass

import numpy as np

from flight_sim_typing import Array1D


@dataclass
class VehicleState(Protocol):
    def pack(self) -> Array1D:
        raise NotImplementedError

    @classmethod
    def unpack(cls, state_array: Array1D) -> "VehicleState":
        raise NotImplementedError


@dataclass
class PointMassState:
    position_ned: Array1D  # (3,)
    velocity_ned: Array1D  # (3,)

    def pack(self) -> Array1D:  # (6,)
        return np.concatenate([self.position_ned, self.velocity_ned])

    @classmethod
    def unpack(cls, state_array: Array1D) -> "PointMassState":
        return cls(
            position_ned=state_array[0:3],
            velocity_ned=state_array[3:6],
        )
