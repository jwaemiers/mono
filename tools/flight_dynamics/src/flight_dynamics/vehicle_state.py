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


@dataclass
class Basic6DOFState:
    translational_position: Array1D  # (3,)
    translational_velocity: Array1D  # (3,)
    orientation_quaternion: Array1D  # (4,)
    rotational_velocity: Array1D  # (3,)

    def pack(self) -> Array1D:  # (6,)
        return np.concatenate(
            [
                self.translational_position,
                self.translational_velocity,
                self.orientation_quaternion,
                self.rotational_velocity,
            ]
        )

    @classmethod
    def unpack(cls, state_array: Array1D) -> "Basic6DOFState":
        return cls(
            translational_position=state_array[0:3],
            translational_velocity=state_array[3:6],
            orientation_quaternion=state_array[6:10],
            rotational_velocity=state_array[10:13],
        )
