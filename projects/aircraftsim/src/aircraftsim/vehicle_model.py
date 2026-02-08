from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from aircraftsim.atmospheric_model import AtmosphericModel

NDArray1D: TypeAlias = np.ndarray
NDArray2D: TypeAlias = np.ndarray


class InertiaTensor:
    def __init__(
        self, Jxx: float, Jyy: float, Jzz: float, Jxy: float, Jxz: float, Jyz: float
    ) -> None:
        self.tensor: NDArray2D = np.array(
            [[Jxx, Jxy, Jxz], [Jxy, Jyy, Jyz], [Jxz, Jyz, Jzz]]
        )

    def to_matrix(self):
        return self.tensor


@dataclass
class GravProperties:
    mass: float
    inertia_tensor: InertiaTensor


@dataclass
class AeroProperties:
    cd: float
    cl: float


class VehicleModel:
    def __init__(
        self,
        gravs: GravProperties,
        aero_properties: AeroProperties,
    ) -> None:
        self.gravs: GravProperties = gravs
        self.aero_properties: AeroProperties = aero_properties

    def get_body_net_force(
        self, time, vehicle_state: NDArray1D, atmospheric_model: AtmosphericModel
    ) -> NDArray1D:
        # raise NotImplementedError
        return np.array([0, 0, 0])

    def get_body_net_moment(
        self, time, vehicle_state: NDArray1D, atmospheric_model: AtmosphericModel
    ) -> NDArray1D:
        # raise NotImplementedError
        return np.array([0, 0, 0])
