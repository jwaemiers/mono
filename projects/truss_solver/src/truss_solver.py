from typing import TypeAlias
from truss import TrussBuilder
import numpy as np
from numpy.typing import NDArray

NDArray1D: TypeAlias = NDArray[np.float64]


class TrussSolver:
    def solve_truss(self) -> NDArray1D:
        x = np.array([1, 2])
        return x
