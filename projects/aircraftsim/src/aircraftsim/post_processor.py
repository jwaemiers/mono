from typing import TypeAlias
import numpy as np
import matplotlib.pyplot as plt

NDArray1D: TypeAlias = np.ndarray


class PostProcessor:
    def __init__(self) -> None:
        self.times: NDArray1D
        self.states: dict[str, NDArray1D]

    def add_solution(self, solution: tuple[NDArray1D, dict[str, NDArray1D]]):
        self.times: NDArray1D = solution[0]
        self.states: dict[str, NDArray1D] = solution[1]

    def standard_plots(self):
        self.plot_state_variable_time_series()

    def plot_state_variable_time_series(self):
        slicers = [slice(0, 6), slice(6, 12)]
        for slicer in slicers:
            fig, axs = plt.subplots(2, 3)
            for ax, name in zip(
                axs.flat,
                list(self.states.keys())[slicer],
            ):
                data = self.states[name]
                ax.plot(self.times, data)
                ax.set_title(name)
                ax.autoscale()
            fig.tight_layout()
            fig.show()
        input()


def convert_ind_to_xy(ind: int, x_max: int, y_max: int) -> tuple:
    invalid_ind = (ind < 0) or (ind > x_max * y_max)
    if invalid_ind:
        raise IndexError

    x = ind % x_max
    y = (ind - x) / x_max

    return x, y


def convert_xy_to_ind(x: int, y: int, x_max: int, y_max: int) -> int:
    invalid_xy = (x < 0) or (y < 0) or (x > x_max) or (y > y_max)
    if invalid_xy:
        raise IndexError

    ind = y * x_max + x
    return ind
