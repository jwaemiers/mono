from typing import TypeAlias

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from numpy import ndarray

Plot: TypeAlias = tuple[Figure, ndarray]


class Plotter:
    def __init__(self, times, states) -> None:
        self.t = times
        self.y = states

    def show(self):
        plt.show()

    def plot_state_vs_time(
        self,
        dofs_to_plot: list[int] | None = None,
        plot: Plot | None = None,
        shape: int | tuple[int, int] | None = None,
    ):
        if dofs_to_plot is None:
            dofs_to_plot = [i for i in range(len(self.y))]

        if shape is None:
            shape = len(dofs_to_plot)
        if isinstance(shape, int):
            shape = (shape, 1)

        if plot is None:
            plot = plt.subplots(*shape, squeeze=False, sharex=True)

        fig, axes = plot
        for dof, ax in zip(dofs_to_plot, axes.flatten()):
            t = self.t
            y = self.y[dof]
            ax: Axes = ax
            ax.plot(t, y)
            ax.set_ylabel(f"{dof=}")
            ax.grid()
        fig.supxlabel("time [s]")
        fig.suptitle("Plot of dofs vs time")
