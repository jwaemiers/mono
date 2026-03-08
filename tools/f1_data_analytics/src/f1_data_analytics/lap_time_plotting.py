from matplotlib import pyplot as plt

import fastf1
from fastf1 import plotting

from plotting_utils import default_plot_decorator


@default_plot_decorator(figsize=(8, 5))
def plot_driver_lap_times(race, plot=None, title=None):
    # Enable Matplotlib patches for plotting timedelta values and load
    # FastF1's dark color scheme
    fig, ax = plot  # type: ignore

    for driver in ("HAM", "PER", "VER", "RUS"):
        laps = race.laps.pick_drivers(driver).pick_quicklaps().reset_index()
        style = plotting.get_driver_style(
            identifier=driver, style=["color", "linestyle"], session=race
        )
        ax.plot(laps["LapTime"], **style, label=driver)

    # add axis labels and a legend
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time")
    plotting.add_sorted_driver_legend(ax, race)
    ax.set_title(title)
    return plot


if __name__ == "__main__":
    race = fastf1.get_session(2021, "Abu Dhabi", "race")
    race.load()
    plot = plot_driver_lap_times(race, title="2021 Abu Dhabi GP Lap Times")
    plt.show()
