import matplotlib.pyplot as plt
from fastf1 import plotting


def default_plot_decorator(**plot_kwargs):
    plotting.setup_mpl(mpl_timedelta_support=True, color_scheme="fastf1")

    def outer(func):
        def inner(*args, plot=None, title=None, **kwargs):
            if plot is None:
                plot = plt.subplots(**plot_kwargs)
            if title is None:
                title = title
            return func(*args, plot, title, **kwargs)

        return inner

    return outer
