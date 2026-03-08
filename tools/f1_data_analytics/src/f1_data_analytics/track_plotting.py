import matplotlib.pyplot as plt
import numpy as np

import fastf1
from fastf1.core import Session

from plotting_utils import default_plot_decorator


@default_plot_decorator()
def plot_track(
    session: Session, title: str | None = None, corner_labels=True, plot=None
):
    circuit_info = session.get_circuit_info()
    if circuit_info is None:
        raise ValueError("No circuit data found")

    fast_lap = session.laps.pick_fastest(only_by_time=False)
    if fast_lap is None:
        raise ValueError("No lap data found")
    pos = fast_lap.get_pos_data()

    track = pos.loc[:, ("X", "Y")].to_numpy()

    track_angle = circuit_info.rotation

    rotated_track = rotate(track, track_angle)

    centre_x = sum(rotated_track[0]) / len(rotated_track[0])
    centre_y = sum(rotated_track[1]) / len(rotated_track[1])
    centre = np.empty_like(rotated_track)
    centre[:, 0] = centre_x
    centre[:, 1] = centre_y
    centred_track = rotated_track - centre

    plot[1].plot(centred_track[:, 0], centred_track[:, 1])

    if corner_labels:
        corner_label_offset = [500, 0]
        for _, corner in circuit_info.corners.iterrows():
            corner_label_text = f"{corner['Number']}{corner['Letter']}"

            offset_angle = corner["Angle"]

            offset_x, offset_y = rotate(corner_label_offset, offset_angle)

            text_x = corner["X"] + offset_x
            text_y = corner["Y"] + offset_y
            text_x, text_y = rotate([text_x, text_y], track_angle)

            track_x, track_y = rotate([corner["X"], corner["Y"]], track_angle)

            plt.plot([track_x, text_x], [track_y, text_y], color="gray")
            plt.scatter(text_x, text_y, s=140, color="gray")
            plt.text(
                text_x,
                text_y,
                corner_label_text,
                va="center_baseline",
                ha="center",
                size="small",
                color="white",
            )

    plt.axis("equal")
    plt.suptitle(title)
    plt.xticks([])
    plt.yticks([])

    return plot


def rotate(vec, angle_deg):
    angle = angle_deg * np.pi / 180
    r = np.array(
        [
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)],
        ]
    )
    return vec @ r


if __name__ == "__main__":
    session = fastf1.get_session(2023, "silversone", "Q")
    session.load()
    plot = plot_track(session, title="Track Map", corner_labels=False)

    session = fastf1.get_session(2026, "australia", "Q")
    session.load()
    plot = plot_track(session, corner_labels=False, plot=plot)
    plt.show()
