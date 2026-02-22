from control_input import ControlInput
# from flight_sim_typing import Array1D


class VehicleModel:
    def __init__(self) -> None:
        self.mass: float
        self.cd: float
        self.cl: float
        self.reference_area: float
        self.max_thrust: float

    def aero_forces(
        self, air_speed: float, air_density: float, control: ControlInput
    ) -> tuple[float, float, float]:  # Lift ,Drag, Thrust

        dynamic_pressure = 0.5 * air_density * air_speed**2

        lift_force_mag = dynamic_pressure * self.cl * self.reference_area

        drag_force_mag = dynamic_pressure * self.cd * self.reference_area

        thurst_force_mag = control.throttle_percent * self.max_thrust

        return lift_force_mag, drag_force_mag, thurst_force_mag
