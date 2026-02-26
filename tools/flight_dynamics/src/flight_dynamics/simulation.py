import numpy as np
from control_input import ControlInput
from dynamics_model import DynamicsModelPointMass
from environment_model import (
    NoAeroConstantGravEnvironment,
    NoWindConstantGravSeaLevelEnvironment,
)
from reference_frame_model import FrameModelFlatEarthNED
from vehicle_model import VehicleModel
from vehicle_state import PointMassState

if __name__ == "__main__":
    no_aero_environment = NoAeroConstantGravEnvironment()
    no_wind_environment = NoWindConstantGravSeaLevelEnvironment()

    no_thrust_control = ControlInput()
    no_thrust_control.throttle_percent = 1.0

    vehicle_model = VehicleModel()
    vehicle_model.mass = 1.0
    vehicle_model.cl = 0.1
    vehicle_model.cd = 0.5
    vehicle_model.reference_area = 1.0
    vehicle_model.max_thrust = 1.0

    flat_earth_ned_frame = FrameModelFlatEarthNED()

    initial_state_vector = np.array([0, 0, 1, 1, 0, 0])
    intial_state = PointMassState.unpack(initial_state_vector)

    dynamics_model = DynamicsModelPointMass(
        vehicle_model, no_aero_environment, flat_earth_ned_frame
    )

    x_dot = dynamics_model.state_derivative(
        time=0.0,
        vehicle_state_vector=initial_state_vector,
        control=no_thrust_control,
    )

    print(x_dot)
