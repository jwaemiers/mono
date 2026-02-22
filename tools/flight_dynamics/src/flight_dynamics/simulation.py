import numpy as np
from control_input import ControlInput
from dynamics_model import DynamicsModelPointMass
from environment_model import (
    NoAeroConstantGravEnvironment,
    NoWindConstantGravSeaLevelEnvironment,
)
from vehicle_model import VehicleModel

if __name__ == "__main__":
    dynamics_model = DynamicsModelPointMass()

    no_aero_environment = NoAeroConstantGravEnvironment()
    no_wind_environment = NoWindConstantGravSeaLevelEnvironment()

    no_thrust_control = ControlInput()
    no_thrust_control.throttle_percent = 0.0

    vehicle_model = VehicleModel()
    vehicle_model.mass = 1.0
    vehicle_model.cl = 0.0
    vehicle_model.cd = 0.5
    vehicle_model.reference_area = 1.0
    vehicle_model.max_thrust = 0.0

    initial_state_vector = np.array([0, 0, 0, 0, 0, 0])

    x_dot = dynamics_model.state_derivative(
        time=0.0,
        vehicle_state=initial_state_vector,
        vehicle=vehicle_model,
        environment=no_aero_environment,
        control=no_thrust_control,
    )

    print(x_dot)
