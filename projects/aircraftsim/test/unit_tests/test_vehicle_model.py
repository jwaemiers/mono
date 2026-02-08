from conftest import (
    get_inertia_tensor,
    get_atmospheric_model_no_aero,
    get_vehicle_model,
    get_vehicle_state,
)


## -- GRAV PROPERTIES TESTS -- ##


## -- INERTIA TENSOR TESTS -- ##
def test_inertia_tensor_is_symmetric():
    inertia_tensor = get_inertia_tensor()
    pass_condition: bool = (
        inertia_tensor.to_matrix() == inertia_tensor.to_matrix().T
    ).all()
    assert pass_condition


## -- AERO PROPERTIES TESTS -- ##


## -- VEHICLE MODEL TESTS -- ##
def test_calculate_body_force_is_3d():
    vehicle_model = get_vehicle_model()
    vehicle_state = get_vehicle_state()
    atmospheric_model = get_atmospheric_model_no_aero()
    body_force = vehicle_model.get_body_net_force(0.0, vehicle_state, atmospheric_model)
    pass_condition: bool = len(body_force) == 3
    assert pass_condition


def test_calculate_body_moment_is_3d():
    vehicle_model = get_vehicle_model()
    vehicle_state = get_vehicle_state()
    atmospheric_model = get_atmospheric_model_no_aero()
    body_force = vehicle_model.get_body_net_moment(
        0.0, vehicle_state, atmospheric_model
    )
    pass_condition: bool = len(body_force) == 3
    assert pass_condition
