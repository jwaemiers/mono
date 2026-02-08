from typing import Callable, TypeAlias
import numpy as np
from aircraftsim.atmospheric_model import AtmosphericModel
from aircraftsim.dynamics_model import DynamicsModel
from aircraftsim.vehicle_model import VehicleModel
from aircraftsim.post_processor import PostProcessor


NDArray1D: TypeAlias = np.ndarray
NDArray2D: TypeAlias = np.ndarray


class AircraftSim:
    def __init__(
        self, vehicle_model, dynamic_model, atmospheric_model, post_processor
    ) -> None:
        self.vehicle_model: VehicleModel = vehicle_model
        self.dynamics_model: DynamicsModel = dynamic_model
        self.atmospheric_model: AtmosphericModel = atmospheric_model
        self.post_processor: PostProcessor = post_processor

    def integrate_vehicle_state(
        self,
        integrate: Callable[
            [
                Callable[[float, NDArray1D], NDArray1D],
                NDArray1D,
                tuple[float, float],
                NDArray1D | None,
            ],
            tuple[NDArray1D, NDArray2D],
        ],
        t_span: tuple[float, float],
        t_eval: NDArray1D | None = None,
    ) -> tuple[NDArray1D, dict[str, NDArray1D]]:
        def func(time: float, vehicle_state: NDArray1D) -> NDArray1D:
            kinematics_derivative = self.dynamics_model.get_kinematics_state_derivative(
                time,
                vehicle_state,
                self.vehicle_model,
                self.atmospheric_model,
            )
            return kinematics_derivative

        initial_condition: NDArray1D = self.dynamics_model.initial_state
        time_series, state_series = integrate(func, initial_condition, t_span, t_eval)
        solution = (
            time_series,
            self.dynamics_model.get_state_variable_time_series_dict(state_series),
        )
        self.post_processor.add_solution(solution)
        return solution

    def get_final_time_state_vector(
        self,
        integrate: Callable[
            [
                Callable[[float, NDArray1D], NDArray1D],
                NDArray1D,
                tuple[float, float],
                NDArray1D | None,
            ],
            tuple[NDArray1D, NDArray2D],
        ],
        end_time: float,
    ) -> NDArray1D:
        time, state_variables = self.integrate_vehicle_state(
            integrate, (0, end_time), np.array([end_time])
        )
        state_vector = np.asarray(list(state_variables.values())).flatten()
        return state_vector
