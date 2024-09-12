from regelum.callback import (
    ScenarioStepLogger, 
    ObjectiveTracker, 
    HistoricalDataCallback,
    HistoricalCallback)
from regelum.policy import Policy
from regelum.scenario import PPO, REINFORCE
from src.scenario import RosMPC, MyPPO

from typing import Dict, Any
import numpy as np

import torch
from typing import Union, Dict, Any
from pathlib import Path
from src.scenario import MyScenario
from rich.logging import RichHandler


class ROSScenarioStepLogger(ScenarioStepLogger):
    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, RosMPC)
            and method == "post_compute_action"
        )

    def on_function_call(self, obj, method: str, output: Dict[str, Any]):
        print("Enter Here")
        
        try:
            with np.printoptions(precision=2, suppress=True):
                self.log(
                    f"runn. objective: {output['running_objective']:.2f}, "
                    f"state est.: {output['estimated_state'][0]}, "
                    f"observation: {output['observation'][0]}, "
                    f"action: {output['action'][0]}, "
                    f"value: {output['current_value']:.4f}, "
                    f"time: {output['time']:.4f} ({100 * output['time']/obj.simulator.time_final:.1f}%), "
                    f"episode: {int(output['episode_id'])}/{obj.N_episodes}, "
                    f"iteration: {int(output['iteration_id'])}/{obj.N_iterations}"
                )
        except Exception as err:
            print(err)
            print("Error Here")

class MyObjectiveTracker(ObjectiveTracker):
    def is_target_event(self, obj, method, output, triggers):
        return (
            isinstance(obj, Policy)
            and method == "post_obj_run"
        )

    def is_done_collecting(self):
        return hasattr(self, "objective")

    def on_function_call(self, obj, method, output):
        # print("post_obj_run:", output["running_objective"], output["current_value"])
        self.running_objective = output["running_objective"]
        self.value = output["current_value"]
        self.objective = np.array([self.value, self.running_objective])
        self.objective_naming = ["Value", "Running objective"]


class CALFHistoricalDataCallback(HistoricalDataCallback):
    def is_target_event(self, obj, method, output, triggers):
        if isinstance(obj, MyScenario) and method == "post_compute_action":
            print("Enter CALFHistoricalDataCallback")   
            return True
        
        return False
    
    def on_function_call(self, obj, method: str, output: Dict[str, Any]):
        if self.observation_components_naming is None:
            self.observation_components_naming = (
                [
                    f"observation_{i + 1}"
                    for i in range(obj.simulator.system.dim_observation)
                ]
                if obj.simulator.system.observation_naming is None
                else obj.simulator.system.observation_naming
            )

        if self.action_components_naming is None:
            self.action_components_naming = (
                [f"action_{i + 1}" for i in range(obj.simulator.system.dim_inputs)]
                if obj.simulator.system.inputs_naming is None
                else obj.simulator.system.inputs_naming
            )

        if self.state_components_naming is None:
            self.state_components_naming = (
                [f"state_{i + 1}" for i in range(obj.simulator.system.dim_state)]
                if obj.simulator.system.state_naming is None
                else obj.simulator.system.state_naming
            )

        if method == "post_compute_action":
            self.add_datum(
                {
                    **{
                        "time": output["time"],
                        "running_objective": output["running_objective"],
                        "current_value": output["current_value"],
                        "episode_id": output["episode_id"],
                        "iteration_id": output["iteration_id"],
                    },
                    **dict(zip(self.action_components_naming, output["action"][0])),
                    **dict(
                        zip(self.state_components_naming, output["estimated_state"][0])
                    ),
                    **{
                        "use_calf": output.get("use_calf"),
                        "critic_new": output.get("critic_new"),
                        "critic_safe": output.get("critic_safe"),
                        "critic_low_kappa": output.get("critic_low_kappa"),
                        "critic_up_kappa": output.get("critic_up_kappa"),
                        "calf_diff": output.get("calf_diff"),
                    },
                }
            )


class PolicyNumpyModelSaver(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_counter = 1

    def is_target_event(self, obj, method, output, triggers):
        return isinstance(obj, MyScenario) and method == "reset_iteration"

    def on_function_call(self, obj, method, outputs):
        save_model(
            self,
            numpy_array=obj.policy.critic_weight_tensor_safe,
            iteration_counter=self.iteration_counter,
        )
        self.iteration_counter += 1

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        identifier = f"Policy_weights_during_episode_{str(iteration_number).zfill(5)}"
        if not self.data.empty:
            self.save_plot(identifier)
            self.insert_column_left("episode", iteration_number)
            self.dump_and_clear_data(identifier)


class PolicyModelSaver(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_counter = 1

    def is_target_event(self, obj, method, output, triggers):
        if (
            (isinstance(obj, REINFORCE) or isinstance(obj, PPO) or isinstance(obj, MyPPO))
            and method == "pre_optimize"
        ):
            which, event, time, episode_counter, iteration_counter = output
            return which == "Policy"

    def on_function_call(self, obj, method, outputs):
        save_nn_model(
            self,
            torch_nn_module=obj.policy.model,
            iteration_counter=self.iteration_counter,
        )
        self.iteration_counter += 1

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        identifier = f"Policy_weights_during_episode_{str(iteration_number).zfill(5)}"
        if not self.data.empty:
            self.save_plot(identifier)
            self.insert_column_left("episode", iteration_number)
            self.dump_and_clear_data(identifier)


class CriticModelSaver(HistoricalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration_counter = 1

    def is_target_event(self, obj, method, output, triggers):
        if (
            (isinstance(obj, PPO) or isinstance(obj, MyPPO))
            and method == "pre_optimize"
        ):
            which, event, time, episode_counter, iteration_counter = output
            return which == "Critic"

    def on_function_call(self, obj, method, outputs):
        save_nn_model(
            self,
            torch_nn_module=obj.critic.model,
            iteration_counter=self.iteration_counter,
        )
        self.iteration_counter += 1

    def on_episode_done(
        self,
        scenario,
        episode_number,
        episodes_total,
        iteration_number,
        iterations_total,
    ):
        identifier = f"Critic_weights_during_episode_{str(iteration_number).zfill(5)}"
        if not self.data.empty:
            self.save_plot(identifier)
            self.insert_column_left("episode", iteration_number)
            self.dump_and_clear_data(identifier)


def save_nn_model(
    cls: Union[PolicyModelSaver, CriticModelSaver],
    torch_nn_module: torch.nn.Module,
    iteration_counter: int,
) -> None:
    torch.save(
        torch_nn_module.state_dict(),
        Path(".callbacks")
        / cls.__class__.__name__
        / f"model_it_{iteration_counter:05}",
    )

def save_model(
    cls: Union[PolicyNumpyModelSaver],
    numpy_array: np.ndarray,
    iteration_counter: int,
) -> None:
    np.save(
        Path(".callbacks")
        / cls.__class__.__name__
        / f"model_it_{iteration_counter:05}.npy",
        numpy_array
    )


class HandlerChecker(ScenarioStepLogger):
    """A callback which allows to log every step of simulation in a scenario."""
    def is_target_event(self, obj, method, output, triggers):
        try:
            if len(self._metadata["logger"].handlers) == 0:
                self._metadata["logger"].addHandler(RichHandler())
        except Exception as err:
            print("Error:", err)
