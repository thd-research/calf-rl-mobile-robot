from regelum import objective
import numpy as np
from typing import Union
from regelum.utils import rg
from regelum.model import ModelQuadLin, PerceptronWithTruncatedNormalNoise, ModelNN
import torch
from regelum.objective import get_gae_advantage


class ThreeWheeledRobotCostWithSpot(objective.RunningObjective):
    def __init__(
        self,
        quadratic_model: ModelQuadLin,
        spot_gain: float,
        spot_x_center: float,
        spot_y_center: float,
        spot_std: float,
    ):
        self.quadratic_model = quadratic_model
        self.spot_gain = spot_gain
        self.spot_x_center = spot_x_center
        self.spot_y_center = spot_y_center
        self.spot_std = spot_std

    def cal_spot_penalty(self, observation):
        return (
            self.spot_gain
            * rg.exp(
                -(
                    (observation[:, 0] - self.spot_x_center) ** 2
                    + (observation[:, 1] - self.spot_y_center) ** 2
                )
                / (2 * self.spot_std**2)
            )
            / (2 * np.pi * self.spot_std**2)
        )

    def __call__(
        self,
        observation,
        action,
        is_save_batch_format: bool = False,
    ):
        spot_cost = self.cal_spot_penalty(observation)

        quadratic_cost = self.quadratic_model(observation, action)
        cost = quadratic_cost + spot_cost

        if is_save_batch_format:
            return cost
        else:
            return cost[0, 0]

        # if is_save_batch_format:
        #     return rg.array(
        #         rg.array(quadratic_cost, prototype=observation),
        #         prototype=observation,
        #     )
        # else:
        #     return quadratic_cost

def ppo_objective(
    policy_model: PerceptronWithTruncatedNormalNoise,
    critic_model: ModelNN,
    observations: torch.FloatTensor,
    actions: torch.FloatTensor,
    times: torch.FloatTensor,
    episode_ids: torch.LongTensor,
    discount_factor: float,
    N_episodes: int,
    running_objectives: torch.FloatTensor,
    cliprange: float,
    initial_log_probs: torch.FloatTensor,
    running_objective_type: str,
    sampling_time: float,
    gae_lambda: float,
    is_normalize_advantages: bool = True,
    entropy_coeff: float = 0.0,
) -> torch.FloatTensor:
    """Calculate PPO objective.

    Args:
        policy_model: The neural network model representing the policy.
        critic_model: The neural network model representing the value function (critic).
        observations: A tensor of observations from the environment.
        actions: A tensor of actions taken by the agent.
        times: A tensor with timestamps for each observation-action pair.
        episode_ids: A tensor with unique identifiers for each episode.
        discount_factor: The factor by which future rewards are discounted.
        N_episodes: The total number of episodes over which the objective is averaged.
        running_objectives: A tensor of accumulated rewards or costs for each timestep.
        cliprange: The range for clipping the probability ratio in the objective function.
        initial_log_probs: The log probabilities of taking the actions at the time
            of sampling, under the policy model before the update.
        running_objective_type (str): Indicates whether the running objectives are 'cost' or 'reward'.
        sampling_time: The timestep used for sampling in the environment.
        gae_lambda: The lambda parameter for GAE, controlling the trade-off between bias and variance.
        is_normalize_advantages: Flag indicating whether to normalize advantage estimates.

    Returns:
        objective for PPO
    """
    assert (
        running_objective_type == "cost" or running_objective_type == "reward"
    ), "running_objective_type can be either 'cost' or 'reward'"

    critic_values = critic_model(observations)
    prob_ratios = torch.exp(
        policy_model.log_pdf(observations, actions) - initial_log_probs.reshape(-1)
    ).reshape(-1, 1)
    if hasattr(policy_model, "entropy"):
        entropies = entropy_coeff * policy_model.entropy(observations).reshape(-1, 1)
    else:
        entropies = torch.zeros_like(prob_ratios)
    clipped_prob_ratios = torch.clamp(prob_ratios, 1 - cliprange, 1 + cliprange)
    objective_value = 0.0
    for episode_idx in torch.unique(episode_ids):
        mask = episode_ids.reshape(-1) == episode_idx
        advantages = get_gae_advantage(
            gae_lambda=gae_lambda,
            running_objectives=running_objectives[mask],
            values=critic_values[mask],
            times=times[mask],
            discount_factor=discount_factor,
            sampling_time=sampling_time,
        )
        if is_normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        objective_value += (
            torch.sum(
                (discount_factor)
                * (
                    torch.maximum(
                        advantages * prob_ratios[mask][:-1],
                        advantages * clipped_prob_ratios[mask][:-1],
                    )
                    - entropies[mask][:-1]
                    if running_objective_type == "cost"
                    else torch.minimum(
                        advantages * prob_ratios[mask][:-1],
                        advantages * clipped_prob_ratios[mask][:-1],
                    )
                    + entropies[mask][:-1]
                )
            )
            / N_episodes
        )

    return objective_value