from regelum.policy import PolicyPPO
import torch
from src.objective import ppo_objective


class MyPolicyPPO(PolicyPPO):
    def objective_function(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        time: torch.Tensor,
        episode_id: torch.Tensor,
        running_objective: torch.Tensor,
        initial_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Define the surrogate objective function used for PPO optimization.

        Args:
            observation: Observations from the environment.
            action: Actions taken by the policy.
            time: Time steps of the action.
            episode_id: Identifiers for episodes within the buffer.
            running_objective: Cumulative reward or cost for each step in the episodes.
            initial_log_probs: Initial policy log probabilities of actions.

        Returns:
            The surrogate objective function value for the PPO algorithm.
        """
        return ppo_objective(
            policy_model=self.model,
            critic_model=self.critic.model,
            observations=observation,
            actions=action,
            times=time,
            discount_factor=self.discount_factor,
            N_episodes=self.N_episodes,
            episode_ids=episode_id.long(),
            running_objectives=running_objective,
            initial_log_probs=initial_log_probs,
            cliprange=self.cliprange,
            running_objective_type=self.running_objective_type,
            sampling_time=self.sampling_time,
            gae_lambda=self.gae_lambda,
            is_normalize_advantages=self.is_normalize_advantages,
            entropy_coeff=self.entropy_coeff,
        )

