python3.10 run.py \
    +seed=4 \
    simulator=ros \
    scenario=ppo_scenario \
    system=3wrobot_kin_customized \
    --single-thread \
    --experiment=ppo_report \
    scenario.N_episodes=1 \
    scenario.N_iterations=1 \
    scenario.policy_n_epochs=50 \
    scenario.critic_n_epochs=50 \
    scenario.policy_opt_method_kwargs.lr=0.005 \
    scenario.policy_model.n_hidden_layers=2 \
    scenario.policy_model.dim_hidden=15 \
    scenario.policy_model.std=0.01 \
    scenario.critic_model.n_hidden_layers=3 \
    scenario.critic_model.dim_hidden=15 \
    scenario.critic_opt_method_kwargs.lr=0.1 \
    scenario.gae_lambda=0 \
    scenario.discount_factor=0.9 \
    scenario.cliprange=0.2 \
    scenario.critic_td_n=1 \
    simulator.time_final=50 \
    common.sampling_time=0.1 \
    scenario.policy_checkpoint_path="/regelum-ws/checkpoints/ppo_240827/policy/model_it_00053" \
    scenario.critic_checkpoint_path="/regelum-ws/checkpoints/ppo_240827/critic/model_it_00053" \
    --interactive