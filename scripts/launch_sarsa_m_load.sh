python3.10 run.py \
            +seed=5 \
            simulator=ros \
            policy=rc_sarsa_m \
            initial_conditions=3wrobot_kin_with_spot \
            +policy.R1_diag="[100, 100, 1, 0, 0]" \
            scenario=my_scenario \
            system=3wrobot_kin_with_spot \
            common.sampling_time=0.1 \
            simulator.time_final=70 \
            scenario.N_iterations=1 \
            --single-thread \
            --experiment=sarsa_m_init_ros \
            policy.critic_desired_decay=1e-6 \
            policy.critic_low_kappa_coeff=1e-1 \
            policy.critic_up_kappa_coeff=4e2 \
            policy.penalty_factor=1e2 \
            policy.step_size_multiplier=5 \
            policy.weight_path="/regelum-ws/checkpoints/sarsa_m_240830/policy/model_it_00033.npy" \
            --interactive