for seed in {1..20}; do
    python3.10 run.py \
            +seed=$seed \
            simulator=ros \
            simulator.time_final=50 \
            initial_conditions=3wrobot_kin_with_spot \
            scenario=my_scenario \
            scenario.N_iterations=50 \
            system=3wrobot_kin_with_spot \
            common.sampling_time=0.1 \
            --single-thread \
            --experiment=sarsa_m_3wrobot_kin_ros \
            policy=rc_sarsa_m \
            +policy.R1_diag="[100, 100, 1, 0, 0]" \
            policy.critic_desired_decay=1e-6 \
            policy.critic_low_kappa_coeff=1e-1 \
            policy.critic_up_kappa_coeff=5e2 \
            policy.penalty_factor=1e2 \
            policy.step_size_multiplier=5
done