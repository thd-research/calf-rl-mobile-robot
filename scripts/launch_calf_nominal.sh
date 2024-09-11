if [[ $1 = "--ros" ]] || [[ $1 = "-r" ]]
    then  
        python3.10 run.py \
                  +seed=7 \
                  simulator=ros \
                  policy=rc_calfq \
                  initial_conditions=3wrobot_kin_with_spot \
                  +policy.nominal_kappa_params="[0.1, 1.5, -.15]" \
                  scenario=my_scenario \
                  system=3wrobot_kin_with_spot \
                  common.sampling_time=0.1 \
                  simulator.time_final=50 scenario.N_iterations=1 \
                  --single-thread \
                  --jobs=-1 \
                  --experiment=calf_inc_penalty \
                  policy.critic_desired_decay=1e-6 \
                  policy.critic_low_kappa_coeff=1e-1 \
                  policy.critic_up_kappa_coeff=1e3 \
                  policy.penalty_factor=1e3 \
                  policy.step_size_multiplier=5 \
                  policy.nominal_only=True \
                  simulator.use_phy_robot=true \
                  --interactive
    else
        python3.10 run.py +seed=7 \
                  policy=rc_calfq \
                  initial_conditions=3wrobot_kin_with_spot \
                  scenario=my_scenario \
                  system=3wrobot_kin_with_spot \
                  common.sampling_time=0.1 \
                  simulator.time_final=40 scenario.N_iterations=1 \
                  --jobs=-1 \
                  --experiment=calf_inc_penalty \
                  policy.critic_desired_decay=1e-6 \
                  policy.critic_low_kappa_coeff=1e-1 \
                  policy.critic_up_kappa_coeff=1e3 \
                  policy.penalty_factor=1e2 \
                  policy.step_size_multiplier=5 \
                  policy.nominal_only=True \
                  --interactive --fps=10
fi
