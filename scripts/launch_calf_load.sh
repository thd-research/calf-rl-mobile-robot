python3.10 run.py \
           +seed=5 \
           simulator=ros \
           policy=rc_calfq \
           initial_conditions=3wrobot_kin_with_spot \
           +policy.nominal_kappa_params="[0.2, 1.5, -.15]" \
           scenario=my_scenario \
           system=3wrobot_kin_with_spot \
           common.sampling_time=0.1 \
           simulator.time_final=50 scenario.N_iterations=1 \
           --single-thread \
           --experiment=calf_report \
           policy.critic_desired_decay=1e-6 \
           policy.critic_low_kappa_coeff=1e-1 \
           policy.critic_up_kappa_coeff=1e3 \
           policy.penalty_factor=1e2 \
           policy.step_size_multiplier=5 \
           policy.weight_path="/regelum-ws/checkpoints/calf_240829/policy/model_it_00015.npy" \
           policy.nominal_only=False \
           simulator.use_phy_robot=false
