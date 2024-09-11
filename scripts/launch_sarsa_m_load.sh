# ROOT="regelum_data/outputs"
# SARSA_M_MODEL_PATH=${ROOT} + "/2024-08-07/11-53-44/0/.callbacks/PolicyNumpyModelSaver/model_it_00009.npy"

if [[ $1 = "--ros" ]] || [[ $1 = "-r" ]]
    then  
    for seed in {2..2}; do
        python3.10 run.py \
                  +seed=$seed \
                  simulator=ros \
                  policy=rc_sarsa_m \
                  initial_conditions=3wrobot_kin_with_spot \
                  +policy.R1_diag="[100, 100, 1, 0, 0]" \
                  scenario=my_scenario \
                  system=3wrobot_kin_with_spot \
                  common.sampling_time=0.1 \
                  simulator.time_final=70 scenario.N_iterations=1 \
                  --single-thread \
                  --jobs=-1 \
                  --experiment=sarsa_m_init_ros \
                  policy.critic_desired_decay=1e-6 \
                  policy.critic_low_kappa_coeff=1e-1 \
                  policy.critic_up_kappa_coeff=4e2 \
                  policy.penalty_factor=1e2 \
                  policy.step_size_multiplier=5 \
                  policy.weight_path="/regelum-playground/regelum_data/outputs/2024-08-30/03-58-04/0/.callbacks/PolicyNumpyModelSaver/model_it_00033.npy" \
                  simulator.use_phy_robot=true \
                  --interactive
    done
    else
        python3.10 run.py +seed=7 \
                  policy=rc_sarsa_m \
                  initial_conditions=3wrobot_kin_with_spot \
                  +policy.R1_diag="[10, 100, 1e-1, 0, 0]" \
                  scenario=my_scenario \
                  system=3wrobot_kin_with_spot \
                  common.sampling_time=0.1 \
                  simulator.time_final=15 scenario.N_iterations=1 \
                  --jobs=-1 \
                  --experiment=sarsa_m_init \
                  policy.critic_desired_decay=1e-6 \
                  policy.critic_low_kappa_coeff=1e-1 \
                  policy.critic_up_kappa_coeff=1e3 \
                  policy.penalty_factor=1e3 \
                  policy.step_size_multiplier=5 \
                  --interactive --fps=10 \
                  policy.weight_path=${CALF_MODEL_PATH}
fi
