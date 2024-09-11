if [[ $1 = "--ros" ]] || [[ $1 = "-r" ]]
    then
        python3.10 run.py \
                    +seed=8 \
                    simulator=ros \
                    initial_conditions=3wrobot_kin_customized \
                    system=3wrobot_kin_with_spot \
                    scenario=mpc_scenario_customized \
                    scenario.running_objective.spot_gain=100 \
                    scenario.prediction_horizon=25 \
                    scenario.prediction_step_size=4 \
                    common.sampling_time=.1 \
                    simulator.time_final=40 \
                    simulator.use_phy_robot=true \
                    --interactive
    else
        python3.10 run.py \
                    initial_conditions=3wrobot_kin_customized \
                    system=3wrobot_kin_with_spot \
                    scenario=mpc_scenario_customized \
                    scenario.running_objective.spot_gain=100 \
                    scenario.prediction_horizon=20 \
                    common.sampling_time=1 \
                    --interactive \
                    --fps=10
fi
