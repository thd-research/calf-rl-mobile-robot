if [[ $1 = "--ros" ]] || [[ $1 = "-r" ]]
    then
        python3.10 run.py \
                    simulator=ros \
                    initial_conditions=3wrobot_kin_customized \
                    system=3wrobot_kin \
                    scenario=mpc_scenario_customized \
                    scenario.prediction_horizon=20 \
                    scenario.prediction_step_size=10 \
                    common.sampling_time=.1 \
                    --interactive
 
    else
        python3.10 run.py \
                initial_conditions=3wrobot_kin_with_spot \
                system=3wrobot_kin \
                scenario=mpc_scenario_customized \
                scenario.running_objective.spot_gain=0 \
                scenario.prediction_horizon=3 \
                common.sampling_time=1 \
                --interactive \
                --fps=10
fi
