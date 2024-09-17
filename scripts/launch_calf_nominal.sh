python3.10 run.py \
        +seed=7 \
        simulator=ros \
        simulator.time_final=50 \
        scenario=my_scenario \
        scenario.N_iterations=1 \
        system=3wrobot_kin_with_spot \
        initial_conditions=3wrobot_kin_with_spot \
        policy=rc_calfq \
        +policy.nominal_kappa_params="[0.2, 1.5, -.15]" \
        policy.nominal_only=True \
        common.sampling_time=0.1 \
        --single-thread \
        --experiment=benchmark
