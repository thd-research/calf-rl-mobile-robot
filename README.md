>**About**

This repository contains the codes for our CALF paper empowered by the `Regelum` framework. The main purpose is to validate CALF performance and it's relative SARSA-m over well-known controllers namely PPO, MPC, and Nominal controller.

For more details, please refer to the paper.

> **Table of Contents**
- [Method Overview](#method-overview)
- [Setup](#setup)
- [Getting Started](#getting-started)
  - [Simulation in Gazebo](#simulation-in-gazebo)
  - [Perform learning process in Gazebo](#perform-learning-process-in-gazebo)
    - [1. CALF](#1-calf)
    - [2. SARSA-m](#2-sarsa-m)
    - [3. PPO](#3-ppo)
  - [Perform Nonimal, MPC controllers and CALF, SARSA-m, PPO controllers with checkpoints](#perform-nonimal-mpc-controllers-and-calf-sarsa-m-ppo-controllers-with-checkpoints)
    - [1. CALF](#1-calf-1)
    - [2. SARSA-m](#2-sarsa-m-1)
    - [3. PPO](#3-ppo-1)
    - [4. Nominal](#4-nominal)
    - [5. MPC](#5-mpc)
  - [Perform the proposed controllers on Turtlebot3 in real-world](#perform-the-proposed-controllers-on-turtlebot3-in-real-world)
    - [Turtlebot setup](#turtlebot-setup)
    - [Perform the proposed controllers](#perform-the-proposed-controllers)

# Method Overview

<img src="media/experiment_diagram.svg" width="1000" />

# Setup
[Installation guildlines](docs/Installation.md)

# Getting Started
This repository is aimed to perform all the tasks in a docker container. Details of running and attaching the docker container can be found [here](docs/Installation.md#3-run-docker).

**ALL THE FOLLOWING COMMANDS HAVE TO BE USED INSIDE DOCKER CONTAINTER.**

## Simulation in Gazebo

To launch Turtlebot3 simulation in Gazebo

``` bash
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
```

## Perform learning process in Gazebo

Open another Terminal, to navigate to the workspace folder inside docker.

``` bash
cd regelum-ws
```

### 1. CALF

Execute the predefined script (for 20 seeds) by using the following command:

```
source scripts/launch_calf.sh -r
```

or directly execute with the seed 7:

```
python3.10 run.py \
           +seed=7 \
           simulator=ros \
           initial_conditions=3wrobot_kin_with_spot \
           policy=rc_calfq \
           +policy.nominal_kappa_params="[0.2, 1.5, -.15]" \
           scenario=my_scenario \
           system=3wrobot_kin_with_spot \
           common.sampling_time=0.1 \
           simulator.time_final=50 \
           scenario.N_iterations=40 \
           --single-thread \
           --experiment=calf_report \
           policy.critic_desired_decay=1e-6 \
           policy.critic_low_kappa_coeff=1e-1 \
           policy.critic_up_kappa_coeff=1e3 \
           policy.penalty_factor=1e2 \
           policy.step_size_multiplier=5 \
           policy.nominal_only=False
```

### 2. SARSA-m

Execute the predefined script (for 20 seeds) by using the following command:

```
source scripts/launch_sarsa_m.sh -r
```

or directly execute with the seed 7:

```
python3.10 run.py \
           +seed=7 \
           simulator=ros \
           policy=rc_sarsa_m \
           initial_conditions=3wrobot_kin_with_spot \
           +policy.R1_diag="[100, 100, 1, 0, 0]" \
           scenario=my_scenario \
           system=3wrobot_kin_with_spot \
           common.sampling_time=0.1 \
           simulator.time_final=50 \
           scenario.N_iterations=50 \
           --single-thread \
           --experiment=sarsa_m_report \
           policy.critic_desired_decay=1e-6 \
           policy.critic_low_kappa_coeff=1e-1 \
           policy.critic_up_kappa_coeff=5e2 \
           policy.penalty_factor=1e2 \
           policy.step_size_multiplier=5
```

### 3. PPO
   
Execute the predefined script (for 20 seeds) by using the following command:

```
source scripts/launch_ppo.sh -r
```

or directly execute with the seed 7:

```
python3.10 run.py \
           +seed=7 \
           simulator=ros \
           scenario=ppo_scenario \
           system=3wrobot_kin_customized \
           --single-thread \
           --experiment=ppo_3wrobot_kin_retry_2708_1 \
           scenario.N_episodes=1 \
           scenario.N_iterations=100 \
           scenario.policy_n_epochs=50 \
           scenario.critic_n_epochs=50 \
           scenario.policy_opt_method_kwargs.lr=0.005 \
           scenario.policy_model.n_hidden_layers=2 \
           scenario.policy_model.dim_hidden=15 \
           scenario.policy_model.std=0.1 \
           scenario.critic_model.n_hidden_layers=3 \
           scenario.critic_model.dim_hidden=15 \
           scenario.critic_opt_method_kwargs.lr=0.1 \
           scenario.gae_lambda=0 \
           scenario.discount_factor=0.9 \
           scenario.cliprange=0.2 \
           scenario.critic_td_n=1 \
           simulator.time_final=50 \
           common.sampling_time=0.1
```

## Perform Nonimal, MPC controllers and CALF, SARSA-m, PPO controllers with checkpoints
To run the robot with a given checkpoint, we reset the value of N_iterations and N_episode to 1 and provide checkpoint paths where weights matrix was stored to `policy.weight_path` or/and `critic.weight_path` if that controller requires.

### 1. CALF

```
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
           --experiment=benchmark \
           policy.critic_desired_decay=1e-6 \
           policy.critic_low_kappa_coeff=1e-1 \
           policy.critic_up_kappa_coeff=1e3 \
           policy.penalty_factor=1e2 \
           policy.step_size_multiplier=5 \
           policy.weight_path="/regelum-ws/checkpoints/calf_240829/policy/model_it_00015.npy" \
           policy.nominal_only=False \
           simulator.use_phy_robot=false \
           --interactive
```

### 2. SARSA-m

```
python3.10 run.py \
            +seed=5 \
            simulator=ros \
            policy=rc_sarsa_m \
            initial_conditions=3wrobot_kin_with_spot \
            +policy.R1_diag="[100, 100, 1, 0, 0]" \
            scenario=my_scenario \
            system=3wrobot_kin_with_spot \
            common.sampling_time=0.1 \
            simulator.time_final=50 \
            scenario.N_iterations=1 \
            --single-thread \
            --experiment=benchmark \
            policy.critic_desired_decay=1e-6 \
            policy.critic_low_kappa_coeff=1e-1 \
            policy.critic_up_kappa_coeff=4e2 \
            policy.penalty_factor=1e2 \
            policy.step_size_multiplier=5 \
            policy.weight_path="/regelum-ws/checkpoints/sarsa_m_240830/policy/model_it_00033.npy" \
            simulator.use_phy_robot=false \
            --interactive

```

### 3. PPO

```
python3.10 run.py \
        +seed=4 \
        simulator=ros \
        scenario=ppo_scenario \
        system=3wrobot_kin_customized \
        --single-thread \
        --experiment=benchmark \
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
        simulator.use_phy_robot=false \
        --interactive
```

### 4. Nominal

```
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
           simulator.use_phy_robot=false \
           --single-thread \
           --experiment=benchmark
```

### 5. MPC

```
python3.10 run.py \
            +seed=8 \
            simulator=ros \
            initial_conditions=3wrobot_kin_customized \
            system=3wrobot_kin_with_spot \
            scenario=mpc_scenario_customized \
            scenario.running_objective.spot_gain=100 \
            scenario.prediction_horizon=10 \
            scenario.prediction_step_size=4 \
            common.sampling_time=.1 \
            simulator.time_final=50 \
            simulator.use_phy_robot=false \
            --interactive \
            --experiment=benchmark
```

## Perform the proposed controllers on Turtlebot3 in real-world

### Turtlebot setup
To control physical turtlebot, we need to connect the PC running Regelum controllers and the turtlebot to the same Wifi access point. 

**NOTE: The Turtlebot in Gazebo is not allowed to run while we run the turtlebot in real-world.**

1. Network configuration: Follow the instruction [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#network-configuration), find the assigned IP address of your PC (i.e. `192.168.122.11`) and execute these commands: (inside docker container)
```
source config_ros_host.sh <your_IP_address>
source ~/.bashrc
```
And then start ROS Master:

```
roscore
```

2. Turtlebot calibration:
Follow the Bring-up instruction [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/bringup/#bringup).

### Perform the proposed controllers

Execute the command mentioned in the [previous section](#perform-nonimal-mpc-controllers-and-calf-sarsa-m-ppo-controllers-with-checkpoints) but change the variable `simulator.use_phy_robot` to `true`.

For example: Nominal controller

```
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
           simulator.use_phy_robot=true \
           --single-thread \
           --experiment=benchmark
```