>**About**

This repository contains the codes for our CALF paper empowered by the [Regelum](https://github.com/osinenkop/regelum-control) framework. The main purpose is to validate CALF performance and it's relative SARSA-m over well-known controllers namely PPO, MPC, and Nominal controller.

For more details, please refer to the paper.

> **Table of Contents**
- [Method Overview](#method-overview)
- [Getting Started](#getting-started)
- [Experimental achievement](#experimental-achievement)
- [Plots](#plots)
  - [PPO](#ppo)
  - [SARSA-m](#sarsar-m)
  - [CALF](#calf)
  - [Parking position error](#parking-position-error)

- [Analysis](#analysis)
- [Remark](#remark)
- [Licence](#licence)
  - [Bibtex reference](#bibtex-reference)

# Method Overview

<img src="media/experiment_diagram.svg" width="1000" />


# Getting Started

All the reproduction steps can be found [here](docs/get_started.md).

# Experimental achievement

At the end of this experiment, the turtlebot is capaple of reaching the goal without passing the "hot" spot.

[![video of controllers trajectories](https://img.youtube.com/vi/RgiDHzE5-w8/0.jpg)](https://www.youtube.com/watch?v=RgiDHzE5-w8)


# Plots

## PPO

PPO has unstable performances. Despite some runs successfully parking nearby the target, the rest keeps a certain distance away from the goal area.

On the left side, `PPO_full` trajectories are raw trajectories from PPO derived mentioned behaviors.

On the right side, supposing that all the robot stops when reaching the area around the target x=0, y=0 a radius of 0.12 meter, `PPO_simplified` depicts top 10 trajectories satisfying that parking condition.

<img src="media/report_PPO_full_trajectory.svg" width="415"/> <img src="media/report_PPO_simplified_trajectory.svg" width="425"/> 

## SARSA-m

These 2 figures show unpredictable runs of SARSA-m with top 20 and top 10 runs having lowest accumulated cost. To see how many runs can reach the goal, please have a look at [this histogram](#parking-position-error).


<img src="media/report_SARSA-m top20_trajectory.svg" width="425"/> <img src="media/report_SARSA-m_trajectory.svg" width="415"/> 

## CALF

Robot always targets the goal with CALF controllers. These 2 figure are top 20 and top 10 runs having lowest accumulated cost.

<img src="media/report_CALF_top20_trajectory.svg" width="425"/> <img src="media/report_CALF_trajectory.svg" width="425"/>

## Parking position error

Here is the comparison of the successfully parking frequency of proposed controllers. Overall, CALF is totally successfull in parking at the goal, and 17 of 20 runs of SARSA-m could reach the goal while PPO needs a condition of stopping at the target area to meet the goal.

<img src="media/combined_hist_top_20_of_all_controllers:_distance_from_goal.svg" /> 
<img src="media/combined_hist_selected_top_10_of_all_controllers:_distance_from_goal.svg" />


# Analysis

A simple notebook was written for controllers performance monitoring and analysis.

Due to the storing stucture of regelum, datetime format to pick up a checkpoint is utilized as below.

To load single checkpoint:
```
start_datetime_str = end_datetime_str = "2024-08-29 16-29-17"
```


To load multi-checkpoints:

```
start_datetime_str = "2024-08-29 16-29-17"
end_datetime_str = "2024-08-30 06-07-04"
```

All possible checkpoints could be found by using [MLFlow](docs/installation.md#monitor-training-progress-and-pick-checkpoints).

The above loading mechanism is used in this [Jupyter Notebook](notebooks/simple_plotting.ipynb) where show trajectories, learning curves, accumulated cost over time, and control signals.

NOTE: `regelum-control` should be installed in a jupyter kernel server.

# Remark

The behavior of all controllers in reaching the goal was entirely driven by the learning process, without relying on traditional path planning methods such as cell decomposition or potential fields.

# Licence

This project is licensed under the terms of the [MIT license](https://github.com/osinenkop/regelum-control/blob/main/LICENSE).

## Bibtex reference

Thank you for citing [Regelum control](https://github.com/osinenkop/regelum-control) if you use any of this code.

```
@misc{regelum2024,
author =   {Pavel Osinenko, Grigory Yaremenko, Georgiy Malaniya, Anton Bolychev},
title =    {Regelum: a framework for simulation, control and reinforcement learning},
howpublished = {\url{https://github.com/osinekop/regelum-control}},
year = {2024},
note = {Licensed under the MIT License}
}
```
