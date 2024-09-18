>**About**

This repository contains the codes for our CALF paper, which is empowered by the [Regelum](https://github.com/osinenkop/regelum-control) framework. The main purpose is to validate CALF performance and its relative SARSA-m over well-known controllers such as PPO, MPC, and Nominal controller.

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

<div align="center">
<img src="media/experiment_diagram.svg" width="850" />
</div>

# Getting Started

All the setting up and reproduction steps can be found [here](docs/get_started.md).

# Experimental achievement

At the end of this [experiment](https://www.youtube.com/watch?v=RgiDHzE5-w8&ab_channel=RomanZashchitin), the Turtlebot is capable of reaching the goal without passing the "hot" spot.

<div align="center">
<img src="media/CALF_SARSA-m turtlebot experiments.gif"/>
</div>

# Plots

## PPO

PPO has unstable performance. Although some runs successfully park near the target, the rest keep a certain distance away from the goal area.

On the left side, `PPO_full` trajectories are raw PPO trajectories showing the above behavior.

On the right side, supposing that the robot stops when reaching the area around the target x=0, y=0 with a radius of 0.12 meters. `PPO_simplified` depicts the top 10 trajectories satisfying that parking condition.

<div align="center">
<img src="media/report_PPO_full_trajectory.svg" width="402.9"/> <img src="media/report_PPO_simplified_trajectory.svg" width="416"/>
</div>

## SARSA-m

These 2 figures show unpredictable runs of SARSA-m, with the top 20 and top 10 runs having the lowest accumulated cost. To see how many runs can reach the goal, please have a look at [this histogram](#parking-position-error).

<div align="center">
<img src="media/report_SARSA-m top20_trajectory.svg" width="402"/> <img src="media/report_SARSA-m_trajectory.svg" width="392"/>
</div>

## CALF

The robot always targets the goal with CALF controllers. These 2 figures are the top 20 and top 10 runs with the lowest accumulated cost.

<div align="center">
<img src="media/report_CALF_top20_trajectory.svg" width="382"/> <img src="media/report_CALF_trajectory.svg" width="382"/>
</div>

## Parking position error

Here is a comparison of the successful parking frequency of the proposed controllers. Overall, CALF is entirely successful in parking at the target, and 17 out of 20 SARSA-m runs could reach the target, while PPO needs a stop condition in the target zone to meet the target.

<!--Here is a comparison of the successful parking frequency of proposed controllers. Overall, CALF is totally successful in parking at the goal, and 17 of 20 runs of SARSA-m could reach the goal, while PPO needs a condition of stopping at the target area to meet the goal.-->

<div align="center">
<img src="media/combined_hist_top_20_of_all_controllers_distance_from_goal.svg" width="359"/> <img src="media/combined_hist_selected_top_10_of_all_controllers_distance_from_goal.svg" width="353"/>
</div>

# Analysis

A simple notebook was written for controllers' performance monitoring and analysis.

Due to Regelum's storage structure, the date-time format for picking up a checkpoint is utilized as follows.

To load a single checkpoint:
```
start_datetime_str = end_datetime_str = "2024-08-29 16-29-17"
```


To load multi-checkpoints:

```
start_datetime_str = "2024-08-29 16-29-17"
end_datetime_str = "2024-08-30 06-07-04"
```

Using [MLFlow](docs/installation.md#monitor-training-progress-and-pick-checkpoints), all possible checkpoints can be found.
<!--All possible checkpoints could be found by using [MLFlow](docs/installation.md#monitor-training-progress-and-pick-checkpoints).-->

The above loading mechanism is used in this [Jupyter Notebook](notebooks/simple_plotting.ipynb), which shows trajectories, learning curves, accumulated cost over time, and control signals.

NOTE: `regelum-control` should be installed in a Jupyter kernel server.

# Remark

We consider the task of mobile robot parking as a benchmarking playground for the studied agents. In general, RL agents apply to any dynamical system, not restricted to settings addressable by traditional path planning. The mobile robot studied poses a canonical example of a non-holonomic control system, hence the interest in it specifically herein. Interested readers may refer to tabular RL, though. Yet, the curse of dimensionality may pose a formidable problem there.

The learning process entirely drove the behavior of all controllers in reaching the goal without relying on traditional path planning methods such as cell decomposition or potential fields.
<!--The behavior of all controllers in reaching the goal was entirely driven by the learning process, without relying on traditional path planning methods such as cell decomposition or potential fields.-->

# Licence

This project is licensed under the terms of the [MIT license](https://github.com/osinenkop/regelum-control/blob/main/LICENSE).


## Reference
Our experiment is based on Regelum with the following credit.
```
@misc{regelum2024,
author =   {Pavel Osinenko, Grigory Yaremenko, Georgiy Malaniya, Anton Bolychev},
title =    {Regelum: a framework for simulation, control and reinforcement learning},
howpublished = {\url{https://github.com/osinekop/regelum-control}},
year = {2024},
note = {Licensed under the MIT License}
}
```

## Bibtex cite

If you use our code for your projects, please give us credit:
Here is the Bibtex entry for our repo

```
@misc{calfrobot2024,
author =   {Grigory Yaremenko, Dmitrii Dobriborsci, Roman Zashchitin, Ruben Contreras Maestre, Ngoc Quoc Huy Hoang, Pavel Osinenko},
title =    {A novel agent with formal goal-reaching guarantees: an experimental study with a mobile robot},
howpublished = {\url{https://github.com/thd-research/calf-rl-mobile-robot}},
year = {2024},
note = {Licensed under the MIT License}
}
```
