# Installation

## Description
Here, you will find your exercises


### 1. Install Docker

You need to go into the downloaded folder and go into the docker folder.


    cd ./docker

If you don't have a Nvidia graphics card, run the command:

    ./install_docker.bash

    
The installation of Docker should be completed correctly. If Docker does not install, it is most likely an error in your utility (apt) settings for downloading and updating from remote repositories. You should solve this problem on your own.


### 2. Building Docker
    
We must first create the environment in which our program will run. We already have a ready-made environment, and we need to build it:

    source ./build_docker.sh

Next, you will begin the building process. You must start again if the process ends incorrectly or with an error.

### 3. Run Docker

To execute the Docker container, use the command:

    source ./run_docker.sh
    
If you need an additional terminal inside Docker, open a new window in the terminal (Ctrl+Shift+T) and use the command:

    source ./into_docker.sh

### 4. Setup ROS workspace

If you are using the `turtlebot3_ws` workspace for the first time, it needs to be built to set up turtlebot3 packages.

After getting into the Docker container, use this command:
    
    source turtlebot3_ws/build.bash
