# Installation

## Description
Here you will find your exercises


### 1. Install Docker

You need to go into the downloaded folder and go into the docker folder.


    cd ./docker

If you don't have an Nvidia graphics card, run the command:

    ./install_docker.bash

    
The installation of docker should complete correctly. If docker does not install, it is most likely an error in your utility (apt) settings for downloading and updating from remote repositories. This is a problem you should solve on your own.


### 2. Building Docker
    
We must first create the environment in which our program will run. We already have a ready-made environment and we just need to building it:

    source ./build_docker.sh

Next will begin the process of building. If the process ends incorrectly or with an error, you must start building again.

### 3. Run Docker

To execute the docker container use command:

    source ./run_docker.sh
    
If you need additional terminal inside of the Docker open new window in the terminal (Ctrl+Shift+T) and use command

    source ./into_docker.sh

### 4. Setup ROS workspace

For the first time using `turtlebot3_ws` workspace, it need to be built to set up turtlebot3 packages.

After getting into the docker container, use this command:
    
    source turtlebot3_ws/build.bash
