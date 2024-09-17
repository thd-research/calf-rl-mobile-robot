#!/bin/bash

xhost +local:docker || true

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

if [[ $1 = "--nvidia" ]] || [[ $1 = "-n" ]]
  then
    docker run --gpus all \
                -ti --rm \
                -e "DISPLAY" \
                -e "QT_X11_NO_MITSHM=1" \
                -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
                -e XAUTHORITY \
                -v /dev:/dev \
                -v $ROOT_DIR/turtlebot3_ws:/turtlebot3_ws \
                -v $ROOT_DIR:/regelum-ws \
               --net=host \
               --privileged \
               --name ros-regelum ros-regelum-rl-img

else

    echo "[!] If you wanna use nvidia gpu, please use script with -n or --nvidia argument"
    docker run  -ti --rm \
                -e "DISPLAY" \
                -e "QT_X11_NO_MITSHM=1" \
                -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
                -e XAUTHORITY \
                -v /dev:/dev \
                -v $ROOT_DIR/turtlebot3_ws:/turtlebot3_ws \
                -v $ROOT_DIR:/regelum-ws \
               --net=host \
               --privileged \
               --name ros-regelum ros-regelum-rl-img
fi

cd $ROOT_DIR/.git && \
  sudo chgrp -R $(id -g -n $(whoami)) . &&\
  sudo chmod -R g+rwX . &&\
  sudo find . -type d -exec chmod g+s '{}' + &&\
  cd ..
