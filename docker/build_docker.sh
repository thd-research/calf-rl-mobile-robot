#!/usr/bin/env bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
EXEC_PATH=$PWD

cd $ROOT_DIR

if [[ $1 = "--nvidia" ]] || [[ $1 = "-n" ]]
  then
    docker build -t ros-regelum-rl-img -f $ROOT_DIR/docker/Dockerfile $ROOT_DIR \
                                       --network=host \
                                       --build-arg from=nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

else
    echo "[!] If you use nvidia gpu, please rebuild with -n or --nvidia argument"
    docker build -t ros-regelum-rl-img -f $ROOT_DIR/docker/Dockerfile $ROOT_DIR \
                                       --network=host \
                                       --build-arg from=ubuntu:20.04
fi

cd $EXEC_PATH

bash req/workspace_building.bash
