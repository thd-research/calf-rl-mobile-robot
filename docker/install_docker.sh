#!/bin/bash

# Install docker
sudo apt update && sudo apt install -y apt-transport-https \
                                  ca-certificates \
                                  curl \
                                  gnupg-agent \
                                  software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io
sudo groupadd docker
sudo usermod -aG docker $USER


# NVIDIA Container Toolkit
if [[ $1 = "--nvidia" ]] || [[ $1 = "-n" ]]
  then
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

      sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
      sudo systemctl restart docker
fi


newgrp docker
