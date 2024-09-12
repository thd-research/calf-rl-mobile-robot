if [ $# -eq 0 ]
    then
        IP_ADDRESS=192.168.122.19
    else
        IP_ADDRESS=$1
fi

echo 'export ROS_MASTER_URI=http://'$IP_ADDRESS':11311' >> ~/.bashrc
echo 'export ROS_HOSTNAME=' $IP_ADDRESS >> ~/.bashrc
