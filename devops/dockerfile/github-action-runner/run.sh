REPO=$1
ACCESS_TOKEN=$2
ARCH=$3
DOCKER_PULL=false
TAG="0.1.0"

if [ $# != 3 ]; then
  echo "Please provide five arguments."
  echo "./runner-start.sh [YourGitRepo][YourGitHubRunnerToken][YourArch]"
  exit -1
fi

# List of Docker container names
# containers=("fedml/action_runner_3.8_$ARCH:0.1.0" "fedml/action_runner_3.9_$ARCH:0.1.0" "fedml/action_runner_3.10_$ARCH:0.1.0" "fedml/action_runner_3.11_$ARCH:0.1.0")
containers=("action_runner_3.8_$ARCH" "action_runner_3.9_$ARCH" "action_runner_3.10_$ARCH" "action_runner_3.11_$ARCH")

# Iterate through each container
for container in "${containers[@]}"; do
    # Find the running container
    if [ "$DOCKER_PULL" = "true" ]; then
        echo "docker pull fedml/$container:$TAG"
        docker pull fedml/$container:$TAG
    fi
    # docker stop `sudo docker ps |grep ${TAG}- |awk -F' ' '{print $1}'`

    running_container=$(docker ps -a | grep $container | awk -F ' ' '{print $1}')

    if [ -n "$running_container" ]; then
        # Stop the running container
        echo "Stopping running container: $container"
        docker rm "$running_container"
    else
        echo "No running container found for: $container"
    fi
    # docker pull $container
    ACT_NAME=$container
    docker run --rm --name $ACT_NAME --env REPO=$REPO --env ACCESS_TOKEN=$ACCESS_TOKEN -d fedml/$container:$TAG

done
echo "Script completed."

