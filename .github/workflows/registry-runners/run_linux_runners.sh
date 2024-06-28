REPO=$1
ACCESS_TOKEN=$2
API_KEY=$3
DOCKER_PULL=false
ARCH=linux64
TAG="0.1.0"

if [ $# != 3 ]; then
  echo "Please provide two arguments."
  echo "./runner-start.sh [YourGitRepo][YourGitHubRunnerToken][API_KEY]"
  exit -1
fi

# List of Docker container names
# containers=("fedml/action_runner_3.8_$ARCH:0.1.0" "fedml/action_runner_3.9_$ARCH:0.1.0" "fedml/action_runner_3.10_$ARCH:0.1.0" "fedml/action_runner_3.11_$ARCH:0.1.0")
containers=("action_runner_3.8_$ARCH" "action_runner_3.9_$ARCH" "action_runner_3.10_$ARCH" "action_runner_3.11_$ARCH")
python_versions=("python3.8" "python3.9" "python3.10" "python3.11")


# Iterate through each container
for container_index in "${!containers[@]}"; do

    container=${containers[$container_index]}
    # Find the running container
    if [ "$DOCKER_PULL" = "true" ]; then
        echo "docker pull fedml/$container:$TAG"
        docker pull fedml/$container:$TAG
    fi
    # docker stop `sudo docker ps |grep ${TAG}- |awk -F' ' '{print $1}'`

    running_container=$(docker ps -a | grep $container | awk -F ' ' '{print $1}')

    if [ -n "$running_container" ]; then
        # Stop the running container
        echo "Stopping running container: $container, $running_container"
        docker stop "$running_container"
    else
        echo "No running container found for: $container"
    fi
    sleep 5
    # docker pull $container
    ACT_NAME=${containers[$container_index]}
    echo "docker run --rm --name $ACT_NAME --env API_KEY=$API_KEY --env REPO=$REPO --env ACCESS_TOKEN=$ACCESS_TOKEN -d fedml/${containers[$container_index]}:$TAG bash ./start.sh ${REPO} ${ACCESS_TOKEN} ${python_versions[$container_index]}"
    docker run --rm --name $ACT_NAME --env API_KEY=$API_KEY --env REPO=$REPO --env ACCESS_TOKEN=$ACCESS_TOKEN -d fedml/${containers[$container_index]}:$TAG bash ./start.sh ${REPO} ${ACCESS_TOKEN} ${python_versions[$container_index]}

done
echo "Script completed."

