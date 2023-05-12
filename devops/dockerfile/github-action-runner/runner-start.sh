REPO=$1
TAG=$2
NUM=$3
ACCESS_TOKEN=$4
LOCAL_DEV_SOURCE_DIR=$5
LOCAL_RELEASE_SOURCE_DIR=$6

if [ $# != 6 ]; then
  echo "Please provide five arguments."
  echo "./runner-start.sh [YourGitRepo] [YourRunnerPrefix] [YourRunnerNum] [YourGitHubRunnerToken] [LocalDevSourceDir] [LocalReleaseSourceDir]"
  exit -1
fi

sudo docker stop `sudo docker ps |grep ${TAG}- |awk -F' ' '{print $1}'`

for((i=1;i<=$NUM;i++));
do
ACT_NAME=$TAG-$i
sudo docker rm $ACT_NAME
sudo docker run --name $ACT_NAME --env REPO=$REPO --env ACCESS_TOKEN=$ACCESS_TOKEN -v $LOCAL_DEV_SOURCE_DIR:/home/actions-runner/fedml-dev -v $LOCAL_RELEASE_SOURCE_DIR:/home/actions-runner/fedml-master -d fedml/github-action-runner:latest
done