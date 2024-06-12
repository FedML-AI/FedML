REPO=$1
TAG=$2
NUM=$3
ACCESS_TOKEN=$4
LOCAL_DATA_DIR=$5

if [ $# != 5 ]; then
  echo "Please provide five arguments."
  echo "./runner-start.sh [YourGitRepo] [YourRunnerPrefix] [YourRunnerNum] [YourGitHubRunnerToken][LocalDataDir]"
  exit -1
fi

sudo docker stop `sudo docker ps |grep ${TAG}- |awk -F' ' '{print $1}'`
sudo docker pull fedml/github-action-runner:latest

for((i=1;i<=$NUM;i++));
do
ACT_NAME=$TAG-$i
sudo docker rm $ACT_NAME
sudo docker run --name $ACT_NAME --env REPO=$REPO --env ACCESS_TOKEN=$ACCESS_TOKEN -v $LOCAL_DATA_DIR:/home/fedml/fedml_data -v $LOCAL_DATA_DIR:/home/actions-runner/fedml_data -d fedml/github-action-runner:latest
done