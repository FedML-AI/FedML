# --exclude='path/to/excluded/dir'
# git clone https://github.com/Qigemingziba/FedML.git 
# git checkout dev/v0.7.0
docker login
docker build -t fedml/github-action-runner_wx:test -f ./DockerfileLight .
docker run --rm fedml/github-action-runner_wx:test