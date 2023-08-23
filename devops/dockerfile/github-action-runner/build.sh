docker build -t fedml/github-action-runner:latest -f ./Dockerfile .
docker login
docker push fedml/github-action-runner:latest