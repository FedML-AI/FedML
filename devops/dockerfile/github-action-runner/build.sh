docker login
docker build -t fedml/github-action-runner-torch:wx_test -f ./Dockerfile .
docker push fedml/github-action-runner-torch:wx_test