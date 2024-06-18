tag="0.1.0"

platform="linux/amd64"

echo "build python:3.11"
docker build --no-cache --platform $platform --build-arg BASE_IMAGE=python:3.11 -t fedml/action_runner_3.11_linux64:$tag -f ./Dockerfile .
echo "build python:3.10"
docker build --no-cache  --platform $platform --build-arg BASE_IMAGE=python:3.10 -t fedml/action_runner_3.10_linux64:$tag -f ./Dockerfile .
echo "build python:3.9"
docker build --no-cache --platform $platform --build-arg BASE_IMAGE=python:3.9 -t fedml/action_runner_3.9_linux64:$tag -f ./Dockerfile .
echo "build python:3.8"
docker build --no-cache --platform $platform --build-arg BASE_IMAGE=python:3.8 -t fedml/action_runner_3.8_linux64:$tag -f ./Dockerfile .
