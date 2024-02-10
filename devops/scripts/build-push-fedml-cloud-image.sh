
version=$1
image_version=$2

if [ ${version} == "dev" ]; then
  docker build --network=host -f ./devops/dockerfile/device-image/Dockerfile-Dev -t fedml/fedml-device-image:${version} .
  docker push fedml/fedml-device-image:${version}

  docker build --network=host -f ./devops/dockerfile/server-agent/Dockerfile-Dev -t fedml/fedml-server-agent:${version} .
  docker push fedml/fedml-server-agent:${version}
elif [ ${version} == "test" ]; then
  echo "test"
  docker build --network=host -f ./devops/dockerfile/device-image/Dockerfile-Test -t fedml/fedml-device-image:${version} .
  docker push fedml/fedml-device-image:${version}

  docker build --network=host -f ./devops/dockerfile/server-agent/Dockerfile-Test -t fedml/fedml-server-agent:${version} .
  docker push fedml/fedml-server-agent:${version}
elif [ ${version} == "release" ]; then
  docker build --network=host -f ./devops/dockerfile/device-image/Dockerfile-Release -t fedml/fedml-device-image:${version} .
  docker push fedml/fedml-device-image:${version}

  docker tag fedml/fedml-device-image:release public.ecr.aws/x6k8q1x9/fedml-device-image:release
  aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
  docker push public.ecr.aws/x6k8q1x9/fedml-device-image:release

  docker build --network=host -f ./devops/dockerfile/server-agent/Dockerfile-Release -t fedml/fedml-server-agent:${version} .
  docker push fedml/fedml-server-agent:${version}
elif [ ${version} == "local" ]; then
  docker build --network=host -f ./devops/dockerfile/device-image/Dockerfile-Local -t fedml/fedml-device-image:${version} .
  docker push fedml/fedml-device-image:${version}

  docker build --network=host -f ./devops/dockerfile/server-agent/Dockerfile-Local -t fedml/fedml-server-agent:${image_version} .
  docker push fedml/fedml-server-agent:${image_version}
fi

