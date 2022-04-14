#!/usr/bin/env bash

echo "Now, you are building the fedml project to be two packages which will be used in the MLOps platform."
echo "The packages will be used for client training and server aggregated."
echo "When the building process is completed, you will find the packages in directories named dist-packages/server and dist-package/client."
echo "Then you may upload the packages on the configuration page in the MLOps platform to start the federated learning flow."
echo "Building..."

cur_dir=`pwd`
client_dest=./mlops-core/fedml-client/package/fedml/
mkdir -p ./mlops-core/fedml-client/package/fedml
rm -rf ./mlops-core/fedml-client/package/fedml/*
cp  -rf ../fedml_api ${client_dest}
cp  -rf ../fedml_core ${client_dest}
cp  -rf ../fedml_experiments ${client_dest}
rm -rf ${client_dest}/fedml_experiments/distributed/fedgkt
rm -rf ${client_dest}/fedml_experiments/standalone/decentralized/doc
rm -rf ${client_dest}/fedml_api/model/cv/pretrained

server_dest=./mlops-core/fedml-server/package/fedml/
mkdir -p ./mlops-core/fedml-server/package/fedml
rm -rf ./mlops-core/fedml-server/package/fedml/*
cp  -rf ../fedml_api ${server_dest}
cp  -rf ../fedml_core ${server_dest}
cp  -rf ../fedml_experiments ${server_dest}
rm -rf ${server_dest}/fedml_experiments/distributed/fedgkt
rm -rf ${server_dest}/fedml_experiments/standalone/decentralized/doc
rm -rf ${server_dest}/fedml_api/model/cv/pretrained

cd ${cur_dir}/mlops-core/fedml-client/
rm -f package.zip
zip -q -r package.zip ./package/*
mkdir -p ${cur_dir}/dist-packages/client/
mv package.zip ${cur_dir}/dist-packages/client/

cd ${cur_dir}/mlops-core/fedml-server/
rm -f package.zip
zip -q -r package.zip ./package/*
mkdir -p ${cur_dir}/dist-packages/server/
mv package.zip ${cur_dir}/dist-packages/server/

echo "You have finished all building process. "
echo "Now you may use dist-packages/client/package.zip and dist-packages/server/package.zip to start your federated learning run."

cp ${cur_dir}/dist-packages/client/package.zip ${cur_dir}/../../FLServer_Agent/devops/deploy/dist-packages/client/ >/dev/null 2>&1
cp ${cur_dir}/dist-packages/server/package.zip ${cur_dir}/../../FLServer_Agent/devops/deploy/dist-packages/server/ >/dev/null 2>&1

cp ${cur_dir}/dist-packages/client/package.zip ${cur_dir}/../../FLClient_Agent/devops/deploy/dist-packages/client/ >/dev/null 2>&1
cp ${cur_dir}/dist-packages/server/package.zip ${cur_dir}/../../FLClient_Agent/devops/deploy/dist-packages/server/ >/dev/null 2>&1
