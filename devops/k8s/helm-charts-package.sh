#!/bin/bash

helm package ./fedml-model-premise-slave --app-version release
helm package ./fedml-model-premise-master --app-version release
helm package ./fedml-edge-client-server/fedml-client-deployment --app-version release
helm package ./fedml-edge-client-server/fedml-server-deployment --app-version release
mv ./fedml-model-premise-slave*.tgz ../../installation/install_on_k8s/fedml-model-premise-slave-latest.tgz
mv ./fedml-model-premise-master*.tgz ../../installation/install_on_k8s/fedml-model-premise-master-latest.tgz
mv ./fedml-client-deployment*.tgz ../../installation/install_on_k8s/fedml-edge-client-server/fedml-client-deployment-latest.tgz
mv ./fedml-server-deployment*.tgz ../../installation/install_on_k8s/fedml-edge-client-server/fedml-server-deployment-latest.tgz