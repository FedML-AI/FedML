#!/bin/bash

FEDML_HELM_CHARTS_BASE_DIR=$1
FEDML_MODEL_PREMISE_PACKAGE=$2
FEDML_MODEL_OPS_VERSION=$3
DEPLOY_NAMESPACE=$4
FEDML_MODEL_OPS_ACCOUNT_ID=$5

instance_count=`helm list --filter $FEDML_MODEL_PREMISE_PACKAGE -n $DEPLOY_NAMESPACE |wc -l`
if [ $instance_count -gt 1 ]; then
  echo "upgrading..."
  echo `pwd`
  rm -f *.tgz
  helm package $FEDML_HELM_CHARTS_BASE_DIR/$FEDML_MODEL_PREMISE_PACKAGE --app-version $FEDML_MODEL_OPS_VERSION
  helm upgrade -n $DEPLOY_NAMESPACE --set env.fedmlAccountId="$FEDML_MODEL_OPS_ACCOUNT_ID" --set env.fedmlVersion=$FEDML_MODEL_OPS_VERSION $FEDML_MODEL_PREMISE_PACKAGE ./$FEDML_MODEL_PREMISE_PACKAGE-*.tgz
else
  echo "installing..."
  rm -f *.tgz
  helm package $FEDML_HELM_CHARTS_BASE_DIR/$FEDML_MODEL_PREMISE_PACKAGE --app-version $FEDML_MODEL_OPS_VERSION
  helm install -n $DEPLOY_NAMESPACE --set env.fedmlAccountId="$FEDML_MODEL_OPS_ACCOUNT_ID" --set env.fedmlVersion=$FEDML_MODEL_OPS_VERSION $FEDML_MODEL_PREMISE_PACKAGE ./$FEDML_MODEL_PREMISE_PACKAGE-*.tgz
fi