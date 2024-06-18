#!/bin/bash

ORGANIZATION=$1
ACCESS_TOKEN=$2
PYTHON_VERSION=$3

echo $ORGANIZATION
echo $ACCESS_TOKEN
echo $PYTHON_VERSION

cd /home/fedml/actions-runner

RUNNER_ALLOW_RUNASROOT="1" ./config.sh --url https://github.com/${ORGANIZATION} --token ${ACCESS_TOKEN} --labels self-hosted,Linux,X64,$PYTHON_VERSION

cleanup() {
    echo "Removing runner..."
    RUNNER_ALLOW_RUNASROOT="1" ./config.sh remove --unattended --token ${ACCESS_TOKEN}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

RUNNER_ALLOW_RUNASROOT="1" ./run.sh & wait $!