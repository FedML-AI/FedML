#!/bin/bash

ORGANIZATION=$1
ACCESS_TOKEN=$2

echo $ORGANIZATION
echo $ACCESS_TOKEN

cd /home/fedml/actions-runner

RUNNER_ALLOW_RUNASROOT="1" ./config.sh --url https://github.com/${ORGANIZATION} --token ${ACCESS_TOKEN}

cleanup() {
    echo "Removing runner..."
    RUNNER_ALLOW_RUNASROOT="1" ./config.sh remove --unattended --token ${ACCESS_TOKEN}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

RUNNER_ALLOW_RUNASROOT="1" ./run.sh & wait $!