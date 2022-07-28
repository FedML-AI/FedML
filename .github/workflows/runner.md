ssh -i amir-github-actions-key.cer ubuntu@54.183.200.162
ssh -i amir-github-actions-key.cer ubuntu@52.53.164.162
ssh -i github_actions.cer ubuntu@54.153.18.24

https://github.com/FedML-AI/FedML/settings/actions/runners/new




mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.293.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.293.0/actions-runner-linux-x64-2.293.0.tar.gz
echo "06d62d551b686239a47d73e99a557d87e0e4fa62bdddcf1d74d4e6b2521f8c10  actions-runner-linux-x64-2.293.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.293.0.tar.gz
./config.sh --url https://github.com/FedML-AI/FedML --token AQLQ4JAGL4KICTWDZSXVXQDCV7GSO
nohup bash run.sh > actions.log 2>&1 &