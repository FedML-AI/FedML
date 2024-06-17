# Run self-host runner in your machine

## Usage

### build images
bash build_batch.sh

### run
bash run.sh [YourGitRepo] [YourGitHubRunnerToken]

For the argument YourGitHubRunnerToken, you may navigate based the following path.

Settings -> Actions -> Runners -> New self-hosted runner. 

In the Configure section, you should find the similar line:
./config.sh --url https://github.com/FedML-AI/FedML --token AXRYPL6G2VHVGDFDQQS5XA3ELYI6M

set YourGitHubRunnerToken to value of --token

## Example
Use the following commands to run 4 runners in the FedML-AI/FedML repo:

bash main.sh FedML-AI/FedML AXRYPLZLZN6XVJB3BAIXSP3EMFC7U

bash main.sh Qigemingziba/FedML AGMK3PYAURK7QSRM475HF6LGN7L6A
