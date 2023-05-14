# Run self-host runner in your machine

## Usage

./runner-start.sh [YourGitRepo] [YourRunnerPrefix] [YourRunnerNum] [YourGitHubRunnerToken] [LocalDevSourceDir] [LocalReleaseSourceDir] [LocalDataDir]

For the argument YourGitHubRunnerToken, you may navigate based the following path.

Settings -> Actions -> Runners -> New self-hosted runner. 

In the Configure section, you should find the similar line:
./config.sh --url https://github.com/FedML-AI/FedML --token AXRYPL6G2VHVGDFDQQS5XA3ELYI6M

set YourGitHubRunnerToken to value of --token


## Example

Use the following commands to run 30 runners in the FedML-AI/FedML repo and run 6 runners in the FedML-AI/Front-End-Auto-Test repo:

./runner-start.sh FedML-AI/FedML fedml-runner 30 AXRYPLY7MQNAC23EZFEKQL3EL5BTW /home/fedml/FedML4GitHubAction-Dev /home/fedml/FedML4GitHubAction /home/fedml/fedml_data
./runner-start.sh FedML-AI/Front-End-Auto-Test webtest-runner 6 AXRYPL3TJFYUZ5M3TMO4GCLEL5CS4 /home/fedml/FedML4GitHubAction-Dev /home/fedml/FedML4GitHubAction /home/fedml/fedml_data

./runner-start.sh FedML-AI/FedML fedml-runner 20 AXRYPLY7MQNAC23EZFEKQL3EL5BTW /home/chaoyanghe/sourcecode/FedML4GitHubAction-Dev /home/chaoyanghe/sourcecode/FedML4GitHubAction /home/chaoyanghe/fedml_data
