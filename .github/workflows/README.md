# 1. Design

![Design](image.png)

##  Design principles

The CI tests need to be comprehensive, covering typical scenarios only, achievable within 5 minutes.

# 2. Registry Self-Host Runners

## 2.1 Linux Runners

### Step1: Build linux images

Build all the linux images for Self-Host Runners.
```
cd registry-runners
bash build_linux_runners.sh
```

### Step2: Specify the token and key.
Find your GitHub runner token and your test-account apikey.

For the argument YourGitHubRunnerToken, Navigate the path `Settings -> Actions -> Runners -> New self-hosted runner` to get.

In the Configure section, you will find the similar line:
./config.sh --url https://github.com/FedML-AI/FedML --token AXRYPL6G2VHVGDFDQQS5XA3ELYI6M to get YourGitHubRunnerToken to value of --token

### Step3: Registry all the runners.
Registry by run `run_linux_runners.sh` script
```
bash run_linux_runners.sh [YourGitRepo] [YourGitHubRunnerToken] [YourTestAccountApiKey]
```
for example
```
bash run_linux_runners.sh FedML-AI/FedML AXRYPLZLZN6XVJB3BAIXSP3EMFC7U 11215dkevvdkegged
```
### Step4: Verify Success

Check if all the runners are registered successfully. Navigate the following path. `Settings -> Actions -> Runners` to check that all your runners are active.

## 2.2 Windows Runners

### Step1: Install Anaconda packages
Install Anaconda or Miniconda on a Windows machine. Anaconda and Miniconda can manage your Python environments.

### Step2: Create python enviroments
Create 4 python environments named `python38`、`python39`、`python310` and `python311` for different runners.
Specify the python version to install.
For example 
```
conda create -n python38 python==3.8
```
### Step3: Create directories 
Create 4 directories named `actions-runner-python38`、`actions-runner-python39`、`actions-runner-python310` and `actions-runner-python311` for different runners.

### Step4: Install the latest runner package. 
Follow the insturction from navigating this path `Settings -> Actions -> Runners -> New self-hosted runner` to add a new Windows runner. Note that you only need to download、extract the files into the directories created in Step 3. Configuration and running will be done through a script later.

### Step5: Registry all the runners.
Run the script from `./registry-runners/windows.ps1` to registry all the runners to your github. Replace the variables `$REPO`、`$ACCESS_TOKEN` and `$WORKPLACE` with actual values. Note that you can get your $ACCESS_TOKEN from the following path `Settings -> Actions -> Runners -> New self-hosted runner.`.
In the Configure section, you will find the similar line: `./config.sh --url https://github.com/FedML-AI/FedML --token AXRYPL6G2VHVGDFDQQS5XA3ELYI6M` to get your `$ACCESS_TOKEN`.

### Step6: Verify Success
Check if the runners are registered successfully by navigate to `Settings -> Actions -> Runners`. Make sure that all your runners are active. 

## 2.3 Mac Runners

# 3. Bind Test Machines

Bind the actual machine to run the test training job. Follow this document to bind your test machines.
https://docs.tensoropera.ai/share-and-earn

Note that we need to bind our machines to the test environment.

Specify the computing resource type to which you have bound your machines. Your job will be scheduled to that machine.

# 4. Trigger

Applying for a PR can trigger all tests automatically.

Run a single test on a specific branch from the GitHub Actions tab.

Schedule daily runs at a specific time by configuring your workflow YAML. You can check the results in the GitHub Actions tab.

# 5. Add a new CI test

Creating a new workflow YAML file, such as CI_launch.yaml or CI_train.yaml, allows you to add a CI test that is different from the current business.

Adding a new CI test to the current business can be done by placing your test in the path python/tests/test_{business}/test_file.py and ensuring that your workflow YAML can run that Python test script.

Ensuring your workflow YAML is configured correctly will enable it to run the new test automatically.

# 6. TODO

Implement the Mac runners.

