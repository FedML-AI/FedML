## 1. test cases for MLOps CLI
When pushing or pulling requests to the FedML repository. The following GitHub action
will be triggered.
1). fedml-build: copy fedml latest codes into the local directory for fedml pip package.
2). mlops-build: build MLOps client and server package.
3). login: call `fedml login` command to login the MLOps platform to wait for MLOps request,
e.g. start run or stop run.
4). start run: simulate sending a start run request to the local client agent.
5). stop run: simulate sending a stop run request to the local client agent.

##