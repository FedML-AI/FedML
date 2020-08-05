# FebMLTrainingExecutor
> EMQ X,Docker,Flask,Gunicorn

# 1 Start EMQX (MacOS)
## 1.1 Install EMQX
```bash
# Add EMQ X tap
brew tap emqx/emqx
# Install EMQ X Broker
brew install emqx
# 
```
## 1.2 Start EMQ X Broker
```bash
# start emqx
emqx start
# get emqx status
emqx_ctl status
```
## 1.3 Drop EMQ X
```bash
# Stop EMQ X Broker
emqx stop
# Uninstall EMQ X Broker
brew uninstall emqx
```
# 2 Quick Start FebMLTrainingExecutor
## 2.1 install dependencies
```bash
pip install -r requirements.txt
```
gunicorn maybe need add to PATH:
```bash
echo 'export PATH="/{your machine path}}/Python/3.7/bin:$PATH"' >> ~/.bash_profile
```
## 2.2 start FebMLTrainingExecutor
```bash
# start TrainingExecutor server
sh svr_ctrl.sh start
# stop TrainingExecutor server
sh svr_ctrl.sh stop
# restart TrainingExecutor server
sh svr_ctrl.sh restart
# query the server status
sh svr_ctrl.sh status
```