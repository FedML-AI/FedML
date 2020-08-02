# FebMLTrainingExecutor
> EMQ X,Docker,Flask,Gunicorn

## 1 Start EMQX (MacOS)
### 1.1 Install EMQX
```bash
# Add EMQ X tap
brew tap emqx/emqx
# Install EMQ X Broker
brew install emqx
# 
```
### 1.2 Start EMQ X Broker
```bash
# start emqx
emqx start
# get emqx status
emqx_ctl status
```
### 1.3 Drop EMQ X
```bash
# Stop EMQ X Broker
emqx stop
# Uninstall EMQ X Broker
brew uninstall emqx
```