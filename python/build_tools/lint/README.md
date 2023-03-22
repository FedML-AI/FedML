# Usage for local test
```
pip install pylint

# pylint --rcfile=.pylintrc --disable=C,R,W ../../fedml/core > pylint_log.txt
# remember to change the code path to your own
bash pylint.sh
```

# Usage on GitHub Action

1. install all related packages

```
pip install "fedml[MPI]"
pip install "fedml[gRPC]"
pip install "fedml[tensorflow]"
pip install "fedml[jax]"
pip install "fedml[mxnet]"
```

2. run pylint
```
pip install pylint
cd FedML/python/build_tools/lint/
bash pylint.sh
```
