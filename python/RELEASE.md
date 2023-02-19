### Local Installation (real-time editable)
```
pip install -e ./


```

### Local Installation-Extra

https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-setuptools-extras

```
# TensorFlow extra installation
pip install -e '.[tensorflow]'

# JAX extra installation
pip install -e '.[jax]'

# MXNet extra installation
pip install -e '.[mxnet]'

# for different communication backends
pip install -e '.[MPI]'
pip install -e '.[gRPC]'

```

### Test
```
pip install twine
python setup.py sdist bdist_wheel
twine check dist/*
twine upload --repository testpypi dist/*
```

### Install test pip
```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fedml
```

### Upgrade test pip
```
pip install -U --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fedml
```

### Release
```
pip install twine
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
```