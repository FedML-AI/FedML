### Local Installation (real-time editable)
```
pip install -e ./
```

### Local Installation-Extra

https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-setuptools-extras

```
pip install -e '.[MPI]'
pip install -e '.[gRPC]'

```


### Release
```
pip install twine
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*

```