Local Installation
```
python setup.py install
```

Release
```
pip install twine
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
```