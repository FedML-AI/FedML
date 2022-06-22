Local Installation (real-time editable)
```
pip install -e ./
```

Release
```
pip install twine
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*

```