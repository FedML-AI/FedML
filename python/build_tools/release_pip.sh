cd ../
rm -rf ./build
rm -rf ./dist
rm -rf ./fedml.egg-info
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*