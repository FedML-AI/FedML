#!/bin/bash

python_site_dir=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`
python_site_dir_pip=`pip3 show fedml |grep Location: |awk -F':' '{print $2}'`

echo "python site dir: $python_site_dir"
echo "python pip dir: $python_site_dir_pip"

exist_python_conda_env=`ls ${python_site_dir} |grep 'No such file'`
if [ -z ${python_site_dir} ]; then
  echo "No conda env"
else
  if [ -z ${exist_python_conda_env} ]; then
    echo "Have conda env"
    cp -Rf ./fedml/fedml-pip/* ${python_site_dir}/fedml > /dev/null 2>&1
  fi
fi

exist_python_pip=`ls ${python_site_dir_pip} |grep 'No such file'`
if [ -z ${python_site_dir_pip} ]; then
  echo "No fedml pip"
else
  if [ -z ${exist_python_pip} ]; then
    echo "Have fedml pip"
    cp -Rf ./fedml/fedml-pip/* ${python_site_dir_pip}/fedml > /dev/null 2>&1
  fi
fi

