#!/bin/bash

python_site_dir=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`
echo "python site dir: $python_site_dir"
cp -Rf ./fedml/fedml-pip/* ${python_site_dir}/fedml
