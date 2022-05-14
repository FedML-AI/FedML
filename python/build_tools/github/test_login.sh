# https://doc.fedml.ai/mlops/api.html

test_build_fedml_package() {
  pip install twine
  pwd=`pwd`
  cd ../../
#  python3 setup.py sdist bdist_wheel
#  twine check ./dist/*
  pip uninstall fedml
  pip install fedml
  python_site_dir=`python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'`
  echo "python site dir: $python_site_dir"
  cp -Rf ./fedml/* ${python_site_dir}/fedml
  cd ${pwd}

  fedml_version=`fedml version |grep "fedml version:"`
  if [ -z "${fedml_version}" ]; then
    echo "Failed to build fedml pip package!"
  else
    echo "Succeeded to build fedml pip package!"
  fi
}

test_mlops_build() {
  client_source_dir=../../examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/client
  client_entry_file=torch_client.py
  server_source_dir=../../examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/server
  server_entry_file=torch_server.py
  config_dir=../../examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/config
  dest_pkg_dir=/tmp/fedml_test

  mkdir -p ${dest_pkg_dir}

  fedml build -t client -sf ${client_source_dir} -ep ${client_entry_file} -cf ${config_dir} -df ${dest_pkg_dir}
  client_pkg_exist=`ls ${dest_pkg_dir}/dist-packages/client-package.zip`
  if [ -z ${client_pkg_exist} ]; then
    echo "Failed to build FedML client package!"
  else
    echo "Succeeded to build FedML client package!"
  fi

  fedml build -t server -sf ${server_source_dir} -ep ${server_entry_file} -cf ${config_dir} -df ${dest_pkg_dir}
  server_pkg_exist=`ls ${dest_pkg_dir}/dist-packages/client-package.zip`
  if [ -z "${server_pkg_exist}" ]; then
    echo "Failed to build FedML server package!"
  else
    echo "Succeeded to build FedML server package!"
  fi
}

test_mlops_login() {
  fedml login 105

  sleep 20

  python3 test_login.py -v release

  sleep 20

  fedml logout
}

test_python_version_with_new_env() {
    python_venv=$1
    python_ver=$2

#    python3 -m venv --clear ${python_venv}
#    python3 -m venv  ${python_venv}
#    source ${python_venv}/bin/activate

    source ~/anaconda3/etc/profile.d/conda.sh
    conda env remove --name ${python_venv}
    conda create -y -n ${python_venv} python=${python_ver}
    conda activate ${python_venv}
    conda install -y -c anaconda mpi4py

    test_build_fedml_package
    #test_mlops_build
    #test_mlops_login

#    source ${python_venv}/bin/deactivate
    conda deactivate
}

test_python_version_with_new_env 'fedml_ci_py_37' '3.7'
