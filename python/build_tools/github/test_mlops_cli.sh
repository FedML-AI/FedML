#!/usr/bin/env bash
# https://doc.fedml.ai/mlops/api.html

test_build_fedml_package() {
  pwd=`pwd`
  cd ../../
#  pip install twine
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

  python3 test_mlops_cli.py -v release

  sleep 5
}

test_python_version_with_new_env() {
    python_venv=$1
    python_ver=$2

    conda_base_dir=`conda info |grep  'base environment' |awk -F':' '{print $2}' |awk -F'(' '{print $1}' |awk -F' ' '{print $1}'`
    conda_env_init="${conda_base_dir}/etc/profile.d/conda.sh"
    source ${conda_env_init}
    conda env remove --name ${python_venv}
    conda create -y -n ${python_venv} python=${python_ver}
    conda activate ${python_venv}
    conda install -y -c anaconda mpi4py

    test_build_fedml_package
    test_mlops_build
    test_mlops_login

    conda deactivate
}

test_logout() {
    python_venv=$1
    python_ver=$2

    conda_base_dir=`conda info |grep  'base environment' |awk -F':' '{print $2}' |awk -F'(' '{print $1}' |awk -F' ' '{print $1}'`
    conda_env_init="${conda_base_dir}/etc/profile.d/conda.sh"
    source ${conda_env_init}
    conda activate ${python_venv}

    fedml logout

    conda deactivate
}

# test action for login and logout
test_action=$1
if [ -z $test_action ]; then
  echo "test mlops cli..."
  test_python_version_with_new_env 'fedml_ci_py_37' '3.7'
elif [ "$test_action" = "logout" ]; then
  echo "test logout..."
  test_logout 'fedml_ci_py_37' '3.7'
  echo "Logout successfully!"
fi
