
WORKSPACE=`pwd`
echo $WORKSPACE
cd $WORKSPACE/examples/federate/quick_start/parrot
python torch_fedavg_mnist_lr_one_line_example.py --cf fedml_config.yaml
python torch_fedavg_mnist_lr_custum_data_and_model_example.py --cf fedml_config.yaml

cd $WORKSPACE/examples/federate/simulation/sp_decentralized_mnist_lr_example
python torch_fedavg_mnist_lr_step_by_step_example.py --cf fedml_config.yaml

cd $WORKSPACE/examples/federate/simulation/sp_fednova_mnist_lr_example
python torch_fednova_mnist_lr_step_by_step_example.py --cf fedml_config.yaml
          
cd $WORKSPACE/examples/federate/simulation/sp_fedopt_mnist_lr_example
python torch_fedopt_mnist_lr_step_by_step_example.py --cf fedml_config.yaml

cd $WORKSPACE/examples/federate/simulation/sp_hierarchicalfl_mnist_lr_example
python torch_hierarchicalfl_mnist_lr_step_by_step_example.py --cf fedml_config.yaml


cd $WORKSPACE/examples/federate/simulation/sp_turboaggregate_mnist_lr_example
python torch_turboaggregate_mnist_lr_step_by_step_example.py --cf fedml_config.yaml 


cd $WORKSPACE/examples/federate/simulation/sp_vertical_mnist_lr_example
python torch_vertical_mnist_lr_step_by_step_example.py --cf fedml_config.yaml 
