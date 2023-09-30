# Install FedML and Prepare the Distributed Environment
```
pip install fedml
```


# Run the example

## mpi hierarchical fl
```
sh run_step_by_step_example.sh 5 config/mnist_lr/fedml_config.yaml
```

## mpi hierarchical fl based on some topology (e.g., 2d_torus, star, complete, isolated, balanced_tree and random)
```
sh run_step_by_step_example.sh 5 config/mnist_lr/fedml_config.yaml
```

