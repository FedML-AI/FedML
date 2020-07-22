# OnDeviceTraining

This is the src of the android project we used to measure the on-device training time of pre-defined DNN.

Note that this project is developed based on the [DL4JIrisClassifierDemo](https://github.com/eclipse/deeplearning4j-examples/tree/master/android/DL4JIrisClassifierDemo), you may need to refer to [http://deeplearning4j.org/](http://deeplearning4j.org/) for more information.



## How to run

The project is built using Android Studio, so the simplest way to run it is importing it to Android Studio and re-building it.



## Guide

- modify the function in `onClick` to measure the on-device trianing time of pre-defined DNN, e.g. `olaf_reddit();`
- modify the hyper-parameter if needed 
- note that we only measure the training time of one batch, so the optimizer, params, initializer do not matter
- functions we used are `olaf_*` 
- run `initialize(String)` first to avoid cold boot