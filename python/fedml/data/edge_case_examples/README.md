## Attack of the Tails: Yes, You Really Can Backdoor Federated Learning
This is for downloading the edge-case examples proposed in the paper "Attack of the Tails: Yes, You Really Can Backdoor Federated Learning " Paper (to Appear in NeurIPS 2020) preprint: [https://arxiv.org/abs/2007.05084]

### Download and Decompress the Dataset
To download and decompress the edge-case examples, you can just run
```
bash .get_data.sh
```

### Model Details on the Edge-case Examples:
- For the CIFAR-10 dataset (Task1 in the paper), we mixed clean labeled CIFAR-10 examples with wrongly labeled Southwest Airline examples (adversarial label ``truck"). The raw examples can be found in `.edge_case_examples/southwest_cifar10/southwest_airlines` where we download the images directly from Google. The first step we did to reprocess the data examples is rescaling them to dimension at `32x32x3` to fit the CIFAR image size. We split the Southwest Airline examples to training (70%) and test (30%) sets and augment each image from 1 copy to 4 copies using rotation at degrees of 90, 180, and 270. The saved datasets for training and test can be found in `southwest_images_new_train.pkl` and `southwest_images_new_test.pkl`
- For the EMNIST dataset (Task2 in the paper), we mixed the clean labeled EMNIST examples with wrongly labeled ``7" examples in the [ARDIS](https://ardisdataset.github.io/ARDIS/) dataset (adversarial label ``1"). To get the ARDIS dataset, one can just run `bash .edge_case_examples/ARDIS/get_ardis_data` and then run `python generating_poisoned_DA.py` (which will generate `poisoned_dataset_fraction_0.1` and `ardis_test_dataset.pt`).
- For the ImageNet dataset (Task3 in the paper), we mixed the clean labeled ImageNet examples with wrongly labeled ``People in traditional Cretan costumes". The raw examples can be found in `.edge_case_examples/cretan_costume_imagenet/greek_examples` where we download the images directly from Google. Simiar to what we did for the Southwest Airline examples, we rescale the ``People in traditional Cretan costumes" examples to `224x224x3` to fit the ImageNet example size. We split the edge case examples into training and set sets and augment each example to 4 examples using the same approach as the aforementioned Southwest Airline examples. The augmented examples can be found in `.edge_case_examples/cretan_costume_imagenet/train` and `.edge_case_examples/cretan_costume_imagenet/test`.
