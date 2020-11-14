## Usage
data_loader.py - for distributed computing and standalone simulation

mnist_mobile_preprocessor.py - for IoT/Mobile training. It splits the dataset into small files, so each client only needs to store a small file, which saves the memory cost on edge devices and also largely recudes the loading time.


## Mobile MNIST Data Preprocessing



path: feel_api/data_preprocessing/MNIST/mnist_mobile_preprocessor.py



### Command Line Execution

`python mnist_mobile_preprocessor.py --client_num_per_round 10 --comm_round 10`



### Output File & Structure

Output Directory: MNIST_mobile_zip

If client_num_per_round (worker number) = 2, comm_round = 8:

MNIST_mobile _zip

​							|- 0.zip (for device 0)

​									|-test/test.json  (8 data samples)

​									|-train/train.json (8 data samples)

​							|-1.zip (for device 1)

​									|-test/test.json  (8 data samples)

​									|-train/train.json (8 data samples)

### Client Sampling Example

For each round of sampling (2 workers, 8 rounds):

client_indexes = [993 859], [507 818], [37 726], [642 577], [544 515],[978 22],[778 334]

Then for device 0, the data includes:

["f_00993", "f_00507", "f_00037", "f_00642", "f_00698", "f_00544", "f_00978", "f_00778"]

For device 1, the data includes:

["f_00859", "f_00818", "f_00726", "f_00762", "f_00577", "f_00515", "f_00022", "f_00334"]
