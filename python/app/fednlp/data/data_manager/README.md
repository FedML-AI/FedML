# Data Manager

What DataManager does is to control the whole workflow from loading data to returning trainnable features. To be specific, DataManager is set up for reading h5py data files and driving preprocessor to convert raw data to features. Besides, DataManager is splitted into four classes according to the task definition. Users can customize their own DataManager by inheriting one of the DataManager classes, specifying data operation functions and embedding the particular preprocessor. The workflow is illustrated as flow.

![avatar](./data_manager_workflow.png)