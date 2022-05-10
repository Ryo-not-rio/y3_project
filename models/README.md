# Code for the baseline model, proposed model and a random model as well as some cached datasets and a saved model
- ./model.py - contains the code for the proposed model
- ./baseline.py - contains code for the baseline model

## Instructions for training and testing each model
Before running any of the model code, ensure parsed_data_raw.hdf5 exists in this directory. If not, run create_h5_dataset() in common.py first.

There are various cache files also contained in this directory to speed up testing and training. It is recommended not to remove them.
To train and test each model, run each file.

For hyperparameter optimization code, refer to ../training.ipynb