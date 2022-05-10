# Code for evaluating various models
Before running any of the model code, ensure parsed_data_raw.hdf5 exists in ../models. If not, run create_h5_dataset() in ../models/common.py first.

All code for evaluation including statistical methods is in backtest.py.
Run backtest.py to see all metrics and backtesting results for all of the models. Other functions for evaluating the NN model such as plotting ROC curves and precision-recall curves are also included in the file.