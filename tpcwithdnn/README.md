# Overview over most important files and classes
## Config files
`default.yml`
- switches on / off different steps of the program, e.g. training or creation of validation data

`config_model_parameters.yml`
- Collection of running and model parameters
    - `common`: general parameters for training, validation, input data, etc.
    - `xgboost`: parameters for 1D training and validation, 1D ML models
        - `params`: parameters for RF, XGB models
        - `nn_params`: parameters for dense NN
    - `dnn`: parameters for 3D CNN model and training


&nbsp;

## Main file
`steer_analysis.py`
- Main script to
    - parse parameters from config files or command line
    - run different parts of the code accroding to parsed parameters
    - examples how to run the code are given in the [wiki](https://github.com/AliceO2Group/TPCwithDNN/wiki/Running-the-correction)


&nbsp;

## General files
`common_settings.py`
- Classes to store common, xgboost and dnn parameters
  - Data members in models classes (optimisers) and validator classes to access parameters

`data_loader.py`
- Obtain input data for 1D, 3D models
- Input data (e.g. corrections, position, IDCs, density, ...) are stored in single numpy arrays
  - One numpy array per variable per random / mean map
  - Procedure of input data creation is given in the [wiki](https://github.com/AliceO2Group/TPCwithDNN/wiki/Creating-input-data-from-simulation)
- 1D input array layout: r, phi, z, derivatives (ddr, ddrphi, ddz), FFT coefficients (real, imag)
- 3D input layout: nd numpy array with dimensions (nphi, nr, nz, 2)
  - Last dimension is number of inputs:
    - 1) Fluctuation corrections
    - 2) Space-charge density (supposed to be derivative of avg. corrections w.r.t. FFT c0 in data)

`logger.py`
- Basic logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)

`debug_utils.py`
- Logging utility for processing time and memory usage

`tree_df_utils.py`
- Methods to convert TTree (in root file) <-> pandas (both directions)

`hadd.py`
- Python implementation of root hadd
- used for creation of cached downsampled input data in `xgboost_optimiser.py`
- Much slower than native root hadd

`optimiser.py`
- Base optimiser class with empty methods
- Base class for `xgboost_optimiser.py` and `dnn_optimiser.py`


&nbsp;

## Files related to 1D fluctuations, 1D IDCs, RF, XGB, NN
`nn_utils.py`
- Methods to set up simple dense neural networks

`xgboost_optimiser.py`
- Main class for training of 1D models (RF, XGB, simple NN), model storage and loading
- Methods to create cache of downsampled input data
  - Cached downsampled data provides faster loading of downsampled input data compared to loading full data of each map and downsampling afterwards

`idc_data_validator.py`
- Create validation data to evaluate trained model performance
  - Use combination of random-mean maps
  - Process all points of each map
  - Dump full information + predicted fluctuation correction
    - Output is split in single files per map, to be merged with `merge_validation_trees.sh`
  - Validation output can be loaded, evaluated and histogrammed interactively with jupyter notebooks in `TPCwithDNN/notebooks/`
- Methods to create nd histograms and pdf maps from validation data offline (non-interactively)
  - Old workflow, now replaced by interactive histogramming in jupyter notebooks

`merge_validation_trees.sh`
- Script to merge split output of `idc_data_validator.py`, using root hadd


&nbsp;

## Files related to 3D fluctuations, 3D IDCs (3D space-charge denisty), CNN
`fluctuation_data_generator.py`
- Class to load input data for training and prediction, using methods from `data_loader.py`

`symmetry_padding_3d.py`
- Custom Keras layer implementing symmetric padding in 3D

`dnn_utils.py`
- Methods to construct a U-Net with variable parameters like number of layers, type of pooling, etc.

`dnn_optimiser.py`
- Train, save and load U-Net models

`data_validator.py`
- Create validation data (true and predicted corrections as function of point, density fluctuation, ...) to evaluate U-Net model performance
- Methods to create nd histograms and pdf maps from validation data offline (non-interactively)
