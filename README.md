# TPC distorsion calibration with Deep Networks

![TPC detector](figures/TPC.png)

## Overview of the software:
This software is meant at providing a fast way for performing space-charge (SC) distorsion corrections using deep networks. In particular, the current version uses UNet to train an input dataset made of SC densities and fluctuations and try to predict distorsions along the R, phi and z axis. 


## How to install the software on aliceml:

Create your own virtual environment and load it: 

```bash
ml-create-virtualenv
ml-activate-virtualenv

```

Activate ROOT taken from the machine installation:
```
ml-activate-root
```

Clone the following package where you can find a set of existing tools useful for this analysis and a setup configuration for downloading all the needed packages:

```bash
git clone https://github.com/ginnocen/MachineLearningHEP.git
```

```bash
cd MachineLearningHEP
pip3 install -e .
```

Now download the TPCwithDNN repository:

```
git clone https://github.com/ginnocen/TPCwithDNN.git
```

Move inside the package to get to the executable script:
```
cd TPCwithDNN/tpcwithdnn
```
To run the exercise you can do:
```
python steer_analysis.py
```
You can define which step of the analysis you want to run by configuring the database:
```
vim default.yaml
```
The parameters of the ML analysis can be configured here:

```
vim database_parameters_DNN_fluctuations.yml
```
