# TPC distorsion calibration with Deep Networks

![TPC detector](figures/TPC.png)

## Overview of the software:
This software is meant at providing a fast way for performing space-charge (SC) distorsion corrections using deep networks. In particular, the current version uses UNet to train an input dataset made of SC densities and fluctuations and try to predict distorsions along the R, phi and z axis. 


## How to install the software:
- first git clone the following package where you can find a set of existing tools useful for this analysis and a setup configuration for downloading all the needed packages:

```bash
git clone https://github.com/ginnocen/MachineLearningHEP.git
```

```bash
cd MachineLearningHEP
pip3 install -e .
```
