# TPC distorsion calibration with Deep Networks

![TPC detector](figures/TPC.png)

## Overview of the software:
This software is meant at providing a fast way for performing space-charge (SC) distorsion corrections using deep networks. In particular, the current version uses UNet to train an input dataset made of SC densities and fluctuations and try to predict distorsions along the R, phi and z axis. 

Authors: M. Ivanov (marian.ivanov@cern.ch), G.M. Innocenti (ginnocen@cern.ch), Rifki Sadikin (rifki.sadikin@cern.ch), D. Sekihata (daiki.sekihata@cern.ch)

The original version of the code was developed by M. Ivanov and R. Sadikin and can be found here https://gitlab.cern.ch/alice-tpc-offline/alice-tpc-notes/-/tree/master/JIRA%2FATO-439%2Fcode%2Fpython

Please find detailed instruction about the analysis and the software package here https://github.com/AliceO2Group/TPCwithDNN/wiki
