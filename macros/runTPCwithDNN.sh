#!/bin/bash
workDir=$1
steeringFile=$2

codeDir=${TPCwithDNN}

cd ${workDir}
time python3 ${codeDir}/tpcwithdnn/steer_analysis.py ${steeringFile}