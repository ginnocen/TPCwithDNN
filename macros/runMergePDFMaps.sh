#!/bin/bash
workDir=$1
case=$2

codeDir=${TPCwithDNN}

cd ${workDir}
time python3 ${codeDir}/macros/merge_pdf_maps.py ${case}