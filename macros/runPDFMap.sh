#!/bin/bash
workDir=$1
case=$2
var=$3
meanid=$4

codeDir=${TPCwithDNN}

cd ${workDir}
time python3 ${codeDir}/macros/create_pdf_map.py ${case} ${var} ${meanid}