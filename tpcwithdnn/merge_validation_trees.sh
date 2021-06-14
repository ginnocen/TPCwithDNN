#!/bin/bash
[[ $# -lt 1 ]] && cat <<HELP_USAGE
    merge_validation_trees.sh - merge input or validation trees created using "docreatevaldata true" in default.yml (function create_data() in tpcwithdnn/data_validator.py or tpcwithdnn/idc_data_validator.py)

    Input:
        Param 1: inputDir - path to output trees, usually <base_dir>/trees (input data trees) or <base_dir>/trees/<model_parameters> (validation data trees)
        Param 2 optional: nTrainEvents - number of training events used for model of validation tree
    Example usage:
         ./merge_validation_trees.sh trees
         ./merge_validation_trees.sh trees 5000
HELP_USAGE
[[ $# -lt 1 ]] && exit

inputDir=$1

# case for second argument (nTrainEvents) provided
if [ $# -eq 2 ]; then
  nTrainEvents=$2
  [[ -z $(ls $(ls -d ${inputDir}/parts/* | head -n 1) | grep "_nEv${nTrainEvents}.root") ]] && echo "No files exist for number of training events specified in arguments!" && exit
  for ifile in $(ls $(ls -d ${inputDir}/parts/* | head -n 1) | grep nEv${nTrainEvents}); do
    hadd -f ${inputDir}/${ifile} $(ls ${inputDir}/parts/*/${ifile})
  done
  exit
fi

# case for one argument (inputDir) provided
for ifile in $(ls $(ls -d ${inputDir}/parts/* | head -n 1)); do
  hadd -f ${inputDir}/${ifile} $(ls ${inputDir}/parts/*/${ifile})
done
