#!/bin/bash

TRAIN='1000, 5000, 10000, 20000, 25000'
TEST='100, 500, 1000, 2000, 1000'
APPLY='5000, 5000, 5000, 5000, 1000'

# From default database settings
sed -i "s/train_events: \[900\]/train_events: \[${TRAIN}\]/g" database_parameters_DNN_fluctuations.yml
sed -i "s/test_events: \[100\]/test_events: \[${TEST}\]/g" database_parameters_DNN_fluctuations.yml
sed -i "s/apply_events: \[6000\]/apply_events: \[${APPLY}\]/g" database_parameters_DNN_fluctuations.yml

echo "90x17x17, ${TRAIN}:${TEST}:${APPLY}"
time python steer_analysis.py > 90_debug.txt 2>&1

sed -i 's/grid_phi: 90/grid_phi: 180/g' database_parameters_DNN_fluctuations.yml
sed -i 's/grid_z: 17/grid_z: 33/g' database_parameters_DNN_fluctuations.yml
sed -i 's/grid_r: 17/grid_r: 33/g' database_parameters_DNN_fluctuations.yml

echo "180x33x33, ${TRAIN}:${TEST}:${APPLY}"
time python steer_analysis.py > 180_debug.txt 2>&1

#sed -i 's/grid_phi: 180/grid_phi: 90/g' database_parameters_DNN_fluctuations.yml
#sed -i 's/grid_z: 33/grid_z: 17/g' database_parameters_DNN_fluctuations.yml
#sed -i 's/grid_r: 33/grid_r: 17/g' database_parameters_DNN_fluctuations.yml
#sed -i 's/train_events: \[5000, 10000, 18000, 25000\]/train_events: \[10000, 25000\]/g' database_parameters_DNN_fluctuations.yml
#sed -i 's/test_events: \[500, 1000, 1800, 1000\]/test_events: \[1000, 1000\]/g' database_parameters_DNN_fluctuations.yml
#sed -i 's/apply_events: \[5000, 5000, 5000, 1000\]/apply_events: \[5000, 1000\]/g' database_parameters_DNN_fluctuations.yml
