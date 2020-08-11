#!/bin/bash
rm performance.list
touch performance.list
for ievent in {1000,5000,10000,20000}; do
  echo "#Title:model.nEv${ievent}" >> performance.list
  echo "$(pwd)/validation/phi180_r33_z33_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1_dophi0_doz0_Nev${ievent}/outputPDFMaps.root" >> performance.list
done
