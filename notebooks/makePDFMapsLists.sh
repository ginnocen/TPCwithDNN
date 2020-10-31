#!/bin/bash
## source ${TPCwithDNN}/notebooks/makePDFMapsLists.sh

DATA_MAIN_DIR="/home/mkabus/TPCwithDNN/tpcwithdnn"

makeValTreesList()
{
  listName=valtrees.list
  rm -f ${listName}
  touch ${listName}
  for ievent in {500,1000,2000,5000}; do
    for mean in 0.9 1.0 1.1 ; do
        echo "#Title:model.nEv${ievent}.mean${mean}" >> ${listName}
        echo "${DATA_MAIN_DIR}/trees/phi90_r17_z17_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1_dophi0_doz0/treeValidation_mean${mean}_nEv${ievent}.root" >> ${listName}
    done
  done
}

makePDFMapsList()
{
  listName=pdfmaps.list
  rm -f ${listName}
  touch ${listName}
  for ievent in {500,1000,2000,5000}; do
    echo "#Title:model.nEv${ievent}" >> ${listName}
    echo "${DATA_MAIN_DIR}/trees/phi90_r17_z17_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1_dophi0_doz0/pdfmaps_nEv${ievent}.root" >> ${listName}
  done
}

makePDFMapsListGPUbenchmark()
{
  listName=gpu_benchmarks/pdfmapsGPUbenchmark.list
  rm ${listName}
  touch ${listName}
  for ievent in {1000,5000,10000}; do
    echo "#Title:ROCM16.nEv${ievent}" >> ${listName}
    echo "$(pwd)//trees/phi180_r33_z33_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1_dophi0_doz0/pdfmaps_nEv${ievent}.root" >> ${listName}
    echo "#Title:CUDA16.nEv${ievent}" >> ${listName}
    echo "$(pwd)/CUDA/SC-33-33-180_memGPU16GB/trees/phi180_r33_z33_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1_dophi0_doz0/pdfmaps_nEv${ievent}.root" >> ${listName}
    echo "#Title:CUDA32.nEv${ievent}" >> ${listName}
    echo "$(pwd)/CUDA/SC-33-33-180/trees/phi180_r33_z33_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1_dophi0_doz0/pdfmaps_nEv${ievent}.root" >> ${listName}
  done
}
