#!/bin/bash
## source ${TPCwithDNN}/notebooks/makePDFMapsLists.sh

makePDFMapsList()
{
  listName=pdfmaps.list
  rm ${listName}
  touch ${listName}
  for ievent in {1000,5000,10000,20000}; do
    echo "#Title:model.nEv${ievent}" >> ${listName}
    echo "$(pwd)/validation/phi180_r33_z33_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1_dophi0_doz0_Nev${ievent}/outputPDFMaps.root" >> ${listName}
  done
}

makePDFMapsListGPUbenchmark()
{
  listName=gpu_benchmarks/pdfmapsGPUbenchmark.list
  rm ${listName}
  touch ${listName}
  for ievent in {1000,5000,10000}; do
    echo "#Title:ROCM16.nEv${ievent}" >> ${listName}
    echo "$(pwd)/ROCM/SC-33-33-180/trees/phi180_r33_z33_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1_dophi0_doz0/pdfmaps_nEv${ievent}.root" >> ${listName}
    echo "#Title:CUDA16.nEv${ievent}" >> ${listName}
    echo "$(pwd)/CUDA/SC-33-33-180_memGPU16GB/trees/phi180_r33_z33_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1_dophi0_doz0/pdfmaps_nEv${ievent}.root" >> ${listName}
    echo "#Title:CUDA32.nEv${ievent}" >> ${listName}
    echo "$(pwd)/CUDA/SC-33-33-180/trees/phi180_r33_z33_filter4_poo0_drop0.00_depth4_batch0_scaler0_useSCMean1_useSCFluc1_pred_doR1_dophi0_doz0/pdfmaps_nEv${ievent}.root" >> ${listName}
  done
}