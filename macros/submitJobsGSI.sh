#!/bin/bash
## source ${TPCwithDNN}/macros/submitJobsGSI.sh

submitTrainingCUDA()
{
  ## go to working directory first before submitting
  workDir=$(pwd)
  codeDir=${TPCwithDNN}
  batchCommand="sbatch --get-user-env --mem-per-cpu=200000 --time=08:00:00 --reservation=hpc_10 --nodelist=lxbk0717 --cpus-per-task=63"

  if [ ! -d logs ]; then
    mkdir logs
  fi

  echo '#!/bin/bash' >scdcalib_train.sh
  echo "nEvTrain=\$((\${SLURM_ARRAY_TASK_ID}*1000))" >>scdcalib_train.sh
  echo "nEvTest=\$((\${nEvTrain}/5))" >>scdcalib_train.sh
  echo "createSteeringXMLFile DNN_fluctuations_nEv\${nEvTrain} train_nEv\${nEvTrain}.yml true false false false false false false false false" >>scdcalib_train.sh
  echo "createParamXMLFile DNN_fluctuations_nEv\${nEvTrain} 180 33 33 \${nEvTrain} \${nEvTest} 3000 CUDA" >>scdcalib_train.sh
  echo "singularity exec --nv /lustre/alice/users/hellbaer/NOTESData/JIRA/ATO-500/SCDcalibML_CUDA.sif ${codeDir}/macros/runTPCwithDNN.sh ${workDir} train_nEv\${nEvTrain}.yml" >>scdcalib_train.sh
  ${batchCommand} --array=1,5,10 -o logs/train_nEv%a000.o -e logs/train_nEv%a000.e scdcalib_train.sh
  rm scdcalib_train.sh
}

submitTrainingCUDA16GB()
{
  ## go to working directory first before submitting
  workDir=$(pwd)
  codeDir=${TPCwithDNN}
  batchCommand="sbatch --get-user-env --mem-per-cpu=200000 --time=08:00:00 --reservation=hpc_10 --nodelist=lxbk0717 --cpus-per-task=63"

  if [ ! -d logs ]; then
    mkdir logs
  fi

  echo '#!/bin/bash' >scdcalib_train.sh
  echo "nEvTrain=\$((\${SLURM_ARRAY_TASK_ID}*1000))" >>scdcalib_train.sh
  echo "nEvTest=\$((\${nEvTrain}/5))" >>scdcalib_train.sh
  echo "createSteeringXMLFile DNN_fluctuations_nEv\${nEvTrain} train_nEv\${nEvTrain}.yml true false false false false false false false false" >>scdcalib_train.sh
  echo "createParamXMLFile DNN_fluctuations_nEv\${nEvTrain} 180 33 33 \${nEvTrain} \${nEvTest} 3000 CUDA" >>scdcalib_train.sh
  echo "sed -i 's/SC\-33\-33\-180/SC\-33\-33\-180\_memGPU16GB/g' database_parameters_DNN_fluctuations_nEv\${nEvTrain}.yml" >>scdcalib_train.sh
  echo "export TPCwithDNNSETMEMLIMIT=16384" >>scdcalib_train.sh
  echo "singularity exec --nv /lustre/alice/users/hellbaer/NOTESData/JIRA/ATO-500/SCDcalibML_CUDA.sif ${codeDir}/macros/runTPCwithDNN.sh ${workDir} train_nEv\${nEvTrain}.yml" >>scdcalib_train.sh
  ${batchCommand} --array=1,5,10 -o logs/train_nEv%a000.o -e logs/train_nEv%a000.e scdcalib_train.sh
  rm scdcalib_train.sh
}

submitTrainingROCM()
{
  ## go to working directory first before submitting
  workDir=$(pwd)
  codeDir=${TPCwithDNN}
  batchCommand="sbatch --get-user-env --mem-per-cpu=200000 --time=08:00:00 --reservation=hpc_10 --nodelist=lxbk0721 --cpus-per-task=63"

  if [ ! -d logs ]; then
    mkdir logs
  fi

  echo '#!/bin/bash' >scdcalib_train.sh
  echo "nEvTrain=\$((\${SLURM_ARRAY_TASK_ID}*1000))" >>scdcalib_train.sh
  echo "nEvTest=\$((\${nEvTrain}/5))" >>scdcalib_train.sh
  echo "createSteeringXMLFile DNN_fluctuations_nEv\${nEvTrain} train_nEv\${nEvTrain}.yml true false false false false false false false false" >>scdcalib_train.sh
  echo "createParamXMLFile DNN_fluctuations_nEv\${nEvTrain} 180 33 33 \${nEvTrain} \${nEvTest} 3000 ROCM" >>scdcalib_train.sh
  echo "singularity exec /lustre/alice/users/hellbaer/NOTESData/JIRA/ATO-500/SCDcalibML_ROCM.sif ${codeDir}/macros/runTPCwithDNN.sh ${workDir} train_nEv\${nEvTrain}.yml" >>scdcalib_train.sh
  ${batchCommand} --array=1,5,10 -o logs/train_nEv%a000.o -e logs/train_nEv%a000.e scdcalib_train.sh
  rm scdcalib_train.sh
}

submitValData()
{
  ## go to working directory first before submitting
  workDir=$(pwd)
  codeDir=${TPCwithDNN}
  batchCommand="sbatch --get-user-env --mem-per-cpu=50000 --time=08:00:00"

  if [ ! -d logs ]; then
    mkdir logs
  fi

  submitFile=scdcalib_valData.sh
  jobPrefix=valData

  echo '#!/bin/bash' >${submitFile}
  echo "nEvTrain=\$((\${SLURM_ARRAY_TASK_ID}*1000))" >>${submitFile}
  echo "createSteeringXMLFile DNN_fluctuations_nEv\${nEvTrain} ${jobPrefix}_nEv\${nEvTrain}.yml false false false false false true false false false" >>${submitFile}
  echo "singularity exec /lustre/alice/users/hellbaer/NOTESData/JIRA/ATO-500/SCDcalibML_ROCM.sif ${codeDir}/macros/runTPCwithDNN.sh ${workDir} ${jobPrefix}_nEv\${nEvTrain}.yml" >>${submitFile}
  ${batchCommand} --array=1,5,10 -o logs/${jobPrefix}_nEv%a000.o -e logs/${jobPrefix}_nEv%a000.e ${submitFile}
  rm ${submitFile}
}

submitPDFMaps()
{
  ## go to working directory first before submitting
  workDir=$(pwd)
  codeDir=${TPCwithDNN}
  batchCommand="sbatch --get-user-env --mem-per-cpu=50000 --time=08:00:00 --ntasks=1"

  if [ ! -d logs ]; then
    mkdir logs
  fi

  submitFile=scdcalib_pdfmaps.sh
  jobPrefix=pdfmaps

  pdfMapList=pdfMaps.list
  createPDFMapList ${pdfMapList}
  numberMaps=$(wc -l ${pdfMapList} | awk '{print $1}')

  echo '#!/bin/bash' >${submitFile}
  echo "imap=\$(cat ${pdfMapList} | sed -n \${SLURM_ARRAY_TASK_ID}p)" >>${submitFile}
  echo "nEvTrain=\$(echo \${imap} | awk -F' ' '{print \$1}')" >>${submitFile}
  echo "nEvTest=\$((\${nEvTrain}/5))" >>${submitFile}
  echo "var=\$(echo \${imap} | awk -F' ' '{print \$2}')" >>${submitFile}
  echo "meanid=\$(echo \${imap} | awk -F' ' '{print \$3}')" >>${submitFile}
  echo "createSteeringXMLFile DNN_fluctuations_nEv\${nEvTrain}_var\${var}_meanid\${meanid} ${jobPrefix}_nEv\${nEvTrain}_var\${var}_meanid\${meanid}.yml false false false false false false false true false" >>${submitFile}
  echo "createParamXMLFile DNN_fluctuations_nEv\${nEvTrain}_var\${var}_meanid\${meanid} 180 33 33 \${nEvTrain} \${nEvTest} 3000 ROCM" >>${submitFile}
  echo "sed -i \"s/pdf\\_map\\_var\\:\\ flucSC/pdf\\_map\\_var\\:\\ \${var}/g\" database_parameters_DNN_fluctuations_nEv\${nEvTrain}_var\${var}_meanid\${meanid}.yml" >>${submitFile}
  echo "sed -i \"s/pdf\\_map\\mean\\_id\\:\\ 0/pdf\\_map\\mean\\_id\\:\\ \${meanid}/g\" database_parameters_DNN_fluctuations_nEv\${nEvTrain}_var\${var}_meanid\${meanid}.yml" >>${submitFile}
  echo "singularity exec /lustre/alice/users/hellbaer/NOTESData/JIRA/ATO-500/SCDcalibML_ROCM.sif ${codeDir}/macros/runTPCwithDNN.sh ${workDir} ${jobPrefix}_nEv\${nEvTrain}_var\${var}_meanid\${meanid}.yml" >>${submitFile}
  echo "rm ${jobPrefix}_nEv\${nEvTrain}_var\${var}_meanid\${meanid}.yml database_parameters_DNN_fluctuations_nEv\${nEvTrain}_var\${var}_meanid\${meanid}.yml" >>${submitFile}
  ${batchCommand} --array=1-${numberMaps} -o logs/${jobPrefix}_%a.o -e logs/${jobPrefix}_%a.e ${submitFile}
  rm ${submitFile}
}

submitMergePDFMaps()
{
  ## go to working directory first before submitting
  workDir=$(pwd)
  codeDir=${TPCwithDNN}
  batchCommand="sbatch --get-user-env --mem-per-cpu=50000 --time=08:00:00 --ntasks=1"

  if [ ! -d logs ]; then
    mkdir logs
  fi

  submitFile=scdcalib_pdfmaps.sh
  jobPrefix=mergepdfmaps

  echo '#!/bin/bash' >${submitFile}
  echo "nEvTrain=\$((\${SLURM_ARRAY_TASK_ID}*1000))" >>${submitFile}
  echo "createSteeringXMLFile DNN_fluctuations_nEv\${nEvTrain} ${jobPrefix}_nEv\${nEvTrain}.yml false false false false false false false false true" >>${submitFile}
  echo "singularity exec /lustre/alice/users/hellbaer/NOTESData/JIRA/ATO-500/SCDcalibML_ROCM.sif ${codeDir}/macros/runTPCwithDNN.sh ${workDir} ${jobPrefix}_nEv\${nEvTrain}.yml" >>${submitFile}
  ${batchCommand} --array=1,5,10 -o logs/${jobPrefix}_nEv%a000.o -e logs/${jobPrefix}_nEv%a000.e ${submitFile}
  rm ${submitFile}
}

createParamXMLFile()
{
  case=$1
  nphi=$2
  nr=$3
  nz=$4
  nEvTrain=$5
  nEvTest=$6
  nEvApply=$7
  node=$8

  fileName=database_parameters_${case}.yml

  echo "${case}:" >${fileName}
  echo "  dirmodel: /lustre/alice/users/hellbaer/NOTESData/JIRA/ATO-439/${node}/SC-${nz}-${nr}-${nphi}/models" >>${fileName}
  echo "  dirval: /lustre/alice/users/hellbaer/NOTESData/JIRA/ATO-439/${node}/SC-${nz}-${nr}-${nphi}/validation" >>${fileName}
  echo "  dirinput_bias: /lustre/alice/users/mkleiner/NOTESData/JIRA/ATO-439/bias" >>${fileName}
  echo "  dirinput_nobias: /lustre/alice/users/mkleiner/NOTESData/JIRA/ATO-439/nobias" >>${fileName}
  echo "  train_bias: true" >>${fileName}
  echo "  test_bias: true" >>${fileName}
  echo "  apply_bias: false" >>${fileName}
  echo "  diroutflattree: /lustre/alice/users/hellbaer/NOTESData/JIRA/ATO-439/${node}/SC-${nz}-${nr}-${nphi}/trees" >>${fileName}
  echo "  dirouthistograms: /lustre/alice/users/hellbaer/NOTESData/JIRA/ATO-439/${node}/SC-${nz}-${nr}-${nphi}/histograms" >>${fileName}
  echo "  tree_events: 27000 # Number of events for validation (random events for nd_validation)" >>${fileName}
  echo "  validate_model: true" >>${fileName}
  echo "  pdf_map_var: flucSC" >>${fileName}
  echo "  pdf_map_mean_id: 0" >>${fileName}
  echo "  grid_phi: ${nphi}" >>${fileName}
  echo "  grid_z: ${nz}" >>${fileName}
  echo "  grid_r: ${nr}" >>${fileName}
  echo "  selopt_input: 0 # 0 is use to train distorsions only for positive z-values" >>${fileName}
  echo "  selopt_output: 0 # 0 is use to predict distorsions only for positive z-values" >>${fileName}
  echo "  opt_train: [1, 1] #first position = meanSC, second = SCfluctuations" >>${fileName}
  echo "  opt_predout: [1, 0, 0] #R, Rphi, z output distorsion predictions" >>${fileName}
  echo "  nameopt_predout: [R, RPhi, Z] #R, Rphi, z output distorsion predictions" >>${fileName}
  echo "  maxrandomfiles: 1000 #this is the total number of random events generated" >>${fileName}
  echo "  range_mean_index: [0,26] # min and max index of mean SC configurations" >>${fileName}
  echo "  train_events: [${nEvTrain}] # events for training (include a mix of SC config)" >>${fileName}
  echo "  test_events: [${nEvTest}]" >>${fileName}
  echo "  apply_events: [${nEvApply}]" >>${fileName}
  echo "  max_events: 27000 # number of all available events" >>${fileName}
  echo "  use_scaler: 0" >>${fileName}
  echo "  filters: 4" >>${fileName}
  echo "  pooling: 0" >>${fileName}
  echo "  batch_size: 27" >>${fileName}
  echo "  shuffle: false" >>${fileName}
  echo "  depth: 4" >>${fileName}
  echo "  batch_normalization: 0" >>${fileName}
  echo "  dropout: 0.0" >>${fileName}
  echo "  epochs: 20" >>${fileName}
  echo "  lossfun: mse" >>${fileName}
  echo "  metrics: mse" >>${fileName}
  echo "  adamlr: 0.001000" >>${fileName}
}
export -f createParamXMLFile

createSteeringXMLFile()
{
  ## create xml file (default.yml) with steering options
  case=$1
  filename=$2
  dotrain=$3
  doapply=$4
  doplot=$5
  dogrid=$6
  doprofile=$7
  docreatevaldata=$8
  docreatepdfmaps=$9
  docreatepdfmapforvariable=${10}
  domergepdfmaps=${11}

  echo "case: ${case}" > ${filename}
  echo "dotrain: ${dotrain}" >> ${filename}
  echo "doapply: ${doapply}" >> ${filename}
  echo "doplot: ${doplot}" >> ${filename}
  echo "dogrid: ${dogrid}" >> ${filename}
  echo "doprofile: ${doprofile}" >> ${filename}
  echo "docreatevaldata: ${docreatevaldata}" >> ${filename}
  echo "docreatepdfmaps: ${docreatepdfmaps}" >> ${filename}
  echo "docreatepdfmapforvariable: ${docreatepdfmapforvariable}" >> ${filename}
  echo "domergepdfmaps: ${domergepdfmaps}" >> ${filename}
}
export -f createSteeringXMLFile

createPDFMapList()
{
  fileName=$1
  if [ -f ${fileName} ]; then
    rm ${fileName}
  fi

  touch ${fileName}
  for nEvTrain in {1000,5000,10000}; do
    echo "${nEvTrain} flucSC 0" >>${fileName}
    echo "${nEvTrain} flucSC 9" >>${fileName}
    echo "${nEvTrain} flucSC 18" >>${fileName}
    echo "${nEvTrain} meanSC 0" >>${fileName}
    echo "${nEvTrain} meanSC 9" >>${fileName}
    echo "${nEvTrain} meanSC 18" >>${fileName}
    echo "${nEvTrain} derRefMeanSC 0" >>${fileName}
    echo "${nEvTrain} derRefMeanSC 9" >>${fileName}
    echo "${nEvTrain} derRefMeanSC 18" >>${fileName}
    echo "${nEvTrain} flucDistR 0" >>${fileName}
    echo "${nEvTrain} flucDistR 9" >>${fileName}
    echo "${nEvTrain} flucDistR 18" >>${fileName}
    echo "${nEvTrain} flucDistRPred 0" >>${fileName}
    echo "${nEvTrain} flucDistRPred 9" >>${fileName}
    echo "${nEvTrain} flucDistRPred 18" >>${fileName}
    echo "${nEvTrain} flucDistRDiff 0" >>${fileName}
    echo "${nEvTrain} flucDistRDiff 9" >>${fileName}
    echo "${nEvTrain} flucDistRDiff 18" >>${fileName}
    echo "${nEvTrain} meanDistR 0" >>${fileName}
    echo "${nEvTrain} meanDistR 9" >>${fileName}
    echo "${nEvTrain} meanDistR 18" >>${fileName}
    echo "${nEvTrain} derRefMeanDistR 0" >>${fileName}
    echo "${nEvTrain} derRefMeanDistR 9" >>${fileName}
    echo "${nEvTrain} derRefMeanDistR 18" >>${fileName}
  done
}
export -f createPDFMapList
