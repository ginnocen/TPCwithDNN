# Singularity containers
Here we can put recipes to build singularity containers.

The container .sif files can be pushed to the singularity cloud https://cloud.sylabs.io/library (11 GB quota). Uploaded containers can easily be pulled using ```singularity pull```.

For the moment, the containers for CUDA (5 GB) and ROCM (6 GB) application are uploaded to Ernst's sylabs account.

## Usage
* Pull container from sylabs cloud first
    ```
    cd ${TPCwithDNN}/containers
    singularity pull library://ehellbar/default/tpcwithdnn:cuda

    singularity pull library://ehellbar/default/tpcwithdnn:rocm
    ```

* Load container shell
    ```
    singularity shell --nv ${TPCwithDNN}/containers/tpcwithdnn_cuda.sif

    singularity shell ${TPCwithDNN}/containers/tpcwithdnn_rocm.sif
    ```

* Execute commands or shell scripts inside the container, e.g.
    ```
    singularity exec --nv ${TPCwithDNN}/containers/tpcwithdnn_cuda.sif ${TPCwithDNN}/macros/runTPCwithDNN.sh ${workDir} default.yml

    singularity exec ${TPCwithDNN}/containers/tpcwithdnn_rocm.sif ${TPCwithDNN}/macros/runTPCwithDNN.sh ${workDir} default.yml
    ```


## Building the container for development

1. Define singularity prefix directory- SINGULARITY_PREFIX
    ```
    export SINGULARITY_CACHEDIR=/tmp/${USER}/
    export SINGULARITY_PREFIX=/tmp/${USER}/JIRA/ATO-500
    ```
    and build directory:
    ```
    mkdir -p $SINGULARITY_PREFIX
    ```
2.  build containers

    2.1 - in sandbox/directory  - to play with container
    ```
    time singularity build --fakeroot --fix-perms --sandbox $SINGULARITY_PREFIX/tpcwithdnn_cuda ${TPCwithDNN}/containers/tpcwithdnn_cuda.def

    time singularity build --fakeroot --fix-perms --sandbox $SINGULARITY_PREFIX/tpcwithdnn_rocm ${TPCwithDNN}/containers/tpcwithdnn_rocm.def

    ```
    2.2 - test new feature and make modification in sandbox container
    * e.g . install additional packages in %environment section
    ```
    singularity shell --fakeroot $SINGULARITY_PREFIX/tpcwithdnn_cuda
    cd $SINGULARITY_PREFIX/tpcwithdnn_cuda

    singularity shell --fakeroot $SINGULARITY_PREFIX/tpcwithdnn_rocm
    cd $SINGULARITY_PREFIX/tpcwithdnn_rocm

    ```
    2.3 - save as sif file to make final container- to installation path
    ```
    time singularity build --fakeroot --fix-perms ${TPCwithDNN}/containers/tpcwithdnn_cuda.sif $SINGULARITY_PREFIX/tpcwithdnn_cuda

    time singularity build --fakeroot --fix-perms ${TPCwithDNN}/containers/tpcwithdnn_rocm.sif $SINGULARITY_PREFIX/tpcwithdnn_rocm
    ```

## Running GPU benchmarks:
* Tensorflow benchmark:
    ```
    singularity shell ${TPCwithDNN}/containers/tpcwithdnn_cuda.sif
    /usr/local/bin/benchmark.sh

    singularity shell ${TPCwithDNN}/containers/tpcwithdnn_rocm.sif
    /usr/local/bin/benchmark.sh

    ```
