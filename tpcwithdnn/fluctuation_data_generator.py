"""
A data generator for lazy input loading for DNN.
"""
# pylint: disable=too-many-instance-attributes, too-many-arguments
import numpy as np

import keras

from tpcwithdnn.data_loader import load_train_apply

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class FluctuationDataGenerator(keras.utils.Sequence):
    """
    The class defining a lazy data generator.
    """

    def __init__(self, list_ids, grid_phi, grid_r, grid_z, batch_size, shuffle,
                 opt_train, opt_predout, z_range, dirinput,
                 use_scaler):
        """
        Initialize the generator.

        :param list list_ids: list of indices of the event files to be used
        :param int grid_phi: grid granularity (number of voxels) along phi-axis
        :param int grid_r: grid granularity (number of voxels) along r-axis
        :param int grid_z: grid granularity (number of voxels) along z-axis
        :param int batch_size: size of the batch, from the config file
        :param bool shuffle: whether to shuffle the data after each epoch, from the config file
        :param list opt_train: list of 2 binary values corresponding to activating the train input
                               of average space charge and space-charge fluctuations, respectively,
                               taken from the config file
        :param list opt_pred: list of 3 binary values corresponding to activating the prediction of
                              r, rphi and z distortion corrections, taken from the config file
        :param list z_range: a list of [min_z, max_z] values, the input is taken from this interval
        :param str dirinput: the directory with the input data, value taken from the config file
        """
        self.list_ids = list_ids
        self.grid_phi = grid_phi
        self.grid_r = grid_r
        self.grid_z = grid_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.opt_train = opt_train
        self.opt_predout = opt_predout
        self.dim_input = sum(self.opt_train)
        self.dim_output = sum(self.opt_predout)
        self.z_range = z_range
        self.dirinput = dirinput
        self.use_scaler = use_scaler

    def __len__(self):
        """
        Get the number of batches per epoch

        :return: number of batches per epoch
        :rtype: int
        """
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate a batch of data at index

        :param int index: index of the batch to generate
        :return: input and output data for the batch
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]
        # Generate data
        inputs, exp_outputs = self.__data_generation(list_ids_temp)
        return inputs, exp_outputs

    def on_epoch_end(self):
        """
        Update indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """
        Generate data corresponding to the list of indices

        :param list list_ids_temp: list of file indices for a given batch
        :return: input and output data corresponding to the indices
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        # Initialization
        inputs = np.empty((self.batch_size, self.grid_phi, self.grid_r,
                           self.grid_z, self.dim_input))
        exp_outputs = np.empty((self.batch_size, self.grid_phi, self.grid_r,
                                self.grid_z, self.dim_output))
        # Generate data
        for i, id_num in enumerate(list_ids_temp):
            # Store
            inputs_i, exp_outputs_i = load_train_apply(self.dirinput, id_num,
                                                       self.z_range,
                                                       self.grid_r, self.grid_phi, self.grid_z,
                                                       self.opt_train, self.opt_predout)
            inputs[i, :, :, :, :] = inputs_i
            exp_outputs[i, :, :, :, :] = exp_outputs_i
        return inputs, exp_outputs
