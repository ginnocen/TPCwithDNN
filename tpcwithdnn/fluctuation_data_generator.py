# pylint: disable=too-many-instance-attributes, too-many-arguments
# pylint: disable=missing-module-docstring, missing-class-docstring
import numpy as np

import tensorflow.keras

from tpcwithdnn.data_loader import load_train_apply

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class FluctuationDataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, list_ids, phi_slice, r_row, z_col, batch_size, shuffle,
                 opt_train, opt_predout, selopt_input, selopt_output, data_dir,
                 use_scaler):
        self.list_ids = list_ids
        self.phi_slice = phi_slice
        self.r_row = r_row
        self.z_col = z_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.opt_train = opt_train
        self.opt_predout = opt_predout
        self.dim_input = sum(self.opt_train)
        self.dim_output = sum(self.opt_predout)
        self.selopt_input = selopt_input
        self.selopt_output = selopt_output
        self.data_dir = data_dir
        self.use_scaler = use_scaler

    def __len__(self):
        """ Number of batches per epoch """
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]
        # Generate data
        inputs, exp_outputs = self.__data_generation(list_ids_temp)
        return inputs, exp_outputs

    def on_epoch_end(self):
        """ Update indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        'Generates data containing batch_size samples'
        # Initialization
        inputs = np.empty((self.batch_size, self.phi_slice, self.r_row, self.z_col, self.dim_input))
        exp_outputs = np.empty((self.batch_size, self.phi_slice, self.r_row,
                                self.z_col, self.dim_output))
        # Generate data
        for i, id_num in enumerate(list_ids_temp):
            # Store
            inputs_i, exp_outputs_i = load_train_apply(self.data_dir, id_num,
                                                       self.selopt_input, self.selopt_output,
                                                       self.r_row, self.phi_slice, self.z_col,
                                                       self.opt_train, self.opt_predout)
            inputs[i, :, :, :, :] = inputs_i
            exp_outputs[i, :, :, :, :] = exp_outputs_i
        return inputs, exp_outputs
