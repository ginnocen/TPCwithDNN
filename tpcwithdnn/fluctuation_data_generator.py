# pylint: disable=invalid-name, too-many-instance-attributes,
# pylint: disable=fixme, pointless-string-statement, too-many-arguments
import numpy as np
import keras
from data_loader import load_train_apply

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class FluctuationDataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, phi_slice, r_row, z_col, batch_size, shuffle,
                 opt_train, opt_predout, selopt_input, selopt_output, data_dir,
                 use_scaler):
        self.list_IDs = list_IDs
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
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.phi_slice, self.r_row, self.z_col, self.dim_input))
        Y = np.empty((self.batch_size, self.phi_slice, self.r_row, self.z_col, self.dim_output))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store
            x_, y_ = load_train_apply(self.data_dir, ID, self.selopt_input, self.selopt_output,
                                    self.r_row, self.phi_slice, self.z_col,
                                    self.opt_train, self.opt_predout)
            X[i, :, :, :, :] = x_
            Y[i, :, :, :, :] = y_
        return X, Y
