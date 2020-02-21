# pylint: disable=line-too-long, invalid-name, too-many-instance-attributes,
# pylint: disable=fixme, pointless-string-statement, too-many-arguments
import numpy as np
import keras
from sklearn.externals import joblib

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class fluctuationDataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, batch_size=32, phi_slice=180, r_row=129, z_col=129, \
        n_channels=3, side=0, shuffle=True, data_dir='data/', use_scaler=False, \
        distortion_type=0):
        self.phi_slice = phi_slice
        self.r_row = r_row
        self.z_col = z_col
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data_dir = data_dir
        self.side = side
        self.use_scaler = use_scaler
        self.distortion_type = distortion_type
        if use_scaler > 0:
            self.scalerSC = joblib.load(self.data_dir + "scalerSC-" + str(use_scaler) + ".save")
            self.scalerDistR = joblib.load(self.data_dir + "scalerDistR-" + str(use_scaler) + ".save")
            self.scalerDistRPhi = joblib.load(self.data_dir + "scalerDistRPhi-" + str(use_scaler) + ".save")
            self.scalerDistZ = joblib.load(self.data_dir + "scalerDistZ-" + str(use_scaler) + ".save")

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
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.phi_slice, self.r_row, self.z_col, 1))
        Y = np.empty((self.batch_size, self.phi_slice, self.r_row, self.z_col, self.n_channels))
        vecZPos = np.load(self.data_dir + str(0) + '-vecZPos.npy')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store
            vecMeanSC = np.load(self.data_dir + str(ID) + '-vecMeanSC.npy')
            vecRandomSC = np.load(self.data_dir + str(ID) + '-vecRandomSC.npy')
            vecMeanDistR = np.load(self.data_dir + str(ID) + '-vecMeanDistR.npy')
            vecRandomDistR = np.load(self.data_dir + str(ID) + '-vecRandomDistR.npy')
            vecMeanDistRPhi = np.load(self.data_dir + str(ID) + '-vecMeanDistRPhi.npy')
            vecRandomDistRPhi = np.load(self.data_dir + str(ID) + '-vecRandomDistRPhi.npy')
            vecMeanDistZ = np.load(self.data_dir + str(ID) + '-vecMeanDistZ.npy')
            vecRandomDistZ = np.load(self.data_dir + str(ID) + '-vecRandomDistZ.npy')

            if self.side == 0:
                vecFluctuationSC = vecMeanSC[vecZPos >= 0] - vecRandomSC[vecZPos >= 0]
                vecFluctuationDistR = vecMeanDistR[vecZPos >= 0] - vecRandomDistR[vecZPos >= 0]
                vecFluctuationDistRPhi = vecMeanDistRPhi[vecZPos >= 0] - vecRandomDistRPhi[vecZPos >= 0]
                vecFluctuationDistZ = vecMeanDistZ[vecZPos >= 0] - vecRandomDistZ[vecZPos >= 0]
            elif self.side == 1:
                vecFluctuationSC = vecMeanSC[vecZPos < 0] - vecRandomSC[vecZPos < 0]
                vecFluctuationDistR = vecMeanDistR[vecZPos < 0] - vecRandomDistR[vecZPos < 0]
                vecFluctuationDistRPhi = vecMeanDistRPhi[vecZPos < 0] - vecRandomDistRPhi[vecZPos < 0]
                vecFluctuationDistZ = vecMeanDistZ[vecZPos < 0] - vecRandomDistZ[vecZPos < 0]
            elif self.side == 2:
                vecFluctuationSC = vecMeanSC - vecRandomSC
                vecFluctuationDistR = vecMeanDistR - vecRandomDistR
                vecFluctuationDistRPhi = vecMeanDistRPhi - vecRandomDistRPhi
                vecFluctuationDistZ = vecMeanDistZ - vecRandomDistZ
            if self.use_scaler > 0:
                vecFluctuationSC_scaled = self.scalerSC.transform(vecFluctuationSC.reshape(1, -1))
                vecFluctuationDistR_scaled = self.scalerDistR.transform(vecFluctuationDistR.reshape(1, -1))
                vecFluctuationDistRPhi_scaled = self.scalerDistRPhi.transform(vecFluctuationDistRPhi.reshape(1, -1))
                vecFluctuationDistZ_scaled = self.scalerDistZ.transform(vecFluctuationDistZ.reshape(1, -1))

                X[i, :, :, :, 0] = vecFluctuationSC_scaled.reshape(self.phi_slice, self.r_row, self.z_col)
                if self.distortion_type == 0:
                    Y[i, :, :, :, 0] = vecFluctuationDistR_scaled.reshape(self.phi_slice, self.r_row, self.z_col)
                elif self.distortion_type == 1:
                    Y[i, :, :, :, 0] = vecFluctuationDistRPhi_scaled.reshape(self.phi_slice, self.r_row, self.z_col)
                else:
                    Y[i, :, :, :, 0] = vecFluctuationDistZ_scaled.reshape(self.phi_slice, self.r_row, self.z_col)
            else:
                X[i, :, :, :, 0] = vecFluctuationSC.reshape(self.phi_slice, self.r_row, self.z_col)
                if self.distortion_type == 0:
                    Y[i, :, :, :, 0] = vecFluctuationDistR.reshape(self.phi_slice, self.r_row, self.z_col)
                elif self.distortion_type == 1:
                    Y[i, :, :, :, 0] = vecFluctuationDistRPhi.reshape(self.phi_slice, self.r_row, self.z_col)
                else:
                    Y[i, :, :, :, 0] = vecFluctuationDistZ.reshape(self.phi_slice, self.r_row, self.z_col)
            return X, Y
