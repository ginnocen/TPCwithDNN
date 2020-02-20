import os
import time
from root_numpy import fill_hist
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import model_from_json
from ROOT import TH1F, TH2F, TFile # pylint: disable=import-error, no-name-in-module
from SymmetricPadding3D import SymmetricPadding3D
from machine_learning_hep.logger import get_logger
from fluctuationDataGenerator import fluctuationDataGenerator
from modelDataCurrentRegressionKerasNoRoot import UNet, GetFluctuation

# pylint: disable=line-too-long, too-many-instance-attributes, too-many-statements
class DnnOptimiser:
    #Class Attribute
    species = "dnnoptimiser"

    def __init__(self, data_param, case):
        print(case)
        self.logger = get_logger()
        print("BUILDING OPTIMISATION")

        self.data_param = data_param
        self.dirmodel = self.data_param["dirmodel"]
        self.dirval = self.data_param["dirval"]
        self.dirinput = self.data_param["dirinput"]
        self.grid_phi = self.data_param["grid_phi"]
        self.grid_z = self.data_param["grid_z"]
        self.grid_r = self.data_param["grid_r"]
        self.filters = self.data_param["filters"]
        self.pooling = self.data_param["pooling"]
        self.batch_size = self.data_param["batch_size"]
        self.n_channels = self.data_param["n_channels"]
        self.shuffle = self.data_param["shuffle"]
        self.depth = self.data_param["depth"]
        self.batch_normalization = self.data_param["batch_normalization"]
        self.side = self.data_param["side"]
        self.dropout = self.data_param["dropout"]
        self.use_scaler = self.data_param["use_scaler"]
        self.distortion_type = self.data_param["distortion_type"]
        self.epochs = self.data_param["ephocs"]
        self.rangeevent_train = self.data_param["rangeevent_train"]
        self.rangeevent_test = self.data_param["rangeevent_test"]

        self.dirinput = self.dirinput + "/%d-%d-%d/" % (self.grid_phi, self.grid_z,
                                                        self.grid_r)
        self.params = {'phi_slice': self.grid_phi,
                       'r_row' : self.grid_r,
                       'z_col' : self.grid_z,
                       'batch_size': self.batch_size,
                       'n_channels': self.n_channels,
                       'shuffle': self.shuffle,
                       'side' : self.side,
                       'data_dir': self.dirinput,
                       'use_scaler': self.use_scaler,
                       'distortion_type': self.distortion_type}

        self.suffix = "phi%d_r%d_z%d_filter%d_poo%d_drop%.2f_depth%d_batch%d_scaler%d_dist%d" % \
                (self.grid_phi, self.grid_r, self.grid_z, self.filters, self.pooling,
                 self.dropout, self.depth, self.batch_normalization,
                 self.use_scaler, self.distortion_type)

    def train(self):
        partition = {'train': np.arange(self.rangeevent_train[1]),
                     'validation': np.arange(self.rangeevent_test[0], self.rangeevent_test[1])}
        training_generator = fluctuationDataGenerator(partition['train'], **self.params)
        validation_generator = fluctuationDataGenerator(partition['validation'], **self.params)
        model = UNet((self.grid_phi, self.grid_r, self.grid_z, 1),
                     depth=self.depth, bathnorm=self.batch_normalization,
                     pool_type=self.pooling, start_ch=self.filters, dropout=self.dropout)
        model.compile(loss="mse", \
        optimizer=Adam(lr=0.001000), \
        metrics=["mse"]) # Mean squared error
        model.summary()

        his = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=True,
                                  epochs=self.epochs, workers=1)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), his.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), his.history["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("plot.png")

        model_json = model.to_json()
        with open("%s/modelLocalCurrent%s.json" % (self.dirmodel, self.suffix), "w") as json_file: \
            json_file.write(model_json)
        model.save_weights("%s/modelLocalCurrent%s.h5" % (self.dirmodel, self.suffix))
        print("Saved model to disk")
        # list all data in history

    def apply(self):
        print("APPLY")
        json_file = open("%s/modelLocalCurrent%s.json" % (self.dirmodel, self.suffix), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, {'SymmetricPadding3D' : SymmetricPadding3D})
        loaded_model.load_weights("%s/modelLocalCurrent%s.h5" % (self.dirmodel, self.suffix))
        os.chdir(self.dirval)
        myfile = TFile.Open("output%s.root" % self.suffix, "recreate")
        for iexperiment in range(self.rangeevent_test[0], self.rangeevent_test[1]):
            indexev = iexperiment
            print(str(indexev))
            [vecFluctuationSC, vecFluctuationDistR, vecFluctuationDistRPhi, vecFluctuationDistZ] = \
                    GetFluctuation(self.grid_phi, self.grid_r, self.grid_z, indexev)
            if self.use_scaler > 0:
                # pylint: disable=undefined-variable
                scalerSC = joblib.load(self.dirinput + "scalerSC-" + str(self.use_scaler) + ".save")
                scalerDistR = joblib.load(self.dirinput + "scalerDistR-" + str(self.use_scaler) + ".save")
                scalerDistRPhi = joblib.load(self.dirinput + "scalerDistRPhi-" + str(self.use_scaler) + ".save")
                scalerDistZ = joblib.load(self.dirinput + "scalerDistZ-" + str(self.use_scaler) + ".save")
                if self.distortion_type == 0:
                    scalerDist = scalerDistR
                elif self.distortion_type == 1:
                    scalerDist = scalerDistRPhi
                else:
                    scalerDist = scalerDistZ
                vecFluctuationSC_scaled = scalerSC.transform(vecFluctuationSC.reshape(1, -1))
            else:
                start = time.time()

            start = time.time()
            if self.use_scaler > 0:
                distortionPredict = loaded_model.predict(vecFluctuationSC_scaled.reshape(1, self.grid_phi, self.grid_r, self.grid_z, 1))
            else:
                distortionPredict = loaded_model.predict(vecFluctuationSC.reshape(1, self.grid_phi, self.grid_r, self.grid_z, 1))
            end = time.time()
            predictTime = end - start
            print("Time to predict: " + str(predictTime) + " s")

            if self.distortion_type == 0:
                distortionNumeric = vecFluctuationDistR
            elif self.distortion_type == 1:
                distortionNumeric = vecFluctuationDistRPhi
            else:
                distortionNumeric = vecFluctuationDistZ
            #distorsionPredict = distortionPredict.reshape(1, self.grid_phi, self.grid_r, self.grid_z, -1)
            if self.use_scaler > 0:
                distortionPredict = scalerDist.inverse_transform(distortionPredict.reshape(1, -1))
            residueMean = \
            np.absolute(distortionNumeric.reshape(1, self.grid_phi, self.grid_r, self.grid_z) - \
                        distortionPredict.reshape(1, self.grid_phi, self.grid_r, self.grid_z)).mean()
            residueStd = np.absolute(distortionNumeric.reshape(1, self.grid_phi, self.grid_r, self.grid_z) - \
                       distortionPredict.reshape(1, self.grid_phi, self.grid_r, self.grid_z)).std()
            print("residueMean\t" + str(residueMean))
            print("residueStd\t" + str(residueStd))

            distortionNumeric_flat = distortionNumeric.flatten()
            distortionPredict_flat = distortionPredict.flatten()
            deltas = (distortionPredict_flat - distortionNumeric_flat)

            h_dist = TH2F("hdist_Ev%d" % iexperiment + self.suffix, "", 100, -3, 3, 100, -3, 3)
            h_deltas = TH1F("hdeltas_Ev%d" % iexperiment + self.suffix, "", 1000, -1., 1.)
            fill_hist(h_dist, np.concatenate((distortionNumeric.reshape(-1, 1), \
                                             distortionPredict.reshape(-1, 1)), axis=1))
            fill_hist(h_deltas, deltas)
            h_dist.Write()
            h_deltas.Write()
        myfile.Close()

    # pylint: disable=no-self-use
    def gridsearch(self):
        print("GRID SEARCH NOT YET IMPLEMENTED")
