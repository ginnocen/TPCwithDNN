import os
from root_numpy import fill_hist
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import model_from_json
from ROOT import TH1F, TH2F, TFile # pylint: disable=import-error, no-name-in-module
from symmetrypadding3d import symmetryPadding3d
from machine_learning_hep.logger import get_logger
from fluctuationDataGenerator import fluctuationDataGenerator
from utilitiesdnn import UNet
from dataloader import loaddata, loadtrain_test 

# pylint: disable=too-many-instance-attributes, too-many-statements, fixme, pointless-string-statement
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
        # prepare the dataset
        self.selopt_input = self.data_param["selopt_input"]
        self.selopt_output = self.data_param["selopt_output"]
        self.opt_train = self.data_param["opt_train"]
        self.opt_predout = self.data_param["opt_predout"]
        self.dim_input = sum(self.opt_train)
        self.dim_output = sum(self.opt_predout)
        self.rangeevent_train = self.data_param["rangeevent_train"]
        self.rangeevent_test = self.data_param["rangeevent_test"]
        self.use_scaler = self.data_param["use_scaler"]
        # DNN config
        self.filters = self.data_param["filters"]
        self.pooling = self.data_param["pooling"]
        self.batch_size = self.data_param["batch_size"]
        self.shuffle = self.data_param["shuffle"]
        self.depth = self.data_param["depth"]
        self.batch_normalization = self.data_param["batch_normalization"]
        self.dropout = self.data_param["dropout"]
        self.epochs = self.data_param["ephocs"]

        self.dirinput = self.dirinput + "/%d-%d-%d/" % (self.grid_phi, self.grid_z,
                                                        self.grid_r)
        self.params = {'phi_slice': self.grid_phi,
                       'r_row' : self.grid_r,
                       'z_col' : self.grid_z,
                       'batch_size': self.batch_size,
                       'shuffle': self.shuffle,
                       'opt_train' : self.opt_train,
                       'opt_predout' : self.opt_predout,
                       'selopt_input' : self.selopt_input,
                       'selopt_output' : self.selopt_output,
                       'data_dir': self.dirinput,
                       'use_scaler': self.use_scaler}

        self.suffix = "phi%d_r%d_z%d_filter%d_poo%d_drop%.2f_depth%d_batch%d_scaler%d" % \
                (self.grid_phi, self.grid_r, self.grid_z, self.filters, self.pooling,
                 self.dropout, self.depth, self.batch_normalization, self.use_scaler)

    def train(self):
        #FIXME: missing random extraction for training and testing events?
        print("I AM DOING TRAINING")
        partition = {'train': np.arange(self.rangeevent_train[1]),
                     'validation': np.arange(self.rangeevent_test[0], self.rangeevent_test[1])}
        training_generator = fluctuationDataGenerator(partition['train'], **self.params)
        validation_generator = fluctuationDataGenerator(partition['validation'], **self.params)
        print("DIMENSION INPUT", self.dim_input)
        model = UNet((self.grid_phi, self.grid_r, self.grid_z, self.dim_input),
                     depth=self.depth, bathnorm=self.batch_normalization,
                     pool_type=self.pooling, start_ch=self.filters, dropout=self.dropout)
        model.compile(loss="mse", optimizer=Adam(lr=0.001000), metrics=["mse"]) # Mean squared error
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
        with open("%s/model%s.json" % (self.dirmodel, self.suffix), "w") as json_file: \
            json_file.write(model_json)
        model.save_weights("%s/model%s.h5" % (self.dirmodel, self.suffix))
        print("Saved model to disk")
        # list all data in history

    def groupbyindices_input(self, arrayflat):
        return arrayflat.reshape(1, self.grid_phi, self.grid_r, self.grid_z, self.dim_input)

    # pylint: disable=fixme
    def apply(self):
        print("APPLY, input size", self.dim_input)
        json_file = open("%s/model%s.json" % (self.dirmodel, self.suffix), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = \
            model_from_json(loaded_model_json, {'symmetryPadding3d' : symmetryPadding3d})
        loaded_model.load_weights("%s/model%s.h5" % (self.dirmodel, self.suffix))
        os.chdir(self.dirval)
        myfile = TFile.Open("output%s.root" % self.suffix, "recreate")
        for iexperiment in range(self.rangeevent_test[0], self.rangeevent_test[1]):
            indexev = iexperiment
            x_, y_ = loadtrain_test(self.dirinput, indexev, self.selopt_input, self.selopt_output,
                                    self.grid_r, self.grid_phi, self.grid_z,
                                    self.opt_train, self.opt_predout)
            X = np.empty((1, self.grid_phi, self.grid_r, self.grid_z, self.dim_input))
            Y = np.empty((1, self.grid_phi, self.grid_r, self.grid_z, self.dim_output))
            X[0, :, :, :, :] = x_
            Y[0, :, :, :, :] = y_

            distortionPredict_group = loaded_model.predict(X)
            distortionPredict_flatm = distortionPredict_group.reshape(-1, 1)
            distortionPredict_flata = distortionPredict_group.flatten()

            distortionNumeric_group = Y
            distortionNumeric_flatm = distortionNumeric_group.reshape(-1, 1)
            distortionNumeric_flata = distortionNumeric_group.flatten()

            deltas = (distortionPredict_flata - distortionNumeric_flata)

            h_dist = TH2F("hdist_Ev%d" % iexperiment + self.suffix, "", 100, -3, 3, 100, -3, 3)
            h_deltas = TH1F("hdeltas_Ev%d" % iexperiment + self.suffix, "", 1000, -1., 1.)
            fill_hist(h_dist, np.concatenate((distortionNumeric_flatm,
                                              distortionPredict_flatm), axis=1))
            fill_hist(h_deltas, deltas)
            h_dist.Write()
            h_deltas.Write()
        myfile.Close()
        print("DONE APPLY")

    # pylint: disable=no-self-use
    def gridsearch(self):
        print("GRID SEARCH NOT YET IMPLEMENTED")
