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

    def get_fluctuation(self, indexev):
        #FIXME
        """ Here we define the functionalties to load the files from the input
        directory which is set in the database. Here below the description of
        the input files:
            - 0-vecZPos.npy, 0-vecRPos.npy, 0-vecPhiPos.npy contains the
            position of the FIXME. There is only one of these files for each
            folder, therefore for each bunch of events
            Input features for training:
            - vecMeanSC.npy: average space charge in each bin of r, rphi and z.
            - vecRandomSC.npy: fluctuation of the space charge.
            Output from the numberical calculations:
            - vecMeanDistR.npy average distorsion along the R axis in the same
              grid. It represents the expected distorsion that an electron
              passing by that region would have as a consequence of the IBF.
            - vecRandomDistR.npy are the correponding fluctuations.
            - All the distorsions along the other directions have a consistent
              naming choice.

        """

        vecZPosFile = self.dirinput + str(0) + '-vecZPos.npy'
        scMeanFile = self.dirinput + str(indexev) + '-vecMeanSC.npy'
        scRandomFile = self.dirinput + str(indexev) + '-vecRandomSC.npy'
        distRMeanFile = self.dirinput + str(indexev) + '-vecMeanDistR.npy'
        distRRandomFile = self.dirinput + str(indexev) + '-vecRandomDistR.npy'
        distRPhiMeanFile = self.dirinput + str(indexev) + '-vecMeanDistRPhi.npy'
        distRPhiRandomFile = self.dirinput + str(indexev) + '-vecRandomDistRPhi.npy'
        distZMeanFile = self.dirinput + str(indexev) + '-vecMeanDistZ.npy'
        distZRandomFile = self.dirinput + str(indexev) + '-vecRandomDistZ.npy'
        vecZPos = np.load(vecZPosFile)
        vecMeanSC = np.load(scMeanFile)
        vecRandomSC = np.load(scRandomFile)
        vecMeanDistR = np.load(distRMeanFile)
        vecRandomDistR = np.load(distRRandomFile)
        vecMeanDistRPhi = np.load(distRPhiMeanFile)
        vecRandomDistRPhi = np.load(distRPhiRandomFile)
        vecMeanDistZ = np.load(distZMeanFile)
        vecRandomDistZ = np.load(distZRandomFile)

        #FIXME: wrong selection?
        """What is the meaning of splitting along the z axis. Why dont we split
        along other axes? In the previous version of this code, there was a
        selection over the z-axis hardcoded. Is there a reason for that? I
        currently have included all the three options, namely <,>, no choice.

        """
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
        return [vecFluctuationSC, vecFluctuationDistR, vecFluctuationDistRPhi, vecFluctuationDistZ]

    def train(self):
        #FIXME: missing random extraction for training and testing events?
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
        with open("%s/model%s.json" % (self.dirmodel, self.suffix), "w") as json_file: \
            json_file.write(model_json)
        model.save_weights("%s/model%s.h5" % (self.dirmodel, self.suffix))
        print("Saved model to disk")
        # list all data in history

    def groupbyindices(self, arrayflat):
        return arrayflat.reshape(1, self.grid_phi, self.grid_r, self.grid_z, 1)

    # pylint: disable=fixme
    def apply(self):
        print("APPLY")
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
            #[vecFluctSC, vecFluctDistR, vecFluctDistRPhi, vecFluctDistZ] = \
            [vecFluctSC_flata, vecFluctDistR_flata, _, _] = self.get_fluctuation(indexev)

            vecFluctSC_group = self.groupbyindices(vecFluctSC_flata)

            distortionPredict_group = loaded_model.predict(vecFluctSC_group)
            distortionPredict_flatm = distortionPredict_group.reshape(-1, 1)
            distortionPredict_flata = distortionPredict_group.flatten()

            #FIXME HARDCODED distorsions only along R are considered
            distortionNumeric_flata = vecFluctDistR_flata
            distortionNumeric_flatm = distortionNumeric_flata.reshape(-1, 1)

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
