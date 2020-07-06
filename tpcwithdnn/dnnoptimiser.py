import os
import sys
import random
from array import array
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from root_numpy import fill_hist
from ROOT import TH1F, TH2F, TFile, TCanvas, gPad # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT, TTree  # pylint: disable=import-error, no-name-in-module
from symmetrypadding3d import symmetryPadding3d
from machine_learning_hep.logger import get_logger
from fluctuationDataGenerator import fluctuationDataGenerator
from utilitiesdnn import UNet
from dataloader import loadtrain_test, loaddata_original
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use("Agg")


# pylint: disable=too-many-instance-attributes, too-many-statements, fixme, pointless-string-statement
# pylint: disable=logging-not-lazy
class DnnOptimiser:
    #Class Attribute
    species = "dnnoptimiser"

    def __init__(self, data_param, case):
        print(case)
        self.logger = get_logger()


        self.data_param = data_param
        self.dirmodel = self.data_param["dirmodel"]
        self.dirval = self.data_param["dirval"]
        self.diroutflattree = self.data_param["diroutflattree"]
        self.dirinput = self.data_param["dirinput"]
        self.grid_phi = self.data_param["grid_phi"]
        self.grid_z = self.data_param["grid_z"]
        self.grid_r = self.data_param["grid_r"]
        # prepare the dataset
        self.selopt_input = self.data_param["selopt_input"]
        self.selopt_output = self.data_param["selopt_output"]
        self.opt_train = self.data_param["opt_train"]
        self.opt_predout = self.data_param["opt_predout"]
        self.nameopt_predout = self.data_param["nameopt_predout"]
        self.dim_input = sum(self.opt_train)
        self.dim_output = sum(self.opt_predout)
        self.maxrandomfiles = self.data_param["maxrandomfiles"]
        self.rangeevent_train = self.data_param["rangeevent_train"]
        self.rangeevent_test = self.data_param["rangeevent_test"]
        self.rangeevent_apply = self.data_param["rangeevent_apply"]
        self.range_mean_index = self.data_param["range_mean_index"]
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

        self.lossfun = self.data_param["lossfun"]
        self.metrics = self.data_param["metrics"]
        self.adamlr = self.data_param["adamlr"]

        if not os.path.isdir("plots"):
            os.makedirs("plots")

        if not os.path.isdir(self.dirmodel):
            os.makedirs(self.dirmodel)

        if not os.path.isdir(self.dirval):
            os.makedirs(self.dirval)

        self.dirinput = self.dirinput + "/SC-%d-%d-%d/" % \
                (self.grid_z, self.grid_r, self.grid_phi)
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
        self.suffix = "%s_useSCMean%d_useSCFluc%d" % \
                (self.suffix, self.opt_train[0], self.opt_train[1])
        self.suffix = "%s_pred_doR%d_dophi%d_doz%d" % \
                (self.suffix, self.opt_predout[0], self.opt_predout[1], self.opt_predout[2])
        self.suffix_ds = "phi%d_r%d_z%d" % \
                (self.grid_phi, self.grid_r, self.grid_z)

        self.logger.info("DnnOptimizer::Init")
        self.logger.info("I am processing the configuration %s", self.suffix)
        if self.dim_output > 1:
            self.logger.fatal("YOU CAN PREDICT ONLY 1 DISTORSION. The sum of opt_predout == 1")
        self.logger.info("This is the list of inputs active for training")
        self.logger.info("(SCMean, SCFluctuations)=(%d, %d)"  % (self.opt_train[0],
                                                                 self.opt_train[1]))

        random.seed(1)
        self.indexmatrix_ev_mean = []
        indexmatrix_ev_mean_dummy = []
        for ievent in np.arange(self.maxrandomfiles):
            for imean in np.arange(self.range_mean_index[0], self.range_mean_index[1] + 1):
                indexmatrix_ev_mean_dummy.append([ievent, imean])

        self.indexmatrix_ev_mean = random.sample(indexmatrix_ev_mean_dummy, \
            self.maxrandomfiles * (self.range_mean_index[1] + 1 - self.range_mean_index[0]))

        self.indexmatrix_ev_mean_train = [self.indexmatrix_ev_mean[index] \
                for index in range(self.rangeevent_train[0], self.rangeevent_train[1])]
        self.indexmatrix_ev_mean_test = [self.indexmatrix_ev_mean[index] \
                for index in range(self.rangeevent_test[0], self.rangeevent_test[1])]
        self.indexmatrix_ev_mean_apply = [self.indexmatrix_ev_mean[index] \
                for index in range(self.rangeevent_apply[0], self.rangeevent_apply[1])]

        gROOT.SetStyle("Plain")
        gROOT.SetBatch()


    # pylint: disable=too-many-locals
    def dumpflattree(self):
        self.logger.warning("DO YOU REALLY WANT TO DO IT? IT TAKES TIME")
        namefileout = "%s/tree%s.root" % (self.diroutflattree, self.suffix_ds)
        myfile = TFile.Open(namefileout, "recreate")

        t = TTree('tvoxels', 'tree with histos')
        indexr = array('i', [0])
        indexphi = array('i', [0])
        indexz = array('i', [0])
        posr = array('f', [0])
        posphi = array('f', [0])
        posz = array('f', [0])
        evtid = array('i', [0])
        meanid = array('i', [0])
        randomid = array('i', [0])
        distmeanr = array('f', [0])
        distmeanrphi = array('f', [0])
        distmeanz = array('f', [0])
        distrndr = array('f', [0])
        distrndrphi = array('f', [0])
        distrndz = array('f', [0])
        t.Branch('indexr', indexr, 'indexr/I')
        t.Branch('indexphi', indexphi, 'indexphi/I')
        t.Branch('indexz', indexz, 'indexz/I')
        t.Branch('posr', posr, 'posr/F')
        t.Branch('posphi', posphi, 'posphi/F')
        t.Branch('posz', posz, 'posz/F')
        t.Branch('distmeanr', distmeanr, 'distmeanr/F')
        t.Branch('distmeanrphi', distmeanrphi, 'distmeanrphi/F')
        t.Branch('distmeanz', distmeanz, 'distmeanz/F')
        t.Branch('distrndr', distrndr, 'distrndr/F')
        t.Branch('distrndrphi', distrndrphi, 'distrndrphi/F')
        t.Branch('distrndz', distrndz, 'distrndz/F')
        t.Branch('evtid', evtid, 'evtid/I')
        t.Branch('meanid', meanid, 'meanid/I')
        t.Branch('randomid', randomid, 'randomid/I')

        for iexperiment in self.indexmatrix_ev_mean:
            print("processing event", iexperiment)
            indexev = iexperiment


            [vecRPos, vecPhiPos, vecZPos,
             _, _,
             vecMeanDistR, vecRandomDistR,
             vecMeanDistRPhi, vecRandomDistRPhi,
             vecMeanDistZ, vecRandomDistZ] = loaddata_original(self.dirinput, indexev)

            vecRPos_ = vecRPos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vecPhiPos_ = vecPhiPos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vecZPos_ = vecZPos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vecMeanDistR_ = vecMeanDistR.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vecRandomDistR_ = vecRandomDistR.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vecMeanDistRPhi_ = vecMeanDistRPhi.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vecRandomDistRPhi_ = vecRandomDistRPhi.reshape(self.grid_phi, self.grid_r,
                                                           self.grid_z*2)
            vecMeanDistZ_ = vecMeanDistZ.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vecRandomDistZ_ = vecRandomDistZ.reshape(self.grid_phi, self.grid_r, self.grid_z*2)

            for indexphi_ in range(self.grid_phi):
                for indexr_ in range(self.grid_r):
                    for indexz_ in range(self.grid_z*2):
                        indexphi[0] = indexphi_
                        indexr[0] = indexr_
                        indexz[0] = indexz_
                        posr[0] = vecRPos_[indexphi_][indexr_][indexz_]
                        posphi[0] = vecPhiPos_[indexphi_][indexr_][indexz_]
                        posz[0] = vecZPos_[indexphi_][indexr_][indexz_]
                        distmeanr[0] = vecMeanDistR_[indexphi_][indexr_][indexz_]
                        distmeanrphi[0] = vecMeanDistRPhi_[indexphi_][indexr_][indexz_]
                        distmeanz[0] = vecMeanDistZ_[indexphi_][indexr_][indexz_]
                        distrndr[0] = vecRandomDistR_[indexphi_][indexr_][indexz_]
                        distrndrphi[0] = vecRandomDistRPhi_[indexphi_][indexr_][indexz_]
                        distrndz[0] = vecRandomDistZ_[indexphi_][indexr_][indexz_]
                        evtid[0] = indexev[0]+ 10000*indexev[1]
                        meanid[0] = indexev[1]
                        randomid[0] = indexev[0]
                        t.Fill()
        myfile.Write()
        myfile.Close()
        print("Tree written in %s" % namefileout)


    def train(self):
        self.logger.info("DnnOptimizer::train")

        partition = {'train': self.indexmatrix_ev_mean_train,
                     'validation': self.indexmatrix_ev_mean_test}
        training_generator = fluctuationDataGenerator(partition['train'], **self.params)
        validation_generator = fluctuationDataGenerator(partition['validation'], **self.params)
        model = UNet((self.grid_phi, self.grid_r, self.grid_z, self.dim_input),
                     depth=self.depth, bathnorm=self.batch_normalization,
                     pool_type=self.pooling, start_ch=self.filters, dropout=self.dropout)
        model.compile(loss=self.lossfun, optimizer=Adam(lr=self.adamlr),
                      metrics=[self.metrics]) # Mean squared error
        model.summary()
        plot_model(model, to_file='plots/model_plot.png', show_shapes=True, show_layer_names=True)
        his = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=True,
                                  epochs=self.epochs, workers=1)
        plt.style.use("ggplot")
        plt.figure()
        plt.yscale('log')
        plt.plot(np.arange(0, self.epochs), his.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), his.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), his.history[self.metrics],
                 label="train_" + self.metrics)
        plt.plot(np.arange(0, self.epochs), his.history["val_" + self.metrics],
                 label="val_" + self.metrics)
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("plots/plot_%s.png" % self.suffix)

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

        myfile = TFile.Open("%s/output%s.root" % (self.dirval, self.suffix), "recreate")
        h_distallevents = TH2F("hdistallevents" + self.suffix, "", 500, -5, 5, 500, -5, 5)
        h_deltasallevents = TH1F("hdeltasallevents" + self.suffix, "", 1000, -1., 1.)
        h_deltasvsdistallevents = TH2F("hdeltasvsdistallevents" + self.suffix, "",
                                       500, -5.0, 5.0, 100, -0.5, 0.5)

        for iexperiment in self.indexmatrix_ev_mean_apply:
            indexev = iexperiment
            x_, y_ = loadtrain_test(self.dirinput, indexev, self.selopt_input, self.selopt_output,
                                    self.grid_r, self.grid_phi, self.grid_z,
                                    self.opt_train, self.opt_predout)
            x_single = np.empty((1, self.grid_phi, self.grid_r, self.grid_z, self.dim_input))
            y_single = np.empty((1, self.grid_phi, self.grid_r, self.grid_z, self.dim_output))
            x_single[0, :, :, :, :] = x_
            y_single[0, :, :, :, :] = y_

            distortionPredict_group = loaded_model.predict(x_single)
            distortionPredict_flatm = distortionPredict_group.reshape(-1, 1)
            distortionPredict_flata = distortionPredict_group.flatten()

            distortionNumeric_group = y_single
            distortionNumeric_flatm = distortionNumeric_group.reshape(-1, 1)
            distortionNumeric_flata = distortionNumeric_group.flatten()
            deltas_flata = (distortionPredict_flata - distortionNumeric_flata)
            deltas_flatm = (distortionPredict_flatm - distortionNumeric_flatm)

            h_dist = TH2F("hdistEv%d_Mean%d" % (iexperiment[0], iexperiment[1]) + self.suffix, \
                          "", 500, -5, 5, 500, -5, 5)
            h_deltasvsdist = TH2F("hdeltasvsdistEv%d_Mean%d" % (iexperiment[0], iexperiment[1]) + \
                                  self.suffix, "", 500, -5.0, 5.0, 100, -0.5, 0.5)
            h_deltas = TH1F("hdeltasEv%d_Mean%d" % (iexperiment[0], iexperiment[1]) \
                            + self.suffix, "", 1000, -1., 1.)
            fill_hist(h_distallevents, np.concatenate((distortionNumeric_flatm, \
                                distortionPredict_flatm), axis=1))
            fill_hist(h_dist, np.concatenate((distortionNumeric_flatm,
                                              distortionPredict_flatm), axis=1))
            fill_hist(h_deltas, deltas_flata)
            fill_hist(h_deltasallevents, deltas_flata)
            fill_hist(h_deltasvsdist,
                      np.concatenate((distortionNumeric_flatm, deltas_flatm), axis=1))
            fill_hist(h_deltasvsdistallevents,
                      np.concatenate((distortionNumeric_flatm, deltas_flatm), axis=1))
            prof = h_deltasvsdist.ProfileX()
            prof.SetName("profiledeltasvsdistEv%d_Mean%d" % \
                (iexperiment[0], iexperiment[1]) + self.suffix)
            h_dist.Write()
            h_deltas.Write()
            h_deltasvsdist.Write()
            prof.Write()

            h1tmp = h_deltasvsdist.ProjectionX("h1tmp")
            hStdDev = h1tmp.Clone("hStdDev_Ev%d_Mean%d" % \
                (iexperiment[0], iexperiment[1]) + self.suffix)
            hStdDev.Reset()
            hStdDev.SetXTitle("Numerical distortion fluctuation (cm)")
            hStdDev.SetYTitle("std.dev. of (Pred. - Num.) distortion fluctuation (cm)")
            nbin = int(hStdDev.GetNbinsX())
            for ibin in range(0, nbin):
                h1diff = h_deltasvsdist.ProjectionY("h1diff", ibin+1, ibin+1, "")
                stddev = h1diff.GetStdDev()
                stddev_err = h1diff.GetStdDevError()
                hStdDev.SetBinContent(ibin+1, stddev)
                hStdDev.SetBinError(ibin+1, stddev_err)
            hStdDev.Write()

        h_distallevents.Write()
        h_deltasallevents.Write()
        h_deltasvsdistallevents.Write()
        profallevents = h_deltasvsdistallevents.ProfileX()
        profallevents.SetName("profiledeltasvsdistallevents" + self.suffix)
        profallevents.Write()

        h1tmp = h_deltasvsdistallevents.ProjectionX("h1tmp")
        hStdDev_allevents = h1tmp.Clone("hStdDev_allevents" + self.suffix)
        hStdDev_allevents.Reset()
        hStdDev_allevents.SetXTitle("Numerical distortion fluctuation (cm)")
        hStdDev_allevents.SetYTitle("std.dev. of (Pred. - Num.) distortion fluctuation (cm)")
        nbin = int(hStdDev_allevents.GetNbinsX())
        for ibin in range(0, nbin):
            h1diff = h_deltasvsdistallevents.ProjectionY("h1diff", ibin+1, ibin+1, "")
            stddev = h1diff.GetStdDev()
            stddev_err = h1diff.GetStdDevError()
            hStdDev_allevents.SetBinContent(ibin+1, stddev)
            hStdDev_allevents.SetBinError(ibin+1, stddev_err)
        hStdDev_allevents.Write()

        myfile.Close()
        print("DONE APPLY")


    @staticmethod
    def plot_distorsion(h_dist, h_deltas, h_deltasvsdist, prof, suffix, namevar):
        cev = TCanvas("canvas" + suffix, "canvas" + suffix,
                      1400, 1000)
        cev.Divide(2, 2)
        cev.cd(1)
        h_dist.GetXaxis().SetTitle("Numeric %s distortion fluctuation (cm)" % namevar)
        h_dist.GetYaxis().SetTitle("Predicted distortion fluctuation (cm)")
        h_dist.Draw("colz")
        cev.cd(2)
        gPad.SetLogy()
        h_deltasvsdist.GetXaxis().SetTitle("Numeric %s distorsion fluctuation (cm)" % namevar)
        h_deltasvsdist.GetYaxis().SetTitle("Entries")
        h_deltasvsdist.ProjectionX().Draw()
        cev.cd(3)
        gPad.SetLogy()
        h_deltas.GetXaxis().SetTitle("(Predicted - Numeric) %s distortion fluctuation (cm)"
                                     % namevar)
        h_deltas.GetYaxis().SetTitle("Entries")
        h_deltas.Draw()
        cev.cd(4)
        prof.GetYaxis().SetTitle("(Predicted - Numeric) %s distortion fluctuation (cm)" % namevar)
        prof.GetXaxis().SetTitle("Numeric %s distortion fluctuation (cm)" % namevar)
        prof.Draw()
        #cev.cd(5)
        #h_deltasvsdist.GetXaxis().SetTitle("Numeric R distorsion (cm)")
        #h_deltasvsdist.GetYaxis().SetTitle("(Predicted - Numeric) R distorsion (cm)")
        #h_deltasvsdist.Draw("colz")
        cev.SaveAs("plots/canvas_%s.pdf" % (suffix))

    # pylint: disable=no-self-use
    def plot(self):
        namevariable = None
        for iname in self.opt_predout:
            if self.opt_predout[iname] == 1:
                namevariable = self.nameopt_predout[iname]

        myfile = TFile.Open("%s/output%s.root" % (self.dirval, self.suffix), "open")
        h_distallevents = myfile.Get("hdistallevents" + self.suffix)
        hdeltasallevents = myfile.Get("hdeltasallevents" + self.suffix)
        h_deltasvsdistallevents = myfile.Get("hdeltasvsdistallevents" + self.suffix)
        profiledeltasvsdistallevents = myfile.Get("profiledeltasvsdistallevents" + self.suffix)
        self.plot_distorsion(h_distallevents, hdeltasallevents, h_deltasvsdistallevents,
                             profiledeltasvsdistallevents, self.suffix, namevariable)

        counter = 0
        for iexperiment in self.indexmatrix_ev_mean_apply:
            suffix_ = "Ev%d_Mean%d%s" % (iexperiment[0], iexperiment[1], self.suffix)
            h_dist = myfile.Get("hdist%s" % suffix_)
            h_deltas = myfile.Get("hdeltas%s" % suffix_)
            h_deltasvsdist = myfile.Get("hdeltasvsdist%s" % suffix_)
            prof = myfile.Get("profiledeltasvsdist%s" % suffix_)
            self.plot_distorsion(h_dist, h_deltas, h_deltasvsdist, prof,
                                 suffix_, namevariable)
            counter = counter + 1
            if counter > 100:
                sys.exit()



    # pylint: disable=no-self-use
    def gridsearch(self):
        print("GRID SEARCH NOT YET IMPLEMENTED")
