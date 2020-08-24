# pylint: disable=too-many-instance-attributes, too-many-statements, too-many-arguments, fixme
# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=protected-access, too-many-locals
import os
import datetime

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model

from root_numpy import fill_hist

from ROOT import TH1F, TH2F, TFile, TCanvas, TLegend, TPaveText, gPad # pylint: disable=import-error, no-name-in-module
from ROOT import gStyle, kWhite, kBlue, kGreen, kRed, kCyan, kOrange, kMagenta # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT  # pylint: disable=import-error, no-name-in-module

from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d
from tpcwithdnn.logger import get_logger
from tpcwithdnn.fluctuation_data_generator import FluctuationDataGenerator
from tpcwithdnn.utilities_dnn import u_net
from tpcwithdnn.data_loader import load_train_apply, get_event_mean_indices

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use("Agg")

class DnnOptimiser:
    # Class Attribute
    species = "dnnoptimiser"

    h_dist_name = "h_dist"
    h_deltas_name = "h_deltas"
    h_deltas_vs_dist_name = "h_deltas_vs_dist"
    profile_name = "profile_deltas_vs_dist"
    h_std_dev_name = "h_std_dev"

    def __init__(self, data_param, case):
        self.logger = get_logger()
        self.logger.info("DnnOptimizer::Init\nCase: %s", case)

        # Dataset config
        self.grid_phi = data_param["grid_phi"]
        self.grid_z = data_param["grid_z"]
        self.grid_r = data_param["grid_r"]

        self.selopt_input = data_param["selopt_input"]
        self.selopt_output = data_param["selopt_output"]
        self.opt_train = data_param["opt_train"]
        self.opt_predout = data_param["opt_predout"]
        self.nameopt_predout = data_param["nameopt_predout"]
        self.dim_input = sum(self.opt_train)
        self.dim_output = sum(self.opt_predout)
        self.use_scaler = data_param["use_scaler"]

        # Directories
        self.dirmodel = data_param["dirmodel"]
        self.dirval = data_param["dirval"]
        train_dir = data_param["dirinput_bias"] if data_param["train_bias"] \
                    else data_param["dirinput_nobias"]
        test_dir = data_param["dirinput_bias"] if data_param["test_bias"] \
                    else data_param["dirinput_nobias"]
        apply_dir = data_param["dirinput_bias"] if data_param["apply_bias"] \
                    else data_param["dirinput_nobias"]
        self.dirinput_train = "%s/SC-%d-%d-%d/" % \
                              (train_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_test = "%s/SC-%d-%d-%d/" % \
                             (test_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_apply = "%s/SC-%d-%d-%d/" % \
                              (apply_dir, self.grid_z, self.grid_r, self.grid_phi)

        # DNN config
        self.filters = data_param["filters"]
        self.pooling = data_param["pooling"]
        self.batch_size = data_param["batch_size"]
        self.shuffle = data_param["shuffle"]
        self.depth = data_param["depth"]
        self.batch_normalization = data_param["batch_normalization"]
        self.dropout = data_param["dropout"]
        self.epochs = data_param["epochs"]
        self.lossfun = data_param["lossfun"]
        self.metrics = data_param["metrics"]
        self.adamlr = data_param["adamlr"]

        self.params = {'phi_slice': self.grid_phi,
                       'r_row' : self.grid_r,
                       'z_col' : self.grid_z,
                       'batch_size': self.batch_size,
                       'shuffle': self.shuffle,
                       'opt_train' : self.opt_train,
                       'opt_predout' : self.opt_predout,
                       'selopt_input' : self.selopt_input,
                       'selopt_output' : self.selopt_output,
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

        if not os.path.isdir("plots"):
            os.makedirs("plots")

        if not os.path.isdir(self.dirmodel):
            os.makedirs(self.dirmodel)

        if not os.path.isdir(self.dirval):
            os.makedirs(self.dirval)

        self.logger.info("I am processing the configuration %s", self.suffix)
        if self.dim_output > 1:
            self.logger.fatal("YOU CAN PREDICT ONLY 1 DISTORSION. The sum of opt_predout == 1")
        self.logger.info("Inputs active for training: (SCMean, SCFluctuations)=(%d, %d)",
                         self.opt_train[0], self.opt_train[1])

        # Parameters for getting input indices
        self.maxrandomfiles = data_param["maxrandomfiles"]
        self.range_mean_index = data_param["range_mean_index"]
        self.indices_events_means = None
        self.partition = None
        self.total_events = 0

        gROOT.SetStyle("Plain")
        gROOT.SetBatch()


    def train(self):
        self.logger.info("DnnOptimizer::train")

        training_generator = FluctuationDataGenerator(self.partition['train'],
                                                      data_dir=self.dirinput_train, **self.params)
        validation_generator = FluctuationDataGenerator(self.partition['validation'],
                                                        data_dir=self.dirinput_test, **self.params)
        model = u_net((self.grid_phi, self.grid_r, self.grid_z, self.dim_input),
                      depth=self.depth, batchnorm=self.batch_normalization,
                      pool_type=self.pooling, start_channels=self.filters, dropout=self.dropout)
        model.compile(loss=self.lossfun, optimizer=Adam(lr=self.adamlr),
                      metrics=[self.metrics]) # Mean squared error

        model.summary()
        plot_model(model, to_file='plots/model_%s_nEv%d.png' % (self.suffix, self.total_events),
                   show_shapes=True, show_layer_names=True)

        log_dir = "logs/" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        model._get_distribution_strategy = lambda: None
        his = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=True,
                                  epochs=self.epochs, workers=1, callbacks=[tensorboard_callback])

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
        plt.savefig("plots/plot_%s_nEv%d.png" % (self.suffix, self.total_events))

        model_json = model.to_json()
        with open("%s/model_%s_nEv%d.json" % (self.dirmodel, self.suffix, self.total_events), "w") \
            as json_file:
            json_file.write(model_json)
        model.save_weights("%s/model_%s_nEv%d.h5" % (self.dirmodel, self.suffix, self.total_events))
        self.logger.info("Saved trained model to disk")


    def apply(self):
        self.logger.info("DnnOptimizer::apply, input size: %d", self.dim_input)

        json_file = open("%s/model_%s_nEv%d.json" % \
                         (self.dirmodel, self.suffix, self.total_events), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = \
            model_from_json(loaded_model_json, {'SymmetryPadding3d' : SymmetryPadding3d})
        loaded_model.load_weights("%s/model_%s_nEv%d.h5" % \
                                  (self.dirmodel, self.suffix, self.total_events))

        myfile = TFile.Open("%s/output_%s_nEv%d.root" % \
                            (self.dirval, self.suffix, self.total_events), "recreate")
        h_dist_all_events = TH2F("%s_all_events_%s" % (self.h_dist_name, self.suffix),
                                 "", 500, -5, 5, 500, -5, 5)
        h_deltas_all_events = TH1F("%s_all_events_%s" % (self.h_deltas_name, self.suffix),
                                   "", 1000, -1., 1.)
        h_deltas_vs_dist_all_events = TH2F("%s_all_events_%s" % \
                                           (self.h_deltas_vs_dist_name, self.suffix),
                                           "", 500, -5.0, 5.0, 100, -0.5, 0.5)

        for iexperiment in self.partition['apply']:
            indexev = iexperiment
            inputs_, exp_outputs_ = load_train_apply(self.dirinput_apply, indexev,
                                                     self.selopt_input, self.selopt_output,
                                                     self.grid_r, self.grid_phi, self.grid_z,
                                                     self.opt_train, self.opt_predout)
            inputs_single = np.empty((1, self.grid_phi, self.grid_r, self.grid_z, self.dim_input))
            exp_outputs_single = np.empty((1, self.grid_phi, self.grid_r,
                                           self.grid_z, self.dim_output))
            inputs_single[0, :, :, :, :] = inputs_
            exp_outputs_single[0, :, :, :, :] = exp_outputs_

            distortion_predict_group = loaded_model.predict(inputs_single)
            distortion_predict_flat_m = distortion_predict_group.reshape(-1, 1)
            distortion_predict_flat_a = distortion_predict_group.flatten()

            distortion_numeric_group = exp_outputs_single
            distortion_numeric_flat_m = distortion_numeric_group.reshape(-1, 1)
            distortion_numeric_flat_a = distortion_numeric_group.flatten()
            deltas_flat_a = (distortion_predict_flat_a - distortion_numeric_flat_a)
            deltas_flat_m = (distortion_predict_flat_m - distortion_numeric_flat_m)

            h_suffix = "Ev%d_Mean%d_%s" % (iexperiment[0], iexperiment[1], self.suffix)
            h_dist = TH2F("%s_%s" % (self.h_dist_name, h_suffix), "", 500, -5, 5, 500, -5, 5)
            h_deltas = TH1F("%s_%s" % (self.h_deltas_name, h_suffix), "", 1000, -1., 1.)
            h_deltas_vs_dist = TH2F("%s_%s" % (self.h_deltas_vs_dist_name, h_suffix), "",
                                    500, -5.0, 5.0, 100, -0.5, 0.5)

            fill_hist(h_dist_all_events, np.concatenate((distortion_numeric_flat_m, \
                                distortion_predict_flat_m), axis=1))
            fill_hist(h_dist, np.concatenate((distortion_numeric_flat_m,
                                              distortion_predict_flat_m), axis=1))
            fill_hist(h_deltas, deltas_flat_a)
            fill_hist(h_deltas_all_events, deltas_flat_a)
            fill_hist(h_deltas_vs_dist,
                      np.concatenate((distortion_numeric_flat_m, deltas_flat_m), axis=1))
            fill_hist(h_deltas_vs_dist_all_events,
                      np.concatenate((distortion_numeric_flat_m, deltas_flat_m), axis=1))

            prof = h_deltas_vs_dist.ProfileX()
            prof.SetName("%s_%s" % (self.profile_name, h_suffix))

            h_dist.Write()
            h_deltas.Write()
            h_deltas_vs_dist.Write()
            prof.Write()

            h1tmp = h_deltas_vs_dist.ProjectionX("h1tmp")
            h_std_dev = h1tmp.Clone("%s_%s" % (self.h_std_dev_name, h_suffix))
            h_std_dev.Reset()
            h_std_dev.SetXTitle("Numerical distortion fluctuation (cm)")
            h_std_dev.SetYTitle("std.dev. of (Pred. - Num.) distortion fluctuation (cm)")
            nbin = int(h_std_dev.GetNbinsX())
            for ibin in range(0, nbin):
                h1diff = h_deltas_vs_dist.ProjectionY("h1diff", ibin+1, ibin+1, "")
                stddev = h1diff.GetStdDev()
                stddev_err = h1diff.GetStdDevError()
                h_std_dev.SetBinContent(ibin+1, stddev)
                h_std_dev.SetBinError(ibin+1, stddev_err)
            h_std_dev.Write()

        h_dist_all_events.Write()
        h_deltas_all_events.Write()
        h_deltas_vs_dist_all_events.Write()
        prof_all_events = h_deltas_vs_dist_all_events.ProfileX()
        prof_all_events.SetName("%s_all_events_%s" % (self.profile_name, self.suffix))
        prof_all_events.Write()

        h1tmp = h_deltas_vs_dist_all_events.ProjectionX("h1tmp")
        h_std_dev_all_events = h1tmp.Clone("%s_all_events_%s" % (self.h_std_dev_name, self.suffix))
        h_std_dev_all_events.Reset()
        h_std_dev_all_events.SetXTitle("Numerical distortion fluctuation (cm)")
        h_std_dev_all_events.SetYTitle("std.dev. of (Pred. - Num.) distortion fluctuation (cm)")
        nbin = int(h_std_dev_all_events.GetNbinsX())
        for ibin in range(0, nbin):
            h1diff = h_deltas_vs_dist_all_events.ProjectionY("h1diff", ibin+1, ibin+1, "")
            stddev = h1diff.GetStdDev()
            stddev_err = h1diff.GetStdDevError()
            h_std_dev_all_events.SetBinContent(ibin+1, stddev)
            h_std_dev_all_events.SetBinError(ibin+1, stddev_err)
        h_std_dev_all_events.Write()

        myfile.Close()
        self.logger.info("Done apply")


    @staticmethod
    def plot_distorsion(h_dist, h_deltas, h_deltas_vs_dist, prof, suffix, opt_name, total_events):
        cev = TCanvas("canvas_%s_nEv%d_%s" % (suffix, total_events, opt_name),
                      "canvas_%s_nEv%d_%s" % (suffix, total_events, opt_name),
                      1400, 1000)
        cev.Divide(2, 2)
        cev.cd(1)
        h_dist.GetXaxis().SetTitle("Numeric %s distortion fluctuation (cm)" % opt_name)
        h_dist.GetYaxis().SetTitle("Predicted distortion fluctuation (cm)")
        h_dist.Draw("colz")
        cev.cd(2)
        gPad.SetLogy()
        h_deltas_vs_dist.GetXaxis().SetTitle("Numeric %s distorsion fluctuation (cm)" % opt_name)
        h_deltas_vs_dist.ProjectionX().Draw()
        h_deltas_vs_dist.GetYaxis().SetTitle("Entries")
        cev.cd(3)
        gPad.SetLogy()
        h_deltas.GetXaxis().SetTitle("(Predicted - Numeric) %s distortion fluctuation (cm)"
                                     % opt_name)
        h_deltas.GetYaxis().SetTitle("Entries")
        h_deltas.Draw()
        cev.cd(4)
        prof.GetYaxis().SetTitle("(Predicted - Numeric) %s distortion fluctuation (cm)" % opt_name)
        prof.GetXaxis().SetTitle("Numeric %s distortion fluctuation (cm)" % opt_name)
        prof.Draw()
        #cev.cd(5)
        #h_deltas_vs_dist.GetXaxis().SetTitle("Numeric R distorsion (cm)")
        #h_deltas_vs_dist.GetYaxis().SetTitle("(Predicted - Numeric) R distorsion (cm)")
        #h_deltas_vs_dist.Draw("colz")
        cev.SaveAs("plots/canvas_%s_nEv%d.pdf" % (suffix, total_events))

    def plot(self):
        self.logger.info("DnnOptimizer::plot")
        for iname, opt in enumerate(self.opt_predout):
            if opt == 1:
                opt_name = self.nameopt_predout[iname]

                myfile = TFile.Open("%s/output_%s_nEv%d.root" % \
                                    (self.dirval, self.suffix, self.total_events), "open")
                h_dist_all_events = myfile.Get("%s_all_events_%s" % (self.h_dist_name, self.suffix))
                h_deltas_all_events = myfile.Get("%s_all_events_%s" % \
                                                 (self.h_deltas_name, self.suffix))
                h_deltas_vs_dist_all_events = myfile.Get("%s_all_events_%s" % \
                                                         (self.h_deltas_vs_dist_name, self.suffix))
                profile_deltas_vs_dist_all_events = \
                    myfile.Get("%s_all_events_%s" % (self.profile_name, self.suffix))
                self.plot_distorsion(h_dist_all_events, h_deltas_all_events,
                                     h_deltas_vs_dist_all_events, profile_deltas_vs_dist_all_events,
                                     self.suffix, opt_name, self.total_events)

                counter = 0
                for iexperiment in self.partition['apply']:
                    h_suffix = "Ev%d_Mean%d_%s" % (iexperiment[0], iexperiment[1], self.suffix)
                    h_dist = myfile.Get("%s_%s" % (self.h_dist_name, h_suffix))
                    h_deltas = myfile.Get("%s_%s" % (self.h_deltas_name, h_suffix))
                    h_deltas_vs_dist = myfile.Get("%s_%s" % (self.h_deltas_vs_dist_name, h_suffix))
                    profile = myfile.Get("%s_%s" % (self.profile_name, h_suffix))
                    self.plot_distorsion(h_dist, h_deltas, h_deltas_vs_dist, profile,
                                         h_suffix, opt_name, self.total_events)
                    counter = counter + 1
                    if counter > 100:
                        return


    # pylint: disable=no-self-use
    def gridsearch(self):
        self.logger.warning("Grid search not yet implemented")


    def setup_canvas(self, hist_name, opt_name, x_label, y_label):
        full_name = "%s_canvas_%s_%s" % (hist_name, self.suffix, opt_name)
        canvas = TCanvas(full_name, full_name, 0, 0, 800, 800)
        canvas.SetMargin(0.12, 0.05, 0.12, 0.05)
        canvas.SetTicks(1, 1)

        frame = canvas.DrawFrame(-5, -0.5, +5, +0.5)
        frame.GetXaxis().SetTitle(x_label)
        frame.GetYaxis().SetTitle(y_label)
        frame.GetXaxis().SetTitleOffset(1.5)
        frame.GetYaxis().SetTitleOffset(1.5)
        frame.GetXaxis().CenterTitle(True)
        frame.GetYaxis().CenterTitle(True)
        frame.GetXaxis().SetTitleSize(0.035)
        frame.GetYaxis().SetTitleSize(0.035)
        frame.GetXaxis().SetLabelSize(0.035)
        frame.GetYaxis().SetLabelSize(0.035)

        leg = TLegend(0.3, 0.65, 0.65, 0.8)
        leg.SetBorderSize(0)
        leg.SetTextSize(0.03)

        return canvas, frame, leg


    def save_canvas(self, canvas, frame, prefix, func_name, file_formats):
        file_name = "%s_wide_%s_%s" % (prefix, func_name, self.suffix)
        for file_format in file_formats:
            canvas.SaveAs("%s.%s" % (file_name, file_format))
        frame.GetYaxis().SetRangeUser(-0.05, +0.05)
        file_name = "%s_zoom_%s_%s" % (prefix, func_name, self.suffix)
        for file_format in file_formats:
            canvas.SaveAs("%s.%s" % (file_name, file_format))


    def add_desc_to_canvas(self):
        txt1 = TPaveText(0.15, 0.8, 0.4, 0.92, "NDC")
        txt1.SetFillColor(kWhite)
        txt1.SetFillStyle(0)
        txt1.SetBorderSize(0)
        txt1.SetTextAlign(12) # middle,left
        txt1.SetTextFont(42) # helvetica
        txt1.SetTextSize(0.04)
        txt1.AddText("#varphi slice = %d, r slice = %d, z slice = %d" % \
                     (self.grid_phi, self.grid_r, self.grid_z))
        if self.opt_train[0] == 1 and self.opt_train[1] == 1:
            txt1.AddText("inputs: #rho_{SC} - <#rho_{SC}>, <#rho_{SC}>")
        elif self.opt_train[1] == 1:
            txt1.AddText("inputs: #rho_{SC} - <#rho_{SC}>")
        txt1.Draw()


    def draw_multievent_hist(self, events_counts, func_label, hist_name, source_hist):
        gStyle.SetOptStat(0)
        gStyle.SetOptTitle(0)

        file_formats = ["pdf"]
        # file_formats = ["png", "eps", "pdf"]
        var_labels = ["dr", "rd#varphi", "dz"]
        colors = [kBlue+1, kGreen+2, kRed+1, kCyan+2, kOrange+7, kMagenta+2]

        for iname, opt in enumerate(self.opt_predout):
            if opt == 1:
                opt_name = self.nameopt_predout[iname]
                var_label = var_labels[iname]

                x_label = "numerical fluctuation (cm), %s" % var_label
                y_label = "%s of (pred. - num.) in %d apply events (cm), %s" % \
                          (func_label, events_counts[0][2], var_label)
                canvas, frame, leg = self.setup_canvas(hist_name, opt_name, x_label, y_label)

                for i, (train_events, _, _, total_events) in enumerate(events_counts):
                    filename = "%s/output_%s_nEv%d.root" % (self.dirval, self.suffix, total_events)
                    self.logger.info("Reading %s...", filename)

                    root_file = TFile.Open(filename, "read")
                    hist = root_file.Get("%s_all_events_%s" % (source_hist, self.suffix))
                    hist.SetDirectory(0)
                    hist.Draw("same")
                    hist.SetMarkerStyle(20)
                    hist.SetMarkerColor(colors[i])
                    hist.SetLineColor(colors[i])

                    # train_events_k = train_events / 1000
                    leg.AddEntry(hist, "N_{ev}^{training} = %d" % train_events, "LP")
                    root_file.Close()

                leg.Draw()
                self.add_desc_to_canvas()
                self.save_canvas(canvas, frame, "20200803", hist_name, file_formats)


    def draw_profile(self, events_counts):
        self.draw_multievent_hist(events_counts, "mean", "profile", self.profile_name)


    def draw_std_dev(self, events_counts):
        self.draw_multievent_hist(events_counts, "std dev", "std_dev", self.h_std_dev_name)


    def set_ranges(self, ranges, total_events):
        self.total_events = total_events

        self.indices_events_means, self.partition = get_event_mean_indices(
            self.maxrandomfiles, self.range_mean_index, ranges)
        self.logger.info("Processing %d events", self.total_events)
