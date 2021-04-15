# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
import pickle
import numpy as np
from xgboost import XGBRFRegressor

from root_numpy import fill_hist # pylint: disable=import-error
from ROOT import TFile # pylint: disable=import-error, no-name-in-module

import tpcwithdnn.plot_utils as plot_utils
from tpcwithdnn.optimiser import Optimiser
from tpcwithdnn.data_loader import load_train_apply_idc

class XGBoostOptimiser(Optimiser):
    name = "xgboost"

    def __init__(self, config):
        super().__init__(config)
        self.config.logger.info("XGBoostOptimiser::Init")
        self.model = XGBRFRegressor(verbosity=1, n_gpus=1, **(self.config.params))

    def train(self):
        self.config.logger.info("XGBoostOptimiser::train")
        inputs, exp_outputs = self.get_train_apply_data_("train")
        self.model.fit(inputs, exp_outputs)
        self.save_model_(self.model)

    def apply(self):
        self.config.logger.info("XGBoostOptimiser::apply, input size: %d", self.config.dim_input)
        self.load_model_()
        inputs, exp_outputs = self.get_train_apply_data_("train")
        pred_outputs = self.model.predict(inputs)
        self.plot_apply_(exp_outputs, pred_outputs)
        self.config.logger.info("Done apply")

    def search_grid(self):
        raise NotImplementedError("Search grid method not implemented yet")

    def save_model_(self, model):
        # Snapshot - can be used for further training
        out_filename = "%s/xgbmodel_%s_nEv%d_snap.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        pickle.dump(model, open(out_filename, 'wb'), protocol=4)
        out_filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        model.save_model(out_filename)

    def load_model_(self):
        # Loading a snapshot
        #filename = "%s/xgbmodel_%s_nEv%d_snap.json" %\
        #        (self.config.dirmodel, self.config.suffix, self.config.train_events)
        #loaded_model = pickle.load(open(filename, 'rb'))
        filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        self.model.load_model(filename)

    def get_train_apply_data_(self, partition):
        inputs = []
        exp_outputs = []
        for indexev in self.config.partition[partition]:
            inputs_single, exp_outputs_single = load_train_apply_idc(self.config.dirinput_train,
                                                       indexev, self.config.input_z_range,
                                                       self.config.output_z_range,
                                                       self.config.opt_predout)
            inputs.append(inputs_single)
            exp_outputs.append(exp_outputs_single)
        print("Different inputs: {} shape of first: {}"
              .format(len(inputs), inputs[0].shape))
        print("Different outputs: {} shape of first: {}"
              .format(len(exp_outputs), exp_outputs[0].shape))
        inputs = np.concatenate(inputs)
        exp_outputs = np.concatenate(exp_outputs)
        print("Inputs concatenated: {} outputs: {}".format(inputs.shape, exp_outputs.shape))
        return inputs, exp_outputs

    def plot_apply_(self, exp_outputs, pred_outputs):
        myfile = TFile.Open("%s/output_%s_nEv%d.root" % \
                            (self.config.dirval, self.config.suffix, self.config.train_events),
                            "recreate")
        h_dist_all_events, h_deltas_all_events, h_deltas_vs_dist_all_events =\
                plot_utils.create_apply_histos(self.config, self.config.suffix, infix="all_events_")
        distortion_numeric_flat_m, distortion_predict_flat_m, deltas_flat_a, deltas_flat_m =\
            plot_utils.get_apply_results_single_event(pred_outputs, exp_outputs)

        fill_hist(h_dist_all_events, np.concatenate((distortion_numeric_flat_m, \
                                                     distortion_predict_flat_m), axis=1))
        fill_hist(h_deltas_all_events, deltas_flat_a)
        fill_hist(h_deltas_vs_dist_all_events,
                  np.concatenate((distortion_numeric_flat_m, deltas_flat_m), axis=1))

        h_dist_all_events.Write()
        h_deltas_all_events.Write()
        h_deltas_vs_dist_all_events.Write()
        prof_all_events = h_deltas_vs_dist_all_events.ProfileX()
        prof_all_events.SetName("%s_all_events_%s" % (self.config.profile_name,
                                                      self.config.suffix))
        prof_all_events.Write()
        plot_utils.fill_std_dev_apply_hist(h_deltas_vs_dist_all_events, self.config.h_std_dev_name,
                                           self.config.suffix, "all_events_")

        myfile.Close()
