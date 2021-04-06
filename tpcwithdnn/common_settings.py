"""
Storing user settings from config files.
"""
# pylint: disable=missing-function-docstring, missing-class-docstring
# pylint: disable=too-many-statements, too-many-instance-attributes
# pylint: disable=too-few-public-methods
import os

import numpy as np

from tpcwithdnn.logger import get_logger
from tpcwithdnn.data_loader import get_event_mean_indices

class CommonSettings:
    def __init__(self, data_param, case):
        self.logger = get_logger()
        self.logger.info("CommonSettings::Init\nCase: %s", case)

        # Dataset config
        self.grid_phi = data_param["grid_phi"]
        self.grid_z = data_param["grid_z"]
        self.grid_r = data_param["grid_r"]

        self.input_z_range = data_param["input_z_range"]
        self.output_z_range = data_param["output_z_range"]
        self.opt_train = data_param["opt_train"]
        self.opt_predout = data_param["opt_predout"]
        self.nameopt_predout = data_param["nameopt_predout"]
        self.dim_input = sum(self.opt_train)
        self.dim_output = sum(self.opt_predout)

        self.validate_model = data_param["validate_model"]
        self.use_scaler = data_param["use_scaler"]

        # Directories
        self.dirmodel = data_param["dirmodel"]
        self.dirval = data_param["dirval"]
        self.dirplots = data_param["dirplots"]
        self.diroutflattree = data_param["diroutflattree"]
        self.dirouthistograms = data_param["dirouthistograms"]
        train_dir = data_param["dirinput_bias"] if data_param["train_bias"] \
                    else data_param["dirinput_nobias"]
        test_dir = data_param["dirinput_bias"] if data_param["test_bias"] \
                    else data_param["dirinput_nobias"]
        apply_dir = data_param["dirinput_bias"] if data_param["apply_bias"] \
                    else data_param["dirinput_nobias"]
        self.dirinput_train = "%s/SC-%d-%d-%d" % \
                              (train_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_test = "%s/SC-%d-%d-%d" % \
                             (test_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_apply = "%s/SC-%d-%d-%d" % \
                              (apply_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_val = "%s/SC-%d-%d-%d" % \
                            (data_param["dirinput_nobias"], self.grid_z, self.grid_r, self.grid_phi)

        # DNN config
        self.filters = data_param["filters"]
        self.pooling = data_param["pooling"]
        self.depth = data_param["depth"]
        self.batch_normalization = data_param["batch_normalization"]
        self.dropout = data_param["dropout"]

        # For optimiser only
        self.batch_size = data_param["batch_size"]
        self.shuffle = data_param["shuffle"]
        self.epochs = data_param["epochs"]
        self.lossfun = data_param["lossfun"]
        if data_param["metrics"] == "rmse":
            self.metrics = "root_mean_squared_error"
        else:
            self.metrics = data_param["metrics"]
        self.adamlr = data_param["adamlr"]

        self.params = {'phi_slice': self.grid_phi,
                       'r_row' : self.grid_r,
                       'z_col' : self.grid_z,
                       'batch_size': self.batch_size,
                       'shuffle': self.shuffle,
                       'opt_train' : self.opt_train,
                       'opt_predout' : self.opt_predout,
                       'input_z_range' : self.input_z_range,
                       'output_z_range' : self.output_z_range,
                       'use_scaler': self.use_scaler}

        self.suffix = "phi%d_r%d_z%d_filter%d_poo%d_drop%.2f_depth%d_batch%d_scaler%d" % \
                (self.grid_phi, self.grid_r, self.grid_z, self.filters, self.pooling,
                 self.dropout, self.depth, self.batch_normalization, self.use_scaler)
        self.suffix = "%s_useSCMean%d_useSCFluc%d" % \
                (self.suffix, self.opt_train[0], self.opt_train[1])
        self.suffix = "%s_pred_doR%d_dophi%d_doz%d" % \
                (self.suffix, self.opt_predout[0], self.opt_predout[1], self.opt_predout[2])
        self.suffix = "%s_input_z%.1f-%.1f" % \
                (self.suffix, self.input_z_range[0], self.input_z_range[1])
        self.suffix = "%s_output_z%.1f-%.1f" % \
                (self.suffix, self.output_z_range[0], self.output_z_range[1])
        self.suffix_ds = "phi%d_r%d_z%d" % \
                (self.grid_phi, self.grid_r, self.grid_z)

        self.logger.info("I am processing the configuration %s", self.suffix)
        if self.dim_output > 1:
            self.logger.fatal("YOU CAN PREDICT ONLY 1 DISTORSION. The sum of opt_predout == 1")
        self.logger.info("Inputs active for training: (SCMean, SCFluctuations)=(%d, %d)",
                         self.opt_train[0], self.opt_train[1])

        # Parameters for getting input indices
        self.maxrandomfiles = data_param["maxrandomfiles"]
        self.tree_events = data_param["tree_events"]
        self.part_inds = None
        self.use_partition = data_param["use_partition"]
        self.range_mean_index = data_param["range_mean_index"]
        self.indices_events_means = None
        self.partition = None
        self.total_events = 0
        self.train_events = 0
        self.test_events = 0
        self.apply_events = 0

        if not os.path.isdir(self.dirmodel):
            os.makedirs(self.dirmodel)
        if not os.path.isdir(self.dirval):
            os.makedirs(self.dirval)
        if not os.path.isdir(self.dirplots):
            os.makedirs(self.dirplots)

        if not os.path.isdir(self.diroutflattree):
            os.makedirs(self.diroutflattree)
        if not os.path.isdir("%s/%s" % (self.diroutflattree, self.suffix)):
            os.makedirs("%s/%s" % (self.diroutflattree, self.suffix))
        if not os.path.isdir("%s/%s" % (self.dirouthistograms, self.suffix)):
            os.makedirs("%s/%s" % (self.dirouthistograms, self.suffix))

    def set_ranges(self, ranges, total_events, train_events, test_events, apply_events):
        self.total_events = total_events
        self.train_events = train_events
        self.test_events = test_events
        self.apply_events = apply_events

        self.indices_events_means, self.partition = get_event_mean_indices(
            self.maxrandomfiles, self.range_mean_index, ranges)

        part_inds = None
        for part in self.partition:
            events_inds = np.array(self.partition[part])
            events_file = "%s/events_%s_%s_nEv%d.csv" % \
                          (self.dirmodel, part, self.suffix, self.train_events)
            np.savetxt(events_file, events_inds, delimiter=",", fmt="%d")
            if self.use_partition != "random" and part == self.use_partition:
                part_inds = events_inds
                self.part_inds = part_inds[(part_inds[:,1] == 0) | (part_inds[:,1] == 9) | \
                                             (part_inds[:,1] == 18)]

        self.logger.info("Processing %d events", self.total_events)
