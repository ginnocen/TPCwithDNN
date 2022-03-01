"""
User settings from the config_model_parameters.yml
"""
# pylint: disable=too-many-instance-attributes, too-few-public-methods, unused-private-member
import os

import numpy as np

from tpcwithdnn.logger import get_logger
from tpcwithdnn.data_loader import get_event_mean_indices

class Singleton(type):
    """
    Singleton type - there will be always one instance of the settings in the whole program.
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        """
        Create new instance if it was not called yet, otherwise, return the existing instance.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class CommonSettings:
    """
    A class to store the common configuration (first section of the configuration file).
    """
    __metaclass__ = Singleton
    name = "common"

    h_dist_name = "h_dist"
    h_deltas_name = "h_deltas"
    h_deltas_vs_dist_name = "h_deltas_vs_dist"
    profile_name = "profile_deltas_vs_dist"
    h_std_dev_name = "h_std_dev"

    def __init__(self, data_param):
        """
        Read and store the parameters from the file.

        :param dict data_param: dictionary of values read from the config file
        """
        self.logger = get_logger()

        # Dataset config
        self.grid_phi = data_param["grid_phi"]
        self.grid_z = data_param["grid_z"]
        self.grid_r = data_param["grid_r"]

        self.z_range = data_param["z_range"]
        self.opt_train = data_param["opt_train"]
        self.opt_predout = data_param["opt_predout"]
        self.opt_usederivative = data_param["opt_usederivative"]
        self.nameopt_predout = data_param["nameopt_predout"]
        self.dim_input = sum(self.opt_train)
        self.dim_output = sum(self.opt_predout)

        self.num_fourier_coeffs_train = data_param["num_fourier_coeffs_train"]
        self.num_fourier_coeffs_apply = data_param["num_fourier_coeffs_apply"]

        if self.dim_output > 1:
            self.logger.fatal("YOU CAN PREDICT ONLY 1 DISTORTION. The sum of opt_predout == 1")
        self.logger.info("Inputs active for training: (SCMean, SCFluctuations)=(%d, %d)",
                         self.opt_train[0], self.opt_train[1])

        # Directories
        self.dirmodel = data_param["dirmodel"]
        self.dirapply = data_param["dirapply"]
        self.dirplots = data_param["dirplots"]
        self.dirtree = data_param["dirtree"]
        self.dirhist = data_param["dirhist"]
        train_dir = data_param["dirinput_bias"] if data_param["train_bias"] \
                    else data_param["dirinput_nobias"]
        val_dir = data_param["dirinput_bias"] if data_param["validation_bias"] \
                    else data_param["dirinput_nobias"]
        apply_dir = data_param["dirinput_bias"] if data_param["apply_bias"] \
                    else data_param["dirinput_nobias"]
        self.dirinput_train = "%s/SC-%d-%d-%d" % \
                              (train_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_validation = "%s/SC-%d-%d-%d" % \
                                   (val_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_apply = "%s/SC-%d-%d-%d" % \
                              (apply_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_nd_val = "%s/SC-%d-%d-%d" % (data_param["dirinput_nobias"],
                               self.grid_z, self.grid_r, self.grid_phi)

        for dirname in (self.dirmodel, self.dirapply, self.dirplots, self.dirtree, self.dirhist):
            if not os.path.isdir(dirname):
                os.makedirs(dirname)

        self.suffix = None
        self.suffix_ds = "phi%d_r%d_z%d" % \
                (self.grid_phi, self.grid_r, self.grid_z)

        # Parameters for getting input indices
        self.maxrandomfiles = data_param["maxrandomfiles"]
        self.nd_val_events = data_param["nd_val_events"]
        self.range_rnd_index_train = data_param["range_rnd_index_train"]
        self.range_rnd_index_nd_val = data_param["range_rnd_index_nd_val"]
        self.rnd_augment = data_param["rnd_augment"]
        self.part_inds = None
        self.nd_val_partition = data_param["nd_val_partition"]
        self.range_mean_index = data_param["range_mean_index"]
        self.indices_events_means = None
        self.partition = None
        self.total_events = 0
        self.train_events = 0
        self.val_events = 0
        self.apply_events = 0

    def __set_ranges(self, ranges, suffix, total_events, train_events, val_events, apply_events):
        """
        Update the event indices ranges for train / validation / apply.
        To be used internally.

        :param dict ranges: dictionary of lists of event indices ranges for train/validation/apply
        :param str suffix: suffix of the output file
        :param int total_events: number of all events used
        :param int train_events: number of events used for training
        :param int val_events: number of events used for validation
        :param int apply_events: number of events used for prediction
        """
        self.total_events = total_events
        self.train_events = train_events
        self.val_events = val_events
        self.apply_events = apply_events

        self.indices_events_means, self.partition = get_event_mean_indices(
            self.range_rnd_index_train, self.range_mean_index, ranges, self.rnd_augment)

        part_inds = None
        for part in self.partition:
            events_inds = np.array(self.partition[part])
            events_file = "%s/events_%s_%s_nEv%d.csv" % \
                          (self.dirmodel, part, suffix, self.train_events)
            np.savetxt(events_file, events_inds, delimiter=",", fmt="%d")
            if self.nd_val_partition != "random" and part == self.nd_val_partition:
                part_inds = events_inds
                self.part_inds = part_inds #[(part_inds[:,1] == 0) | (part_inds[:,1] == 9) | \
                                           #  (part_inds[:,1] == 18)]

        self.logger.info("Processing %d events", self.total_events)


class DNNSettings:
    """
    A class for the UNet-specific settings.
    Instead of inheriting CommonSettings, it stores a reference.
    """
    name = "dnn"

    def __init__(self, common_settings, data_param):
        """
        Read and store the parameters from the file.

        :param obj common_settings: CommonSettings singleton instance
        :param dict data_param: dictionary of values read from the config file
        """
        self.common_settings = common_settings
        self.logger.info("DNNSettings::Init")

        self.per_event_hists = True

        self.use_scaler = data_param["use_scaler"]

        # DNN config
        self.filters = data_param["filters"]
        self.pool_type = data_param["pool_type"]
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

        self.params = {'grid_phi': self.grid_phi,
                       'grid_r' : self.grid_r,
                       'grid_z' : self.grid_z,
                       'batch_size': self.batch_size,
                       'shuffle': self.shuffle,
                       'opt_train' : self.opt_train,
                       'opt_predout' : self.opt_predout,
                       'z_range' : self.z_range,
                       'use_scaler': self.use_scaler}

        self.suffix = "phi%d_r%d_z%d_filter%d_poo%s_drop%.2f_depth%d_batch%d_scaler%d" % \
                (self.grid_phi, self.grid_r, self.grid_z, self.filters, self.pool_type,
                 self.dropout, self.depth, 1 if self.batch_normalization else 0,
                 1 if self.use_scaler else 0)
        self.suffix = "%s_useSCMean%d_useSCFluc%d" % \
                (self.suffix, self.opt_train[0], self.opt_train[1])
        self.suffix = "%s_pred_doR%d_dophi%d_doz%d" % \
                (self.suffix, self.opt_predout[0], self.opt_predout[1], self.opt_predout[2])
        self.suffix = "%s_input_z%.1f-%.1f" % \
                (self.suffix, self.z_range[0], self.z_range[1])

        if not os.path.isdir("%s/%s" % (self.dirtree, self.suffix)):
            os.makedirs("%s/%s" % (self.dirtree, self.suffix))
        if not os.path.isdir("%s/%s" % (self.dirhist, self.suffix)):
            os.makedirs("%s/%s" % (self.dirhist, self.suffix))

        self.logger.info("I am processing the configuration %s", self.suffix)

    def __getattr__(self, name):
        """
        A Python hack to refer to the fields of the stored CommonSettings instance.

        :param str name: name of the requested instance attribute
        """
        try:
            return getattr(self.common_settings, name)
        except AttributeError as attr_err:
            raise AttributeError("'DNNSettings' object has no attribute '%s'" % name) from attr_err

    def set_ranges(self, ranges, total_events, train_events, val_events, apply_events):
        """
        A wrapper around internal set_ranges().

        :param dict ranges: dictionary of lists of event indices ranges for train/validation/apply
        :param int total_events: number of all events used
        :param int train_events: number of events used for training
        :param int val_events: number of events used for validation
        :param int apply_events: number of events used for prediction
        """
        self._CommonSettings__set_ranges(ranges, self.suffix, total_events, train_events,
                                         val_events, apply_events)

class XGBoostSettings:
    name = "xgboost"

    def __init__(self, common_settings, data_param):
        """
        Read and store the parameters from the file.

        :param obj common_settings: CommonSettings singleton instance
        :param dict data_param: dictionary of values read from the config file
        """
        self.common_settings = common_settings
        self.logger.info("XGBoostSettings::Init")

        self.params = data_param["params"]

        self.per_event_hists = False
        self.downsample = data_param["downsample"]
        self.downsample_npoints = data_param["downsample_npoints"]
        self.plot_train = data_param["plot_train"]
        self.train_plot_npoints = data_param["train_plot_npoints"]

        self.cache_train = data_param["cache_train"]
        self.cache_events = data_param["cache_events"]
        self.cache_file_size = data_param["cache_file_size"]
        self.dircache = data_param["dircache"]
        if not os.path.isdir(self.dircache):
            os.makedirs(self.dircache)

        self.cache_suffix = "cache_phi%d_r%d_z%d" % (self.grid_phi, self.grid_r, self.grid_z)
        if self.rnd_augment:
            self.cache_suffix = "%s_rndaugment" % self.cache_suffix
        if self.downsample:
            self.cache_suffix = "%s_dpoints%d" % \
                (self.cache_suffix, self.downsample_npoints)
        self.cache_suffix = "%s_ftrain%d" % \
            (self.cache_suffix, self.num_fourier_coeffs_train)

        self.suffix = "phi%d_r%d_z%d_nest%d_depth%d_lr%.3f_tm-%s" % \
                (self.grid_phi, self.grid_r, self.grid_z, self.params["n_estimators"],
                 self.params["max_depth"], self.params["learning_rate"],
                 self.params["tree_method"])
        self.suffix = "%s_g%.2f_weight%.1f_d%.1f_sub%.2f" % \
                (self.suffix, self.params["gamma"], self.params["min_child_weight"],
                 self.params["max_delta_step"], self.params["subsample"])
        self.suffix = "%s_colTree%.1f_colLvl%.1f_colNode%.1f" %\
                (self.suffix, self.params["colsample_bynode"], self.params["colsample_bytree"],
                 self.params["colsample_bylevel"])
        self.suffix = "%s_a%.1f_l%.5f_scale%.1f_base%.2f" %\
                (self.suffix, self.params["reg_alpha"], self.params["reg_lambda"],
                 self.params["scale_pos_weight"], self.params["base_score"])
        self.suffix = "%s_pred_doR%d_dophi%d_doz%d" % \
                (self.suffix, self.opt_predout[0], self.opt_predout[1], self.opt_predout[2])
        self.suffix = "%s_input_z%.1f-%.1f" % \
                (self.suffix, self.z_range[0], self.z_range[1])
        if self.rnd_augment:
            self.suffix = "%s_rndaugment" % self.suffix
        if self.downsample:
            self.suffix = "%s_dpoints%d" % \
                (self.suffix, self.downsample_npoints)
        self.suffix = "%s_ftrain%d" % \
            (self.suffix, self.num_fourier_coeffs_train)

        if not os.path.isdir("%s/%s" % (self.dirtree, self.suffix)):
            os.makedirs("%s/%s" % (self.dirtree, self.suffix))
        if not os.path.isdir("%s/%s" % (self.dirhist, self.suffix)):
            os.makedirs("%s/%s" % (self.dirhist, self.suffix))

        self.logger.info("I am processing the configuration %s", self.suffix)

    def __getattr__(self, name):
        """
        A Python hack to refer to the fields of the stored CommonSettings instance.

        :param str name: name of the requested instance attribute
        """
        try:
            return getattr(self.common_settings, name)
        except AttributeError as attr_err:
            raise AttributeError("'XGBoostSettings' object has no attribute '%s'" % name) \
                from attr_err

    def set_ranges(self, ranges, total_events, train_events, val_events, apply_events):
        """
        A wrapper around internal set_ranges().

        :param dict ranges: dictionary of lists of event indices ranges for train/validation/apply
        :param int total_events: number of all events used
        :param int train_events: number of events used for training
        :param int val_events: number of events used for validation
        :param int apply_events: number of events used for prediction
        """
        self._CommonSettings__set_ranges(ranges, self.suffix, total_events, train_events,
                                         val_events, apply_events)

    def set_cache_ranges(self):
        """
        Set ranges for caching.
        """
        ranges = {"train": [0, self.cache_events],
                  "validation": [self.cache_events, self.cache_events],
                  "apply": [self.cache_events, self.cache_events]}
        self.set_ranges(ranges, self.cache_events, self.cache_events, 0, 0)
