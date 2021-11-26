"""
The main script for the TPC calibration with DNN and BDT.
"""
# pylint: disable=fixme, too-many-statements, too-many-branches
import os
import argparse
from timeit import default_timer as timer

# Needs to be set before any tensorflow import to suppress logging
# pylint: disable=wrong-import-position
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set only once for full workflow
SEED = 12345
os.environ["PYTHONHASHSEED"] = str(SEED)
import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

import matplotlib
matplotlib.use("Agg")

# To suppress ROOT-specific command line help
from ROOT import PyConfig # pylint: disable=import-error
PyConfig.IgnoreCommandLineOptions = True

import yaml

import tpcwithdnn.check_root # pylint: disable=unused-import
from tpcwithdnn.logger import get_logger
from tpcwithdnn.debug_utils import log_time, log_total_memory_usage
from tpcwithdnn.common_settings import CommonSettings, XGBoostSettings, DNNSettings
# Currently, DataValidator is used for old data (used by DNN).
# It should be removed once DNN is adapted to the IDC data.
# from tpcwithdnn.data_validator import DataValidator
from tpcwithdnn.idc_data_validator import IDCDataValidator

def setup_tf():
    """
    Limit GPU memory usage.
    """
    if os.environ.get("TPCwithDNNSETMEMLIMIT"):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], \
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit= \
                    int(os.environ.get("TPCwithDNNSETMEMLIMIT")))])
                # for gpu in gpus:
                #     tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger = get_logger()
                logger.error(e)

def init_models(config_parameters):
    """
    Initialize the models and/or intermediate correction activated in the config file.

    :param dict config_parameters: dictionary of values read from the config file
    :return: prepared models, correction algorithm and data validator
    :rtype: tuple(list, obj, obj)
    """
    models = []
    corr = None
    common_settings = CommonSettings(config_parameters["common"])
    if config_parameters["xgboost"]["active"] is True:
        from tpcwithdnn.xgboost_optimiser import XGBoostOptimiser # pylint: disable=import-outside-toplevel
        config = XGBoostSettings(common_settings, config_parameters["xgboost"])
        config.params["random_state"] = SEED
        model = XGBoostOptimiser(config)
        models.append(model)
    if config_parameters["dnn"]["active"] is True:
        setup_tf()
        from tpcwithdnn.dnn_optimiser import DnnOptimiser # pylint: disable=import-outside-toplevel
        config = DNNSettings(common_settings, config_parameters["dnn"])
        model = DnnOptimiser(config)
        models.append(model)
    # TODO: Add the correction function / class here
    if config_parameters["corr"]["active"] is True:
        corr = None
    dataval = IDCDataValidator()
    return models, corr, dataval

def get_events_counts(train_events, val_events, apply_events):
    """
    Verify and zip requested numbers of events.

    :param list train_events: list of numbers of train events from the config file
    :param list val_events: list of numbers of validation events from the config file
    :param list apply_events: list of numbers of apply events from the config file
    :return: zipped (train, validation, apply) numbers
    :rtype: zip
    """
    if len(train_events) != len(val_events) or \
       len(train_events) != len(apply_events):
        raise ValueError("Different number of ranges specified for train/validation/apply")
    return zip(train_events, val_events, apply_events)

def run_model_and_val(model, dataval, default, config_parameters):
    """
    Launch the correction and validation steps activated in the config files,
    for a single set of triain/validation/apply numbers of events.

    :param obj model: instance of the current model (optimizer)
    :param obj dataval: instance of data validator
    :param dict default: dictionary with values from the default.yml configuration file
    :param dict config_parameters: dictionary of values read from the config file
    """
    dataval.set_model(model)
    if default["dotrain"] is True:
        start = timer()
        model.train()
        end = timer()
        log_time(start, end, "train")
    if default["dobayes"] is True:
        start = timer()
        model.bayes_optimise()
        end = timer()
        log_time(start, end, "bayes")
    if default["doapply"] is True:
        start = timer()
        model.apply()
        end = timer()
        log_time(start, end, "apply")
    if default["doplot"] is True:
        model.plot()
    if default["dogrid"] is True:
        model.search_grid()
    if default["docreatendvaldata"] is True:
        dataval.create_data()
    if default["docreatepdfmaps"] is True:
        dataval.create_nd_histograms()
        dataval.create_pdf_maps()
        dataval.merge_pdf_maps()
    if default["docreatepdfmapforvariable"] is True:
        dataval.create_nd_histogram(config_parameters["pdf_map_var"], \
                                      config_parameters["pdf_map_mean_id"])
        dataval.create_pdf_map(config_parameters["pdf_map_var"], \
                                 config_parameters["pdf_map_mean_id"])
    if default["docreatepdfmapformeanid"] is True:
        dataval.create_nd_histograms_meanid(config_parameters["pdf_map_mean_id"])
        dataval.create_pdf_maps_meanid(config_parameters["pdf_map_mean_id"])
        dataval.merge_pdf_maps_meanid(config_parameters["pdf_map_mean_id"])
    if default["domergepdfmaps"] is True:
        dataval.merge_pdf_maps()

def main():
    """ The global main function """
    logger = get_logger()
    logger.info("Starting TPC ML...")

    log_total_memory_usage("Initial memory usage")

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", "--config", dest="config_file", default="config_model_parameters.yml",
                        type=str, help="path to the *.yml configuration file")
    parser.add_argument("-s", "--steer", dest="steer_file", default="default.yml",
                        type=str, help="path to the *.yml steering file")
    # parameters for steer file
    parser.add_argument("--dotrain", action="store_true", default=argparse.SUPPRESS,
                        help="Perform the training")
    parser.add_argument("--docreatendvaldata", action="store_true", default=argparse.SUPPRESS,
                        help="Create validation data trees")
    parser.add_argument("--docache", action="store_true", default=argparse.SUPPRESS,
                        help="Cache training data if not already existing")
    # parameters for config file
    parser.add_argument("--rndaugment", action="store_true", default=argparse.SUPPRESS,
                        help="Use random-random augmentation for training")
    parser.add_argument("--ntrain1d", dest="train_events_oned", type=int, default=argparse.SUPPRESS,
                        help="Set custom number of training events")
    parser.add_argument("--nval", dest="nd_val_events", type=int, default=argparse.SUPPRESS,
                        help="Set custom number of max nd validation events")
    parser.add_argument("--dnpoints", dest="downsample_npoints", type=int,
                        default=argparse.SUPPRESS, help="Set number of downsampling points")
    parser.add_argument("--nestimators", dest="n_estimators", type=int, default=argparse.SUPPRESS,
                        help="Set the number of trees for xgboost models")
    parser.add_argument("--maxdepth", dest="max_depth", type=int, default=argparse.SUPPRESS,
                        help="Set maximum depth of trees for xgboost models")
    parser.add_argument("--nfouriertrain", dest="num_fourier_coeffs_train", type=int,
                        default=argparse.SUPPRESS, help="Set number of Fourier coefficients" \
                        " to take from the 1D IDC train input")
    parser.add_argument("--nfourierapply", dest="num_fourier_coeffs_apply", type=int,
                        default=argparse.SUPPRESS, help="Set number of Fourier coefficients" \
                        " to take from the 1D IDC apply input")
    # parameters for caching
    parser.add_argument("--cache-events", dest="cache_events", type=int, default=argparse.SUPPRESS,
                        help="Set the number of events to cache")
    parser.add_argument("--cache-train", action="store_true", default=argparse.SUPPRESS,
                        help="Use cached data for training")
    parser.add_argument("--cache-file-size", dest="cache_file_size", type=int,
                        default=argparse.SUPPRESS,
                        help="Set the number of events per single temporary cache file")
    args = parser.parse_args()

    logger.info("Using configuration: %s steer file: %s", args.config_file, args.steer_file)

    with open(args.steer_file, "r", encoding="utf-8") as steer_data:
        default = yaml.safe_load(steer_data)
    with open(args.config_file, "r", encoding="utf-8") as config_data:
        config_parameters = yaml.safe_load(config_data)

    logger.info("Arguments provided: %s", str(args))
    if "dotrain" in args:
        default["dotrain"] = True
    if "docreatendvaldata" in args:
        default["docreatendvaldata"] = True
    if "docache" in args:
        default["docache"] = True
    #
    if "rndaugment" in args:
        config_parameters["common"]["rnd_augment"] = True
    if "train_events_oned" in args:
        config_parameters["xgboost"]["train_events"] = [args.train_events_oned]
    if "nd_val_events" in args:
        config_parameters["common"]["nd_val_events"] = args.nd_val_events
    if "downsample_npoints" in args:
        config_parameters["xgboost"]["downsample"] = True
        config_parameters["xgboost"]["downsample_npoints"] = args.downsample_npoints
    if "n_estimators" in args:
        config_parameters["xgboost"]["params"]["n_estimators"] = args.n_estimators
    if "max_depth" in args:
        config_parameters["xgboost"]["params"]["max_depth"] = args.max_depth
    if "num_fourier_coeffs_train" in args:
        config_parameters["common"]["num_fourier_coeffs_train"] = args.num_fourier_coeffs_train
    if "num_fourier_coeffs_apply" in args:
        config_parameters["common"]["num_fourier_coeffs_apply"] = args.num_fourier_coeffs_apply
    #
    if "cache_events" in args:
        config_parameters["xgboost"]["cache_events"] = args.cache_events
    if "cache_train" in args:
        config_parameters["xgboost"]["cache_train"] = True
    if "cache_file_size" in args:
        config_parameters["xgboost"]["cache_file_size"] = args.cache_file_size

    models, corr, dataval = init_models(config_parameters)
    events_counts = (get_events_counts(config_parameters[model.name]["train_events"],
                                       config_parameters[model.name]["validation_events"],
                                       config_parameters[model.name]["apply_events"])
                        for model in models)
    ranges_rnd = config_parameters["common"]["range_rnd_index_train"]
    ranges_mean = config_parameters["common"]["range_mean_index"]
    if config_parameters["common"]["rnd_augment"]:
        max_available_events = (ranges_rnd[1] + 1 - ranges_rnd[0]) * (ranges_rnd[1] - ranges_rnd[0])
    else:
        max_available_events = (ranges_rnd[1] + 1 - ranges_rnd[0]) * \
            (ranges_mean[1] + 1 - ranges_mean[0])

    for model in models:
        if default["docache"] is True and model.name == "xgboost":
            start = timer()
            model.cache_train_data()
            end = timer()
            log_time(start, end, "cache")
    for model, model_events_counts in zip(models, events_counts):
        all_events_counts = []
        for (train_events, val_events, apply_events) in model_events_counts:
            total_events = train_events + val_events + apply_events
            if total_events > max_available_events:
                logger.warning("Too big number of events requested: %d available: %d",
                               total_events, max_available_events)
                continue

            all_events_counts.append((train_events, val_events, apply_events, total_events))

            ranges = {"train": [0, train_events],
                      "validation": [train_events, train_events + val_events],
                      "apply": [train_events + val_events, total_events]}
            model.config.set_ranges(ranges, total_events, train_events, val_events, apply_events)

            run_model_and_val(model, dataval, default, config_parameters["common"])

            # TODO: apply the correction and save in files
            if corr is not None:
                pass

        if default["doprofile"] is True:
            model.draw_profile(all_events_counts)

    logger.info("Program finished.")

if __name__ == "__main__":
    main()
