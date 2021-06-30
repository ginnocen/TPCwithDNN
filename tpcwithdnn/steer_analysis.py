"""
main script for doing tpc calibration with dnn
"""
# pylint: disable=fixme
# pylint: disable=too-many-statements
import os
import argparse
from timeit import default_timer as timer

# Needs to be set before any tensorflow import to suppress logging
# pylint: disable=wrong-import-position
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set only once for full workflow
SEED = 12345
os.environ['PYTHONHASHSEED'] = str(SEED)
import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

import matplotlib
matplotlib.use("Agg")

import yaml

import tpcwithdnn.check_root # pylint: disable=unused-import
from tpcwithdnn.logger import get_logger
from tpcwithdnn.debug_utils import log_time, log_total_memory_usage
from tpcwithdnn.common_settings import CommonSettings, XGBoostSettings, DNNSettings
# from tpcwithdnn.data_validator import DataValidator
from tpcwithdnn.idc_data_validator import IDCDataValidator

def setup_tf():
    # optionally limit GPU memory usage
    if os.environ.get('TPCwithDNNSETMEMLIMIT'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], \
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit= \
                    int(os.environ.get('TPCwithDNNSETMEMLIMIT')))])
                # for gpu in gpus:
                #     tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger = get_logger()
                logger.error(e)

def init_models(config_parameters):
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

def get_events_counts(train_events, test_events, apply_events):
    if len(train_events) != len(test_events) or \
       len(train_events) != len(apply_events):
        raise ValueError("Different number of ranges specified for train/test/apply")
    return zip(train_events, test_events, apply_events)

def run_model_and_val(model, dataval, default, config_parameters):
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
    if default["docreatevaldata"] is True:
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
    parser.add_argument("--dotrain", action='store_true', default=argparse.SUPPRESS,
                        help="Perform the training")
    parser.add_argument("--docreateinputdata", action='store_true', default=argparse.SUPPRESS,
                        help="Create input data trees")
    parser.add_argument("--docreatevaldata", action='store_true', default=argparse.SUPPRESS,
                        help="Create validation data trees")
    parser.add_argument("--ntrain1d", dest='train_events_oned', type=int, default=argparse.SUPPRESS,
                        help="Set custom number of training events")
    parser.add_argument("--nval", dest='val_events', type=int, default=argparse.SUPPRESS,
                        help="Set custom number of validation events")
    parser.add_argument("--frac", dest='downsample_fraction', type=float, default=argparse.SUPPRESS,
                        help="Set downsampling fraction if --downsample is set")
    parser.add_argument("--nestimators", dest='n_estimators', type=int, default=argparse.SUPPRESS,
                        help="Set the number of trees for xgboost models")
    parser.add_argument("--maxdepth", dest='max_depth', type=int, default=argparse.SUPPRESS,
                        help="Set maximum depth of trees for xgboost models")
    args = parser.parse_args()

    logger.info("Using configuration: %s steer file: %s", args.config_file, args.steer_file)

    with open(args.steer_file, "r") as steer_data:
        default = yaml.safe_load(steer_data)
    with open(args.config_file, "r") as config_data:
        config_parameters = yaml.safe_load(config_data)

    logger.info("Arguments provided: %s", str(args))
    if "dotrain" in args:
        default['dotrain'] = True
    if "train_events_oned" in args:
        config_parameters['xgboost']['train_events'] = [args.train_events_oned]
    if "docreateinputdata" in args or "docreatevaldata" in args:
        default['docreatevaldata'] = True
        config_parameters['common']['validate_model'] = False
    if "docreatevaldata" in args:
        config_parameters['common']['validate_model'] = True
    if "val_events" in args:
        config_parameters['common']['val_events'] = args.val_events
    if "downsample_fraction" in args:
        config_parameters['xgboost']['downsample'] = True
        config_parameters['xgboost']['downsample_fraction'] = args.downsample_fraction
    if "n_estimators" in args:
        config_parameters['xgboost']['params']['n_estimators'] = args.n_estimators
    if "max_depth" in args:
        config_parameters['xgboost']['params']['max_depth'] = args.max_depth

    # FIXME: Do we need these commented lines anymore?
    #dirmodel = config_parameters["common"]["dirmodel"]
    #dirval = config_parameters["common"]["dirval"]
    #dirinput = config_parameters["common"]["dirinput"]

    # NOTE
    # checkdir and checkmakedir not yet implemented. Was previously used from
    # machine_learning_hep package but is now the only thing required from there.
    # Easy to adapt an implementation like that to avoid heavy dependency
    # on machine_learning_hep

    #counter = 0
    #if dotraining is True:
    #    counter = counter + checkdir(dirmodel)
    #if dotesting is True:
    #    counter = counter + checkdir(dirval)
    #if counter < 0:
    #    sys.exit()

    #if dotraining is True:
    #    checkmakedir(dirmodel)
    #if dotesting is True:
    #    checkmakedir(dirval)

    models, corr, dataval = init_models(config_parameters)
    events_counts = (get_events_counts(config_parameters[model.name]["train_events"],
                                       config_parameters[model.name]["test_events"],
                                       config_parameters[model.name]["apply_events"])
                        for model in models)
    max_available_events = config_parameters["common"]["max_events"]

    for model, model_events_counts in zip(models, events_counts):
        all_events_counts = []
        for (train_events, test_events, apply_events) in model_events_counts:
            total_events = train_events + test_events + apply_events
            if total_events > max_available_events:
                logger.warning("Too big number of events requested: %d available: %d",
                               total_events, max_available_events)
                continue

            all_events_counts.append((train_events, test_events, apply_events, total_events))

            ranges = {"train": [0, train_events],
                      "test": [train_events, train_events + test_events],
                      "apply": [train_events + test_events, total_events]}
            model.config.set_ranges(ranges, total_events, train_events, test_events, apply_events)

            run_model_and_val(model, dataval, default, config_parameters["common"])

            # TODO: apply the correction and save in files
            if corr is not None:
                pass

        if default["doprofile"] is True:
            model.draw_profile(all_events_counts)

    logger.info("Program finished.")

if __name__ == "__main__":
    main()
