"""
main script for doing tpc calibration with dnn
"""

import os
import yaml

# Needs to be set before any tensorflow import to suppress logging
# pylint: disable=wrong-import-position
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tpcwithdnn.check_root # pylint: disable=unused-import
from tpcwithdnn.logger import get_logger
from tpcwithdnn.dnn_optimiser import DnnOptimiser
from tpcwithdnn.data_validator import DataValidator

## optionally limit GPU memory usage
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

# pylint: disable=too-many-locals, too-many-branches
def main():
    """ The global main function """
    logger = get_logger()
    logger.info("Starting TPC ML...")

    with open("default.yml", 'r') as default_data:
        default = yaml.safe_load(default_data)
    case = default["case"]
    with open("database_parameters_%s.yml" % case, 'r') as parameters_data:
        db_parameters = yaml.safe_load(parameters_data)

    #dirmodel = db_parameters[case]["dirmodel"]
    #dirval = db_parameters[case]["dirval"]
    #dirinput = db_parameters[case]["dirinput"]

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

    myopt = DnnOptimiser(db_parameters[case], case)
    mydataval = DataValidator(db_parameters[case], case)

    #if dotraining is True:
    #    checkmakedir(dirmodel)
    #if dotesting is True:
    #    checkmakedir(dirval)

    if len(db_parameters[case]["train_events"]) != len(db_parameters[case]["test_events"]) or \
       len(db_parameters[case]["train_events"]) != len(db_parameters[case]["apply_events"]):
        raise ValueError("Different number of ranges specified for train/test/apply")
    events_counts = zip(db_parameters[case]["train_events"],
                        db_parameters[case]["test_events"],
                        db_parameters[case]["apply_events"])
    max_available_events = db_parameters[case]["max_events"]

    all_events_counts = []

    for (train_events, test_events, apply_events) in events_counts:
        total_events = train_events + test_events + apply_events
        if total_events > max_available_events:
            print("Too big number of events requested: %d available: %d" % \
                  (total_events, max_available_events))
            continue

        all_events_counts.append((train_events, test_events, apply_events, total_events))

        ranges = {"train": [0, train_events],
                  "test": [train_events, train_events + test_events],
                  "apply": [train_events + test_events, total_events]}
        myopt.set_ranges(ranges, total_events)
        mydataval.set_ranges(ranges, total_events)

        if default["dotrain"] is True:
            myopt.train()
        if default["doapply"] is True:
            myopt.apply()
        if default["doplot"] is True:
            myopt.plot()
        if default["dogrid"] is True:
            myopt.gridsearch()

    if default["doprofile"] is True:
        myopt.draw_profile(all_events_counts)

    if default["docreatevaldata"] is True:
        mydataval.create_data()

    logger.info("Program finished.")

if __name__ == "__main__":
    main()
