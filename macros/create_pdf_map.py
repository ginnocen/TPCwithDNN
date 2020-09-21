"""
script to create single nd histogram
"""

import sys
import os
import yaml

# Needs to be set before any tensorflow import to suppress logging
# pylint: disable=wrong-import-position
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tpcwithdnn.check_root  # pylint: disable=unused-import
from tpcwithdnn.logger import get_logger
from tpcwithdnn.data_validator import DataValidator


def main():
    """
    Input arguments:
    sys.argv[1]: case
    sys.argv[2]: variable to process
    sys.argv[3]: mean_id
    """
    logger = get_logger()
    logger.info("Starting TPC ML...")

    if len(sys.argv) < 4:
        logger.info("Insufficient number of arguments. Needs to be: [case, var, mean_id]. Stop.")
        sys.exit()
    case = str(sys.argv[1])
    var = str(sys.argv[2])
    mean_id = int(sys.argv[3])

    with open("database_parameters_%s.yml" % case, 'r') as parameters_data:
        db_parameters = yaml.safe_load(parameters_data)

    mydataval = DataValidator(db_parameters[case], case)

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
        mydataval.set_ranges(ranges, total_events, train_events, test_events, apply_events)

        mydataval.create_nd_histogram(var, mean_id)
        mydataval.create_pdf_map(var, mean_id)

    logger.info("Program finished.")


if __name__ == "__main__":
    main()
