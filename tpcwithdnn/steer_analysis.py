"""
main script for doing tpc calibration with dnn
"""

import yaml
from machine_learning_hep.logger import get_logger
#from machine_learning_hep.utilities import checkdir, checkmakedir
from dnn_optimiser import DnnOptimiser

def main():
    """ The global main function """
    logger = get_logger()
    logger.info("Starting TPC ML...")

    with open("default.yml", 'r') as default_data:
        default = yaml.safe_load(default_data)
    case = default["case"]
    df_parameters = "database_parameters_%s.yml" % case
    with open(df_parameters, 'r') as parameters_data:
        db_parameters = yaml.safe_load(parameters_data)

    #dirmodel = db_parameters[case]["dirmodel"]
    #dirval = db_parameters[case]["dirval"]
    #dirinput = db_parameters[case]["dirinput"]

    dodumpflattree = default["dumpflattree"]
    dotrain = default["dotrain"]
    doapply = default["doapply"]
    doplot = default["doplot"]
    dogrid = default["dogrid"]

    #counter = 0
    #if dotraining is True:
    #    counter = counter + checkdir(dirmodel)
    #if dotesting is True:
    #    counter = counter + checkdir(dirval)
    #if counter < 0:
    #    sys.exit()

    myopt = DnnOptimiser(db_parameters[case], case)

    #if dotraining is True:
    #    checkmakedir(dirmodel)
    #if dotesting is True:
    #    checkmakedir(dirval)

    if dodumpflattree is True:
        myopt.dumpflattree()
    if dotrain is True:
        myopt.train()
    if doapply is True:
        myopt.apply()
    if doplot is True:
        myopt.plot()
    if dogrid is True:
        myopt.gridsearch()

    logger.info("Program finished.")

if __name__ == "__main__":
    main()
