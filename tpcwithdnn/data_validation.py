# pylint: disable=too-many-instance-attributes, too-many-statements, too-many-arguments, fixme
# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
import os
from array import array
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from root_numpy import fill_hist
from ROOT import TH1F, TH2F, TFile, TCanvas, TLegend, TPaveText, gPad # pylint: disable=import-error, no-name-in-module
from ROOT import gStyle, kWhite, kBlue, kGreen, kRed, kCyan, kOrange, kMagenta # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT, TTree  # pylint: disable=import-error, no-name-in-module
from symmetry_padding_3d import SymmetryPadding3d
from machine_learning_hep.logger import get_logger
from fluctuation_data_generator import FluctuationDataGenerator
from utilities_dnn import u_net
from data_loader import load_train_apply, load_data_original, get_event_mean_indices

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use("Agg")

class DataValidation:
    # Class Attribute
    # TODO: What is this for?
    species = "datavalidation"

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
        self.diroutflattree = data_param["diroutflattree"]

