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
    # FIXME: here I just copied something from the dnn_analyzer. It is likely
    # more information (e.g. the model name). You can just copy what you need
    # from the dnn_optimiser and delete what you dont need.

    def __init__(self, data_param, case):
        self.logger = get_logger()
        self.logger.info("DataValidation::Init\nCase: %s", case)

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

    def create_data(self):
        # FIXME : as you can imagine this is a complete duplication of what we
        # have in the dnn optimiser. But once this class is finished, we will
        # remove that part of code from the dnn_analyzer. Most likely also the
        # plotting code will be moved here. For the moment lets just keep the
        # code duplication.

        self.logger.info("DataValidation::dumpflattree")
        self.logger.warning("DO YOU REALLY WANT TO DO IT? IT TAKES TIME")
        outfile_name = "%s/tree%s.root" % (self.diroutflattree, self.suffix_ds)
        myfile = TFile.Open(outfile_name, "recreate")

        tree = TTree('tvoxels', 'tree with histos')
        indexr = array('i', [0])
        indexphi = array('i', [0])
        indexz = array('i', [0])
        posr = array('f', [0])
        posphi = array('f', [0])
        posz = array('f', [0])
        evtid = array('i', [0])
        meanid = array('i', [0])
        randomid = array('i', [0])
        distmeanr = array('f', [0])
        distmeanrphi = array('f', [0])
        distmeanz = array('f', [0])
        distrndr = array('f', [0])
        distrndrphi = array('f', [0])
        distrndz = array('f', [0])
        tree.Branch('indexr', indexr, 'indexr/I')
        tree.Branch('indexphi', indexphi, 'indexphi/I')
        tree.Branch('indexz', indexz, 'indexz/I')
        tree.Branch('posr', posr, 'posr/F')
        tree.Branch('posphi', posphi, 'posphi/F')
        tree.Branch('posz', posz, 'posz/F')
        tree.Branch('distmeanr', distmeanr, 'distmeanr/F')
        tree.Branch('distmeanrphi', distmeanrphi, 'distmeanrphi/F')
        tree.Branch('distmeanz', distmeanz, 'distmeanz/F')
        tree.Branch('distrndr', distrndr, 'distrndr/F')
        tree.Branch('distrndrphi', distrndrphi, 'distrndrphi/F')
        tree.Branch('distrndz', distrndz, 'distrndz/F')
        tree.Branch('evtid', evtid, 'evtid/I')
        tree.Branch('meanid', meanid, 'meanid/I')
        tree.Branch('randomid', randomid, 'randomid/I')

        for counter, indexev in enumerate(self.indices_events_means_train):
            self.logger.info("processing event: %d [%d, %d]", counter, indexev[0], indexev[1])

            # TODO: Should it be for train or apply data?
            [vec_r_pos, vec_phi_pos, vec_z_pos,
             _, _,
             vec_mean_dist_r, vec_random_dist_r,
             vec_mean_dist_rphi, vec_random_dist_rphi,
             vec_mean_dist_z, vec_random_dist_z] = load_data_original(self.dirinput_train, indexev)

            vec_r_pos = vec_r_pos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_phi_pos = vec_phi_pos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_z_pos = vec_z_pos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_mean_dist_r = vec_mean_dist_r.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_random_dist_r = vec_random_dist_r.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_mean_dist_rphi = vec_mean_dist_rphi.reshape(self.grid_phi, self.grid_r,
                                                            self.grid_z*2)
            vec_random_dist_rphi = vec_random_dist_rphi.reshape(self.grid_phi, self.grid_r,
                                                                self.grid_z*2)
            vec_mean_dist_z = vec_mean_dist_z.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_random_dist_z = vec_random_dist_z.reshape(self.grid_phi, self.grid_r, self.grid_z*2)

            for cur_indexphi in range(self.grid_phi):
                for cur_indexr in range(self.grid_r):
                    for cur_indexz in range(self.grid_z*2):
                        indexphi[0] = cur_indexphi
                        indexr[0] = cur_indexr
                        indexz[0] = cur_indexz
                        posr[0] = vec_r_pos[cur_indexphi][cur_indexr][cur_indexz]
                        posphi[0] = vec_phi_pos[cur_indexphi][cur_indexr][cur_indexz]
                        posz[0] = vec_z_pos[cur_indexphi][cur_indexr][cur_indexz]
                        distmeanr[0] = vec_mean_dist_r[cur_indexphi][cur_indexr][cur_indexz]
                        distmeanrphi[0] = vec_mean_dist_rphi[cur_indexphi][cur_indexr][cur_indexz]
                        distmeanz[0] = vec_mean_dist_z[cur_indexphi][cur_indexr][cur_indexz]
                        distrndr[0] = vec_random_dist_r[cur_indexphi][cur_indexr][cur_indexz]
                        distrndrphi[0] = vec_random_dist_rphi[cur_indexphi][cur_indexr][cur_indexz]
                        distrndz[0] = vec_random_dist_z[cur_indexphi][cur_indexr][cur_indexz]
                        evtid[0] = indexev[0] + 10000*indexev[1]
                        meanid[0] = indexev[1]
                        randomid[0] = indexev[0]
                        tree.Fill()

            # Set as you want
            if counter == 10:
                break

        myfile.Write()
        myfile.Close()
        self.logger.info("Tree written in %s", outfile_name)


       # FIXME: HERE YOU WOULD NEED TO LOAD THE MODEL, APPLY THE MODEL TO THE DATA AND
       # FILL NEW COLUMNS THAT CONTAIN e.g. THE PREDICTED DISTORTION
       # FLUCTUATIONS. HERE IS WHERE THE CODE OF ERNST SHOULD BE INSERTED.

