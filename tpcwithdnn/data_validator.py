# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=fixme, too-many-statements, too-many-instance-attributes
import os
import gzip
import pickle
import subprocess
from array import array

import matplotlib
import numpy as np
import pandas as pd
from keras.models import model_from_json
from root_pandas import read_root

from ROOT import gROOT, TFile, TTree  # pylint: disable=import-error, no-name-in-module

from tpcwithdnn.logger import get_logger
from tpcwithdnn.data_loader import load_data_original, get_event_mean_indices
from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d
from tpcwithdnn.data_loader import load_data_apply_nd, load_data_derivatives_ref_mean

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use("Agg")

class DataValidator:
    # Class Attribute
    species = "data validator"

    # FIXME: here I just copied something from the dnn_analyzer. It is likely
    # more information (e.g. the model name). You can just copy what you need
    # from the dnn_optimiser and delete what you dont need.
    def __init__(self, data_param, case):
        self.logger = get_logger()
        self.logger.info("DataValidator::Init\nCase: %s", case)

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

        self.validate_model = data_param["validate_model"]
        self.use_scaler = data_param["use_scaler"]

        # Directories
        self.dirmodel = data_param["dirmodel"]
        self.dirval = data_param["dirval"]
        self.diroutflattree = data_param["diroutflattree"]
        train_dir = data_param["dirinput_bias"] if data_param["train_bias"] \
                    else data_param["dirinput_nobias"]
        test_dir = data_param["dirinput_bias"] if data_param["test_bias"] \
                    else data_param["dirinput_nobias"]
        apply_dir = data_param["dirinput_bias"] if data_param["apply_bias"] \
                    else data_param["dirinput_nobias"]
        self.dirinput_train = "%s/SC-%d-%d-%d/" % \
                              (train_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_test = "%s/SC-%d-%d-%d/" % \
                             (test_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_apply = "%s/SC-%d-%d-%d/" % \
                              (apply_dir, self.grid_z, self.grid_r, self.grid_phi)

        # DNN config
        self.filters = data_param["filters"]
        self.pooling = data_param["pooling"]
        self.depth = data_param["depth"]
        self.batch_normalization = data_param["batch_normalization"]
        self.dropout = data_param["dropout"]

        self.suffix = "phi%d_r%d_z%d_filter%d_poo%d_drop%.2f_depth%d_batch%d_scaler%d" % \
                (self.grid_phi, self.grid_r, self.grid_z, self.filters, self.pooling,
                 self.dropout, self.depth, self.batch_normalization, self.use_scaler)
        self.suffix = "%s_useSCMean%d_useSCFluc%d" % \
                (self.suffix, self.opt_train[0], self.opt_train[1])
        self.suffix = "%s_pred_doR%d_dophi%d_doz%d" % \
                (self.suffix, self.opt_predout[0], self.opt_predout[1], self.opt_predout[2])
        self.suffix_ds = "phi%d_r%d_z%d" % \
                (self.grid_phi, self.grid_r, self.grid_z)

        self.logger.info("I am processing the configuration %s", self.suffix)
        if self.dim_output > 1:
            self.logger.fatal("YOU CAN PREDICT ONLY 1 DISTORSION. The sum of opt_predout == 1")
        self.logger.info("Inputs active for training: (SCMean, SCFluctuations)=(%d, %d)",
                         self.opt_train[0], self.opt_train[1])

        # Parameters for getting input indices
        self.maxrandomfiles = data_param["maxrandomfiles"]
        self.range_mean_index = data_param["range_mean_index"]
        self.indices_events_means = None
        self.partition = None
        self.total_events = 0
        self.train_events = 0
        self.tree_events = data_param["tree_events"]
        self.tree_means = daa_param["tree_means"]

        if not os.path.isdir("input_plots"):
            os.makedirs("input_plots")
        if not os.path.isdir(self.dirval + "/" + self.suffix):
            os.makedirs(self.dirval + "/" + self.suffix)

        gROOT.SetStyle("Plain")
        gROOT.SetBatch()



    def set_ranges(self, ranges, train_events, total_events):
        self.total_events = total_events
        self.train_events = train_events

        self.indices_events_means, self.partition = get_event_mean_indices(
            self.maxrandomfiles, self.range_mean_index, ranges)


    # pylint: disable=too-many-locals
    def create_data(self):
        # FIXME : as you can imagine this is a complete duplication of what we
        # have in the dnn optimiser. But once this class is finished, we will
        # remove that part of code from the dnn_analyzer. Most likely also the
        # plotting code will be moved here. For the moment lets just keep the
        # code duplication.

        self.logger.info("DataValidator::create_data")
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

        for counter, indexev in enumerate(self.partition["apply"]):
            self.logger.info("processing event: %d [%d, %d]", counter, indexev[0], indexev[1])

            # TODO: Should it be for train or apply data?
            [vec_r_pos, vec_phi_pos, vec_z_pos,
             _, _, # Omitted: vec_mean_sc, vec_random_sc
             vec_mean_dist_r, vec_random_dist_r,
             vec_mean_dist_rphi, vec_random_dist_rphi,
             vec_mean_dist_z, vec_random_dist_z] = load_data_original(self.dirinput_apply, indexev)

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

            if counter + 1 == self.tree_events:
                break

        myfile.Write()
        myfile.Close()
        self.logger.info("Tree written in %s", outfile_name)


    def create_nd_validation_data(self):
        self.logger.info("Create data for multi-dimensional analysis, input size: %d",
                         self.dim_input)

        # TODO: Should it be dirinput_train, test, apply?
        arr_der_ref_mean_sc, mat_der_ref_mean_dist = \
            load_data_derivatives_ref_mean(self.dirinput_apply, self.selopt_input,
                                           self.opt_predout)

        if self.validate_model:
            json_file = open("%s/model_%s_nEv%d.json" % \
                             (self.dirmodel, self.suffix, self.total_events), "r")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = \
                model_from_json(loaded_model_json, {'SymmetryPadding3d' : SymmetryPadding3d})
            loaded_model.load_weights("%s/model_%s_nEv%d.h5" % \
                                      (self.dirmodel, self.suffix, self.total_events))

        dist_names = np.array(self.nameopt_predout)[np.array([self.opt_predout[0],
                                                              self.opt_predout[1],
                                                              self.opt_predout[2]]) > 0]
        column_names = np.array(["phi", "r", "z",
                                 "flucSC", "meanSC", "deltaSC", "derRefMeanSC"])
        for dist_name in dist_names:
            column_names = np.append(column_names, ["flucDist" + dist_name,
                                                    "meanDist" + dist_name,
                                                    "derRefMeanDist" + dist_name,
                                                    "flucDist" + dist_name + "Pred"])

        # FIXME: Some function to calculate proper factor or make it a configurable?
        for imean, factor in zip(self.tree_means, [1.0, 1.1, 0.9]):
            filename = "%s/%s/outputValidation_mean%f.root" % (self.dirval, self.suffix, factor)
            if os.path.isfile(filename):
                os.remove(filename)

            for irnd in np.arange(self.tree_events):
                imap = [irnd, imean]
                # TODO: Should it be dirinput_train, test, apply?
                (arr_r_pos, arr_phi_pos, arr_z_pos, arr_mean_sc, arr_fluc_sc,
                 mat_mean_dist, mat_fluc_dist) = \
                    load_data_apply_nd(self.dirinput_apply, imap, self.selopt_input,
                                       self.opt_predout)
                arr_index_random = np.empty(arr_z_pos.size)
                arr_index_random[:] = imap[0]
                arr_index_mean = np.empty(arr_z_pos.size)
                arr_index_mean[:] = imap[1]
                arr_delta_sc = np.empty(arr_z_pos.size)
                arr_delta_sc[:] = sum(arr_fluc_sc) / sum(arr_mean_sc)

                df_single_map = pd.DataFrame({"indexRnd" : arr_index_random,
                                              "indexMean" : arr_index_mean,
                                              column_names[0] : arr_phi_pos,
                                              column_names[1] : arr_r_pos,
                                              column_names[2] : arr_z_pos,
                                              column_names[3] : arr_fluc_sc,
                                              column_names[4] : arr_mean_sc,
                                              column_names[5] : arr_delta_sc,
                                              column_names[6] : arr_der_ref_mean_sc})

                input_single = np.empty((1, self.grid_phi, self.grid_r, self.grid_z,
                                         self.dim_input))
                index_fill_input = 0
                if self.opt_train[0] == 1:
                    input_single[0, :, :, :, index_fill_input] = arr_mean_sc.reshape(self.grid_phi,
                                                                                     self.grid_r,
                                                                                     self.grid_z)
                    index_fill_input = index_fill_input + 1
                if self.opt_train[1] == 1:
                    input_single[0, :, :, :, index_fill_input] = arr_fluc_sc.reshape(self.grid_phi,
                                                                                     self.grid_r,
                                                                                     self.grid_z)
                    index_fill_input = index_fill_input + 1

                if self.validate_model:
                    mat_fluc_dist_predict_group = loaded_model.predict(input_single)
                    mat_fluc_dist_predict = np.empty((self.dim_output, mat_fluc_dist[0, :].size))
                    for ind_dist in range(self.dim_output):
                        mat_fluc_dist_predict[ind_dist, :] = \
                            mat_fluc_dist_predict_group[0, :, :, :, ind_dist].flatten()
                        df_single_map[column_names[7 + ind_dist * 4]] = mat_fluc_dist[ind_dist, :]
                        df_single_map[column_names[8 + ind_dist * 4]] = mat_mean_dist[ind_dist, :]
                        df_single_map[column_names[9 + ind_dist * 4]] = \
                            mat_der_ref_mean_dist[ind_dist, :]
                        df_single_map[column_names[10 + ind_dist * 4]] = \
                            mat_fluc_dist_predict[ind_dist, :]

                df_single_map.to_root(filename, key="validation", mode="a", store_index=False)

        self.logger.info("Done create_nd_validation_data")
