# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=too-many-statements, too-many-instance-attributes
import os
import matplotlib
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from root_pandas import to_root # pylint: disable=import-error, unused-import

from tpcwithdnn.logger import get_logger
from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d
from tpcwithdnn.data_loader import load_data_original, get_event_mean_indices
from tpcwithdnn.data_loader import load_data_derivatives_ref_mean

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use("Agg")

class DataValidator:
    # Class Attribute
    species = "data validator"

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
        self.dirinput_val = "%s/SC-%d-%d-%d/" % \
                            (data_param["dirinput_nobias"], self.grid_z, self.grid_r, self.grid_phi)

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
        self.tree_events = data_param["tree_events"]

        if not os.path.isdir(self.diroutflattree):
            os.makedirs(self.diroutflattree)


    def set_ranges(self, ranges, total_events):
        self.total_events = total_events

        self.indices_events_means, self.partition = get_event_mean_indices(
            self.maxrandomfiles, self.range_mean_index, ranges)


    # pylint: disable=too-many-locals, too-many-branches
    def create_data(self):
        self.logger.info("DataValidator::create_data")

        tree_filename = "%s/tree%s.root" % (self.diroutflattree, self.suffix_ds)
        if os.path.isfile(tree_filename):
            os.remove(tree_filename)

        vec_der_ref_mean_sc, mat_der_ref_mean_dist = \
            load_data_derivatives_ref_mean(self.dirinput_val, self.selopt_input)

        dist_names = np.array(self.nameopt_predout)[np.array(self.opt_predout) > 0]
        column_names = np.array(["eventId", "meanId", "randomId", "r", "phi", "z",
                                 "flucSC", "meanSC", "randomSC", "deltaSC", "derRefMeanSC"])
        for dist_name in self.nameopt_predout:
            column_names = np.append(column_names, ["flucDist" + dist_name,
                                                    "meanDist" + dist_name,
                                                    "randomDist" + dist_name,
                                                    "derRefMeanDist" + dist_name])
        if self.validate_model:
            json_file = open("%s/model_%s_nEv%d.json" % \
                             (self.dirmodel, self.suffix, self.total_events), "r")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = \
                model_from_json(loaded_model_json, {'SymmetryPadding3d' : SymmetryPadding3d})
            loaded_model.load_weights("%s/model_%s_nEv%d.h5" % \
                                      (self.dirmodel, self.suffix, self.total_events))

            for dist_name in dist_names:
                column_names = np.append(column_names, ["flucDist" + dist_name + "Pred"])

        counter = 0
        for imean in [0, 9, 18]:
            for irnd in range(self.maxrandomfiles):
                counter = counter + 1
                self.logger.info("processing event: %d [%d, %d]", counter, imean, irnd)

                [vec_r_pos, vec_phi_pos, vec_z_pos,
                 vec_mean_sc, vec_random_sc,
                 vec_mean_dist_r, vec_rand_dist_r,
                 vec_mean_dist_rphi, vec_rand_dist_rphi,
                 vec_mean_dist_z, vec_rand_dist_z] = load_data_original(self.dirinput_val,
                                                                        [irnd, imean])

                if self.selopt_input == 0:
                    vec_sel_z = vec_z_pos > 0
                elif self.selopt_input == 1:
                    vec_sel_z = vec_z_pos < 0
                elif self.selopt_input == 2:
                    vec_sel_z = vec_z_pos

                vec_z_pos = vec_z_pos[vec_sel_z]
                vec_r_pos = vec_r_pos[vec_sel_z]
                vec_phi_pos = vec_phi_pos[vec_sel_z]
                vec_mean_sc = vec_mean_sc[vec_sel_z]
                vec_random_sc = vec_random_sc[vec_sel_z]
                vec_mean_dist_r = vec_mean_dist_r[vec_sel_z]
                vec_mean_dist_rphi = vec_mean_dist_rphi[vec_sel_z]
                vec_mean_dist_z = vec_mean_dist_z[vec_sel_z]
                vec_rand_dist_r = vec_rand_dist_r[vec_sel_z]
                vec_rand_dist_rphi = vec_rand_dist_rphi[vec_sel_z]
                vec_rand_dist_z = vec_rand_dist_z[vec_sel_z]

                mat_mean_dist = np.array((vec_mean_dist_r, vec_mean_dist_rphi, vec_mean_dist_z))
                mat_rand_dist = np.array((vec_rand_dist_r, vec_rand_dist_rphi, vec_rand_dist_z))
                mat_fluc_dist = mat_mean_dist - mat_rand_dist

                vec_index_random = np.empty(vec_z_pos.size)
                vec_index_random[:] = irnd
                vec_index_mean = np.empty(vec_z_pos.size)
                vec_index_mean[:] = imean
                vec_index = np.empty(vec_z_pos.size)
                vec_index[:] = irnd + 10000 * imean
                vec_fluc_sc = vec_mean_sc - vec_random_sc
                vec_delta_sc = np.empty(vec_z_pos.size)
                vec_delta_sc[:] = sum(vec_fluc_sc) / sum(vec_mean_sc)

                df_single_map = pd.DataFrame({column_names[0] : vec_index,
                                              column_names[1] : vec_index_mean,
                                              column_names[2] : vec_index_random,
                                              column_names[3] : vec_phi_pos,
                                              column_names[4] : vec_r_pos,
                                              column_names[5] : vec_z_pos,
                                              column_names[6] : vec_fluc_sc,
                                              column_names[7] : vec_mean_sc,
                                              column_names[8] : vec_delta_sc,
                                              column_names[9] : vec_der_ref_mean_sc})

                for ind_dist in range(3):
                    df_single_map[column_names[10 + ind_dist * 3]] = mat_fluc_dist[ind_dist, :]
                    df_single_map[column_names[11 + ind_dist * 3]] = mat_mean_dist[ind_dist, :]
                    df_single_map[column_names[12 + ind_dist * 3]] = \
                        mat_der_ref_mean_dist[ind_dist, :]

                if self.validate_model:
                    input_single = np.empty((1, self.grid_phi, self.grid_r, self.grid_z,
                                             self.dim_input))
                    index_fill_input = 0
                    if self.opt_train[0] == 1:
                        input_single[0, :, :, :, index_fill_input] = \
                            vec_mean_sc.reshape(self.grid_phi, self.grid_r, self.grid_z)
                        index_fill_input = index_fill_input + 1
                    if self.opt_train[1] == 1:
                        input_single[0, :, :, :, index_fill_input] = \
                            vec_fluc_sc.reshape(self.grid_phi, self.grid_r, self.grid_z)

                    mat_fluc_dist_predict_group = loaded_model.predict(input_single)
                    mat_fluc_dist_predict = np.empty((self.dim_output, vec_fluc_sc.size))
                    for ind_dist in range(self.dim_output):
                        mat_fluc_dist_predict[ind_dist, :] = \
                            mat_fluc_dist_predict_group[0, :, :, :, ind_dist].flatten()
                        df_single_map[column_names[19 + ind_dist]] = \
                            mat_fluc_dist_predict[ind_dist, :]

                df_single_map.to_root(tree_filename, key="validation", mode="a", store_index=False)

                if counter == self.tree_events:
                    self.logger.info("Tree written in %s", tree_filename)
                    return

        self.logger.info("Tree written in %s", tree_filename)
