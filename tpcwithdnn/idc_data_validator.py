# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=too-many-statements, too-many-instance-attributes
# pylint: disable=fixme
import os
import matplotlib
import numpy as np
import pandas as pd
from root_pandas import to_root, read_root  # pylint: disable=import-error, unused-import

from tpcwithdnn.logger import get_logger
from tpcwithdnn.data_loader import load_data_original_idc
from tpcwithdnn.data_loader import load_data_derivatives_ref_mean_idc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use("Agg")

class IDCDataValidator:
    # Class Attribute
    species = "IDC data validator"

    def __init__(self, data_param, case):
        self.logger = get_logger()
        self.logger.info("IDCDataValidator::Init\nCase: %s", case)

        # Dataset config
        self.grid_phi = data_param["grid_phi"]
        self.grid_z = data_param["grid_z"]
        self.grid_r = data_param["grid_r"]

        self.input_z_range = data_param["input_z_range"]
        self.output_z_range = data_param["output_z_range"]
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
        self.dirouthistograms = data_param["dirouthistograms"]
        train_dir = data_param["dirinput_bias"] if data_param["train_bias"] \
                    else data_param["dirinput_nobias"]
        test_dir = data_param["dirinput_bias"] if data_param["test_bias"] \
                    else data_param["dirinput_nobias"]
        apply_dir = data_param["dirinput_bias"] if data_param["apply_bias"] \
                    else data_param["dirinput_nobias"]
        grid_str_dash = "%d-%d-%d" % (self.grid_z, self.grid_r, self.grid_phi)
        grid_str = "%d_%d_%d" % (self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_train = "%s/SC-%s/%s" % \
                              (train_dir, grid_str_dash, grid_str)
        self.dirinput_test = "%s/SC-%s/%s" % \
                             (test_dir, grid_str_dash, grid_str)
        self.dirinput_apply = "%s/SC-%s/%s" % \
                              (apply_dir, grid_str_dash, grid_str)
        self.dirinput_val = "%s/SC-%s/%s" % \
                            (data_param["dirinput_nobias"], grid_str_dash, grid_str)

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
        self.suffix = "%s_input_z%.1f-%.1f" % \
                (self.suffix, self.input_z_range[0], self.input_z_range[1])
        self.suffix = "%s_output_z%.1f-%.1f" % \
                (self.suffix, self.output_z_range[0], self.output_z_range[1])
        self.suffix_ds = "phi%d_r%d_z%d" % \
                (self.grid_phi, self.grid_r, self.grid_z)

        self.logger.info("I am processing the configuration %s", self.suffix)
        if self.dim_output > 1:
            self.logger.fatal("YOU CAN PREDICT ONLY 1 DISTORSION. The sum of opt_predout == 1")
        self.logger.info("Inputs active for training: (SCMean, SCFluctuations)=(%d, %d)",
                         self.opt_train[0], self.opt_train[1])

        # Parameters for getting input indices
        self.maxrandomfiles = data_param["maxrandomfiles"]
        self.train_events = 0
        self.tree_events = data_param["tree_events"]
        self.part_inds = None
        self.use_partition = data_param["use_partition"]

        if not os.path.isdir(self.diroutflattree):
            os.makedirs(self.diroutflattree)
        if not os.path.isdir("%s/%s" % (self.diroutflattree, self.suffix)):
            os.makedirs("%s/%s" % (self.diroutflattree, self.suffix))
        if not os.path.isdir("%s/%s" % (self.dirouthistograms, self.suffix)):
            os.makedirs("%s/%s" % (self.dirouthistograms, self.suffix))

    def set_ranges(self, train_events):
        self.train_events = train_events

        if self.use_partition != 'random':
            events_file = "%s/events_%s_%s_nEv%d.csv" % (self.dirmodel, self.use_partition,
                                                         self.suffix, self.train_events)
            part_inds = np.genfromtxt(events_file, delimiter=",")
            self.part_inds = part_inds[(part_inds[:,1] == 0) | (part_inds[:,1] == 5) | \
                                         (part_inds[:,1] == 2)]

    # pylint: disable=too-many-locals
    def create_data_for_event(self, imean, irnd, column_names, vec_der_ref_mean_sc,
                              mat_der_ref_mean_corr, tree_filename):
        [vec_r_pos, vec_phi_pos, vec_z_pos,
         mean_zero_idc, random_zero_idc,
         mean_one_idc, random_one_idc,
         vec_mean_sc, vec_random_sc,
         vec_mean_dist_r, vec_rand_dist_r,
         vec_mean_dist_rphi, vec_rand_dist_rphi,
         vec_mean_dist_z, vec_rand_dist_z,
         vec_mean_corr_r, vec_rand_corr_r,
         vec_mean_corr_rphi, vec_rand_corr_rphi,
         vec_mean_corr_z, vec_rand_corr_z] = load_data_original_idc(self.dirinput_val,
                                                                    [irnd, imean])

        vec_sel_z = (self.input_z_range[0] <= vec_z_pos) & (vec_z_pos < self.input_z_range[1])
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
        vec_mean_corr_r = vec_mean_corr_r[vec_sel_z]
        vec_mean_corr_rphi = vec_mean_corr_rphi[vec_sel_z]
        vec_mean_corr_z = vec_mean_corr_z[vec_sel_z]
        vec_rand_corr_r = vec_rand_corr_r[vec_sel_z]
        vec_rand_corr_rphi = vec_rand_corr_rphi[vec_sel_z]
        vec_rand_corr_z = vec_rand_corr_z[vec_sel_z]

        mat_mean_dist = np.array((vec_mean_dist_r, vec_mean_dist_rphi, vec_mean_dist_z))
        mat_rand_dist = np.array((vec_rand_dist_r, vec_rand_dist_rphi, vec_rand_dist_z))
        mat_fluc_dist = mat_mean_dist - mat_rand_dist

        mat_mean_corr = np.array((vec_mean_corr_r, vec_mean_corr_rphi, vec_mean_corr_z))
        mat_rand_corr = np.array((vec_rand_corr_r, vec_rand_corr_rphi, vec_rand_corr_z))
        mat_fluc_corr = mat_mean_corr - mat_rand_corr

        vec_mean_zero_idc = np.empty(vec_z_pos.size)
        vec_mean_zero_idc[:] = mean_zero_idc
        vec_random_zero_idc = np.empty(vec_z_pos.size)
        vec_random_zero_idc[:] = random_zero_idc

        # TODO: How to save 1D IDCs together with the rest?
        # The arrays need to be of the same length as the other vectors.
        vec_mean_one_idc = np.empty(vec_z_pos.size)
        vec_mean_one_idc[:mean_one_idc.size] = mean_one_idc
        vec_mean_one_idc[mean_one_idc.size:] = 0.
        vec_random_one_idc = np.empty(vec_z_pos.size)
        vec_random_one_idc[:random_one_idc.size] = random_one_idc
        vec_random_one_idc[random_one_idc.size:] = 0.

        vec_index_random = np.empty(vec_z_pos.size)
        vec_index_random[:] = irnd
        vec_index_mean = np.empty(vec_z_pos.size)
        vec_index_mean[:] = imean
        vec_index = np.empty(vec_z_pos.size)
        vec_index[:] = irnd + 1000 * imean

        vec_fluc_sc = vec_mean_sc - vec_random_sc
        vec_delta_sc = np.empty(vec_z_pos.size)
        vec_delta_sc[:] = sum(vec_fluc_sc) / sum(vec_mean_sc)

        vec_fluc_zero_idc = vec_mean_zero_idc - vec_random_zero_idc
        vec_fluc_zero_idc[mean_zero_idc.size:] = 0.
        vec_fluc_one_idc = vec_mean_one_idc - vec_random_one_idc
        vec_fluc_one_idc[mean_one_idc.size:] = 0.
        vec_delta_one_idc = sum(vec_fluc_one_idc) / sum(vec_mean_one_idc)

        df_single_map = pd.DataFrame({column_names[0] : vec_index,
                                      column_names[1] : vec_index_mean,
                                      column_names[2] : vec_index_random,
                                      column_names[3] : vec_r_pos,
                                      column_names[4] : vec_phi_pos,
                                      column_names[5] : vec_z_pos,
                                      column_names[6] : vec_fluc_sc,
                                      column_names[7] : vec_mean_sc,
                                      column_names[8] : vec_delta_sc,
                                      column_names[9] : vec_der_ref_mean_sc,
                                      column_names[10] : vec_fluc_one_idc,
                                      column_names[11] : vec_mean_one_idc,
                                      column_names[12] : vec_delta_one_idc,
                                      column_names[13] : vec_fluc_zero_idc,
                                      column_names[14] : vec_mean_zero_idc})

        for ind_dist in range(3):
            df_single_map[column_names[15 + ind_dist * 5]] = mat_fluc_dist[ind_dist, :]
            df_single_map[column_names[16 + ind_dist * 5]] = mat_mean_dist[ind_dist, :]
            df_single_map[column_names[17 + ind_dist * 5]] = \
                mat_der_ref_mean_corr[ind_dist, :]
            df_single_map[column_names[18 + ind_dist * 5]] = mat_fluc_corr[ind_dist, :]
            df_single_map[column_names[19 + ind_dist * 5]] = mat_mean_corr[ind_dist, :]

        df_single_map.to_root(tree_filename, key="validation", mode="a", store_index=False)

    # pylint: disable=too-many-locals, too-many-branches
    def create_data(self):
        self.logger.info("DataValidator::create_data")

        vec_der_ref_mean_sc, mat_der_ref_mean_corr = \
            load_data_derivatives_ref_mean_idc(self.dirinput_val, self.input_z_range)

        column_names = np.array(["eventId", "meanId", "randomId", "r", "phi", "z",
                                 "flucSC", "meanSC", "deltaSC", "derRefMeanSC",
                                 "fluc1DIDC", "mean1DIDC", "delta1DIDC",
                                 "fluc0DIDC", "mean0DIDC"])
        for dist_name in self.nameopt_predout:
            column_names = np.append(column_names, ["flucDist" + dist_name,
                                                    "meanDist" + dist_name,
                                                    "derRefMeanCorr" + dist_name,
                                                    "flucCorr" + dist_name,
                                                    "meanCorr" + dist_name])

        for imean, mean_factor in zip([0, 2, 5], [1.0, 0.94, 1.06]):
            tree_filename = "%s/treeInput_mean%.2f_%s.root" \
                            % (self.diroutflattree, mean_factor, self.suffix_ds)

            if os.path.isfile(tree_filename):
                os.remove(tree_filename)

            counter = 0
            if self.use_partition != 'random':
                for ind_ev in self.part_inds:
                    if ind_ev[1] != imean:
                        continue
                    irnd = ind_ev[0]
                    self.logger.info("processing event: %d [%d, %d]", counter, imean, irnd)
                    self.create_data_for_event(imean, irnd, column_names, vec_der_ref_mean_sc,
                                               mat_der_ref_mean_corr, tree_filename)
                    counter = counter + 1
                    if counter == self.tree_events:
                        break
            else:
                for irnd in range(self.maxrandomfiles):
                    self.logger.info("processing event: %d [%d, %d]", counter, imean, irnd)
                    self.create_data_for_event(imean, irnd, column_names, vec_der_ref_mean_sc,
                                               mat_der_ref_mean_corr, tree_filename)
                    counter = counter + 1
                    if counter == self.tree_events:
                        break

            self.logger.info("Tree written in %s", tree_filename)
