# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=too-many-statements, too-many-instance-attributes
# pylint: disable=fixme
import os
import numpy as np
import pandas as pd
from root_pandas import to_root, read_root  # pylint: disable=import-error, unused-import

from tpcwithdnn.logger import get_logger
from tpcwithdnn.data_validator import DataValidator
from tpcwithdnn.data_loader import load_data_original_idc
from tpcwithdnn.data_loader import load_data_derivatives_ref_mean_idc
from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d

class IDCDataValidator(DataValidator):
    # Class Attribute
    species = "IDC data validator"

    def __init__(self):
        super().__init__()
        logger = get_logger()
        logger.info("IDCDataValidator::Init")
        self.model = None
        self.config = None

    def set_model(self, model):
        self.model = model
        self.config = model.config

    # pylint: disable=too-many-locals
    def create_data_for_event(self, imean, irnd, column_names, vec_der_ref_mean_sc,
                              mat_der_ref_mean_dist, loaded_model, tree_filename):
        [vec_r_pos, vec_phi_pos, vec_z_pos,
         mean_zero_idc, random_zero_idc,
         mean_one_idc, random_one_idc,
         vec_mean_sc, vec_random_sc,
         vec_mean_dist_r, vec_rand_dist_r,
         vec_mean_dist_rphi, vec_rand_dist_rphi,
         vec_mean_dist_z, vec_rand_dist_z,
         vec_mean_corr_r, vec_rand_corr_r,
         vec_mean_corr_rphi, vec_rand_corr_rphi,
         vec_mean_corr_z, vec_rand_corr_z] = load_data_original_idc(self.config.dirinput_val,
                                                                    [irnd, imean])

        vec_sel_z = (self.config.input_z_range[0] <= vec_z_pos) &\
                    (vec_z_pos < self.config.input_z_range[1])
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
                mat_der_ref_mean_dist[ind_dist, :]
            df_single_map[column_names[18 + ind_dist * 5]] = mat_fluc_corr[ind_dist, :]
            df_single_map[column_names[19 + ind_dist * 5]] = mat_mean_corr[ind_dist, :]

        # FIXME: Copied from old data validator, to be updated
        if self.config.validate_model:
            input_single = np.empty((1, self.config.grid_phi, self.config.grid_r,
                                     self.config.grid_z, self.config.dim_input))
            index_fill_input = 0
            if self.config.opt_train[0] == 1:
                input_single[0, :, :, :, index_fill_input] = \
                    vec_mean_sc.reshape(self.config.grid_phi, self.config.grid_r,
                                        self.config.grid_z)
                index_fill_input = index_fill_input + 1
            if self.config.opt_train[1] == 1:
                input_single[0, :, :, :, index_fill_input] = \
                    vec_fluc_sc.reshape(self.config.grid_phi, self.config.grid_r,
                                        self.config.grid_z)

            mat_fluc_dist_predict_group = loaded_model.predict(input_single)
            mat_fluc_dist_predict = np.empty((self.config.dim_output, vec_fluc_sc.size))
            for ind_dist in range(self.config.dim_output):
                mat_fluc_dist_predict[ind_dist, :] = \
                    mat_fluc_dist_predict_group[0, :, :, :, ind_dist].flatten()
                df_single_map[column_names[19 + ind_dist]] = \
                    mat_fluc_dist_predict[ind_dist, :]

        df_single_map.to_root(tree_filename, key="validation", mode="a", store_index=False)

    # pylint: disable=too-many-locals, too-many-branches
    def create_data(self):
        self.config.logger.info("DataValidator::create_data")

        vec_der_ref_mean_sc, mat_der_ref_mean_corr = \
            load_data_derivatives_ref_mean_idc(self.config.dirinput_val, self.config.input_z_range)

        dist_names = np.array(self.config.nameopt_predout)[np.array(self.config.opt_predout) > 0]
        column_names = np.array(["eventId", "meanId", "randomId", "r", "phi", "z",
                                 "flucSC", "meanSC", "deltaSC", "derRefMeanSC",
                                 "fluc1DIDC", "mean1DIDC", "delta1DIDC",
                                 "fluc0DIDC", "mean0DIDC"])
        for dist_name in self.config.nameopt_predout:
            column_names = np.append(column_names, ["flucDist" + dist_name,
                                                    "meanDist" + dist_name,
                                                    "derRefMeanCorr" + dist_name,
                                                    "flucCorr" + dist_name,
                                                    "meanCorr" + dist_name])
        if self.config.validate_model:
            from tensorflow.keras.models import model_from_json # pylint: disable=import-outside-toplevel
            json_file = open("%s/model_%s_nEv%d.json" % \
                             (self.config.dirmodel, self.config.suffix,
                              self.config.train_events), "r")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = \
                model_from_json(loaded_model_json, {'SymmetryPadding3d' : SymmetryPadding3d})
            loaded_model.load_weights("%s/model_%s_nEv%d.h5" % \
                                      (self.config.dirmodel, self.config.suffix,
                                       self.config.train_events))

            for dist_name in dist_names:
                column_names = np.append(column_names, ["flucDist" + dist_name + "Pred"])
        else:
            loaded_model = None

        for imean, mean_factor in zip([0, 2, 5], [1.0, 0.94, 1.06]):
            tree_filename = "%s/treeInput_mean%.2f_%s.root" \
                            % (self.config.diroutflattree, mean_factor, self.config.suffix_ds)

            if os.path.isfile(tree_filename):
                os.remove(tree_filename)

            counter = 0
            if self.config.use_partition != 'random':
                for ind_ev in self.config.part_inds:
                    if ind_ev[1] != imean:
                        continue
                    irnd = ind_ev[0]
                    self.config.logger.info("processing event: %d [%d, %d]", counter, imean, irnd)
                    self.create_data_for_event(imean, irnd, column_names, vec_der_ref_mean_sc,
                                               mat_der_ref_mean_corr, loaded_model, tree_filename)
                    counter = counter + 1
                    if counter == self.config.val_events:
                        break
            else:
                for irnd in range(self.config.maxrandomfiles):
                    self.config.logger.info("processing event: %d [%d, %d]", counter, imean, irnd)
                    self.create_data_for_event(imean, irnd, column_names, vec_der_ref_mean_sc,
                                               mat_der_ref_mean_corr, loaded_model, tree_filename)
                    counter = counter + 1
                    if counter == self.config.val_events:
                        break

            self.config.logger.info("Tree written in %s", tree_filename)
