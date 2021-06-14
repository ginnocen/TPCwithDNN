# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=too-many-statements
import os
import shutil
import gzip
import pickle
import math
import numpy as np
import pandas as pd
from RootInteractive.Tools.histoNDTools import makeHistogram  # pylint: disable=import-error, unused-import
from RootInteractive.Tools.makePDFMaps import makePdfMaps  # pylint: disable=import-error, unused-import

from tpcwithdnn.logger import get_logger
from tpcwithdnn.utilities import pandas_to_tree, tree_to_pandas
from tpcwithdnn.data_loader import load_data_original
from tpcwithdnn.data_loader import load_data_derivatives_ref_mean

class DataValidator:
    name = "data validator"
    mean_ids = (0, 9, 18)
    mean_factors = (1.0, 1.1, 0.9)

    def __init__(self):
        logger = get_logger()
        logger.info("DataValidator::Init")
        self.model = None
        self.config = None

    def set_model(self, model):
        self.model = model
        self.config = model.config

    def create_data_for_event(self, index_mean_id, irnd, column_names, vec_der_ref_mean_sc,
                              mat_der_ref_mean_dist, loaded_model, dir_name):
        [vec_r_pos, vec_phi_pos, vec_z_pos,
         vec_mean_sc, vec_random_sc,
         vec_mean_dist_r, vec_rand_dist_r,
         vec_mean_dist_rphi, vec_rand_dist_rphi,
         vec_mean_dist_z, vec_rand_dist_z] = load_data_original(
             self.config.dirinput_val,
            [irnd, self.mean_ids[index_mean_id]])

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

        mat_mean_dist = np.array((vec_mean_dist_r, vec_mean_dist_rphi, vec_mean_dist_z))
        mat_rand_dist = np.array((vec_rand_dist_r, vec_rand_dist_rphi, vec_rand_dist_z))
        mat_fluc_dist = mat_rand_dist - mat_mean_dist

        vec_index_random = np.empty(vec_z_pos.size)
        vec_index_random[:] = irnd
        vec_index_mean = np.empty(vec_z_pos.size)
        vec_index_mean[:] = self.mean_ids[index_mean_id]
        vec_index = np.empty(vec_z_pos.size)
        vec_index[:] = irnd + 1000 * self.mean_ids[index_mean_id]
        vec_fluc_sc = vec_random_sc - vec_mean_sc
        vec_delta_sc = np.empty(vec_z_pos.size)
        vec_delta_sc[:] = sum(vec_fluc_sc) / sum(vec_mean_sc)

        df_single_map = pd.DataFrame({column_names[0] : vec_index,
                                      column_names[1] : vec_index_mean,
                                      column_names[2] : vec_index_random,
                                      column_names[3] : vec_r_pos,
                                      column_names[4] : vec_phi_pos,
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

        tree_filename = "%s/%d/treeInput_mean%.2f_%s.root" \
            % (dir_name, irnd, self.mean_factors[index_mean_id], self.config.suffix_ds)
        if self.config.validate_model:
            tree_filename = "%s/%d/treeValidation_mean%.2f_nEv%d.root" \
                            % (dir_name, irnd, self.mean_factors[index_mean_id],
                               self.config.train_events)

        if not os.path.isdir("%s/%d" % (dir_name, irnd)):
            os.makedirs("%s/%d" % (dir_name, irnd))

        pandas_to_tree(df_single_map, tree_filename, 'validation')

    # pylint: disable=too-many-locals, too-many-branches
    def create_data(self):
        self.config.logger.info("DataValidator::create_data")

        vec_der_ref_mean_sc, mat_der_ref_mean_dist = \
            load_data_derivatives_ref_mean(self.config.dirinput_val, self.config.input_z_range)

        dist_names = np.array(self.config.nameopt_predout)[np.array(self.config.opt_predout) > 0]
        column_names = np.array(["eventId", "meanId", "randomId", "r", "phi", "z",
                                 "flucSC", "meanSC", "deltaSC", "derRefMeanSC"])
        for dist_name in self.config.nameopt_predout:
            column_names = np.append(column_names, ["flucDist" + dist_name,
                                                    "meanDist" + dist_name,
                                                    "derRefMeanDist" + dist_name])
        if self.config.validate_model:
            loaded_model = self.model.load_model()
            for dist_name in dist_names:
                column_names = np.append(column_names, ["flucDist" + dist_name + "Pred"])
        else:
            loaded_model = None

        dir_name = "%s/parts" % (self.config.diroutflattree)
        if self.config.validate_model:
            dir_name = "%s/%s/parts" % (self.config.diroutflattree, self.config.suffix)
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)

        for index_mean_id in range(0, len(self.mean_ids)):
            counter = 0
            if self.config.use_partition != 'random':
                for ind_ev in self.config.part_inds:
                    if ind_ev[1] != self.mean_ids[index_mean_id]:
                        continue
                    irnd = ind_ev[0]
                    self.config.logger.info("processing event: %d [%d, %d]",
                                            counter, self.mean_ids[index_mean_id], irnd)
                    self.create_data_for_event(index_mean_id, irnd, column_names,
                                               vec_der_ref_mean_sc, mat_der_ref_mean_dist,
                                               loaded_model, dir_name)
                    counter = counter + 1
                    if counter == self.config.val_events:
                        break
            else:
                for irnd in range(self.config.maxrandomfiles):
                    self.config.logger.info("processing event: %d [%d, %d]",
                                            counter, self.mean_ids[index_mean_id], irnd)
                    self.create_data_for_event(index_mean_id, irnd, column_names,
                                               vec_der_ref_mean_sc, mat_der_ref_mean_dist,
                                               loaded_model, dir_name)
                    counter = counter + 1
                    if counter == self.config.val_events:
                        break

            self.config.logger.info("Trees written in %s", dir_name)


    def get_pdf_map_variables_list(self):
        dist_names_list = np.array(self.config.nameopt_predout) \
            [np.array([self.config.opt_predout[0], self.config.opt_predout[1],
                       self.config.opt_predout[2]]) > 0]

        var_list = ['flucSC', 'meanSC', 'derRefMeanSC']
        for dist_name in dist_names_list:
            var_list.append('flucDist' + dist_name + 'Pred')
            var_list.append('flucDist' + dist_name)
            var_list.append('meanDist' + dist_name)
            var_list.append('derRefMeanDist' + dist_name)
            var_list.append('flucDist' + dist_name + 'Diff')

        return var_list


    def create_nd_histogram(self, var, mean_id):
        """
        Create nd histograms for given variable and mean id
        var: string of the variable name
        mean_id: index of mean map. Only 0 (factor=1.0), 9 (factor=1.1) and 18 (factor=0.9) working.
        """
        self.config.logger.info("DataValidator::create_nd_histogram, var = %s, mean_id = %d",
                                var, mean_id)
        if mean_id not in (0, 9, 18):
            self.config.logger.error("Code implementation only designed for mean ids 0, 9, 18.")
            self.config.logger.fatal("Exiting...")
        mean_factor = 1 + 0.1 * (mean_id != 0) * (1 - 2 * (mean_id == 18))

        column_names = ['phi', 'r', 'z', 'deltaSC']
        diff_index = var.find("Diff")
        if diff_index == -1:
            column_names.append(var)
        else:
            column_names = column_names + [var[:diff_index], var[:diff_index] + "Pred"]

        df_val = tree_to_pandas("%s/%s/treeValidation_mean%.1f_nEv%d.root"
                                % (self.config.diroutflattree, self.config.suffix, mean_factor,
                                   self.config.train_events),
                                'validation', column_names)
        if diff_index != -1:
            df_val[var] = \
                df_val[var[:diff_index] + "Pred"] - df_val[var[:diff_index]]

        # Definition string for nd histogram required by makeHistogram function in RootInteractive
        # 1) variables from data frame
        # 2) cut selection
        # 3) histogram name and binning in each dimension
        # E.g. "var1:var2:var3:#cut_selection>>histo_name(n1,min1,max1,n2,min2,max2,n3,min3,max3)"
        histo_string = "%s:phi:r:z:deltaSC" % (var) + \
                       ":#r>0" + \
                       ">>%s" % (var) + \
                       "(%d,%.4f,%.4f," % (200, df_val[var].min(), df_val[var].max()) + \
                       "180,0.0,6.283," + \
                       "33,83.5,254.5," + \
                       "40,0,250," + \
                       "%d,%.4f,%.4f)" % (10, df_val['deltaSC'].min(), df_val['deltaSC'].max())
        output_file_name = "%s/%s/ndHistogram_%s_mean%.1f_nEv%d.gzip" \
            % (self.config.dirouthistograms, self.config.suffix, var, mean_factor,
               self.config.train_events)
        with gzip.open(output_file_name, 'wb') as output_file:
            pickle.dump(makeHistogram(df_val, histo_string), output_file)
        output_file.close()
        self.config.logger.info("Nd histogram %s written to %s.", histo_string, output_file_name)


    def create_nd_histograms_meanid(self, mean_id):
        """
        Create nd histograms for given mean id
        mean_id: index of mean map. Only 0 (factor=1.0), 9 (factor=1.1) and 18 (factor=0.9) working.
        """
        for var in self.get_pdf_map_variables_list():
            self.create_nd_histogram(var, mean_id)


    def create_nd_histograms(self):
        """
        Create nd histograms for mean maps with id 0, 9, 18
        """
        for mean_id in [0, 9, 18]:
            self.create_nd_histograms_meanid(mean_id)


    def create_pdf_map(self, var, mean_id):
        """
        Create a pdf map for given variable and mean id
        var: string of the variable name
        mean_id: index of mean map. Only 0 (factor=1.0), 9 (factor=1.1) and 18 (factor=0.9) working.
        """
        self.config.logger.info("DataValidator::create_pdf_map, var = %s, mean_id = %d",
                                var, mean_id)
        if mean_id not in (0, 9, 18):
            self.config.logger.error("Code implementation only designed for mean ids 0, 9, 18.")
            self.config.logger.fatal("Exiting...")
        mean_factor = 1 + 0.1 * (mean_id != 0) * (1 - 2 * (mean_id == 18))

        input_file_name = "%s/%s/ndHistogram_%s_mean%.1f_nEv%d.gzip" \
            % (self.config.dirouthistograms, self.config.suffix, var, mean_factor,
               self.config.train_events)
        with gzip.open(input_file_name, 'rb') as input_file:
            histo = pickle.load(input_file)

        output_file_name = "%s/%s/pdfmap_%s_mean%.1f_nEv%d.root" \
            % (self.config.diroutflattree, self.config.suffix, var, mean_factor,
               self.config.train_events)
        dim_var = 0
        # slices: (start_bin, stop_bin, step, grouping) for each histogram dimension
        slices = ((0, histo['H'].shape[0], 1, 0),
                  (0, histo['H'].shape[1], 1, 0),
                  (0, histo['H'].shape[2], 1, 0),
                  (0, histo['H'].shape[3], 1, 0),
                  (0, histo['H'].shape[4], 1, 0))
        df_pdf_map = makePdfMaps(histo, slices, dim_var)
        pandas_to_tree(df_pdf_map, output_file_name, histo['name'])
        self.config.logger.info("Pdf map %s written to %s.", histo['name'], output_file_name)


    def create_pdf_maps_meanid(self, mean_id):
        """
        Create pdf maps for given mean id
        mean_id: index of mean map. Only 0 (factor=1.0), 9 (factor=1.1) and 18 (factor=0.9) working.
        """
        for var in self.get_pdf_map_variables_list():
            self.create_pdf_map(var, mean_id)


    def create_pdf_maps(self):
        """
        Create pdf maps for mean maps with id 0, 9, 18
        """
        for mean_id in [0, 9, 18]:
            self.create_pdf_maps_meanid(mean_id)


    def merge_pdf_maps(self, mean_ids=None):
        """
        Merge pdf maps for different variables into one file
        """
        self.config.logger.info("DataValidator::merge_pdf_maps")

        if mean_ids is None:
            mean_ids = [0, 9, 18]
        mean_ids_to_factors = {0: 1.0, 9: 1.1, 18: 0.9}
        mean_factors = [mean_ids_to_factors[mean_id] for mean_id in mean_ids]

        df_merged = pd.DataFrame()
        for mean_factor in mean_factors:
            input_file_name_0 = "%s/%s/pdfmap_flucSC_mean%.1f_nEv%d.root" \
                % (self.config.diroutflattree, self.config.suffix, mean_factor,
                   self.config.train_events)
            df = tree_to_pandas(input_file_name_0, 'flucSC', [".*Bin"])
            df['fsector'] = df['phiBinCenter'] / math.pi * 9
            df['meanMap'] = mean_factor
            for var in self.get_pdf_map_variables_list():
                input_file_name = "%s/%s/pdfmap_%s_mean%.1f_nEv%d.root" \
                    % (self.config.diroutflattree, self.config.suffix, var, mean_factor,
                       self.config.train_events)
                df_temp = tree_to_pandas(input_file_name, var, [".*"], [".*Bin"])
                for col in list(df_temp.keys()):
                    df[var + '_' + col] = df_temp[col]
            df_merged = df_merged.append(df, ignore_index=True)

        output_file_name = "%s/%s/pdfmaps_nEv%d.root" \
            % (self.config.diroutflattree, self.config.suffix, self.config.train_events)
        pandas_to_tree(df_merged, output_file_name, 'pdfmaps')
        self.config.logger.info("Pdf maps written to %s.", output_file_name)

    def merge_pdf_maps_meanid(self, mean_id):
        """
        Merge pdf maps for given mean id
        mean_id: index of mean map. Only 0 (factor=1.0), 9 (factor=1.1) and 18 (factor=0.9) working.
        """
        if mean_id not in (0, 9, 18):
            self.config.logger.error("Code implementation only designed for mean ids 0, 9, 18.")
            self.config.logger.fatal("Exiting...")
        self.merge_pdf_maps([mean_id])
