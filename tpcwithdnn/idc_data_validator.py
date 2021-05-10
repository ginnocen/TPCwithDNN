# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=too-many-statements, fixme
import os
import gzip
import pickle
import math
import numpy as np
import pandas as pd
from root_pandas import to_root, read_root  # pylint: disable=import-error, unused-import
from RootInteractive.Tools.histoNDTools import makeHistogram  # pylint: disable=import-error, unused-import
from RootInteractive.Tools.makePDFMaps import makePdfMaps  # pylint: disable=import-error, unused-import

from tpcwithdnn.logger import get_logger
from tpcwithdnn.data_validator import DataValidator
from tpcwithdnn.data_loader import load_data_original_idc, filter_idc_data
from tpcwithdnn.data_loader import load_data_derivatives_ref_mean_idc, load_event_idc

class IDCDataValidator(DataValidator):
    name = "IDC data validator"
    mean_ids = (0, 27, 36)
    mean_factors = (0, 1.06, 0.94)

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
         num_mean_zero_idc_a, num_mean_zero_idc_c, num_random_zero_idc_a, num_random_zero_idc_c,
         vec_mean_one_idc_a, vec_mean_one_idc_c, vec_random_one_idc_a, vec_random_one_idc_c,
         vec_mean_sc, vec_random_sc,
         vec_mean_dist_r, vec_random_dist_r,
         vec_mean_dist_rphi, vec_random_dist_rphi,
         vec_mean_dist_z, vec_random_dist_z,
         vec_mean_corr_r, vec_random_corr_r,
         vec_mean_corr_rphi, vec_random_corr_rphi,
         vec_mean_corr_z, vec_random_corr_z] = load_data_original_idc(self.config.dirinput_val,
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
        vec_random_dist_r = vec_random_dist_r[vec_sel_z]
        vec_random_dist_rphi = vec_random_dist_rphi[vec_sel_z]
        vec_random_dist_z = vec_random_dist_z[vec_sel_z]
        vec_mean_corr_r = vec_mean_corr_r[vec_sel_z]
        vec_mean_corr_rphi = vec_mean_corr_rphi[vec_sel_z]
        vec_mean_corr_z = vec_mean_corr_z[vec_sel_z]
        vec_random_corr_r = vec_random_corr_r[vec_sel_z]
        vec_random_corr_rphi = vec_random_corr_rphi[vec_sel_z]
        vec_random_corr_z = vec_random_corr_z[vec_sel_z]

        mat_mean_dist = np.array((vec_mean_dist_r, vec_mean_dist_rphi, vec_mean_dist_z))
        mat_random_dist = np.array((vec_random_dist_r, vec_random_dist_rphi, vec_random_dist_z))
        mat_fluc_dist = mat_mean_dist - mat_random_dist

        mat_mean_corr = np.array((vec_mean_corr_r, vec_mean_corr_rphi, vec_mean_corr_z))
        mat_random_corr = np.array((vec_random_corr_r, vec_random_corr_rphi, vec_random_corr_z))
        mat_fluc_corr = mat_mean_corr - mat_random_corr

        # TODO: How to save 1D IDCs together with the rest?
        # The arrays need to be of the same length as the other vectors.
        data_a = (vec_mean_one_idc_a, vec_random_one_idc_a,
                  num_mean_zero_idc_a, num_random_zero_idc_a)
        data_c = (vec_mean_one_idc_c, vec_random_one_idc_c,
                  num_mean_zero_idc_c, num_random_zero_idc_c)
        mean_one_idc, random_one_idc, mean_zero_idc, random_zero_idc =\
            filter_idc_data(data_a, data_c, self.config.input_z_range) # pylint: disable=unbalanced-tuple-unpacking

        vec_mean_zero_idc = np.empty(vec_z_pos.size)
        vec_mean_zero_idc[:] = np.tile(mean_zero_idc, vec_z_pos.size // mean_zero_idc.size)
        vec_random_zero_idc = np.empty(vec_z_pos.size)
        vec_random_zero_idc[:] = np.tile(random_zero_idc, vec_z_pos.size // mean_zero_idc.size)

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

        if self.config.validate_model:
            inputs_single, _ = load_event_idc(self.config.dirinput_train,
                                              [irnd, imean], self.config.input_z_range,
                                              self.config.output_z_range,
                                              self.config.opt_predout)
            df_single_map[column_names[30]] = loaded_model.predict(inputs_single)

        df_single_map.to_root(tree_filename, key="validation", mode="a", store_index=False)

    # pylint: disable=too-many-locals, too-many-branches
    def create_data(self):
        self.config.logger.info("DataValidator::create_data")

        vec_z_pos = np.load("%s/Pos/vecZPos.npy" % self.config.dirinput_val)
        vec_sel_z = (self.config.input_z_range[0] <= vec_z_pos) &\
                       (vec_z_pos < self.config.input_z_range[1])
        vec_der_ref_mean_sc, mat_der_ref_mean_corr = \
            load_data_derivatives_ref_mean_idc(self.config.dirinput_val, vec_sel_z)

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
            loaded_model = self.model.load_model()
            for dist_name in dist_names:
                column_names = np.append(column_names, ["flucDist" + dist_name + "Pred"])
        else:
            loaded_model = None

        for imean, mean_factor in zip(self.mean_ids, self.mean_factors):
            tree_filename = "%s/treeInput_mean%.2f_%s.root" \
                            % (self.config.diroutflattree, mean_factor, self.config.suffix_ds)
            if self.config.validate_model:
                tree_filename = "%s/%s/treeValidation_mean%.1f_nEv%d.root" \
                                % (self.config.diroutflattree, self.config.suffix, mean_factor,
                                   self.config.train_events)

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
        mean_id: index of mean map.
        Only 0 (factor=1.00), 27 (factor=1.06) and 36 (factor=0.94) working.
        """
        self.config.logger.info("DataValidator::create_nd_histogram, var = %s, mean_id = %d",
                                var, mean_id)
        self.check_mean_id_(mean_id)
        mean_factor = self.get_mean_factor_(mean_id)

        column_names = ['phi', 'r', 'z', 'deltaSC']
        diff_index = var.find("Diff")
        if diff_index == -1:
            column_names.append(var)
        else:
            column_names = column_names + [var[:diff_index], var[:diff_index] + "Pred"]

        df_val = read_root("%s/%s/treeValidation_mean%.1f_nEv%d.root"
                           % (self.config.diroutflattree, self.config.suffix, mean_factor,
                              self.config.train_events),
                           key='validation', columns=column_names)
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
        mean_id: index of mean map.
        Only 0 (factor=1.00), 27 (factor=1.06) and 36 (factor=0.94) working.
        """
        for var in self.get_pdf_map_variables_list():
            self.create_nd_histogram(var, mean_id)

    def create_nd_histograms(self):
        """
        Create nd histograms for mean maps with id 0, 27, 36
        """
        for mean_id in self.mean_ids:
            self.create_nd_histograms_meanid(mean_id)

    def create_pdf_map(self, var, mean_id):
        """
        Create a pdf map for given variable and mean id
        var: string of the variable name
        mean_id: index of mean map.
        Only 0 (factor=1.00), 27 (factor=1.06) and 36 (factor=0.94) working.
        """
        self.config.logger.info("DataValidator::create_pdf_map, var = %s, mean_id = %d",
                                var, mean_id)
        self.check_mean_id_(mean_id)
        mean_factor = self.get_mean_factor_(mean_id)

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
        # set the index name to retrieve the name of the variable of interest later
        df_pdf_map.index.name = histo['name']
        df_pdf_map.to_root(output_file_name, key=histo['name'], mode='w', store_index=True)
        self.config.logger.info("Pdf map %s written to %s.", histo['name'], output_file_name)


    def create_pdf_maps_meanid(self, mean_id):
        """
        Create pdf maps for given mean id
        mean_id: index of mean map.
        Only 0 (factor=1.00), 27 (factor=1.06) and 36 (factor=0.94) working.
        """
        for var in self.get_pdf_map_variables_list():
            self.create_pdf_map(var, mean_id)

    def create_pdf_maps(self):
        """
        Create pdf maps for mean maps with id 0, 27, 36
        """
        for mean_id in self.mean_ids:
            self.create_pdf_maps_meanid(mean_id)

    def merge_pdf_maps(self, mean_ids=None):
        """
        Merge pdf maps for different variables into one file
        """
        self.config.logger.info("DataValidator::merge_pdf_maps")

        if mean_ids is None:
            mean_ids = self.mean_ids
        mean_factors = [self.mean_factors[mean_ids.index(mean_id)] for mean_id in mean_ids]

        df_merged = pd.DataFrame()
        for mean_factor in mean_factors:
            input_file_name_0 = "%s/%s/pdfmap_flucSC_mean%.1f_nEv%d.root" \
                % (self.config.diroutflattree, self.config.suffix, mean_factor,
                   self.config.train_events)
            df = read_root(input_file_name_0, columns="*Bin*")
            df['fsector'] = df['phiBinCenter'] / math.pi * 9
            df['meanMap'] = mean_factor
            for var in self.get_pdf_map_variables_list():
                input_file_name = "%s/%s/pdfmap_%s_mean%.1f_nEv%d.root" \
                    % (self.config.diroutflattree, self.config.suffix, var, mean_factor,
                       self.config.train_events)
                df_temp = read_root(input_file_name, ignore="*Bin*")
                for col in list(df_temp.keys()):
                    df[var + '_' + col] = df_temp[col]
            df_merged = df_merged.append(df, ignore_index=True)

        output_file_name = "%s/%s/pdfmaps_nEv%d.root" \
            % (self.config.diroutflattree, self.config.suffix, self.config.train_events)
        df_merged.to_root(output_file_name, key='pdfmaps', mode='w', store_index=False)
        self.config.logger.info("Pdf maps written to %s.", output_file_name)

    def merge_pdf_maps_meanid(self, mean_id):
        """
        Merge pdf maps for given mean id
        mean_id: index of mean map.
        Only 0 (factor=1.00), 27 (factor=1.06) and 36 (factor=0.94) working.
        """
        self.check_mean_id_(mean_id)
        self.merge_pdf_maps([mean_id])

    def check_mean_id_(self, mean_id):
        if mean_id not in self.mean_ids:
            self.config.logger.error("Code implementation only designed for mean ids 0, 27, 36.")
            self.config.logger.fatal("Exiting...")

    def get_mean_factor_(self, mean_id):
        return self.mean_factors[self.mean_ids.index(mean_id)]
