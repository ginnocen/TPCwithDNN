"""
Validation module for IDC input and optimization output.

It is adapted to both DNN and XGBoost optimizers, but, since DNN does not use IDC data yet,
the old validator must be used for DNN validation.
"""
# pylint: disable=too-many-statements, fixme
import os
import gzip
import pickle
import math
import numpy as np
import pandas as pd
from RootInteractive.Tools.histoNDTools import makeHistogram  # pylint: disable=import-error, unused-import
from RootInteractive.Tools.makePDFMaps import makePdfMaps  # pylint: disable=import-error, unused-import

from tpcwithdnn.logger import get_logger
from tpcwithdnn.tree_df_utils import pandas_to_tree, tree_to_pandas
from tpcwithdnn.data_loader import load_data_original_idc, get_input_oned_idc_single_map
from tpcwithdnn.data_loader import filter_idc_data, mat_to_vec, get_fourier_coeffs
from tpcwithdnn.hadd import hadd

class IDCDataValidator:
    """
    The class for an IDC data validator.
    """
    name = "IDC data validator"
    mean_ids = (0, 9, 18, 27, 36)
    mean_factors = (1.00, 1.03, 0.97, 1.06, 0.94)

    def __init__(self):
        """
        Initialize the validator
        """
        logger = get_logger()
        logger.info("IDCDataValidator::Init")
        self.model = None
        self.config = None

    def set_model(self, model):
        """
        Set model and configuration to be tested.

        :param Optimizer model: an Optimizer class instance (DNN or XGBoost)
        """
        self.model = model
        self.config = model.config

    # pylint: disable=too-many-locals
    def create_data_for_event(self, mean_id, irnd, loaded_model):
        """
        Generate and save into file the input validation data for a given event pair

        :param int mean_id: index of a mean map
        :param int irnd: index of a random map
        :param obj loaded_model: the proper loaded model - either keras.Model or
                                 xgboost.sklearn.XGBModel
        :returns: path to the output file
        :rtype: str
        """
        [vec_r_pos, vec_phi_pos, vec_z_pos,
         vec_mean_sc, vec_random_sc,
         *_,
         vec_mean_corr_r, vec_random_corr_r,
         vec_mean_corr_rphi, vec_random_corr_rphi,
         vec_mean_corr_z, vec_random_corr_z,
         vec_der_ref_mean_sc, mat_der_ref_mean_corr,
         num_mean_zerod_idc_a, num_mean_zerod_idc_c, num_random_zerod_idc_a, num_random_zerod_idc_c,
         vec_mean_oned_idc_a, vec_mean_oned_idc_c, vec_random_oned_idc_a, vec_random_oned_idc_c] = \
            load_data_original_idc(self.config.dirinput_nd_val,
                                   [irnd, mean_id],
                                   self.config.z_range, False)
                                   # here we still use mean maps as references (hardcoded)


        mat_mean_corr = np.array((vec_mean_corr_r, vec_mean_corr_rphi, vec_mean_corr_z))
        mat_random_corr = np.array((vec_random_corr_r, vec_random_corr_rphi, vec_random_corr_z))
        mat_fluc_corr = mat_random_corr - mat_mean_corr

        # TODO:
        # Proper format of position-dependent and independent data when both A and C side data used
        data_a = (vec_mean_oned_idc_a, vec_random_oned_idc_a,
                  num_mean_zerod_idc_a, num_random_zerod_idc_a)
        data_c = (vec_mean_oned_idc_c, vec_random_oned_idc_c,
                  num_mean_zerod_idc_c, num_random_zerod_idc_c)
        vec_mean_oned_idc, vec_random_oned_idc, num_mean_zerod_idc, num_random_zerod_idc =\
            filter_idc_data(data_a, data_c, self.config.z_range) # pylint: disable=unbalanced-tuple-unpacking
        vec_fluc_oned_idc = vec_random_oned_idc - vec_mean_oned_idc
        num_fluc_zerod_idc = num_random_zerod_idc - num_mean_zerod_idc
        dft_coeffs = get_fourier_coeffs(vec_fluc_oned_idc,
                                        self.config.num_fourier_coeffs_train,
                                        self.config.num_fourier_coeffs_apply)

        vec_index_random = np.empty(vec_z_pos.size)
        vec_index_random[:] = irnd
        vec_index_mean = np.empty(vec_z_pos.size)
        vec_index_mean[:] = mean_id
        vec_index = np.empty(vec_z_pos.size)
        vec_index[:] = irnd + 1000 * mean_id

        vec_fluc_sc = vec_random_sc - vec_mean_sc
        vec_delta_sc = sum(vec_fluc_sc) / sum(vec_mean_sc)

        df_single_map = pd.DataFrame({"eventId": vec_index.astype('int32'),
                                      "meanId": vec_index_mean.astype('int32'),
                                      "randomId": vec_index_random.astype('int32'),
                                      "r": vec_r_pos.astype('float32'),
                                      "phi": vec_phi_pos.astype('float32'),
                                      "z": vec_z_pos.astype('float32'),
                                      "flucSC": vec_fluc_sc.astype('float32'),
                                      "meanSC": vec_mean_sc.astype('float32'),
                                      "deltaSC": np.tile((vec_delta_sc),
                                                         vec_z_pos.size).astype('float32'),
                                      "derRefMeanSC": vec_der_ref_mean_sc.astype('float32'),
                                      "fluc0DIDC": np.tile((num_fluc_zerod_idc),
                                                           vec_z_pos.size).astype('float32'),
                                      "mean0DIDC": np.tile((num_mean_zerod_idc),
                                                           vec_z_pos.size).astype('float32')})
        for ind_dist, dist_name in enumerate(self.config.nameopt_predout):
            df_single_map["flucCorr" + dist_name] = \
                mat_fluc_corr[ind_dist, :].astype('float32')
            df_single_map["meanCorr" + dist_name] = \
                mat_mean_corr[ind_dist, :].astype('float32')
            df_single_map["derRefMeanCorr" + dist_name] = \
                mat_der_ref_mean_corr[ind_dist, :].astype('float32')
        for i_coeff, (coeff_real, coeff_imag) in enumerate(zip(dft_coeffs[0::2], dft_coeffs[1::2])):
            df_single_map["c%d_real" % i_coeff] = np.tile((coeff_real),
                                                          vec_z_pos.size).astype('float32')
            df_single_map["c%d_imag" % i_coeff] = np.tile((coeff_imag),
                                                          vec_z_pos.size).astype('float32')

        vec_der_ref_mean_corr,  = mat_to_vec(self.config.opt_predout, (mat_der_ref_mean_corr,))
        inputs = get_input_oned_idc_single_map(vec_r_pos, vec_phi_pos, vec_z_pos,
                                                vec_der_ref_mean_corr, dft_coeffs)
        df_single_map["flucCorrRPred"] = loaded_model.predict(inputs).astype('float32')

        dir_name = "%s/%s/parts/%d" % (self.config.dirtree, self.config.suffix, irnd)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        tree_filename = "%s/validation_mean%.2f_nEv%d.root" \
            % (dir_name, self.mean_factors[self.mean_ids.index(mean_id)], self.config.train_events)
        pandas_to_tree(df_single_map, tree_filename, 'validation')
        return tree_filename


    # pylint: disable=too-many-locals, too-many-branches
    def create_data(self):
        self.config.logger.info("DataValidator::create_data")

        loaded_model = self.model.load_model()
        dir_name = "%s/%s" % (self.config.dirtree, self.config.suffix)

        for mean_id in self.mean_ids:
            counter = 0
            out_file_names = []
            merged_file_name = "%s/validation_mean%.2f_nEv%d.root" %\
                               (dir_name, self.mean_factors[self.mean_ids.index(mean_id)],
                                self.config.train_events)

            if self.config.nd_val_partition != 'random':
                for ind_ev in self.config.part_inds:
                    if ind_ev[1] != mean_id:
                        continue
                    irnd = ind_ev[0]
                    self.config.logger.info("processing event: %d [%d, %d]",
                                            counter, mean_id, irnd)
                    out_name = self.create_data_for_event(mean_id, irnd, loaded_model)
                    out_file_names.append(out_name)

                    counter = counter + 1
                    if counter == self.config.nd_val_events:
                        break
            else:
                for irnd in np.arange(self.config.range_rnd_index_nd_val[0],
                                      self.config.range_rnd_index_nd_val[1] + 1):
                    self.config.logger.info("processing event: %d [%d, %d]",
                                            counter, mean_id, irnd)
                    out_name = self.create_data_for_event(mean_id, irnd, loaded_model)
                    out_file_names.append(out_name)

                    counter = counter + 1
                    if counter == self.config.nd_val_events:
                        break

            hadd(out_file_names, merged_file_name)
            for out_file in out_file_names:
                os.remove(out_file)
            self.config.logger.info("Merged trees for mean %d written in %s",
                                    mean_id, merged_file_name)

        self.config.logger.info("All trees written in %s", dir_name)

    def get_pdf_map_variables_list(self):
        dist_names_list = np.array(self.config.nameopt_predout) \
            [np.array([self.config.opt_predout[0], self.config.opt_predout[1],
                       self.config.opt_predout[2]]) > 0]

        var_list = ['flucSC', 'meanSC', 'derRefMeanSC']
        for dist_name in dist_names_list:
            var_list.append('flucCorr' + dist_name + 'Pred')
            var_list.append('flucCorr' + dist_name)
            var_list.append('meanCorr' + dist_name)
            var_list.append('derRefMeanCorr' + dist_name)
            var_list.append('flucCorr' + dist_name + 'Diff')

        return var_list

    def create_nd_histogram(self, var, mean_id):
        """
        Create nd histograms for given variable and mean id
        Only 0 (factor=1.00), 9 (factor=1.03), 18 (factor=0.97),
        27 (factor=1.06) and 36 (factor=0.94) working.

        :param str var: variable name
        :param int mean_id: index of mean map.
        """
        self.config.logger.info("DataValidator::create_nd_histogram, var = %s, mean_id = %d",
                                var, mean_id)
        self.__check_mean_id(mean_id)
        mean_factor = self.__get_mean_factor(mean_id)

        column_names = ['phi', 'r', 'z', 'deltaSC']
        diff_index = var.find("Diff")
        if diff_index == -1:
            column_names.append(var)
        else:
            column_names = column_names + [var[:diff_index], var[:diff_index] + "Pred"]

        df_val = tree_to_pandas("%s/%s/treeValidation_mean%.2f_nEv%d.root"
                                % (self.config.dirtree, self.config.suffix, mean_factor,
                                   self.config.train_events),
                                'validation', columns=column_names)
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
        output_file_name = "%s/%s/ndHistogram_%s_mean%.2f_nEv%d.gzip" \
            % (self.config.dirhist, self.config.suffix, var, mean_factor,
               self.config.train_events)
        with gzip.open(output_file_name, 'wb') as output_file:
            pickle.dump(makeHistogram(df_val, histo_string), output_file)
        output_file.close()
        self.config.logger.info("Nd histogram %s written to %s.", histo_string, output_file_name)

    def create_nd_histograms_meanid(self, mean_id):
        """
        Create nd histograms for given mean id
        Only 0 (factor=1.00), 9 (factor=1.03), 18 (factor=0.97),
        27 (factor=1.06) and 36 (factor=0.94) working.

        :param int mean_id: index of mean map.
        """
        for var in self.get_pdf_map_variables_list():
            self.create_nd_histogram(var, mean_id)

    def create_nd_histograms(self):
        """
        Create nd histograms for mean maps with id 0, 9, 18, 27, 36
        """
        for mean_id in self.mean_ids:
            self.create_nd_histograms_meanid(mean_id)

    def create_pdf_map(self, var, mean_id):
        """
        Create a pdf map for given variable and mean id
        Only 0 (factor=1.00), 9 (factor=1.03), 18 (factor=0.97),
        27 (factor=1.06) and 36 (factor=0.94) working.

        :param str var: variable name
        :param int mean_id: index of mean map.
        """
        self.config.logger.info("DataValidator::create_pdf_map, var = %s, mean_id = %d",
                                var, mean_id)
        self.__check_mean_id(mean_id)
        mean_factor = self.__get_mean_factor(mean_id)

        input_file_name = "%s/%s/ndHistogram_%s_mean%.2f_nEv%d.gzip" \
            % (self.config.dirhist, self.config.suffix, var, mean_factor,
               self.config.train_events)
        with gzip.open(input_file_name, 'rb') as input_file:
            histo = pickle.load(input_file)

        output_file_name = "%s/%s/pdfmap_%s_mean%.2f_nEv%d.root" \
            % (self.config.dirtree, self.config.suffix, var, mean_factor,
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
        Only 0 (factor=1.00), 9 (factor=1.03), 18 (factor=0.97),
        27 (factor=1.06) and 36 (factor=0.94) working.

        :param int mean_id: index of mean map.
        """
        for var in self.get_pdf_map_variables_list():
            self.create_pdf_map(var, mean_id)

    def create_pdf_maps(self):
        """
        Create pdf maps for mean maps with id 0, 9, 18, 27, 36
        """
        for mean_id in self.mean_ids:
            self.create_pdf_maps_meanid(mean_id)

    def merge_pdf_maps(self, mean_ids=None):
        """
        Merge pdf maps for different variables into one file

        :param list mean_ids: list of indices of mean maps, whose corresponding pdf maps
                              will be merged
        """
        self.config.logger.info("DataValidator::merge_pdf_maps")

        if mean_ids is None:
            mean_ids = self.mean_ids
        mean_factors = [self.mean_factors[mean_ids.index(mean_id)] for mean_id in mean_ids]

        df_merged = pd.DataFrame()
        for mean_factor in mean_factors:
            input_file_name_0 = "%s/%s/pdfmap_flucSC_mean%.2f_nEv%d.root" \
                % (self.config.dirtree, self.config.suffix, mean_factor,
                   self.config.train_events)
            df = tree_to_pandas(input_file_name_0, 'flucSC', filter_name="*Bin*")
            df['fsector'] = df['phiBinCenter'] / math.pi * 9
            df['meanMap'] = mean_factor
            for var in self.get_pdf_map_variables_list():
                input_file_name = "%s/%s/pdfmap_%s_mean%.2f_nEv%d.root" \
                    % (self.config.dirtree, self.config.suffix, var, mean_factor,
                       self.config.train_events)
                df_temp = tree_to_pandas(input_file_name, var,
                                         filter_name="/(means|medians|rms|entries)/")
                for col in list(df_temp.columns):
                    df[var + '_' + col] = df_temp[col]
            df_merged = df_merged.append(df, ignore_index=True)

        output_file_name = "%s/%s/pdfmaps_nEv%d.root" \
            % (self.config.dirtree, self.config.suffix, self.config.train_events)
        pandas_to_tree(df_merged, output_file_name, 'pdfmaps')
        self.config.logger.info("Pdf maps written to %s.", output_file_name)

    def merge_pdf_maps_meanid(self, mean_id):
        """
        Merge pdf maps for given mean id
        Only 0 (factor=1.00), 9 (factor=1.03), 18 (factor=0.97),
        27 (factor=1.06) and 36 (factor=0.94) working.

        :param int mean_id: index of mean map.
        """
        self.__check_mean_id(mean_id)
        self.merge_pdf_maps([mean_id])

    def __check_mean_id(self, mean_id):
        """
        A shortcut to check mean map id, as many functions are designed only for specific ids.

        :param int mean_id: index of mean map
        """
        if mean_id not in self.mean_ids:
            self.config.logger.error("Code implementation only designed for mean ids",
                                     self.mean_ids)
            self.config.logger.fatal("Exiting...")

    def __get_mean_factor(self, mean_id):
        """
        Convert mean map id to a mean factor

        :param int mean_id: index of mean map
        :return: mean factor
        :rtype: double
        """
        return self.mean_factors[self.mean_ids.index(mean_id)]
