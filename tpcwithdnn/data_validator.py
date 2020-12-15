# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=too-many-statements, too-many-instance-attributes
import sys
import os
import gzip
import pickle
import math
import matplotlib
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from root_pandas import to_root, read_root  # pylint: disable=import-error, unused-import
from RootInteractive.Tools.histoNDTools import makeHistogram  # pylint: disable=import-error, unused-import
from RootInteractive.Tools.makePDFMaps import makePdfMaps  # pylint: disable=import-error, unused-import

from tpcwithdnn.logger import get_logger
from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d
from tpcwithdnn.data_loader import load_data_original
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
        self.dirouthistograms = data_param["dirouthistograms"]
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

        events_file = "%s/events_%s_%s_nEv%d.csv" % (self.dirmodel, self.use_partition,
                                                     self.suffix, self.train_events)
        part_inds = np.genfromtxt(events_file, delimiter=",")
        self.part_inds = part_inds[(part_inds[:,1] == 0) | (part_inds[:,1] == 9) | \
                                     (part_inds[:,1] == 18)]

    def create_data_for_event(self, imean, irnd, column_names, vec_der_ref_mean_sc,
                              mat_der_ref_mean_dist, loaded_model, tree_filename):
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
        vec_index[:] = irnd + 1000 * imean
        vec_fluc_sc = vec_mean_sc - vec_random_sc
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

    # pylint: disable=too-many-locals, too-many-branches
    def create_data(self):
        self.logger.info("DataValidator::create_data")

        vec_der_ref_mean_sc, mat_der_ref_mean_dist = \
            load_data_derivatives_ref_mean(self.dirinput_val, self.selopt_input)

        dist_names = np.array(self.nameopt_predout)[np.array(self.opt_predout) > 0]
        column_names = np.array(["eventId", "meanId", "randomId", "r", "phi", "z",
                                 "flucSC", "meanSC", "deltaSC", "derRefMeanSC"])
        for dist_name in self.nameopt_predout:
            column_names = np.append(column_names, ["flucDist" + dist_name,
                                                    "meanDist" + dist_name,
                                                    "derRefMeanDist" + dist_name])
        if self.validate_model:
            json_file = open("%s/model_%s_nEv%d.json" % \
                             (self.dirmodel, self.suffix, self.train_events), "r")
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = \
                model_from_json(loaded_model_json, {'SymmetryPadding3d' : SymmetryPadding3d})
            loaded_model.load_weights("%s/model_%s_nEv%d.h5" % \
                                      (self.dirmodel, self.suffix, self.train_events))

            for dist_name in dist_names:
                column_names = np.append(column_names, ["flucDist" + dist_name + "Pred"])

        for imean, mean_factor in zip([0, 9, 18], [1.0, 1.1, 0.9]):
            tree_filename = "%s/treeInput_mean%.1f_%s.root" \
                            % (self.diroutflattree, mean_factor, self.suffix_ds)
            if self.validate_model:
                tree_filename = "%s/%s/treeValidation_mean%.1f_nEv%d.root" \
                                % (self.diroutflattree, self.suffix, mean_factor, self.train_events)

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
                                               mat_der_ref_mean_dist, loaded_model, tree_filename)
                    counter = counter + 1
                    if counter == self.tree_events:
                        break
            else:
                for irnd in range(self.maxrandomfiles):
                    self.logger.info("processing event: %d [%d, %d]", counter, imean, irnd)
                    self.create_data_for_event(imean, irnd, column_names, vec_der_ref_mean_sc,
                                               mat_der_ref_mean_dist, loaded_model, tree_filename)
                    counter = counter + 1
                    if counter == self.tree_events:
                        break

            self.logger.info("Tree written in %s", tree_filename)


    def get_pdf_map_variables_list(self):
        dist_names_list = np.array(self.nameopt_predout) \
            [np.array([self.opt_predout[0], self.opt_predout[1], self.opt_predout[2]]) > 0]

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
        self.logger.info("DataValidator::create_nd_histogram, var = %s, mean_id = %d", var, mean_id)
        if mean_id not in (0, 9, 18):
            self.logger.info("Code implementation only designed for mean ids 0, 9, 18. Exiting...")
            sys.exit()
        mean_factor = 1 + 0.1 * (mean_id != 0) * (1 - 2 * (mean_id == 18))

        column_names = ['phi', 'r', 'z', 'deltaSC']
        diff_index = var.find("Diff")
        if diff_index == -1:
            column_names.append(var)
        else:
            column_names = column_names + [var[:diff_index], var[:diff_index] + "Pred"]

        df_val = read_root("%s/%s/treeValidation_mean%.1f_nEv%d.root"
                           % (self.diroutflattree, self.suffix, mean_factor, self.train_events),
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
            % (self.dirouthistograms, self.suffix, var, mean_factor, self.train_events)
        with gzip.open(output_file_name, 'wb') as output_file:
            pickle.dump(makeHistogram(df_val, histo_string), output_file)
        output_file.close()
        self.logger.info("Nd histogram %s written to %s.", histo_string, output_file_name)


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
        self.logger.info("DataValidator::create_pdf_map, var = %s, mean_id = %d", var, mean_id)
        if mean_id not in (0, 9, 18):
            self.logger.info("Code implementation only designed for mean ids 0, 9, 18. Exiting...")
            sys.exit()
        mean_factor = 1 + 0.1 * (mean_id != 0) * (1 - 2 * (mean_id == 18))

        input_file_name = "%s/%s/ndHistogram_%s_mean%.1f_nEv%d.gzip" \
            % (self.dirouthistograms, self.suffix, var, mean_factor, self.train_events)
        with gzip.open(input_file_name, 'rb') as input_file:
            histo = pickle.load(input_file)

        output_file_name = "%s/%s/pdfmap_%s_mean%.1f_nEv%d.root" \
            % (self.diroutflattree, self.suffix, var, mean_factor, self.train_events)
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
        self.logger.info("Pdf map %s written to %s.", histo['name'], output_file_name)


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
        self.logger.info("DataValidator::merge_pdf_maps")

        if mean_ids is None:
            mean_ids = [0, 9, 18]
        mean_ids_to_factors = {0: 1.0, 9: 1.1, 18: 0.9}
        mean_factors = [mean_ids_to_factors[mean_id] for mean_id in mean_ids]

        df_merged = pd.DataFrame()
        for mean_factor in mean_factors:
            input_file_name_0 = "%s/%s/pdfmap_flucSC_mean%.1f_nEv%d.root" \
                % (self.diroutflattree, self.suffix, mean_factor, self.train_events)
            df = read_root(input_file_name_0, columns="*Bin*")
            df['fsector'] = df['phiBinCenter'] / math.pi * 9
            df['meanMap'] = mean_factor
            for var in self.get_pdf_map_variables_list():
                input_file_name = "%s/%s/pdfmap_%s_mean%.1f_nEv%d.root" \
                    % (self.diroutflattree, self.suffix, var, mean_factor, self.train_events)
                df_temp = read_root(input_file_name, ignore="*Bin*")
                for col in list(df_temp.keys()):
                    df[var + '_' + col] = df_temp[col]
            df_merged = df_merged.append(df, ignore_index=True)

        output_file_name = "%s/%s/pdfmaps_nEv%d.root" \
            % (self.diroutflattree, self.suffix, self.train_events)
        df_merged.to_root(output_file_name, key='pdfmaps', mode='w', store_index=False)
        self.logger.info("Pdf maps written to %s.", output_file_name)

    def merge_pdf_maps_meanid(self, mean_id):
        """
        Merge pdf maps for given mean id
        mean_id: index of mean map. Only 0 (factor=1.0), 9 (factor=1.1) and 18 (factor=0.9) working.
        """
        if mean_id not in (0, 9, 18):
            self.logger.info("Code implementation only designed for mean ids 0, 9, 18. Exiting...")
            sys.exit()
        self.merge_pdf_maps([mean_id])
