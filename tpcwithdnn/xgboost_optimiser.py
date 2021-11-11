"""
XGBoost optimizer for 1D IDC distortion correction
"""
from timeit import default_timer as timer

from itertools import chain
import math
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot3
from xgboost import XGBRFRegressor

from sklearn.metrics import mean_squared_error

from ROOT import TFile # pylint: disable=import-error, no-name-in-module

from tpcwithdnn import plot_utils
from tpcwithdnn.debug_utils import log_time, log_memory_usage, log_total_memory_usage
from tpcwithdnn.tree_df_utils import pandas_to_tree, tree_to_pandas
from tpcwithdnn.optimiser import Optimiser
from tpcwithdnn.data_loader import load_data_oned_idc, get_input_names_oned_idc
from tpcwithdnn.hadd import hadd

class XGBoostOptimiser(Optimiser):
    """
    XGBoost optimizer class, with the interface defined by the Optimiser parent class
    """
    name = "xgboost"

    def __init__(self, config):
        """
        Initialize the optimizer. No more action needed that in the base class.

        :param CommonSettings config: a singleton settings object
        """
        super().__init__(config)
        self.config.logger.info("XGBoostOptimiser::Init")

    def train(self):
        """
        Train the optimizer.
        """
        self.config.logger.info("XGBoostOptimiser::train")
        model = XGBRFRegressor(verbosity=1, **(self.config.params))
        start = timer()
        inputs, exp_outputs = self.get_data_("train")
        end = timer()
        log_time(start, end, "for loading training data")
        log_memory_usage(((inputs, "Input train data"), (exp_outputs, "Output train data")))
        log_total_memory_usage("Memory usage after loading data")
        if self.config.plot_train:
            inputs_val, outputs_val = self.get_data_("validation")
            log_memory_usage(((inputs_val, "Input validation data"),
                              (outputs_val, "Output validation data")))
            log_total_memory_usage("Memory usage after loading validation data")
            self.plot_train_(model, inputs, exp_outputs, inputs_val, outputs_val)
        start = timer()
        model.fit(inputs, exp_outputs)
        end = timer()
        log_time(start, end, "actual train")
        model.get_booster().feature_names = get_input_names_oned_idc(
            self.config.num_fourier_coeffs_train)
        self.plot_feature_importance_(model)
        self.save_model(model)

    def apply(self):
        """
        Apply the optimizer.
        """
        self.config.logger.info("XGBoostOptimiser::apply, input size: %d", self.config.dim_input)
        loaded_model = self.load_model()
        inputs, exp_outputs = self.get_data_("apply")
        log_memory_usage(((inputs, "Input apply data"), (exp_outputs, "Output apply data")))
        log_total_memory_usage("Memory usage after loading apply data")
        start = timer()
        pred_outputs = loaded_model.predict(inputs)
        end = timer()
        log_time(start, end, "actual predict")
        self.plot_apply_(exp_outputs, pred_outputs)
        self.config.logger.info("Done apply")

    def search_grid(self):
        """
        Perform grid search to find the best model configuration.

        :raises NotImplementedError: the method not implemented yet for XGBoost
        """
        raise NotImplementedError("Search grid method not implemented yet")

    def bayes_optimise(self):
        """
        Perform Bayesian optimization to find the best model configuration.

        :raises NotImplementedError: the method not implemented yet for XGBoost
        """
        raise NotImplementedError("Bayes optimise method not implemented yet")

    def save_model(self, model):
        """
        Save the model to a JSON file. Saves also plots of feature importances into separate files.

        :param xgboost.sklearn.XGBModel model: the XGBoost model to be saved
        """
        # Snapshot - can be used for further training
        out_filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        with open(out_filename, "wb") as out_file:
            pickle.dump(model, out_file, protocol=4)


    def load_model(self):
        """
        Load the XGBoost model from a JSON file

        :return: the loaded model
        :rtype: xgboost.sklearn.XGBModel
        """
        # Loading a snapshot
        filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        with open(filename, "rb") as file:
            model = pickle.load(file)
        return model

    def get_data_(self, partition):
        """
        Load the full input data for a XGBoost optimization.
        Function used internally.

        :param str partition: name of partition, one from "train", "validation", "apply"
        :return: tuple of inputs and expected outputs
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        num_fourier_coeffs_apply = self.config.num_fourier_coeffs_apply
        downsample = False
        if partition == "train":
            downsample = self.config.downsample
            # Take all Fourier coefficients for training
            num_fourier_coeffs_apply = self.config.num_fourier_coeffs_train
            if self.config.dump_train and self.config.train_events <= self.config.cache_events:
                return self.get_cache_(partition, downsample, num_fourier_coeffs_apply)
        return self.get_partition_(partition, downsample, num_fourier_coeffs_apply, slice(None))

    def get_partition_(self, partition, downsample, num_fourier_coeffs_apply, part_range):
        """
        Load the input data from given partition. Function used internally.

        :param str partition: name of partition, one from "train", "validation", "apply"
        :param bool downsample: whether to downsample the data
        :param int num_fourier_coeffs_apply: number of Fourier coefficients for applying
        :param slice part_range: range of data to be taken from the partition,
                                 e.g., slice(0, 10) for first 10 event indices from the partition
        :return: tuple of inputs and expected outputs
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        dirinput = getattr(self.config, "dirinput_" + partition)
        inputs = []
        exp_outputs = []
        for indexev in self.config.partition[partition][part_range]:
            inputs_single, exp_outputs_single = load_data_oned_idc(self.config, dirinput,
                                                           indexev, downsample,
                                                           self.config.num_fourier_coeffs_train,
                                                           num_fourier_coeffs_apply)
            inputs.append(inputs_single)
            exp_outputs.append(exp_outputs_single)
        inputs = np.concatenate(inputs)
        exp_outputs = np.concatenate(exp_outputs)

        return inputs, exp_outputs

    def cache_train_data(self):
        """
        Cache train data if it is not cached.
        """
        self.get_cache_("train", self.config.downsample, self.config.num_fourier_coeffs_train)

    def read_cache_from_file(self, filepath):
        """
        Read the cached data from the file specified. Function used internally.

        :param str filepath: full path to the cache file
        :return: tuple of inputs and expected outputs
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        self.config.logger.info("Reading cache from %s", filepath)
        with uproot3.open(filepath) as root_file:
            print("File keys: {}".format(root_file.keys()))
            print("Tree: {}".format(root_file["cache"]))
            input_data = root_file["cache"].pandas.df(["*"])
        #input_data = tree_to_pandas(filepath, "cache", ["*"])
        print(input_data)
        inputs = input_data.filter(regex="^(?!exp).*").to_numpy()
        exp_outputs = input_data.filter(like="exp").to_numpy()
        self.config.logger.info("Data read from cache: %s", filepath)
        return inputs, exp_outputs

    def get_batch_range_(self, partition, batch_size):
        """
        A generator to calculate the next batch of data from the partition.

        :param str partition: name of partition, one from "train", "validation", "apply"
        :param int batch_size: size of one batch
        :return: a slice with minimum (inclusive) and maximum (exclusive) indices for the batch
        :rtype: slice
        """
        data_size = len(self.config.partition[partition])
        last_full_end = int(np.floor(data_size / batch_size)) * batch_size
        for i in range(0, last_full_end, batch_size):
            yield slice(i, i + 1)
        if last_full_end < data_size:
            yield slice(last_full_end, data_size)

    def get_cache_(self, partition, downsample, num_fourier_coeffs_apply):
        """
        Get the cached data from given partition. Function used internally.

        :param str partition: name of partition, one from "train", "validation", "apply"
        :param bool downsample: whether to downsample the data
        :param int num_fourier_coeffs_apply: number of Fourier coefficients for applying
        :return: tuple of inputs and expected outputs
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        self.config.logger.info("Searching for cached data")
        filename = "%s_%sEv%d" % (self.config.cache_suffix, partition, self.config.cache_events)
        full_path = "%s/%s" % (self.config.dirinput_cache, filename)
        cache_file = "%s.root" % full_path
        try:
            inputs, exp_outputs = self.read_cache_from_file(cache_file)
            self.config.logger.info("Found cache: %s", cache_file)
            return inputs, exp_outputs
        except FileNotFoundError:
            self.config.logger.info("Cache: %s does not exist, saving new cache", cache_file)
            fourier_names = list(chain.from_iterable(("Fourier real %d" % i, "Fourier imag %d" % i)
                                 for i in range(self.config.num_fourier_coeffs_train)))
            batch_file_names = []
            batch_size = self.config.cache_file_size
            for i, part_range in enumerate(self.get_batch_range_(partition, batch_size)):
                inputs, exp_outputs = self.get_partition_(partition, downsample,
                                                          num_fourier_coeffs_apply, part_range)
                input_data = np.hstack((inputs, exp_outputs.reshape(-1, 1)))
                cache_data = pd.DataFrame(input_data,
                                          columns=["r", "phi", "z", "der mean corr"] +\
                                                  fourier_names + ["exp correction fluctuations"])
                batch_file = "%s_%d.root" % (full_path, i)
                batch_file_names.append(batch_file)
                pandas_to_tree(cache_data, batch_file, "cache")
                self.config.logger.info("Cache: %s saved", batch_file)
            hadd(batch_file_names, cache_file)
            #for batch_file in batch_file_names:
            #    os.remove(batch_file)
            self.config.logger.info("Merged cache: %s saved", cache_file)
            return self.read_cache_from_file(cache_file)

    def plot_feature_importance_(self, model):
        """
        Create plots and text file of feature importances (total gain, weight, gain).

        :param xgboost.sklearn.XGBModel model: the XGBoost model to use
        """
        score_total_gain = model.get_booster().get_score(importance_type='total_gain')
        score_gain = model.get_booster().get_score(importance_type='gain')
        score_weight = model.get_booster().get_score(importance_type='weight')
        out_filename_feature_importance = "%s/feature_importance_%s_nEv%d.txt" %\
            (self.config.dirmodel, self.config.suffix, self.config.train_events)

        # importances sorted according to list of feature names
        feature_sorted = []
        total_gain_sorted = []
        gain_sorted = []
        weight_sorted = []
        for feature in model.get_booster().feature_names:
            if feature in score_total_gain.keys():
                feature_sorted.append(feature)
                total_gain_sorted.append(score_total_gain[feature])
                gain_sorted.append(score_gain[feature])
                weight_sorted.append(score_weight[feature])

        with open(out_filename_feature_importance, 'w', encoding="utf-8") as file_name:
            print("Feature importances", file=file_name)
            print("Feature:   Total gain   -   Gain   -   Weight", file=file_name)
            for index, feature in enumerate(feature_sorted):
                print("%s:   %.4f   -  %.4f   -   %.4f" %
                      (feature, total_gain_sorted[index],
                       gain_sorted[index], weight_sorted[index]),
                      file=file_name)

        df_importance = pd.DataFrame({'total_gain': total_gain_sorted,
                                      'gain': gain_sorted,
                                      'weight': weight_sorted}, index=feature_sorted)
        bar_colors = ['tab:orange', 'tab:green', 'tab:blue']
        for importance_type, bar_color in zip(df_importance.columns, bar_colors):
            px = 1/plt.rcParams['figure.dpi']
            df_importance.plot(kind='bar', y=importance_type, log=True,
                                    color=bar_color, figsize=(1200*px, 400*px))
            plt.title("n_estimators: %d, max_depth: %d, down_npoints: %.3f, train_maps: %d" %
                      (self.config.params["n_estimators"],
                       self.config.params["max_depth"],
                       self.config.downsample_npoints,
                       self.config.train_events))
            plt.tight_layout()
            plt.ylim(
                bottom=math.pow(10, math.floor(math.log(df_importance[importance_type].min(), 10))))
            plt.savefig("%s/figImportances_%s_%s_nEv%d.pdf" %
                        (self.config.dirplots, importance_type, self.config.suffix,
                         self.config.train_events))


    def plot_apply_(self, exp_outputs, pred_outputs):
        """
        Create result histograms in the output ROOT file after applying the model.
        Function used internally.

        :param np.ndarray exp_outputs: vector of expected outputs
        :param np.ndarray pred_outputs: vector of network predictions
        """
        myfile = TFile.Open("%s/output_%s_nEv%d.root" % \
                            (self.config.dirapply, self.config.suffix, self.config.train_events),
                            "recreate")
        h_dist_all_events, h_deltas_all_events, h_deltas_vs_dist_all_events =\
                plot_utils.create_apply_histos(self.config, self.config.suffix, infix="all_events_")
        distortion_numeric_flat_m, distortion_predict_flat_m, deltas_flat_a, deltas_flat_m =\
            plot_utils.get_apply_results_single_event(pred_outputs, exp_outputs)
        plot_utils.fill_apply_tree(h_dist_all_events, h_deltas_all_events,
                                   h_deltas_vs_dist_all_events,
                                   distortion_numeric_flat_m, distortion_predict_flat_m,
                                   deltas_flat_a, deltas_flat_m)

        for hist in (h_dist_all_events, h_deltas_all_events, h_deltas_vs_dist_all_events):
            hist.Write()
        plot_utils.fill_profile_apply_hist(h_deltas_vs_dist_all_events, self.config.profile_name,
                                           self.config.suffix)
        plot_utils.fill_std_dev_apply_hist(h_deltas_vs_dist_all_events, self.config.h_std_dev_name,
                                           self.config.suffix, "all_events_")

        myfile.Close()

    def plot_train_(self, model, x_train, y_train, x_val, y_val):
        """
        Plot the learning curve for 1D calibration.
        Function used internally.

        :param xgboost.sklearn.XGBModel model: the XGBoost model to be checked
        :param np.ndarray x_train: input data for training
        :param np.ndarray y_train: expected training output
        :param np.ndarray x_train: input data for validation
        :param np.ndarray y_train: expected validation output
        """
        plt.figure()
        train_errors, val_errors = [], []
        data_size = len(x_train)
        size_per_event = int(data_size / self.config.train_events)
        step = int(data_size / self.config.train_plot_npoints)
        checkpoints = np.arange(start=size_per_event, stop=data_size, step=step)
        for ind, checkpoint in enumerate(checkpoints):
            model.fit(x_train[:checkpoint], y_train[:checkpoint])
            y_train_predict = model.predict(x_train[:checkpoint])
            y_val_predict = model.predict(x_val)
            train_errors.append(mean_squared_error(y_train_predict, y_train[:checkpoint]))
            val_errors.append(mean_squared_error(y_val_predict, y_val))
            if ind in (0, self.config.train_plot_npoints // 2, self.config.train_plot_npoints - 1):
                self.plot_results_(y_train[:checkpoint], y_train_predict, "train-%d" % ind)
                self.plot_results_(y_val, y_val_predict, "val-%d" % ind)
        self.config.logger.info("Memory usage during plot train")
        log_total_memory_usage()
        plt.plot(checkpoints, np.sqrt(train_errors), ".", label="train")
        plt.plot(checkpoints, np.sqrt(val_errors), ".", label="validation")
        plt.ylim([0, np.amax(np.sqrt(val_errors)) * 2])
        plt.title("Learning curve BDT")
        plt.xlabel("Training set size")
        plt.ylabel("RMSE")
        plt.legend(loc="lower left")
        plt.savefig("%s/learning_plot_%s_nEv%d.png" % (self.config.dirplots, self.config.suffix,
                                                       self.config.train_events))
        plt.clf()

    def plot_results_(self, exp_outputs, pred_outputs, infix):
        """
        Plot the diagram of predicted vs expected 1D calibration output after applying the model.
        Function used internally.

        :param np.ndarray exp_outputs: vector of expected outputs
        :param np.ndarray pred_outputs: vector of network predictions
        :param str infix: the string to be inserted into the output file name
        """
        plt.figure()
        plt.plot(exp_outputs, pred_outputs, ".")
        plt.xlabel("Expected output")
        plt.ylabel("Predicted output")
        plt.savefig("%s/num-exp-%s_%s_nEv%d.png" % (self.config.dirplots, infix, self.config.suffix,
                                                   self.config.train_events))
        plt.clf()
