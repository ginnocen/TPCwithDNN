"""
XGBoost optimizer for 1D IDC distortion correction
"""
from timeit import default_timer as timer

import array
from itertools import chain
import math
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRFRegressor
from xgboost import XGBRegressor

import tensorflow 
from tensorflow import keras

from sklearn.metrics import mean_squared_error

from ROOT import TFile, TTree # pylint: disable=import-error, no-name-in-module

from tpcwithdnn import plot_utils
from tpcwithdnn.debug_utils import log_time, log_memory_usage, log_total_memory_usage
from tpcwithdnn.tree_df_utils import pandas_to_tree, tree_to_pandas
from tpcwithdnn.optimiser import Optimiser
from tpcwithdnn.data_loader import load_data_oned_idc, get_input_names_oned_idc
from tpcwithdnn.hadd import hadd

from tpcwithdnn.nn_utils import nn_1d

class Fitter():
    """
    class for ML network switching. Still work in progress(2.6.2022).
    """
    def __init__(self, config):
        if config.xgbtype=="XGB":
            self.model = XGBRegressor(verbosity=1, **(config.params))
            config.logger.info("XGBoostOptimiser::train, XGBoost type: XGBRegressor")
        elif config.xgbtype=="RF":
            self.model = XGBRFRegressor(verbosity=1, **(config.params))
            config.logger.info("XGBoostOptimiser::train, XGBoost type: XGBRFRegressor")
        elif config.xgbtype=="NN":
            #self.model = nn_1d()
            # self.model will be declared in fitModel()
            config.logger.info("XGBoostOptimiser::train, XGBoost type: Neural Network(keras)")
        else:
            config.logger.fatal("Unknown Optimiser type! (param xgbtype must be 'XGB', 'RF' or 'NN')")
 
    def fitModel(self, config, inputs, exp_outputs, inputs_val, outputs_val):
        if config.xgbtype=="XGB" or config.xgbtype=="RF":
            self.model.fit(inputs, exp_outputs)
        elif config.xgbtype=="NN":
            self.model=nn_1d(self, config, inputs, exp_outputs, inputs_val, outputs_val)
            #config.logger.fatal("Neural network not implemented yet!")
        else:
            config.logger.fatal("Unknown optimiser type! (param xgbtype must be 'XGB', 'RF' or 'NN')")


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
        self.fourierMean = np.empty(0)   #Member arrays to store Mean and Std. Dev. of the fourier coefficents in the training dataset.
        self.fourierStdDev = np.empty(0) # Used in function normalize_inputs to standardize the inputs for the neural network.
        self.config.logger.info("XGBoostOptimiser::Init")

    def train(self):
        """
        Train the optimizer.
        """
        self.config.logger.info("XGBoostOptimiser::train") 
        fitter = Fitter(self.config)
        start = timer()
#<<<<<<< HEAD
        inputs, exp_outputs, *_ = self.__get_data("train")
        inputs_val, outputs_val, *_ = self.__get_data("validation")
        #if self.config.xgbtype=="NN":
            #inputs = self.normalize_inputs(inputs)
            #inputs_val = self.normalize_inputs(inputs_val)
            #inputs = self.ver_normalize_inputs(inputs, "train")
            #inputs_val = self.ver_normalize_inputs(inputs_val, "validation")
#=======
#        inputs, exp_outputs, *_ = self.__get_data("train")
#>>>>>>> cda465f115b7880e0edb78c7d66497c1e856c59b
        end = timer()
        log_time(start, end, "for loading training data")
        log_memory_usage(((inputs, "Input train data"), (exp_outputs, "Output train data")))
        log_memory_usage(((inputs_val, "Input validation data"), (outputs_val, "Output validation data")))
        log_total_memory_usage("Memory usage after loading data")
        """
        if self.config.plot_train:
            inputs_val, outputs_val, *_ = self.__get_data("validation")
            log_memory_usage(((inputs_val, "Input validation data"),
                              (outputs_val, "Output validation data")))
            log_total_memory_usage("Memory usage after loading validation data")
            self.__plot_train(model, inputs, exp_outputs, inputs_val, outputs_val)
        """
        start = timer()
        fitter.fitModel(self.config, inputs, exp_outputs, inputs_val, outputs_val)
        end = timer()
        log_time(start, end, "actual train")
        if self.config.xgbtype=="XGB" or self.config.xgbtype=="RF":
            fitter.model.get_booster().feature_names = get_input_names_oned_idc(
                self.config.opt_usederivative, self.config.num_fourier_coeffs_train)
            self.__plot_feature_importance(fitter.model)
            self.save_model(fitter.model)
        elif self.config.xgbtype=="NN":
            self.save_model(fitter.model)

    def apply(self):
        """
        Apply the optimizer.
        """
        self.config.logger.info("XGBoostOptimiser::apply, input size: %d", self.config.dim_input)
        loaded_model = self.load_model()
#<<<<<<< HEAD
        inputs, exp_outputs, indices, _= self.__get_data("apply")
        if self.config.xgbtype=="NN":
            pos = np.empty((inputs.shape[0], 3))
            for i in range(3):
                pos[:, i] = inputs[:,i]
            #inputs = self.normalize_inputs(inputs)
            #inputs = self.ver_normalize_inputs(inputs, "apply")
#=======
#        inputs, exp_outputs, *_ = self.__get_data("apply")
#>>>>>>> cda465f115b7880e0edb78c7d66497c1e856c59b
        log_memory_usage(((inputs, "Input apply data"), (exp_outputs, "Output apply data")))
        log_total_memory_usage("Memory usage after loading apply data")
        start = timer()
        pred_outputs = loaded_model.predict(inputs)
        end = timer()
        log_time(start, end, "actual predict")
        if self.config.xgbtype=="NN":
            for i in range(3):
                inputs[:,i] = pos[:,i]
        self.__plot_apply(inputs, exp_outputs, pred_outputs, indices)
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
        if self.config.xgbtype=="XGB":
            out_filename = "%s/xgbmodel_%s_nEv%d.json" %\
                    (self.config.dirmodel, self.config.suffix, self.config.train_events)
            with open(out_filename, "wb") as out_file:
                pickle.dump(model, out_file, protocol=4)

        elif self.config.xgbtype=="RF":
            out_filename = "%s/RFmodel_%s_nEv%d.json" %\
                    (self.config.dirmodel, self.config.suffix, self.config.train_events)
            with open(out_filename, "wb") as out_file:
                pickle.dump(model, out_file, protocol=4)
        # For Keras neural network, only the architecture of the model can be saved into a json file. By using model.save, 
        # we can save the entire information including the set of weight values, etc.
        elif self.config.xgbtype=="NN":
            out_filename = "%s/NNmodel_%s_nEv%d" %\
                    (self.config.dirmodel, self.config.suffix, self.config.train_events)
            model.save(out_filename)
            np.save(out_filename+"/fourierCoeffMean", self.fourierMean)
            np.save(out_filename+"/fourierCoeffStdDev", self.fourierStdDev)


    def load_model(self):
        """
        Load the XGBoost model from a JSON file

        :return: the loaded model
        :rtype: xgboost.sklearn.XGBModel
        """
        if self.config.xgbtype=="XGB":
            filename = "%s/xgbmodel_%s_nEv%d.json" %\
                    (self.config.dirmodel, self.config.suffix, self.config.train_events)
            with open(filename, "rb") as file:
                model = pickle.load(file)
        if self.config.xgbtype=="RF":
            filename = "%s/RFmodel_%s_nEv%d.json" %\
                    (self.config.dirmodel, self.config.suffix, self.config.train_events)
            with open(filename, "rb") as file:
                model = pickle.load(file)
        if self.config.xgbtype=="NN":
            filename = "%s/NNmodel_%s_nEv%d" %\
                    (self.config.dirmodel, self.config.suffix, self.config.train_events)
            model = keras.models.load_model(filename)
            fourierCoeffMean = np.load(filename+"/fourierCoeffMean.npy")
            fourierCoeffStdDev = np.load(filename+"/fourierCoeffStdDev.npy")
            if fourierCoeffMean.shape==self.fourierMean.shape:
                self.fourierMean = fourierCoeffMean
            elif self.fourierMean.shape==(0,):
                self.fourierMean = np.hstack([self.fourierMean, fourierCoeffMean])
            else:
                print("self.fourierMean.shape=", self.fourierMean.shape)
                print("Shape of data read from numpy file:", fourierCoeffMean.shape)
                self.config.logger.fatal("Number of fourier coefficents used for the last training does not match that used for training of the loaded model.")
            if fourierCoeffStdDev.shape==self.fourierStdDev.shape:
                self.fourierStdDev = fourierCoeffStdDev
            elif self.fourierStdDev.shape==(0,):
                self.fourierStdDev = np.hstack([self.fourierStdDev, fourierCoeffStdDev])
            else:
                self.config.logger.fatal("Number of fourier coefficents used for the last training does not match that used for training of the loaded model.")
       # Loading a snapshot
        #filename = "%s/xgbmodel_%s_nEv%d.json" %\
        #        (self.config.dirmodel, self.config.suffix, self.config.train_events)
        #with open(filename, "rb") as file:
        #    model = pickle.load(file)
        return model

    def cache_train_data(self):
        """
        Cache train data if it is not cached.
        """
        self.config.logger.info("Searching for cached data")
        filename = "%s_cacheEv%d" % (self.config.cache_suffix, self.config.cache_events)
        full_path = "%s/%s" % (self.config.dircache, filename)
        cache_file = "%s.root" % full_path
        try:
            with open(cache_file, encoding="utf-8") as _ :
                self.config.logger.info("Found cache: %s", cache_file)
        except FileNotFoundError:
            self.config.set_cache_ranges()
            self.__save_cache(full_path, "train", self.config.downsample,
                             self.config.num_fourier_coeffs_train)

    def normalize_inputs(self, inputs):
        """
        Standardize the input data for the neural network (it may be effective for XGBoost or RF as well)
        :inputs: Input for the network.
        """
        #n = inputs.shape[1]
        n_points = inputs.shape[0]
        # inputs[:,0]~[:,2] are the positions. [:,3] is the derivative and [:,4] is the real part of the 0th Fourier coefficent.
        # This means [:,5] is the imaginary part of the 0th Fourier coeffficent, which is always 0 (i.e. std. dev. is also 0).
        inputs[:,0] = (inputs[:,0]-169) / 86 # Trying to map r= 83 to -1 and r=255 to 1
        inputs[:,1] = (inputs[:,1] / math.pi) - 1 # Map  phi=0 to -1 and phi=2*pi to 1
        inputs[:,2] = (inputs[:,2] / 125)  - 1 # Map z=0 to -1 and z=250 to 1

        for i in range(n_points):
            t = inputs[i,4:]
            inputs[i,4:]=(inputs[i,4:]-t.mean())/t.std()
        return inputs

    def ver_normalize_inputs(self, inputs, partition):
        """
        Test function for a different way of doing the normalization for Neural Network.
        Standardize the input data for the neural network (it may be effective for XGBoost or RF as well)
        :inputs: Input for the network.
        :partition: Partition. "train", "validation", "apply"
        """
        n_coeffs = inputs.shape[1] - 4
        if partition=="train":
            foMean = np.empty(n_coeffs)
            foStdDev = np.empty(n_coeffs)
            for i in range(4, inputs.shape[1]):
                if i==5:
                    foMean[i-4] = 0
                    foStdDev[i-4] = 0
                else:
                    t = inputs[:,i]
                    foMean[i-4] = t.mean()
                    foStdDev[i-4] = t.std()
            self.fourierMean = np.hstack([self.fourierMean, foMean])
            self.fourierStdDev = np.hstack([self.fourierStdDev, foStdDev])
        #n_points = inputs.shape[0]
        # inputs[:,0]~[:,2] are the positions. [:,3] is the derivative and [:,4] is the real part of the 0th Fourier coefficent.
        # This means [:,5] is the imaginary part of the 0th Fourier coeffficent, which is always 0 (i.e. std. dev. is also 0).
        inputs[:,0] = (inputs[:,0]-169) / 86 # Trying to map r= 83 to -1 and r=255 to 1
        inputs[:,1] = (inputs[:,1] / math.pi) - 1 # Map  phi=0 to -1 and phi=2*pi to 1
        inputs[:,2] = (inputs[:,2] / 125)  - 1 # Map z=0 to -1 and z=250 to 1
        for i in range(4, inputs.shape[1]):
            if i==5:
                inputs[:,i] = inputs[:,i] 
            else:
                inputs[:,i] = (inputs[:,i]-self.fourierMean[i-4])/self.fourierStdDev[i-4]
        return inputs

    def __get_data(self, partition):
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
            if self.config.cache_train and self.config.train_events <= self.config.cache_events:
                return self.__get_cache()
            if self.config.cache_train:
                self.config.logger.warning("Cache insufficient, the train data will be read " +
                                           "from the original files")
        elif partition=="validation":
            downsample = self.config.downsample

        return self.__get_partition(partition, downsample, num_fourier_coeffs_apply, slice(None))

    def __get_partition(self, partition, downsample, num_fourier_coeffs_apply, part_range):
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
        indices = []
#<<<<<<< HEAD
        z_min = self.config.z_range[0]
        z_max = self.config.z_range[1]
        if partition=="train" and self.config.z_range[1] > 249:
            self.config.z_range[1] = 248
        if partition=="train" and self.config.z_range[0] < -249:
            self.config.z_range[0] = -248
        print(self.config.z_range)
#=======
        mean_values = []
#>>>>>>> cda465f115b7880e0edb78c7d66497c1e856c59b
        for indexev in self.config.partition[partition][part_range]:
            inputs_single, exp_outputs_single, mean_values_single  = load_data_oned_idc(
                self.config, dirinput,
                indexev, downsample,
                self.config.num_fourier_coeffs_train,
                num_fourier_coeffs_apply)
            inputs.append(inputs_single)
            exp_outputs.append(exp_outputs_single)
            mean_values.append(mean_values_single)

            vec_index_random = np.empty(inputs_single.shape[0])
            vec_index_random[:] = indexev[0]
            vec_index_mean = np.empty(inputs_single.shape[0])
            vec_index_mean[:] = indexev[1]
            vec_index = np.empty(inputs_single.shape[0])
            vec_index[:] = indexev[0] + 1000 * indexev[1]
            indices_stacked = np.dstack((vec_index.astype('int32'), vec_index_mean.astype('int32'),
                                         vec_index_random.astype('int32')))
            indices.append(indices_stacked.reshape(-1, 3))
        inputs = np.concatenate(inputs)
        exp_outputs = np.concatenate(exp_outputs)
        mean_values = np.concatenate(mean_values)
        indices = np.concatenate(indices)
#<<<<<<< HEAD
        self.config.z_range[0] = z_min
        self.config.z_range[1] = z_max
        print(inputs[:,2].max())
        print(self.config.z_range)
        return inputs, exp_outputs, indices, mean_values
#=======

#        return inputs, exp_outputs, indices, mean_values
#>>>>>>> cda465f115b7880e0edb78c7d66497c1e856c59b

    def __save_cache(self, full_path, partition, downsample, num_fourier_coeffs_apply):
        """
        Save the cache for given partition. Function used internally.

        :param str full_path: full path to the target cache file without the '.root' extension
        :param str partition: name of partition, one from "train", "validation", "apply"
        :param bool downsample: whether to downsample the data
        :param int num_fourier_coeffs_apply: number of Fourier coefficients for applying
        """
        cache_file = "%s.root" % full_path
        self.config.logger.info("Saving new cache to %s", cache_file)
        fourier_names = list(chain.from_iterable(("c%d_real" % i, "c%d_imag" % i)
                             for i in range(self.config.num_fourier_coeffs_train)))
        dist_names = np.array(self.config.nameopt_predout)
        sel_dist_names = np.array(self.config.opt_predout) > 0
        sel_der_names = np.array(self.config.opt_usederivative) > 0
        fluc_corr_names = ["flucCorr" + dist_name for dist_name in dist_names[sel_dist_names]]
        der_ref_mean_corr_names = ["derRefMeanCorr" +
                                   dist_name for dist_name in dist_names[sel_der_names]]
        mean_value_names = ["meanCorrR", "meanCorrRPhi", "meanCorrZ", "mean0DIDC"]
        batch_file_names = []
        batch_size = self.config.cache_file_size
        for i, part_range in enumerate(self.__get_batch_range(partition, batch_size)):
            inputs, exp_outputs, indices, mean_values = self.__get_partition(partition, downsample,
                                                               num_fourier_coeffs_apply,
                                                               part_range)

            input_data = np.hstack((indices, inputs, mean_values, exp_outputs.reshape(-1, 1)))
            cache_data = pd.DataFrame(input_data,
                                      columns=["eventId", "meanId", "randomId", "r", "phi", "z"] +
                                      der_ref_mean_corr_names + fourier_names + mean_value_names +
                                      fluc_corr_names)
            batch_file = "%s_%d.root" % (full_path, i)
            batch_file_names.append(batch_file)
            pandas_to_tree(cache_data, batch_file, "cache")
            self.config.logger.info("Cache: %s saved", batch_file)
        hadd(batch_file_names, cache_file)
        for batch_file in batch_file_names:
            os.remove(batch_file)
        self.config.logger.info("Merged cache: %s saved", cache_file)

    def __get_batch_range(self, partition, batch_size):
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
            yield slice(i, i + batch_size)
        if last_full_end < data_size:
            yield slice(last_full_end, data_size)

    def __read_cache_from_file(self, filepath, events_count):
        """
        Read the cached data from the file specified. Function used internally.

        :param str filepath: full path to the cache file
        :param int events_count: number of events to read
        :return: tuple of inputs, expected outputs and event indices
        :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        self.config.logger.info("Reading cache from %s", filepath)
        points_per_event = self.config.downsample_npoints
        if not self.config.downsample:
            points_per_event = self.config.phi * self.config.r * self.config.z
        data_size = points_per_event * events_count
        input_data = tree_to_pandas(filepath, "cache")
        input_data = input_data.iloc[:data_size, :]
        indices = input_data[["eventId", "meanId", "randomId"]].to_numpy()
        inputs = input_data.filter(regex="^(?!flucCorr).*")
        inputs = inputs.drop(["eventId", "meanId", "randomId"], axis=1)
        dist_names = np.array(self.config.nameopt_predout)
        unsel_der_names = np.array(self.config.opt_usederivative) == 0
        der_ref_mean_corr_names = ["derRefMeanCorr" +
                                   dist_name for dist_name in dist_names[unsel_der_names]]
        inputs = inputs.drop(der_ref_mean_corr_names, axis=1)
        mean_values_names = ["meanCorrR", "meanCorrRPhi", "meanCorrZ", "mean0DIDC"]
        inputs = inputs.drop(mean_values_names, axis=1)
        inputs = inputs.to_numpy()
        exp_outputs = input_data.filter(like="flucCorr").to_numpy()
        self.config.logger.info("Data read from cache: %s", filepath)
        return inputs, exp_outputs, indices

    def __get_cache(self):
        """
        Get the cached data. Function used internally.

        :return: tuple of inputs, expected outputs and event indices
        :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        self.config.logger.info("Searching for cached data")

        filename = "%s_cacheEv%d" % (self.config.cache_suffix, self.config.cache_events)
        full_path = "%s/%s" % (self.config.dircache, filename)
        cache_file = "%s.root" % full_path
        try:
            inputs, exp_outputs, indices = self.__read_cache_from_file(cache_file,
                                                                self.config.train_events)
        except FileNotFoundError:
            self.config.logger.fatal("Cache: %s does not exist, no data for training!", cache_file)

        return inputs, exp_outputs, indices

    def __plot_feature_importance(self, model):
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
            plt.savefig("%s/figImp_%s_%s_nEv%d.pdf" %
                        (self.config.dirplots, importance_type, self.config.suffix,
                         self.config.train_events))


    def __plot_apply(self, inputs, exp_outputs, pred_outputs, indices):
        """
        Create result histograms in the output ROOT file after applying the model.
        Function used internally.

        :param np.ndarray inputs: vector of inputs
        :param np.ndarray exp_outputs: vector of expected outputs
        :param np.ndarray pred_outputs: vector of network predictions
        """
        myfile = TFile.Open("%s/output_%s_nEv%d.root" % (self.config.dirapply, self.config.suffix, self.config.train_events), "recreate")
        h_dist_all_events, h_deltas_all_events, h_deltas_vs_dist_all_events =\
                plot_utils.create_apply_histos(self.config, self.config.suffix, infix="all_events_")
        distortion_numeric_flat_m, distortion_predict_flat_m, deltas_flat_a, deltas_flat_m =\
            plot_utils.get_apply_results_single_event(pred_outputs, exp_outputs)
        plot_utils.fill_apply_tree(h_dist_all_events, h_deltas_all_events,
                                   h_deltas_vs_dist_all_events,
                                   distortion_numeric_flat_m, distortion_predict_flat_m,
                                   deltas_flat_a, deltas_flat_m)

        # Create a TTree with all the Apply data.
        if self.config.apply_tree:
            tree = TTree("apply_tree", "Apply Results")

            eventID_array = array.array("l", [0])
            PosR_array = array.array("f", [0])
            PosPhi_array = array.array("f", [0])
            PosZ_array = array.array("f", [0])
            exp_out_array = array.array("f", [0])
            pred_out_array = array.array("f", [0])

            tree.Branch("EventID", eventID_array, "EventID/L")
            tree.Branch("PosR", PosR_array, "PosR/F")
            tree.Branch("PosPhi", PosPhi_array, "PosPhi/F")
            tree.Branch("PosZ", PosZ_array, "PosZ/F")
            tree.Branch("exp_out", exp_out_array, "exp_out/F")
            tree.Branch("pred_out", pred_out_array, "pred_out/F")

            nPerEvent = len(inputs) // self.config.apply_events
            for i in range(0, len(pred_outputs)):
                eventID_array[0]= i // nPerEvent
                PosR_array[0]=inputs[i,0]
                PosPhi_array[0]=inputs[i,1]
                PosZ_array[0]=inputs[i,2]
                exp_out_array[0]=exp_outputs[i]
                pred_out_array[0]=pred_outputs[i]
                tree.Fill()

            tree.Write()

        for hist in (h_dist_all_events, h_deltas_all_events, h_deltas_vs_dist_all_events):
            hist.Write()
        plot_utils.fill_profile_apply_hist(h_deltas_vs_dist_all_events, self.config.profile_name,
                                           self.config.suffix)
        plot_utils.fill_std_dev_apply_hist(h_deltas_vs_dist_all_events, self.config.h_std_dev_name,
                                           self.config.suffix, "all_events_")

        myfile.Close()

    def __plot_train(self, model, x_train, y_train, x_val, y_val):
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
                self.__plot_results(y_train[:checkpoint], y_train_predict, "train-%d" % ind)
                self.__plot_results(y_val, y_val_predict, "val-%d" % ind)
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

    def __plot_results(self, exp_outputs, pred_outputs, infix):
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
        plt.savefig("%s/num-exp-%s_%s_fapply%d_nEv%d.png" % (self.config.dirplots, infix,
                    self.config.suffix, self.config.num_fourier_coeffs_apply,
                    self.config.train_events))
        plt.clf()
