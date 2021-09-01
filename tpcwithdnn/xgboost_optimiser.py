"""
XGBoost optimizer for 1D IDC distortion correction
"""
from timeit import default_timer as timer

import pickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRFRegressor

from sklearn.metrics import mean_squared_error
import xgboost

from ROOT import TFile # pylint: disable=import-error, no-name-in-module

from tpcwithdnn import plot_utils
from tpcwithdnn.debug_utils import log_time, log_memory_usage, log_total_memory_usage
from tpcwithdnn.optimiser import Optimiser
from tpcwithdnn.data_loader import load_data_oned_idc, get_input_names_oned_idc

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
            log_memory_usage(((inputs_val, "Input val data"), (outputs_val, "Output val data")))
            log_total_memory_usage("Memory usage after loading val data")
            self.plot_train_(model, inputs, exp_outputs, inputs_val, outputs_val)
        start = timer()
        model.fit(inputs, exp_outputs)
        end = timer()
        log_time(start, end, "actual train")
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
        model.get_booster().feature_names = get_input_names_oned_idc()
        out_filename_feature_importance = "%s/feature_importance_%s_nEv%d.txt" %\
            (self.config.dirmodel, self.config.suffix, self.config.train_events)
        score_total_gain = model.get_booster().get_score(importance_type='total_gain')
        indices_total_gain = np.flip(np.argsort(list(score_total_gain.values())))
        score_gain = model.get_booster().get_score(importance_type='gain')
        indices_gain = np.flip(np.argsort(list(score_gain.values())))
        score_weight = model.get_booster().get_score(importance_type='weight')
        indices_weight = np.flip(np.argsort(list(score_weight.values())))
        with open(out_filename_feature_importance, 'w', encoding="utf-8") as file_name:
            print("Feature importances", file=file_name)
            print("Total gain   -   Gain   -   Weight", file=file_name)
            for index_total_gain, index_gain, index_weight in zip(indices_total_gain,
                                                                  indices_gain,
                                                                  indices_weight):
                print("%s: %.4f   -" % (list(score_total_gain.keys())[index_total_gain],
                                        list(score_total_gain.values())[index_total_gain]),
                      "   %s: %.4f   -" % (list(score_gain.keys())[index_gain],
                                           list(score_gain.values())[index_gain]),
                      "   %s: %.4f" % (list(score_weight.keys())[index_weight],
                                       list(score_weight.values())[index_weight]),
                      file=file_name)

        font_size = 5
        for importance_type, score, indices in zip(['total_gain', 'gain', 'weight'],
                                        [score_total_gain, score_gain, score_weight],
                                        [indices_total_gain, indices_gain, indices_weight]):
            num_features = min(40, len(indices))
            xgboost.plot_importance(model, importance_type=importance_type, xlabel=importance_type,
                                    log=True, max_num_features=num_features, grid=False,
                                    show_values=False, height=0.5)
            plt.yticks(fontsize=font_size)
            plt.gca().grid(b=True, which='both', axis='x', linewidth=0.4)
            for i, index in enumerate(indices[:num_features]):
                plt.text(list(score.values())[index],
                         num_features - 1.275 + 1 / num_features - i,
                         str("%.3e" % list(score.values())[index]),
                        fontsize=font_size)
            plt.savefig("%s/figImportances_%s_%s_nEv%d.pdf" %
                        (self.config.dirplots, importance_type, self.config.suffix,
                         self.config.train_events))

        # Snapshot - can be used for further training
        out_filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        with open(out_filename, "wb", encoding="utf-8") as out_file:
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
        with open(filename, "rb", encoding="utf-8") as file:
            model = pickle.load(file)
        return model

    def get_data_(self, partition):
        """
        Load the full input data for a XGBoost optimization.
        Function used internally.

        :param dict partition: dictionary of pairs of event indices
                               for training / validation / apply
        :return: tuple of inputs and expected outputs
        :rtype: tuple(np.ndarray, np.ndarray)
        """
        downsample = self.config.downsample if partition == "train" else False
        inputs = []
        exp_outputs = []
        for indexev in self.config.partition[partition]:
            inputs_single, exp_outputs_single = load_data_oned_idc(self.config.dirinput_train,
                                                                   indexev, self.config.z_range,
                                                                   self.config.opt_predout,
                                                                   downsample,
                                                                   self.config.downsample_frac,
                                                                   self.config.rnd_augment)
            inputs.append(inputs_single)
            exp_outputs.append(exp_outputs_single)
        inputs = np.concatenate(inputs)
        exp_outputs = np.concatenate(exp_outputs)
        return inputs, exp_outputs

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
