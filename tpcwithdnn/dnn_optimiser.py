# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=protected-access
import matplotlib.pyplot as plt

import numpy as np

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model

from ROOT import TFile # pylint: disable=import-error, no-name-in-module

from tpcwithdnn import plot_utils
from tpcwithdnn.optimiser import Optimiser
from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d
from tpcwithdnn.fluctuation_data_generator import FluctuationDataGenerator
from tpcwithdnn.utilities_dnn import u_net
from tpcwithdnn.data_loader import load_train_apply

class DnnOptimiser(Optimiser):
    name = "dnn"

    def __init__(self, config):
        super().__init__(config)
        self.config.logger.info("DnnOptimiser::Init")

    def train(self):
        self.config.logger.info("DnnOptimiser::train")

        training_generator = FluctuationDataGenerator(self.config.partition['train'],
                                                      data_dir=self.config.dirinput_train,
                                                      **self.config.params)
        validation_generator = FluctuationDataGenerator(self.config.partition['validation'],
                                                        data_dir=self.config.dirinput_test,
                                                        **self.config.params)
        model = u_net((self.config.grid_phi, self.config.grid_r, self.config.grid_z,
                       self.config.dim_input),
                      depth=self.config.depth, batchnorm=self.config.batch_normalization,
                      pool_type=self.config.pool_type, start_channels=self.config.filters,
                      dropout=self.config.dropout)
        if self.config.metrics == "root_mean_squared_error":
            metrics = RootMeanSquaredError()
        else:
            metrics = self.config.metrics
        model.compile(loss=self.config.lossfun, optimizer=Adam(lr=self.config.adamlr),
                      metrics=[metrics]) # Mean squared error

        model.summary()
        plot_model(model, to_file='%s/model_%s_nEv%d.png' % \
                   (self.config.dirplots, self.config.suffix, self.config.train_events),
                   show_shapes=True, show_layer_names=True)

        #log_dir = "logs/" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_dir = 'logs/' + '%s_nEv%d' % (self.config.suffix, self.config.train_events)
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        model._get_distribution_strategy = lambda: None
        his = model.fit(training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        epochs=self.config.epochs, callbacks=[tensorboard_callback])

        self.plot_train_(his)
        self.save_model(model)

    def apply(self):
        self.config.logger.info("DnnOptimiser::apply, input size: %d", self.config.dim_input)
        loaded_model = self.load_model()

        myfile = TFile.Open("%s/output_%s_nEv%d.root" % \
                            (self.config.dirapply, self.config.suffix, self.config.train_events),
                            "recreate")
        h_dist_all_events, h_deltas_all_events, h_deltas_vs_dist_all_events =\
                plot_utils.create_apply_histos(self.config, self.config.suffix, infix="all_events_")

        for indexev in self.config.partition['apply']:
            inputs_, exp_outputs_ = load_train_apply(self.config.dirinput_apply, indexev,
                                                     self.config.z_range,
                                                     self.config.grid_r, self.config.grid_phi,
                                                     self.config.grid_z,
                                                     self.config.opt_train,
                                                     self.config.opt_predout)
            inputs_single = np.empty((1, self.config.grid_phi, self.config.grid_r,
                                      self.config.grid_z, self.config.dim_input))
            exp_outputs_single = np.empty((1, self.config.grid_phi, self.config.grid_r,
                                           self.config.grid_z, self.config.dim_output))
            inputs_single[0, :, :, :, :] = inputs_
            exp_outputs_single[0, :, :, :, :] = exp_outputs_

            distortion_predict_group = loaded_model.predict(inputs_single)

            distortion_numeric_flat_m, distortion_predict_flat_m, deltas_flat_a, deltas_flat_m =\
                plot_utils.get_apply_results_single_event(distortion_predict_group,
                                                          exp_outputs_single)
            plot_utils.fill_apply_tree_single_event(self.config, indexev,
                                                    distortion_numeric_flat_m,
                                                    distortion_predict_flat_m,
                                                    deltas_flat_a, deltas_flat_m)
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
        self.config.logger.info("Done apply")

    def search_grid(self):
        raise NotImplementedError("Search grid method not implemented yet")

    def bayes_optimise(self):
        raise NotImplementedError("Bayes optimise method not implemented yet")

    def save_model(self, model):
        model_json = model.to_json()
        with open("%s/model_%s_nEv%d.json" % (self.config.dirmodel, self.config.suffix,
                                              self.config.train_events), "w") \
            as json_file:
            json_file.write(model_json)
        model.save_weights("%s/model_%s_nEv%d.h5" % (self.config.dirmodel, self.config.suffix,
                                                     self.config.train_events))
        self.config.logger.info("Saved trained DNN model to disk")

    def load_model(self):
        with open("%s/model_%s_nEv%d.json" % \
                  (self.config.dirmodel, self.config.suffix, self.config.train_events), "r") as f:
            loaded_model_json = f.read()
        loaded_model = \
            model_from_json(loaded_model_json, {'SymmetryPadding3d' : SymmetryPadding3d})
        loaded_model.load_weights("%s/model_%s_nEv%d.h5" % \
                                  (self.config.dirmodel, self.config.suffix,
                                   self.config.train_events))
        return loaded_model

    def plot_train_(self, his):
        plt.style.use("ggplot")
        plt.figure()
        plt.yscale('log')
        plt.plot(np.arange(0, self.config.epochs), his.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.config.epochs), his.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.config.epochs), his.history[self.config.metrics],
                 label="train_" + self.config.metrics)
        plt.plot(np.arange(0, self.config.epochs), his.history["val_" + self.config.metrics],
                 label="val_" + self.config.metrics)
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("%s/learning_plot_%s_nEv%d.png" % (self.config.dirplots, self.config.suffix,
                                                       self.config.train_events))
