# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
import pickle
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor

from tpcwithdnn.optimiser import Optimiser
from tpcwithdnn.data_loader import load_train_apply_idc

class XGBoostOptimiser(Optimiser):
    name = "xgboost"

    def __init__(self, config):
        super().__init__(config)
        self.config.logger.info("XGBoostOptimiser::Init")
        self.model = XGBRFRegressor(verbosity=1, n_gpus=0, **(self.config.params))

    def train(self):
        self.config.logger.info("XGBoostOptimiser::train")
        for indexev in self.config.partition['train']:
            inputs, exp_outputs = load_train_apply_idc(self.config.dirinput_train, indexev,
                                                       self.config.input_z_range,
                                                       self.config.output_z_range,
                                                       self.config.opt_predout)
            self.model.fit(inputs, exp_outputs)
        self.save_model_(self.model)

    def save_model_(self, model):
        out_filename = "%s/xgbmodel_%s_nEv%d_snap.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        pickle.dump(model, open(out_filename, 'wb'), protocol=4)
        out_filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        model.save_model(out_filename)

    def apply(self):
        self.config.logger.info("XGBoostOptimiser::apply, input size: %d", self.config.dim_input)
        for indexev in self.config.partition['apply']:
            inputs, exp_outputs = load_train_apply_idc(self.config.dirinput_apply, indexev,
                                                       self.config.input_z_range,
                                                       self.config.output_z_range,
                                                       self.config.opt_predout)
            pred_outputs = self.model.predict(inputs)
            print("pred output size: {}".format(pred_outputs.shape))

    def load_model_(self):
        filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        self.model.load_model(filename)

    def search_grid(self):
        raise NotImplementedError("Search grid method not implemented yet")
