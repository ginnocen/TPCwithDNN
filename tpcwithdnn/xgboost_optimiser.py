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
        inputs, exp_outputs = get_train_apply_data_("train")
        self.model.fit(inputs, exp_outputs)
        self.save_model_(self.model)

    def get_train_apply_data_(partition):
        inputs = []
        exp_outputs = []
        for indexev in self.config.partition[partition]:
            inputs_single, exp_outputs_single = load_train_apply_idc(self.config.dirinput_train,
                                                       indexev, self.config.input_z_range,
                                                       self.config.output_z_range,
                                                       self.config.opt_predout)
            inputs.append(inputs_single)
            exp_outputs.append(exp_outputs_single)
        print("Different inputs: {} shape of first: {}".format(len(inputs), inputs[0].shape))
        print("Different outputs: {} shape of first: {}".format(len(exp_outputs), exp_outputs[0].shape))
        inputs = np.concatenate(inputs)
        exp_outputs = np.concatenate(exp.outputs)
        print("Inputs concatenated: {} outputs: {}".format(inputs.shape, exp_outputs.shape))
        return inputs, exp_outputs

    def save_model_(self, model):
        # Snapshot - can be used for further training
        out_filename = "%s/xgbmodel_%s_nEv%d_snap.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        pickle.dump(model, open(out_filename, 'wb'), protocol=4)
        out_filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        model.save_model(out_filename)

    def apply(self):
        self.config.logger.info("XGBoostOptimiser::apply, input size: %d", self.config.dim_input)
        self.load_model_()
        inputs, exp_outputs = get_train_apply_data_("train")
        pred_outputs = self.model.predict(inputs)
        self.config.logger.info("Done apply")

    def load_model_(self):
        # Loading a snapshot
        #filename = "%s/xgbmodel_%s_nEv%d_snap.json" %\
        #        (self.config.dirmodel, self.config.suffix, self.config.train_events)
        #loaded_model = pickle.load(open(filename, 'rb'))
        filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        self.model.load_model(filename)

    def search_grid(self):
        raise NotImplementedError("Search grid method not implemented yet")
