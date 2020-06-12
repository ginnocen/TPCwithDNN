import os
from os.path import exists
from os import makedirs
import sys
import numpy as np
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from machine_learning_hep.io import parse_yaml, dump_yaml_from_dict
from machine_learning_hep.logger import get_logger
from machine_learning_hep.do_variations import modify_dictionary
from machine_learning_hep.optimisation.bayesian_opt import BayesianOpt
from utilitiesdnn import UNet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use("TkAgg")

def compile_model(model, model_config):
    model.compile(loss=model_config["lossfun"], optimizer=Adam(lr=model_config["adamlr"]),
                  metrics=[model_config["metrics"]]) # Mean squared error
    model.summary()

def construct_model(model_config, model_config_overwrite=None):
    if model_config_overwrite:
        model_config = deepcopy(model_config_nominal)
        modify_dictionary(model_config, model_config_overwrite)

    model = UNet((model_config["grid_phi"], model_config["grid_r"], model_config["grid_z"],
                  model_config["dim_input"]),
                 depth=model_config["depth"], bathnorm=model_config["batch_normalization"],
                 pool_type=model_confnig["pooling"], start_ch=model_config["filters"],
                 dropout=model_config["dropout"])
    compile_model(model, model_config)
    return model, model_config


def make_fit_generator_kwargs(train_gen, val_gen, **kwargs):
    kwargs["generator"] = train_gen
    kwargs["validation_data"] = val_gen
    return kwargs


def fit_model(model, fit_method=None, **fit_kwargs):
    """Fit the model

    Fit model from data directly or from data generators
    

    Args:
        model: Keras TF model
            model to be fit
        fit_method: str
            name of the fit method to be used (either "fit" or "fit_generator"
        fit_kwargs:
            train_data: n_samples x n_features numpy array like (optional)
                training data to be used
            val_data: n_samples_val x n_features numpy array like (optional)
                validation data to be used
            train_gen: training data generator (optional)
            val_gen: validation data generator (optional)
    """
    if fit_method not in ("fit", "fit_generator"):
        get_logger().fatal("Unknown fit method %s", fit_method)

    # This is just to really make sure the fit method exists for this model
    if not hasattr(model, fit_method):
        get_logger().fatal("The provided model does not have the fit method %s", fit_method)

    return getattr(model, fit_method)(**fit_kwargs)


def make_out_dir(out_dir, suffix=0):
    if not exists(out_dir):
        makedirs(out_dir)
        return out_dir
    dir_name, base_name = split(out_dir)
    base_name = f"base_name_{suffix}"
    suffix += 1
    return make_model_out_dir(join(dir_name, base_name), suffix)


def save_model(model, model_config, out_dir):
    out_dir = make_out_dir(out_dir)
    get_logger().info("Save model config and weights at %s", out_dir)
    model_json = model.to_json()
    save_path = join(out_dir, "model.json")
    with open(save_path), "w") as json_file: \
        json_file.write(model_json)
    save_path = join(out_dir, "model.h5")
    model.save_weights(save_path)
    save_path = join(out_dir, "model_config.yaml")
    dump_yaml_from_dict(model_config, save_path)


def load_model(in_dir):
    if not exists(in_dir):
        get_logger().fatal("Directory %s does not exist. Cannot load model", in_dir)
    json_path = join(in_dir, MODEL_JSON)
    weights_path = join(in_dir, MODEL_H5)
    config_path = join(in_dir, MODEL_CONFIG)
    if not exists(json_path) or not exists(weights_path) or not exists(config_path):
        get_logger().fatal("Make sure there is there are all files to load a model from: %s",
                           str((json_path, weights_path, config_path)))

    model = None
    with open(json_path, "r") as json_file:
        model_arch = json_file.read()
        model = model_from_json(model_arch)
    model.load_weights(weights_path)
    return model, parse_yaml(config_path)



class KerasClassifierBayesianOpt(BayesianOpt): # pylint: disable=too-many-instance-attributes


    def __init__(self, model_config, space):
        super().__init__(model_config, space)
        self.model_config_tmp = None
        self.space_tmp = None

        self.train_gen = None
        self.val_gen = None

        # Set scores here explicitly
        self.scoring = ["mse"]
        self.scoring_opt = "mse"


    def trial_(self, space_drawn):
        """One trial

        This does one fit attempt (no CV at the moment) with the given parameters

        """
        model, params = construct_model(self.model_config, space_drawn)
        fit_kwargs = make_fit_generator_kwargs(self.train_gen, self.val_gen,
                                               **{"use_multiprocessing": True,
                                                  "epochs": self.model_config["epochs"]}):
        history = fit_model(model, "fit_generator", **fit_kwargs)

        res = {f"train_{self.scoring_opt}": [history.history[self.scoring_opt]],
               f"test_{self.scoring_opt}": [history.history[f"val_{self.scoring_opt}"]]}
        return res, model, params


    def finalise(self):
        pass


    def save_model_(self, model, out_dir):
        save_model(model, model_config, out_dir)
