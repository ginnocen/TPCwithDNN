from keras.optimizers import Adam
from keras.initializers import he_uniform, he_normal

from hyperopt import hp
from hyperopt.pyll import scope


def make_model_config(grid_phi, grid_r, grid_z, dim_input,
                      depth, batch_normalization, pooling, filters, dropout,
                      loss, learning_rate, epochs):
    return {"model_args": ((grid_phi, grid_r, grid_z, dim_input),),
            "model_kwargs": {"depth": depth,
                             "bathnorm": batch_normalization,
                             "pool_type": pooling,
                             "start_ch": filters,
                             "dropout": dropout},
            "compile": {"loss": loss,
                        "optimizer": {"callable__": Adam, "kwargs": {"lr": learning_rate}},
                        "metrics": [loss]},
            "fit": {"workers": 1,
                    "use_multiprocessing": True,
                    "epochs": epochs}}


def make_opt_space():
    return {"compile": 
            {"optimizer": {"kwargs": {"lr": hp.uniform("m_learning_rate", 0.0005, 0.01)}},
             "loss": hp.choice("m_loss", ["mse", "mean_squared_logarithmic_error"])},
             "model_kwargs": {"bathnorm": hp.choice("m_bathnorm", [0, 1]),
                              "initializer": hp.choice("m_kernel_initializer", [he_uniform(), he_normal(), None]),
                              "dropout": hp.uniform("m_dropout", 0., 0.5),
                              "start_ch": scope.int(hp.quniform("m_start_ch", 2, 8, 1)),
                              "depth": scope.int(hp.quniform("m_depth", 2, 4, 1))},
             "fit": {"epochs": scope.int(hp.quniform("m_epochs", 2, 4, 1))}}



