"""
Utilities for constructing the neural network for 1D correction.
"""
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

import matplotlib.pyplot as plt

def nn_1d(config, inputs, exp_outputs):
    """
    Construction of a simple neural network without loading validation data.
     :param CommonSettings config: config of XGBoost optimiser.
     :param np.ndarray inputs: input training data.
     :param np.ndarray exp_outputs: expected output data for network training.

     :return: Neural network model
     :rtype: tensorflow.keras.model
    """
    model = Sequential()
    model.add(Dense(units=inputs.shape[1], input_dim=inputs.shape[1], activation='linear'))
    for _ in range(config.nn_params["n_hidden_layers"]):
        model.add(Dense(units=config.nn_params["n_neurons"]*inputs.shape[1], activation=config.nn_params['hidden_activation']))
        model.add(Dropout(rate=config.nn_params['dropout_rate']))
    model.add(Dense(units=1))

    model.summary()

    model.compile(optimizer=config.nn_params['optimizer'],\
                  loss=config.nn_params['loss'], metrics=config.nn_params['metrics'])

    config.logger.info("nn_1d(), Neural network model compiled!")

    model.fit(inputs, exp_outputs, epochs=config.nn_params["epochs"],\
              batch_size=config.nn_params["batch_size"], validation_split=0.0)

    config.logger.info("nn_1d(), Neural network model.fit succeeded!")
    return model

def nn_1d_with_validation(config, inputs, exp_outputs, inputs_val, outputs_val):
    """
    Construction of a simple neural network.
    :param CommonSettings config: config of  XGBoost optimiser.
    :param np.ndarray inputs: input training data.
    :param np.ndarray exp_outputs: expected output for data for network training.
    :param np.ndarray inputs_val: input validation data used for visualizing network performance
    :param np.ndarray outputs_val: expected output for the validation dataset

    :return: Neural network model
    :rtype: tensorflow.keras.model
    """
    if not os.path.isdir("%s/learning_plot_%s_nEv%d" % (config.dirplots, config.suffix, config.train_events)):
        os.makedirs("%s/learning_plot_%s_nEv%d" % (config.dirplots, config.suffix, config.train_events))

    model = Sequential()
    model.add(Dense(units=inputs.shape[1], input_dim=inputs.shape[1], activation='linear'))
    for _ in range(config.nn_params["n_hidden_layers"]):
        model.add(Dense(units=config.nn_params["n_neurons"]*inputs.shape[1], activation=config.nn_params['hidden_activation']))
        model.add(Dropout(rate=config.nn_params['dropout_rate']))
    model.add(Dense(units=1))

    model.summary()

    model.compile(optimizer=config.nn_params['optimizer'],\
                  loss=config.nn_params['loss'], metrics=config.nn_params['metrics'])

    config.logger.info("nn_1d(), Neural network model compiled!")

    history = model.fit(inputs, exp_outputs, validation_data=(inputs_val, outputs_val),\
                        epochs=config.nn_params["epochs"],
                        batch_size=config.nn_params["batch_size"])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss comparison during model training')
    plt.ylabel('%s' % config.nn_params['loss'])
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.savefig("%s/learning_plot_%s_nEv%d/training_loss.eps" % (config.dirplots, config.suffix, config.train_events))
    plt.savefig("%s/learning_plot_%s_nEv%d/training_loss.pdf" % (config.dirplots, config.suffix, config.train_events))

    config.logger.info("nn_1d(), Neural network model.fit succeeded!")
    return model
