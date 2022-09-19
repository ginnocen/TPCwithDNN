"""
Utilities for constructing the neural network for 1D correction.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, InputLayer

import matplotlib.pyplot as plt

def nn_1d_without_validation(self, config, inputs, exp_outputs):
    """
    Construction of a simple neural network without loading validation data.
    Perhaps change this version to nn_1d and the original one to nn_1d_with_validation in the future?
    """
    model = Sequential()
    model.add(Dense(units=inputs.shape[1], input_dim=inputs.shape[1], activation='linear'))
    for i in range(config.nn_params["n_hidden_layers"]):
        model.add(Dense(units=inputs.shape[1], activation=config.nn_params['hidden_activation']))
    model.add(Dense(units=1))

    model.summary()

    model.compile(optimizer=config.nn_params['optimizer'], loss=config.nn_params['loss'], metrics=config.nn_params['metrics'])

    config.logger.info("nn_1d(), Neural network model compilation succeeded! Proceeding to training.")

    model.fit(inputs, exp_outputs, epochs=config.nn_params["epochs"], batch_size=config.nn_params["batch_size"])

    config.logger.info("nn_1d(), Neural network model.fit succeeded!")
    return model

def nn_1d(self, config, inputs, exp_outputs, inputs_val, outputs_val):
    """
    Construction of a simple neural network.
    """
    print(inputs.shape, inputs_val.shape)
    model = Sequential()
    model.add(Dense(units=inputs.shape[1], input_dim=inputs.shape[1], activation='linear'))
    for i in range(config.nn_params["n_hidden_layers"]):
        model.add(Dense(units=inputs.shape[1], activation=config.nn_params['hidden_activation']))
    model.add(Dense(units=1))

    model.summary()

    model.compile(optimizer=config.nn_params['optimizer'], loss=config.nn_params['loss'], metrics=config.nn_params['metrics'])

    config.logger.info("nn_1d(), Neural network model compilation succeeded! Proceeding to training.")

    history = model.fit(inputs, exp_outputs, validation_data=(inputs_val, outputs_val), epochs=config.nn_params["epochs"], batch_size=config.nn_params["batch_size"])
    #model.fit(inputs, exp_outputs, validation_data=(inputs_val, outputs_val), epochs=config.nn_params["epochs"], batch_size=config.nn_params["batch_size"])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss comparison during model training')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.savefig("./NN_training/training_loss.pdf")
    plt.savefig("./NN_training/training_loss.eps")

    config.logger.info("nn_1d(), Training drawn in Training/training_loss.pdf")
    config.logger.info("nn_1d(), Neural network model.fit succeeded!")
    return model
