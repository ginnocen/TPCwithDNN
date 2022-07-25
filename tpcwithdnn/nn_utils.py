"""
Utilities for constructing the neural network for 1D correction.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, InputLayer

def nn_1d(self, config, inputs, exp_outputs, inputs_val, outputs_val):
    """
    Construction of a simple neural network.
    """
    model = Sequential()
    model.add(Dense(units=inputs.shape[1], input_dim=inputs.shape[1], activation='linear'))
    model.add(Dense(units=inputs.shape[1], activation=config.nn_params['hidden_activation']))
    model.add(Dense(units=inputs.shape[1], activation=config.nn_params['hidden_activation']))
    model.add(Dense(units=1))

    model.summary()

    model.compile(optimizer=config.nn_params['optimizer'], loss=config.nn_params['loss'], metrics=config.nn_params['metrics'])

    config.logger.info("nn_1d(), Neural network model compilation succeeded! Proceeding to training.")

    model.fit(inputs, exp_outputs, validation_data=(inputs_val, outputs_val), epochs=5, batch_size=30)

    config.logger.info("nn_1d(), Neural network model.fit succeeded!")
    return model
