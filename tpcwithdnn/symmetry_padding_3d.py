# pylint: disable=missing-module-docstring, missing-class-docstring, fixme
import tensorflow as tf
# import keras.backend as K
from tensorflow.keras.layers import Layer

class SymmetryPadding3d(Layer):
    def __init__(self, padding=None,
                 mode='SYMMETRIC', data_format="channels_last", **kwargs):
        self.data_format = data_format
        self.padding = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]] if padding is None else padding
        self.mode = mode
        super().__init__(**kwargs)
        self.output_dim = None

    # pylint: disable=arguments-differ
    def call(self, inputs):
        pad = [[0, 0]] + self.padding + [[0, 0]]
        # TODO: Do we expect other backends as well?
        # if K.backend() == "tensorflow":
        paddings = tf.constant(pad)
        #print(inputs.shape)
        #print(paddings)
        out = tf.pad(inputs, paddings, self.mode)
        # else:
        #    raise Exception("Backend " + K.backend() + "not implemented")
        self.output_dim = [(out.shape[0], out.shape[1], out.shape[2], out.shape[3], out.shape[4])]
        return out

    def compute_output_shape(self, input_shape):
        return self.output_dim

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format, 'mode': self.mode}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
