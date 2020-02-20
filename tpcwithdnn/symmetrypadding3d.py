import keras.backend as K
from keras.layers import Layer
import tensorflow as tf

# pylint: disable=line-too-long, invalid-name
class symmetryPadding3d(Layer):
    def __init__(self, padding=[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC', data_format="channels_last", **kwargs):
        self.data_format = data_format
        self.padding = padding
        self.mode = mode
        super(symmetryPadding3d, self).__init__(**kwargs)
        self.output_dim = None

    def build(self, input_shape):
        super(symmetryPadding3d, self).build(input_shape)

    def call(self, inputs):
        pad = [[0, 0]] + [i for i in self.padding] + [[0, 0]]
        if K.backend() == "tensorflow":
            paddings = tf.constant(pad)
            print(inputs.shape)
            print(paddings)
            out = tf.pad(inputs, paddings, self.mode)
        else:
            raise Exception("Backend " + K.backend() + "not implemented")
        self.output_dim = [(out.shape[0], out.shape[1], out.shape[2], out.shape[3], out.shape[4])]
        return out

    def compute_output_shape(self, input_shape):
        return  self.output_dim

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format, 'mode': self.mode}
        base_config = super(symmetryPadding3d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
