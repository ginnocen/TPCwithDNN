"""
Custom symmetric cube padding.
"""
import tensorflow as tf
from keras.layers import Layer

class SymmetryPadding3d(Layer):
    """
    Custom Keras layer implementing symmetric padding in 3D.
    """
    def __init__(self, padding=None,
                 mode='SYMMETRIC', data_format="channels_last", **kwargs):
        """
        Initialize the layer.

        :param list padding: list of three 2-element lists describing how many values should be
                             inserted before and after tensor contents in each dimension.
                             See Tensorflow documentation for a more detailed description.
                             Default: unit padding in each dimension.
        :param str mode: padding mode, chosen from Tensorflow, default: SYMMETRIC
        :param str data_format: what is the dimension order, default: channels_last, i.e.,
                                shape (batch_size, height, width, channels)
        """
        self.data_format = data_format
        self.padding = [[1, 1], [1, 1], [1, 1]] if padding is None else padding
        self.mode = mode
        super().__init__(**kwargs)
        self.output_dim = None

    # pylint: disable=arguments-differ
    def call(self, inputs):
        """
        Defines computations made by the layer during the forward pass through the network.

        :param tf.Tensor inputs: input Tensorflow tensor from the previous layer.
        :return: padded input tensor
        :rtype: tf.Tensor
        """
        pad = [[0, 0]] + self.padding + [[0, 0]]
        paddings = tf.constant(pad)
        out = tf.pad(inputs, paddings, self.mode)
        self.output_dim = [(out.shape[0], out.shape[1], out.shape[2], out.shape[3], out.shape[4])]
        return out

    def compute_output_shape(self):
        """
        Overriden function from the base class to get the output shape.

        :return: shape of the output data from the layer
        :rtype: list
        """
        return self.output_dim

    def get_config(self):
        """
        Overriden base function for obtaining layer settings, used, e.g.,
        for saving the layer to JSON.

        :return: layer parameters
        :rtype: dict
        """
        config = {'padding': self.padding, 'data_format': self.data_format, 'mode': self.mode}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
