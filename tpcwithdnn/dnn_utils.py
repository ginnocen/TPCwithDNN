"""
Utilities for constructing U-Net.
"""
# pylint: disable=too-many-arguments, invalid-name
# pylint: disable=consider-using-f-string

from keras.models import Model
from keras.layers import Input, concatenate, UpSampling3D
from keras.layers import AveragePooling3D, Conv3DTranspose
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv3D, MaxPooling3D

from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d

#https://github.com/mimrtl/DeepRad-Tools/blob/master/Examples/Unet.py
def conv_block(m, dim, activation, batchnorm, residual, dropout=0):
    """
    Create a convolutional block of layers.

    ;param keras.Layer m: the previous layer
    :param int dim: dimension of the convolutional layer,
                    the convolution kernel will have dimension dim x dim x dim
    :param str activation: name of the activation function
    :param bool batchnorm: whether to apply batch normalization
    :param bool residual: whether to apply residual connection
    :param double dropout: dropout magnitude from the range [0, 1), 0 means no dropout. default: 0
    :return: a convolutional block
    """
    n = Conv3D(dim, 3, activation=activation, padding='same', kernel_initializer="normal")(m)
    n = BatchNormalization()(n) if batchnorm else n
    n = Dropout(dropout)(n) if dropout else n
    n = Conv3D(dim, 3, activation=activation, padding='same', kernel_initializer="normal")(n)
    n = BatchNormalization()(n) if batchnorm else n
    return concatenate([m, n]) if residual else n

def level_block(m, dim, depth, inc_rate, activation, dropout, batchnorm, pool_type,
                upconv, residual):
    """
    Recursively create U-Net blocks, starting from the top level (network input / output)
    and increasing depth by one with each recursive call.

    :param keras.Layer m: the network input layer
    :param int dim: dimension of the convolutional layer,
                    the convolution kernel will have dimension dim x dim x dim
    :param int depth: number of convolutional blocks
    :param double inc_rate: rate of dimension increment for the convolutional block
                            E.g., with inc_rate 2, if the first conv block has layers 4x4x4,
                            the next blocks will have layers 8x8x8, 16x16x16, and so on.
    :param str activation: name of the activation function in the convolutional layers
    :param double dropout: dropout magnitude from the range [0, 1), 0 means no dropout
    :param bool batchnorm: whether to apply batch normalization
    :param str pool_type: short name of the pooling function, from: max, avg, conv
    :param bool upconv: whether to use the upsampling or transposed convolution
    :param bool residual: whether to apply residual connection
    :return: U-Net network model without the very first input and the very last output layer
    :rtype: keras.Model
    """
    if depth > 0:
        n = conv_block(m, dim, activation, batchnorm, residual)
        if pool_type == "max":
            m = MaxPooling3D(pool_size=(2, 2, 2))(n)
        elif pool_type == "avg":
            m = AveragePooling3D(pool_size=(2, 2, 2))(n)
        else: # pool_type == "conv"
            Conv3D(dim, 3, strides=2, padding='same')(n)

        m = level_block(m, int(inc_rate*dim), depth-1, inc_rate, activation, dropout, batchnorm,
                        pool_type, upconv, residual)

        if upconv:
            m = UpSampling3D(size=(2, 2, 2))(m)
            diff_phi = n.shape[1] - m.shape[1]
            diff_r = n.shape[2] - m.shape[2]
            diff_z = n.shape[3] - m.shape[3]
            padding = [[int(diff_phi), 0], [int(diff_r), 0], [int(diff_z), 0]]
            if diff_phi != 0:
                m = SymmetryPadding3d(padding=padding, mode="SYMMETRIC")(m)
            elif (diff_r != 0 or diff_z != 0):
                m = SymmetryPadding3d(padding=padding, mode="CONSTANT")(m)
        else:
            m = Conv3DTranspose(dim, 3, strides=2, activation=activation,
                                padding='same')(m)
        n = concatenate([n, m])
        m = conv_block(n, dim, activation, batchnorm, residual)
    else:
        m = conv_block(m, dim, activation, batchnorm, residual, dropout)
    return m

def u_net(input_shape, start_channels=4, depth=4, inc_rate=2.0, activation="relu", dropout=0.2,
          batchnorm=False, pool_type="max", upconv=True, residual=False):
    """
    Build an U-Net network of given size and type.

    :param int input_shape: shape of the input vector (phi x r x z x dim_input from the config file)
    :param int start_channels: size of the initial convolutional layer, default: 4
                               the convolution kernel will have dimension dim x dim x dim
    :param int depth: number of convolutional blocks, default: 4
    :param double inc_rate: rate of dimension increment for the convolutional block, default: 2.0
                            E.g., if the first conv block has layers 4x4x4, the next blocks
                            will have layers 8x8x8, 16x16x16, and so on.
    :param str activation: name of the activation function in the convolutional layers,
                           default: relu
    :param double dropout: dropout magnitude from the range [0, 1), 0 means no dropout,
                           default: 0.2
    :param bool batchnorm: whether to apply batch normalization, default: False
    :param str pool_type: short name of the pooling function, from: max, avg, conv, default: max
    :param bool upconv: whether to use the upsampling or transposed convolution, default: True
    :param bool residual: whether to apply residual connection, default: False
    :return: Full U-Net network model
    :rtype: keras.Model
    """
    i = Input(shape=input_shape)
    output = level_block(i, start_channels, depth, inc_rate, activation, dropout, batchnorm,
                         pool_type, upconv, residual)
    output = Conv3D(1, 1, activation="linear", padding="same", kernel_initializer="normal")(output)
    return Model(inputs=i, outputs=output)
