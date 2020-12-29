# pylint: disable=too-many-arguments, invalid-name
# pylint: disable=missing-module-docstring, missing-function-docstring
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, UpSampling3D, Add, Activation
from tensorflow.keras.layers import AveragePooling3D, Conv3DTranspose
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv3D, MaxPooling3D

from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d

#https://github.com/mimrtl/DeepRad-Tools/blob/master/Examples/Unet.py
def conv_block(m, dim, activation, batchnorm, residual, dropout=0):
    shortcut = Conv3D(dim, 1, kernel_initializer='normal')(m)
    shortcut = BatchNormalization()(shortcut)
    n = Conv3D(dim, 3, activation=activation, padding='same', kernel_initializer="normal")(m)
    n = BatchNormalization()(n) if batchnorm else n
    n = Dropout(dropout)(n) if dropout else n
    n = Conv3D(dim, 3, padding='same', kernel_initializer="normal")(n)
    n = BatchNormalization()(n) if batchnorm else n
    n = Add()([shortcut, n]) if residual else n
    return Activation(activation)(n)

def level_block(m, dim, depth, inc_rate, activation, dropout, batchnorm, pool_type,
                upconv, residual, use_dilated):
    if depth > 0:
        n = conv_block(m, dim, activation, batchnorm, residual, dropout)

        if pool_type == 0:
            m = MaxPooling3D(pool_size=(2, 2, 2))(n)
        elif pool_type == 1:
            m = AveragePooling3D(pool_size=(2, 2, 2))(n)
        else:
            Conv3D(dim, 3, strides=2, padding='same')(n)

        m = level_block(m, int(inc_rate*dim), depth-1, inc_rate, activation, dropout, batchnorm,
                        pool_type, upconv, residual, use_dilated)
        if upconv:
            m = UpSampling3D(size=(2, 2, 2))(m)
        elif not upconv and use_dilated:
            m = Conv3DTranspose(dim, 4, strides=2, activation=activation,
                                padding='same')(m)
        else:
            m = Conv3DTranspose(dim, 3, strides=2, activation=activation,
                                padding='same')(m)
        diff_phi = n.shape[1] - m.shape[1]
        diff_r = n.shape[2] - m.shape[2]
        diff_z = n.shape[3] - m.shape[3]
        padding = [[int(diff_phi), 0], [int(diff_r), 0], [int(diff_z), 0]]
        if diff_phi != 0:
            m = SymmetryPadding3d(padding=padding, mode="SYMMETRIC")(m)
        elif (diff_r != 0 or diff_z != 0):
            m = SymmetryPadding3d(padding=padding, mode="CONSTANT")(m)
        if depth == 2 and use_dilated:
            n = dilated_dense_net(n, dim)
        n = concatenate([n, m])
        m = conv_block(n, dim, activation, batchnorm, residual, dropout)
    else:
        m = conv_block(m, dim, activation, batchnorm, residual, dropout)
    return m

def u_net(input_shape, start_channels=4, depth=4, inc_rate=2.0, activation="relu", dropout=0.2,
          batchnorm=False, pool_type=0, upconv=True, residual=False, use_dilated=False):
    i = Input(shape=input_shape)
    output = level_block(i, start_channels, depth, inc_rate, activation, dropout, batchnorm,
                         pool_type, upconv, residual, use_dilated)
    output = Conv3D(1, 1, activation="linear", padding="same", kernel_initializer="normal")(output)
    return Model(inputs=i, outputs=output)

#pylint:disable=unused-argument
def simple_net(input_shape, start_channels=4, depth=4, inc_rate=2.0, activation="relu",
               dropout=0.2, batchnorm=False, pool_type=0, upconv=True, residual=False):
    print("SimpleNet is just an attempt. Be patient :)")
    print("the input data size is", input_shape)
    myinput = Input(shape=input_shape)
    print(input_shape)
    conv1 = Conv3D(4, (4, 4, 4), activation="relu", padding="same",
                   kernel_initializer="normal")(myinput)
    conv2 = Conv3D(24, (4, 4, 4), activation="relu", padding="same",
                   kernel_initializer="normal")(conv1)
    conv3 = Conv3D(12, (2, 2, 2), activation="relu", padding="same",
                   kernel_initializer="normal")(conv2)
    conv4 = Conv3D(1, 1, activation="linear", padding="same",
                   kernel_initializer="normal")(conv3)
    return Model(inputs=myinput, outputs=conv4)


def dilated_dense_net(m, dim, activation="relu", dropout=0.5, sub_blocks=3):
    dilation_rate = 1
    x = m
    for dummy in range(sub_blocks):
        n = Conv3D(dim, 1, activation=activation, kernel_initializer='normal')(x)
        n = Dropout(dropout)(n) if dropout else n
        n = Conv3D(int(dim/4), 3, activation=activation, padding='same',
                        kernel_initializer='normal', dilation_rate=dilation_rate)(n)
        n = Dropout(dropout)(n) if dropout else n
        n = concatenate([n, x])
        x = n
        n = Conv3D(dim, 1, activation=activation, kernel_initializer='normal')(x)
        n = Dropout(dropout)(n) if dropout else n
        n = Conv3D(int(dim/4), 3, activation=activation, padding='same',
                        kernel_initializer='normal', dilation_rate=dilation_rate)(n)
        n = Dropout(dropout)(n) if dropout else n
        n = concatenate([n, x])
        x = n
        dilation_rate *= 2
    return n
    