from keras.models import Model
from keras.layers import Input, concatenate, UpSampling3D
from keras.layers import AveragePooling3D, Conv3DTranspose
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv3D, MaxPooling3D
from symmetrypadding3d import symmetryPadding3d

#https://github.com/mimrtl/DeepRad-Tools/blob/master/Examples/Unet.py
# pylint: disable=line-too-long, invalid-name
def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv3D(dim, 3, activation=acti, padding='same', kernel_initializer="normal")(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv3D(dim, 3, activation=acti, padding='same', kernel_initializer="normal")(n)
    n = BatchNormalization()(n) if bn else n
    return concatenate([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, pool_type, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        if pool_type == 0:
            m = MaxPooling3D(pool_size=(2, 2, 2))(n)
        elif pool_type == 1:
            m = AveragePooling3D(pool_size=(2, 2, 2))(n)
        else:
            Conv3D(dim, 3, strides=2, padding='same')(n)

        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, pool_type, up, res)

        if up:
            m = UpSampling3D(size=(2, 2, 2))(m)
            diff_phi = n.shape[1] - m.shape[1]
            diff_r = n.shape[2] - m.shape[2]
            diff_z = n.shape[3] - m.shape[3]
            if diff_phi != 0:
                m = symmetryPadding3d(padding=((int(diff_phi), 0), (int(diff_r), 0), (int(diff_z), 0)), mode="SYMMETRIC")(m)
            elif (diff_r != 0 or diff_z != 0):
                m = symmetryPadding3d(padding=((int(diff_phi), 0), (int(diff_r), 0), (int(diff_z), 0)), mode="CONSTANT")(m)
        else:
            m = Conv3DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = concatenate([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m

def UNet(input_shape, start_ch=4, depth=4, inc_rate=2.0, activation="relu", dropout=0.2, bathnorm=False, pool_type=0, upconv=True, residual=False):
    i = Input(shape=input_shape)
    output_r = level_block(i, start_ch, depth, inc_rate, activation, dropout, bathnorm, pool_type, upconv, residual)
    output_r = Conv3D(1, 1, activation="linear", padding="same", kernel_initializer="normal")(output_r)

    output_rphi = level_block(i, start_ch, depth, inc_rate, activation, dropout, bathnorm, pool_type, upconv, residual)
    output_rphi = Conv3D(1, 1, activation="linear", padding="same", kernel_initializer="normal")(output_rphi)

    output_z = level_block(i, start_ch, depth, inc_rate, activation, dropout, bathnorm, pool_type, upconv, residual)
    output_z = Conv3D(1, 1, activation="linear", padding="same", kernel_initializer="normal")(output_z)
    #o = concatenate([output_r, output_rphi, output_z])
    return Model(inputs=i, outputs=output_r)
