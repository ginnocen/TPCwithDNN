import matplotlib.pyplot as plt # matplotlib module to make the nice plots
import numpy as np  # numpy module
import pandas as pd # pandas dataframe
import keras
from os.path import isfile
from keras.models import Sequential, Model
from keras.layers import Input, concatenate, Flatten, UpSampling3D, AveragePooling3D, ZeroPadding3D
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv3D, MaxPooling3D
import gc
from keras.models import model_from_json
import time
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import os
import h5py
import itertools
from SymmetricPadding3D import SymmetricPadding3D

def GetFluctuation(phiSlice,rRow,zColumn,id,side=0):
	"""
	Get fluctuation id
	"""
	fluctuationDir = os.environ['FLUCTUATIONDIR']
	dataDir = fluctuationDir + 'data/' + str(phiSlice) + '-' + str(rRow) + '-' + str(zColumn) + '/'
	vecZPosFile  = dataDir + str(0) + '-vecZPos.npy'
	scMeanFile = dataDir + str(id) + '-vecMeanSC.npy'
	scRandomFile = dataDir + str(id) + '-vecRandomSC.npy'
	distRMeanFile = dataDir + str(id) + '-vecMeanDistR.npy'
	distRRandomFile = dataDir + str(id) + '-vecRandomDistR.npy'
	distRPhiMeanFile = dataDir + str(id) + '-vecMeanDistRPhi.npy'
	distRPhiRandomFile = dataDir + str(id) + '-vecRandomDistRPhi.npy'
	distZMeanFile = dataDir + str(id) + '-vecMeanDistZ.npy'
	distZRandomFile = dataDir + str(id) + '-vecRandomDistZ.npy'
	vecZPos = np.load(vecZPosFile)
	vecMeanSC = np.load(scMeanFile)
	vecRandomSC = np.load(scRandomFile)
	vecMeanDistR = np.load(distRMeanFile)
	vecRandomDistR = np.load(distRRandomFile)
	vecMeanDistRPhi = np.load(distRPhiMeanFile)
	vecRandomDistRPhi = np.load(distRPhiRandomFile)
	vecMeanDistZ = np.load(distZMeanFile)
	vecRandomDistZ = np.load(distZRandomFile)
	vecFluctuationSC = vecMeanSC[vecZPos >= 0] - vecRandomSC[vecZPos >= 0]
	vecFluctuationDistR = vecMeanDistR[vecZPos >= 0] - vecRandomDistR[vecZPos >= 0]
	vecFluctuationDistRPhi = vecMeanDistRPhi[vecZPos >= 0] - vecRandomDistRPhi[vecZPos >= 0]
	vecFluctuationDistZ= vecMeanDistZ[vecZPos >= 0] - vecRandomDistZ[vecZPos >= 0]
	return [vecFluctuationSC,vecFluctuationDistR,vecFluctuationDistRPhi,vecFluctuationDistZ]

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
		if (pool_type == 0):
			m = MaxPooling3D(pool_size=(2,2,2))(n)
		elif (pool_type == 1):
			m = AveragePooling3D(pool_size=(2,2,2))(n)
		else:
			Conv3D(dim, 3, strides=2, padding='same')(n)

		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, pool_type, up, res)

		if up:
			m = UpSampling3D(size=(2,2,2))(m)
			diff_phi = n.shape[1] - m.shape[1]
			diff_r = n.shape[2] - m.shape[2]
			diff_z = n.shape[3] - m.shape[3]
			if (diff_phi != 0):
				m = SymmetricPadding3D(padding=((int(diff_phi),0),(int(diff_r),0),(int(diff_z),0)),mode="SYMMETRIC")(m)
			elif ((diff_r !=0) or (diff_z != 0)):
				m = SymmetricPadding3D(padding=((int(diff_phi),0),(int(diff_r),0),(int(diff_z),0)),mode="CONSTANT")(m)

		#	m = Conv3D(dim, 3, activation=acti, padding='same',kernel_initializer="normal")(m)
		else:
                    m = Conv3DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = concatenate([n, m])
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(input_shape,start_ch=4,depth=4,inc_rate=2.0,activation="relu",dropout=0.2,bathnorm=False,pool_type=0,upconv=True,residual=False):
	i = Input(shape=input_shape)
	output_r = level_block(i,start_ch,depth,inc_rate,activation,dropout,bathnorm,pool_type,upconv,residual)
	output_r = Conv3D(1,1, activation="linear",padding="same",kernel_initializer="normal")(output_r)

	output_rphi = level_block(i,start_ch,depth,inc_rate,activation,dropout,bathnorm,pool_type,upconv,residual)
	output_rphi = Conv3D(1,1, activation="linear", padding="same", kernel_initializer="normal")(output_rphi)


	output_z = level_block(i,start_ch,depth,inc_rate,activation,dropout,bathnorm,pool_type,upconv,residual)
	output_z	= Conv3D(1,1, activation="linear",padding="same", kernel_initializer="normal")(output_z)
	o = concatenate([output_r,output_rphi,output_z])
	return Model(inputs=i,outputs=output_r)

