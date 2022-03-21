# -*- coding: utf-8 -*-
"""

@author: Zhiqian Zhao
"""

import os
import pandas as pd

import random
import numpy as np
import heapq
from scipy import stats
from scipy.io import loadmat,savemat
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from keras import backend as k
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from keras import layers
from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten, Dropout, Conv1D, Activation, AveragePooling1D,LSTM,MaxPooling1D
from tensorflow.keras.layers import Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import concatenate

from tensorflow.keras.layers import Conv2DTranspose,LeakyReLU,add,Reshape
from  tensorflow.keras.layers import GlobalAveragePooling1D
from  tensorflow.keras.models import Model
import keras.backend as K
from keras import regularizers
from  tensorflow.keras.layers import  GlobalAveragePooling2D, Concatenate,MaxPooling2D, GlobalMaxPooling2D,ZeroPadding2D,ZeroPadding1D
from keras.layers.core import Reshape, Permute





def onehot(labels):
    n_sample = len(labels)
    n_class =max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


mat = loadmat('D:\\国际论文\\论文资料\SPQ转子故障\\TWO_WDCNN_DECNN\\数据\\SPQidataset变工况4类5万1千2.mat')
X_train = mat['X_train']
X_test = mat['X_test']
X_val=mat['X_val']
'''
y_train=mat[' y_train']
y_test=mat['y_test']
y_val=mat[' y_val']
'''
y_train=np.array(mat[' y_train'].tolist()[0])
y_test=np.array(mat['y_test'].tolist()[0])
y_val=np.array(mat[' y_va'].tolist()[0]) 


y_train = onehot(y_train )
y_test = onehot(y_test)
y_val = onehot(y_val)


scaler = MinMaxScaler()
X_train_minmax = scaler.fit_transform(X_train.T).T
X_test_minmax = scaler.fit_transform(X_test.T).T    
X_val_minmax = scaler.fit_transform(X_val.T).T    
X_train=X_train_minmax
X_test=X_test_minmax
X_val=X_val_minmax 

X_test=X_test[:,:,np.newaxis]
X_train=X_train[:,:,np.newaxis]
X_val=X_val[:,:,np.newaxis]
'''
savemat('D:\\转子动力学论文\\SPQ转子故障\\SPQzhuanzidataset.mat',{'X_train':X_train,\
                                                         'X_test':X_test,'X_val':X_val,' y_train': y_train,\
                                                             'y_test':y_test,'y_val':y_val})
'''

channel_1, channel_2 = [], []
earlystop = EarlyStopping(monitor= 'val_loss', min_delta=0 , patience=400, verbose=0, mode='min')

# Block 1

# AMPCNN channel 1
seq_len1 = X_train.shape[1]
sens1 = X_train.shape[2]
# Create input shape based on sequence length
input_shape1 = (seq_len1, sens1)
left_input1 = Input(input_shape1)

# Block 1
conv1 = Conv1D(filters=16, kernel_size=128, strides=8, padding='same')(left_input1)
batch1 = BatchNormalization()(conv1)
activation1 = Activation("relu")(batch1)
poo11 = AveragePooling1D(strides=2)(activation1)
#channel_1.append(GlobalAveragePooling1D()(activation1))

# channel 2
conv21 = Conv1D(filters=16, kernel_size=12, strides=4, padding='same')(left_input1)
batch21 = BatchNormalization()(conv21)
activation21 = Activation("relu")(batch21)
pool21 = AveragePooling1D(strides=2)(activation21)
#channel_2.append(GlobalAveragePooling1D()(activation21))


def compute_euclidean_match_score(l_r):
    l, r = l_r
    denominator = 1. + K.sqrt(
        -2 * K.batch_dot(l, r, axes=[2, 2]) +
        K.expand_dims(K.sum(K.square(l), axis=2), 2) +
        K.expand_dims(K.sum(K.square(r), axis=2), 1)
    )
    denominator = K.maximum(denominator, K.epsilon())
    return 1. / denominator



mul = compute_euclidean_match_score([poo11,pool21])
mulT = Permute((2,1))(mul)

d1_1 =  Dense(units = 16)(mul)  #framesize = 80
d2_1 =  Dense(units = 16)(mulT)
'''
x0 = Permute((2,1))(poo11)
x1 = Permute((2,1))(pool21)

x0 = Reshape(( x0.shape[1], x0.shape[2], 1))(x0)
x1 = Reshape(( x1.shape[1], x1.shape[2], 1))(x1)
 
d1_1 = Reshape(( d1_1.shape[2], d1_1.shape[1], 1))(d1_1)
d2_1 = Reshape(( d2_1.shape[2], d2_1.shape[1], 1))(d2_1)

'''
  
conv11 = concatenate([poo11,d1_1],axis=2)
conv212 = concatenate([pool21,d2_1],axis=2)




# Block 2
conv2 = Conv1D(filters=32, kernel_size=3, strides =1, padding='same')(conv11)
batch2 = BatchNormalization()(conv2)
activation2 = Activation("relu")(batch2)
pool2 = AveragePooling1D(strides=2)(activation2)
#channel_1.append(GlobalAveragePooling1D()(activation2))
# Block 3
conv3 = Conv1D(filters=64, kernel_size=3, strides =1, padding='same')(pool2)
batch3 = BatchNormalization()(conv3)
activation3 = Activation("relu")(batch3)
pool3 = AveragePooling1D(strides=2)(activation3)
#channel_1.append(GlobalAveragePooling1D()(activation3))
# Block 4
conv4 = Conv1D(filters=64, kernel_size=3, strides =1, padding='same')(pool3)
batch4 = BatchNormalization()(conv4)
activation4 = Activation("relu")(batch4)
pool4 = AveragePooling1D(strides=2)(activation4)        
channel_1.append(GlobalAveragePooling1D()(activation4))
# Block 5
conv5 = Conv1D(filters=64, kernel_size=3, strides =1)(pool4)
batch5 = BatchNormalization()(conv5)
activation5 = Activation("relu")(batch5)
pool5 = AveragePooling1D(strides=2)(activation5)
flatten1 = GlobalAveragePooling1D()(pool5)
channel_1.append(GlobalAveragePooling1D()(activation5))



# AMPCNN channel2

# Block 2
conv22 = Conv1D(filters=32, kernel_size=6, strides =4, padding='same')(conv212)
batch22 = BatchNormalization()(conv22)
activation22 = Activation("relu")(batch22)
pool22 = AveragePooling1D(strides=2)(activation22)
#channel_2.append(GlobalAveragePooling1D()(activation22))
# Block 3
conv23 = Conv1D(filters=64, kernel_size=3, strides =2, padding='same')(pool22)
batch23 = BatchNormalization()(conv23)
activation23 = Activation("relu")(batch23)
#pool3 = AveragePooling1D(strides=2)(activation3)
#channel_2.append(GlobalAveragePooling1D()(activation23))
'''
# Block 4
conv4 = Conv1D(filters=128, kernel_size=3, strides =2)(pool3)
batch4 = BatchNormalization()(conv4)
activation4 = Activation("relu")(batch4)
pool4 = AveragePooling1D(strides=2)(activation4)        
'''
pool3g= Reshape(target_shape=[1,8, 64])(activation23)   


Dconv1 = Conv2DTranspose(filters=64, kernel_size=[3,1], strides =[2,1], padding='same')(pool3g)
batch25 = BatchNormalization()(Dconv1)
activation25 = Activation("relu")(batch25)
pool24 = AveragePooling2D(pool_size=2,strides=2)(activation25)
#channel_2.append(GlobalAveragePooling2D()(activation25))
Dconv2 = Conv2DTranspose(filters=32, kernel_size=[3,1], strides =[2,1], padding='same')(pool24)
batch26 = BatchNormalization()(Dconv2)
activation26 = Activation("relu")(batch26)
pool25 = AveragePooling2D(strides=2)(activation26)
channel_2.append(GlobalAveragePooling2D()(activation26))
Dconv3 =Conv2DTranspose(filters=16, kernel_size=[3,1], strides =[2,1], padding='same')(pool25)
batch27 = BatchNormalization()(Dconv3)
activation27= Activation("relu")(batch27)
channel_2.append(GlobalAveragePooling2D()(activation27))
batch28 = BatchNormalization()(activation27)
flatten2 = GlobalAveragePooling2D()(batch28)
h1 = channel_1.pop(-1)
if channel_1:
  h1 = concatenate([h1] + channel_1)
  
h2 = channel_2.pop(-1)
if channel_2:
  h2 = concatenate([h2] + channel_2)

merged = concatenate([h1, h2])
batch29 = BatchNormalization()(merged)
# Add final fully connected layers
dense1 = Dense(128,activation='sigmoid')(batch29)
dropout2 = Dropout(0.5)(dense1)
output = Dense(4, activation = "sigmoid")(dropout2)

AMPCNN_multi = Model(inputs=[left_input1],outputs=output)

print(AMPCNN_multi.summary())

print(AMPCNN_multi.count_params())



# initialize optimizer and random generator within one fold
tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=156324)
#sdg=tf.keras.optimizers.SGD(lr=0.001)
ADAM=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-8)
AMPCNN_multi.compile(optimizer=ADAM,
            loss='mean_squared_error',
            metrics=['accuracy'])



# Fit the model
'''
from tensorflow.keras.models import load_model
f=load_model('D:\\转子动力学论文\\SPQ转子故障\\APCNN99.58%.h5')
wdcnn_multi=f'''

history=AMPCNN_multi.fit([X_train], y_train,validation_data = ([X_test],y_test), epochs =300, batch_size = 2560, verbose=1, 
             callbacks =[earlystop], shuffle = True)
