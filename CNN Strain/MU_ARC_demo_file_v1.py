# -*- coding: utf-8 -*-
"""
Updated on Mon Nov 18 11:13:19 2024

@author: Dr. Benjamin Vien
Demonstration Only: for Strainv2 XX 100k_UNET Version L and Non
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import scipy.io
import mat73

#%%Load trained model and validation data.

new_model2 = tf.keras.models.load_model('2024_output\model_save2024_v2xx_100k_unet_L_non.keras')
mat_v = scipy.io.loadmat('strain2024_validation_data_100_true.mat')

datax2=mat_v['datax'];
datay2 =mat_v['datay_xx'];  

#%% Prediction of Validation Set
decoded_imgs2 = new_model2.predict(datax2)

#%% Show Plots
n = 5
valstart=0;
for i in range(1, n + 1):
        # Display True Strain Fields
        ax = plt.subplot(2, n, i)
        plt.imshow(datay2[i-1+valstart].reshape(32, 32),cmap='RdBu',
               alpha=1)
        plt.colorbar()
        
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Predicted Strain Fields
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs2[i-1+valstart].reshape(32, 32),cmap='RdBu',
               alpha=1)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.colorbar()
plt.show()
metric_valnn=new_model2.evaluate(datax2,datay2, verbose=0, return_dict=True)
print(metric_valnn)
print('Completed!')