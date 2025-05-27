import tensorflow as tf
from timeit import default_timer as timer
import numpy as np
import cv2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
import shutil
from tensorflow.keras.layers import concatenate
from sklearn.metrics import confusion_matrix, f1_score
from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']="0"
import numpy as np
import pandas as pd 
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import math



def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):

    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)
  
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:

        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection




def upsampling_block(expansive_input, contractive_input, n_filters=32):
 
    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 (3,3),    # Kernel size
                 strides=(2,2),
                 padding='same')(expansive_input)

    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,   # Number of filters
                 (3,3),     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,  # Number of filters
                 (3,3),   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv)

    
    return conv



def build_unet(input_size=(96, 128, 3), n_filters=32, n_classes=23):

    inputs = Input(input_size)
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*4)
    cblock4 = conv_block(cblock3[0], n_filters*8, dropout_prob=0.3) 
    cblock5 = conv_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=None) 
    ublock6 = upsampling_block(cblock5[0], cblock4[1],n_filters*8)
   
    ublock7 = upsampling_block(ublock6, cblock3[1],n_filters*4)
    ublock8 = upsampling_block(ublock7, cblock2[1],n_filters*2)
    ublock9 = upsampling_block(ublock8, cblock1[1],n_filters )

    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model