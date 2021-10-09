# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 17:02:37 2021

This model is based on our previous research: 
https://doi.org/10.1016/j.bspc.2020.102203

@author: LI Fanfan, 3rd Ph.D in School of Biomedical Engineering of Dalian University of Technology.
"""

from keras.models import Model
from keras.layers import Conv1D,MaxPooling1D
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling1D
from keras.layers.merge import concatenate
import keras

#%%
def CCN_SE_M(model_input,classes):   

    x1 = Conv1D(64,125*10,strides=50,padding='same')(model_input)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(64,1,strides=1,padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(pool_size=3,strides=3)(x1)
    x1 = Dropout(0.2)(x1)
    print(x1.shape)
    
    #x_se1 = SeNetBlock()(x)
    x2 = Conv1D(64,125,strides=50,padding='same')(model_input)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv1D(64,1,strides=1,padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(pool_size=3,strides=3)(x2)
    x2 = Dropout(0.2)(x2)
    print(x2.shape)
    x = concatenate([x1,x2],axis=2)
 
    x = Conv1D(128,3,strides=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128,1,strides=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3,strides=3)(x)
    x = Dropout(0.2)(x)
 
    x = Conv1D(256,3,strides=3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256,1,strides=1,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    se = GlobalAveragePooling1D()(x)        
    se = Dense(int(se.shape[-1]) // 2,use_bias=False,activation=keras.activations.relu)(se)
    se = Dropout(0.25)(se)
    se = Dense(int(x.shape[-1]),use_bias=False,activation=keras.activations.hard_sigmoid)(se)
    se = Dropout(0.25)(se)
    se_out = keras.layers.Multiply()([x,se])
    x = Flatten()(se_out)
 
    x = Dense(200)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
#    x = GlobalMaxPooling1D()(x_se3)
    x = Dense(classes)(x)
    model_out = Activation('softmax')(x)
    model = Model(inputs=model_input,outputs=model_out)
    return model