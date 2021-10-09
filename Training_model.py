# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:39:06 2021

@author:  LI Fanfan, 3rd Ph.D in School of Biomedical Engineering of Dalian University of Technology.
"""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.optimizer_v2 import adam
from keras.callbacks import ModelCheckpoint#,EarlyStopping
from CCN_SE_M import CCN_SE_M
from sklearn.preprocessing import OneHotEncoder
from batch_data_generate import batch_generator
from keras.utils.multi_gpu_utils import multi_gpu_model
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

seed=np.random.seed(2018)
#%% loading data
dir_path = input('Please input the path of your processed samples:')
extra_dir_path = input('Please input the extra path for class-balance strategy:')

data_files = os.listdir(dir_path)
results_path = os.path.join(dir_path,'mild-results-balance-900')
if not os.path.exists(results_path):
    os.makedirs(results_path)
save_model_path = os.path.join(results_path,'weight model')
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

#%%
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

#%%
batch_size = 512
epochs = 150 
classes = 5

labels = ['W', 'N1', 'N2', 'N3', 'REM']

all_cv_acc = []
all_cv_k = []
all_cv_f1 = []

All_W = []
All_N1 = []
All_N2 = []
All_N3 = []
All_R = []

oenc=OneHotEncoder(sparse=False)
for num_file in range(0,len(data_files)):
    file = data_files[num_file]
    print('file name:',file)
    cv_path = os.path.join(dir_path,file)
    train_samples = os.path.join(cv_path,'train data.p')
    val_samples = os.path.join(cv_path,'val data.p')
    with open(train_samples,'rb') as f_train:        
        tr_data = pickle.load(f_train)
 
    with open(val_samples,'rb') as f_val:        
        va_data = pickle.load(f_val)

    #    le_y_val = le.fit(y_val)
    Y_val = oenc.fit_transform(va_data[1].reshape(-1,1))
    ## SE_CNN MODEL
    # with strategy.scope():
    inputshape = tr_data[0].shape[1:]
    model_input = Input(shape=inputshape)
    model = CCN_SE_M(model_input=model_input,classes=classes)
    par_model = multi_gpu_model(model, gpus=2)
    adam = adam.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
    par_model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['acc'])
    par_model.summary()       
    savemodel_file = os.path.join(save_model_path,'{}.hdf5'.format(file))
    # checkpoint = [EarlyStopping(monitor='val_acc',patience=30),
    #               ParallelModelCheckpoint(model,savemodel_file,monitor='val_acc',mode='max',
    #                                       save_weights_only=False,save_best_only=True,verbose=2,period=1)]
    for epoch in range(epochs):
        cc = list(zip(tr_data[0],tr_data[1]))
        random.shuffle(cc)
        tr_data[0][:],tr_data[1][:] = zip(*cc)
        del cc
        # steps_per_epoch = len(tr_data[0])//batch_size
        # for batch_count in steps_per_epoch:
        for (batch_data,onehot_lab) in batch_generator(tr_data[0],tr_data[1],batch_size,extra_dir_path):
        
            h = par_model.train_on_batch(next(batch_data,onehot_lab))
    
    ##ploting training curve of accuracy and loss
        plt.plot(h.history['acc'])
        # plt.plot(h.history['val_acc'])
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Train'],loc='lower right')
        plt.savefig(os.path.join(results_path,'{} balance accuracy.png'.format(file)),dpi=300)
        plt.show()
         
        plt.plot(h.history['loss'])
        # plt.plot(h.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train'],loc='upper right')
        plt.savefig(os.path.join(results_path,'{} balance loss.png'.format(file)),dpi=300)
        plt.show()
    
     #save model
    model.save(savemodel_file)