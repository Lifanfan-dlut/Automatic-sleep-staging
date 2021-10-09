# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 13:50:50 2021

This codeee is used to test the model.

@author: LI Fanfan, 3rd Ph.D in School of Biomedical Engineering of Dalian University of Technology.
"""

import os
import glob
import pickle
import numpy as np
import keras
from sklearn import metrics
from hypnogram import plot_hypnogram
import matplotlib.pyplot as plt
from pyriemann.utils.viz import plot_confusion_matrix
from sleep_quality import sleep_quality
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

seed=np.random.seed(2018)

#%%  data path
model_path = input('Please input the model path:')
results_path = input('Please input the results path:')
test_path = input('Please input the test data path:')

model_files = glob.glob(os.path.join(model_path,'weight model','*.hdf5'))
test_files = glob.glob(os.path.join(test_path,'*.p'))

hyp_path = os.path.join(results_path,'results-balance-900','test hypnogram')
if not os.path.exists(hyp_path):
    os.makedirs(hyp_path)
    
cm_path = os.path.join(results_path,'results-balance-900','CM')
if not os.path.exists(cm_path):
    os.makedirs(cm_path)     
#%%
labels = ['W', 'N1', 'N2', 'N3', 'REM']
classes = 5
all_cv_acc = []
all_cv_k = []
all_cv_f1 = []

All_W = []
All_N1 = []
All_N2 = []
All_N3 = []
All_R = []

true_quality = []
pre_quality = []

for model_file in model_files:
    test_model = keras.models.load_model(model_file)
    test_model.summary()
    test_cv = os.path.basename(model_file)[:-5]
    
    for test_file in test_files:
            
        file_name = os.path.basename(test_file)
        
        with open(test_file,'rb') as f_test:
            test_sample = pickle.load(f_test)
        test_label = np.array(test_sample[1])
        # d_ind = np.where(test_label==4)[0]
        # test_label[d_ind] = 3
        # r_ind = np.where(test_label==5)[0]
        # test_label[r_ind] = 4
        y_pre_pro = test_model.predict(np.expand_dims(np.array(test_sample[0]),axis=2))
        y_pre = np.argmax(y_pre_pro,axis=1)
        
       
        
        acc_score = metrics.accuracy_score(test_label,y_pre)
        all_cv_acc.append(acc_score)
        
        f1_score = metrics.f1_score(test_label,y_pre,average='macro')
        all_cv_f1.append(f1_score)
        
        k_score = metrics.cohen_kappa_score(test_label,y_pre)
        all_cv_k.append(k_score)
        
        cm = metrics.confusion_matrix(test_label,y_pre)
        cm_nor = np.zeros((classes,classes))#normalize confusion matrix
        
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                cm_nor[r,c] = cm[r,c]/sum(cm[r,:])
     
        plt.figure(0)
        plot_confusion_matrix(test_label,y_pre,labels)
        plt.savefig(os.path.join(cm_path,'{}-cm-{}.png'.format(test_cv,file_name[6:-2])),dpi=300)
        
        pre_quality_index = sleep_quality(y_pre)
        pre_quality.append(pre_quality_index)
        true_quality_index = sleep_quality(test_label)
        true_quality.append(true_quality_index)
    
        #Each stage accuracy
        W_acc = cm_nor[0,0]
        N1_acc = cm_nor[1,1]
        N2_acc = cm_nor[2,2]
        N3_acc = cm_nor[3,3]
        R_acc = cm_nor[4,4]
        
        All_W.append(W_acc)
        All_N1.append(N1_acc)
        All_N2.append(N2_acc)
        All_N3.append(N3_acc)
        All_R.append(R_acc)
        
        algor_hyp_name = '{} algorithm hypn of {}'.format(test_cv,file_name[6:-2])
        true_hyp_name = 'manal hypn of {}'.format(file_name[6:-2])
        if test_cv == 'cv1':
            plot_hypnogram(stages=test_label, figpath=hyp_path, fname=true_hyp_name+'.png', color='b',
                            legend='Manual Scoring', labels=labels,
                            title='Manual Sleep Scoring for {}.'.format(file_name[6:-2]))
            plot_hypnogram(stages=y_pre, figpath=hyp_path, fname=algor_hyp_name+'.png', color='r',
                            legend='Algorithm Scoring', labels=labels,
                            title='Algorithm Sleep Scoring for {}. Accuracy is {:.2f}%'.format(file_name[6:-2],acc_score*100))
        else:
            plot_hypnogram(stages=y_pre, figpath=hyp_path, fname=algor_hyp_name+'.png', color='r',
                            legend='Algorithm Scoring', labels=labels,
                            title='Algorithm Sleep Scoring for {}. Accuracy is {:.2f}%'.format(file_name[6:-2],acc_score*100))
        
#%%metrics

cv_acc = np.reshape(np.array(all_cv_acc),(5,-1))
cv_f1 = np.reshape(np.array(all_cv_f1),(5,-1))
cv_k = np.reshape(np.array(all_cv_k),(5,-1))


mean_cv_acc = np.mean(cv_acc,axis=1)
mean_cv_f1 = np.mean(cv_f1,axis=1)               
mean_cv_k = np.mean(cv_k,axis=1)  

mean_acc = np.mean(mean_cv_acc)
mean_f1 = np.mean(mean_cv_f1)   
mean_k = np.mean(mean_cv_k)  

print('Mean accuracy:',mean_acc)
print('Mean f1:',mean_f1)
print('Mean k:',mean_k)

np.save(os.path.join(model_path,'temp_cv_acc'),cv_acc)
np.save(os.path.join(model_path,'temp_cv_f1'),cv_f1)
np.save(os.path.join(model_path,'temp_cv_k'),cv_k)


#SAVE WAKW
All_cv_sub_W = np.array(All_W).reshape((5,len(test_files)))
np.save(os.path.join(model_path,'temp_All_cv_sub_W'),All_cv_sub_W)

#SAVE N1
All_cv_sub_N1 = np.array(All_N1).reshape((5,len(test_files)))
np.save(os.path.join(model_path,'temp_All_cv_sub_N1'),All_cv_sub_N1)

#SAVE N2
All_cv_sub_N2 = np.array(All_N2).reshape((5,len(test_files)))
np.save(os.path.join(model_path,'temp_All_cv_sub_N2'),All_cv_sub_N2)

#SAVE N3
All_cv_sub_N3 = np.array(All_N3).reshape((5,len(test_files)))
np.save(os.path.join(model_path,'temp_All_cv_sub_N3'),All_cv_sub_N3)

#SAVE REM
All_cv_sub_R = np.array(All_R).reshape((5,len(test_files)))
np.save(os.path.join(model_path,'temp_All_cv_sub_R'),All_cv_sub_R)

#SAVE QUALITY
true_quality = np.array(true_quality)
pre_quality = np.array(pre_quality)
np.save(os.path.join(model_path,'true_sleep_quality'),true_quality)
np.save(os.path.join(model_path,'pre_sleep_quality'),pre_quality)
