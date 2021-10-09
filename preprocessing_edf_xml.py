# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:38:00 2021

This function is used to preprocess the dataset. Parameters in this function denote:
(1) base_dir: the directory of data and label.
(2) channel: EEG channel used to score sleep (['EEG(sec)', 'EEG(SEC)','EEG2','EEG 2']).
(3) d_len: styles of input data length.
(4) cla: classification categories.

It should be noted that:

Firstly, the para d_len is set to indicate the input length. 
(1) 30 -- normal, one-to-one
(2) 900 -- 90s input and its label is the center stage, many-to-one


Then, when running this code, you should input the file path where you will save the preprocessed files. 
According to the recommend in the console, you should also provide the file name of edf data and xml data.

Next, This function also provides the preprocessing for three-stage classfication, i.e. Wake(w), NREM, REM.
It will be useful for the AHI calculation, which needs to calculate the number of hypopnea and apnea events
during NREM and REM stages. So, at the beginning of running this function, you should set para cla.
(1) cla == 3 means three stages classification (Wake(w), NREM, REM).
(2) cla == 5 means five stages classification (Wake(w), N1, N2, N3, REM)
 
@author: LI Fanfan, 3rd Ph.D in School of Biomedical Engineering of Dalian University of Technology.
"""

import os
import pickle
import glob
import read_edf_xml
import numpy as np

#%%

def prepare_dataset(base_dir,channel,d_len,cla):

    #read and preprocessed data
    save_dir = input('Please input the file path to save the preprocessed data: ')
    edfdata = 'data'
    edf_dir = os.path.join(base_dir,edfdata)
    edf_names = glob.glob(os.path.join(edf_dir,'*.edf'))
    annfile = 'label'
    hyp_dir = os.path.join(base_dir,annfile)
    hyp_names = glob.glob(os.path.join(hyp_dir,'*-nsrr.xml'))
    print('number of records:', len(edf_names))
    print('number of hypnograms:', len(hyp_names))
    
    assert len(edf_names) == len(hyp_names)
    #loop on records
    for edf_file in edf_names:
        edf_name = os.path.basename(edf_file)
        subject_name = edf_name[:-4]
        hyp_name = os.path.join(hyp_dir,subject_name+'-nsrr.xml')
        save_path = os.path.join(save_dir,'single chan preprocessed data')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if edf_file is not None:
            try:
                
                print('#####################')            
                EEG_samples,stages,channel_sf = read_edf_xml.read_edf_and_annot(edf_file,hyp_name)
                #verify that there is no other stage
                labels,counts = np.unique(stages,return_counts=True)
                print('labels and counts:'),
                print(labels,counts)

                assert len(EEG_samples)==len(stages),'the length must be equal'
                
                 
                    
                if cla == 3:
                    #converting labels 1,2,3 into "1" and 4 into "2"
                    print('Three states classification processing...')
                    indices4 = np.where(stages==4)
                    stages[indices4] = 2
                    indices3 = np.where(stages==3)
                    stages[indices3] = 1
                    indices2 = np.where(stages==2)
                    stages[indices2] = 1
                   
                    cl,cnts = np.unique(stages,return_counts=True)
                    print('lables and counts for three classes classification: ', cl,cnts)
                    
                if cla == 5:
                    #converting labels 1,2,3 into "1" and 4 into "2"
                    print('Five states classification processing...')
                    cl,cnts = np.unique(stages,return_counts=True)
                    print('lables and counts for five classes classification: ', cl,cnts)
                                                   
                               
                # preparing 30s data and label
                if d_len == 30:
                    save_file = os.path.join(save_path,'{}cla 30s preprocessing'.format(cla),subject_name+'.p')
                    data = EEG_samples,stages
                    print('saving...')                    
                    if not os.path.exists(os.path.dirname(save_file)):
                        os.makedirs(os.path.dirname(save_file))
                    with open(save_file,'wb') as fp:
                        pickle.dump(data,fp)
                
                    
                # preparing 90s-C data and label
                if d_len == 900:
                    save_file = os.path.join(save_path,'{}cla 90s Center preprocessing'.format(cla),subject_name+'.p')
                    data_len = int(channel_sf*90)
                    new_samples = np.zeros((0,data_len))
                    new_stages = stages[1:-1]
                    for idx in range(len(stages)-2):                                                                
                        sample_before = EEG_samples[idx,:]
                        sample = EEG_samples[idx+1,:]
                        sample_after = EEG_samples[idx+2,:]
                        joint_sample = np.concatenate((sample_before,sample,sample_after),axis=0)
                        joint_sample = joint_sample.reshape((1,-1))
                        new_samples = np.concatenate((new_samples,joint_sample),axis=0)
        
                    assert len(new_samples)==len(new_stages),'the length must be equal'
                    
                    data = new_samples,new_stages
                    print('saving...')
                    if not os.path.exists(os.path.dirname(save_file)):
                        os.makedirs(os.path.dirname(save_file))
                    with open(save_file,'wb') as fp:
                        pickle.dump(data,fp)
                
                                        
            except FileNotFoundError:
                print('File not found.')
            except AssertionError:
                print('AssertionError. File {} has different length of data and label'.format(subject_name))
                print('Skipping this patient')
                
            except ValueError:
                print('ValueError. Sampling rate of file {} is not 125'.format(subject_name))               
                print('Skipping this patient')
                    
            finally:
                pass