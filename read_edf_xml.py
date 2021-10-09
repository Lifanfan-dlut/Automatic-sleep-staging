# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:37:21 2021

This code is used to read signals and annotations at the same time and trim the non-stage epochs.

@author: LI Fanfan, 3rd Ph.D in School of Biomedical Engineering of Dalian University of Technology.
"""

import read_edf
import read_xml
import numpy as np

def read_edf_and_annot(edf_file,hyp_file):
    
    samples,channel_sf = read_edf.read_edfrecord(edf_file)
    stages = read_xml.read_annot_regex(hyp_file)
    stages = np.array(stages)
    stages = stages.reshape((-1,1))
    
    # processing lables as recommended in AASM

    d_ind = np.where(stages==4)[0]
    stages[d_ind] = 3
    r_ind = np.where(stages==5)[0]
    stages[r_ind] = 4
    
    # deleting abnormal epochs
    
    lab_type,counts = np.unique(stages,return_counts=True)
    
    sleep_stages = [0,1,2,3,4]
    
    if set(sleep_stages) >= set(lab_type)==True:
        stages = stages
        samples = samples
    else:        
        #deleting abnormal labels
        Other_index = [index for index, other_stage in enumerate(stages) if other_stage not in sleep_stages]
        stages = np.delete(stages,Other_index,axis=0)        
        samples = np.delete(samples,Other_index,axis=0)
        
    #Trimming the extra 30minutes wake stages at the beginning of the sleep
    first_nonzero_ind = np.nonzero(stages)[0][0]
    if first_nonzero_ind > 60:
        print('Trimming......')
        samples = samples[60:,:]
        stages = stages[60:]
    #keeping the last 15minutes wake sages.
    last_nonzero_ind = np.nonzero(stages)[0][-1]
    if (len(stages)-last_nonzero_ind-1)>30:
        print('Trimming......')
        samples = samples[:(last_nonzero_ind+31),:]
        stages = stages[:(last_nonzero_ind+31)]
    assert len(samples)==len(stages)

    assert len(samples) == len(stages)
    return samples,stages,channel_sf