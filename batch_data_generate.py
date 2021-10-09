# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:17:33 2021

@author: LI Fanfan, 2nd Ph.D in School of Biomedical Engineering of Dalian University of Technology.
"""
import numpy as np
import os
import glob
import pickle
import random
# import time
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

#%%
def batch_generator(data,label,batch_size,extra_stage_path):
    data_size = data.shape[0]
    # print('data size: ',data_size)
    normal_stage=[0,1,2,3,4]
    data,label = shuffle(data,label,random_state=2018)
    batch_count = 0
    while True:

        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            # cc = list(zip(data,label))
            # random.shuffle(cc)
            # data[:],label[:] = zip(*cc)
            # del cc
        start = batch_count*batch_size
        end = start + batch_size
        
        batch_data, batch_label = data[start:end],label[start:end]
        del start,end
        lab_type,batch_lab_count = np.unique(batch_label,return_counts=True)
        lab_type = list(lab_type)
        batch_lab_count = list(batch_lab_count)
            
        if len(lab_type) < 5:
            ret = [i for i in normal_stage if i not in lab_type]
            lab_type += ret
            lab_type.sort()
            insert_index = [lab_type.index(j) for j in lab_type if j in ret]
            for m in insert_index:   
                batch_lab_count.insert(m,0)
        average_count = batch_size//5
        
        for count_num in range(len(batch_lab_count)):
            stage_count = batch_lab_count[count_num]
            if stage_count>=average_count:
                del_count = stage_count - average_count
                del_index = random.sample(list(np.where(batch_label==count_num)[0]),del_count)
                batch_data = np.delete(batch_data,del_index,axis=0)
                batch_label = np.delete(batch_label,del_index,axis=0)

            else:
                extra_count = int(average_count-stage_count)

                if count_num==0:
                    extra_stage_file = 'N0 stage'
                if count_num==1:
                    extra_stage_file = 'N1 stage'
                if count_num==2:
                    extra_stage_file = 'N2 stage'
                if count_num==3:
                    extra_stage_file = 'N3 stage'
                if count_num==4:
                    extra_stage_file = 'N4 stage'
                # print(extra_stage_files[p])
                stage_path = glob.glob(os.path.join(extra_stage_path,extra_stage_file,'*.p'))
                with open(stage_path[0],'rb') as fe:
                    extra_data = pickle.load(fe)
                # new_index = np.random.permutation(np.array(extra_data[0]).shape[0])
                extra_index = random.sample(list(range(len(extra_data[0]))),extra_count)
                batch_data = np.concatenate((batch_data,np.expand_dims(np.array(extra_data[0])[extra_index],axis=2)))
                batch_label= np.concatenate((batch_label,np.array([count_num]*extra_count)))
                del extra_data
        # assert len(batch_data)==len(batch_label)
        # batch_data,batch_label = shuffle(batch_data,batch_label,random_state=2018)
        oenc=OneHotEncoder(categories='auto',sparse=False)
        onehot_lab = oenc.fit_transform(batch_label.reshape(-1,1))
        batch_count += 1
        yield (batch_data,onehot_lab)
