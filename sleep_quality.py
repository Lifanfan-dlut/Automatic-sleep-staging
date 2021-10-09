# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:31:51 2021

This function is used to calculate parameters about sleep quality. The followings are included:

TST, SE, DST, LST, RST, SL, RL, WASO, AHI

@author: LI Fanfan, 3rd Ph.D in School of Biomedical Engineering of Dalian University of Technology.
"""
#%%
import numpy as np

def sleep_quality(stages):
    #Total Recording Time(TRT)
    if type(stages)==list:
        stages = np.array(stages)
    TRT = len(stages)*30/60
    # print('Predicted Total Recording Time in minutes:',TRT)
    #Number of Each Stage
    N1_index = np.where(stages==1)[0]
    Num_N1 = len(N1_index)
    N2_index = np.where(stages==2)[0]
    Num_N2 = len(N2_index)
    N3_index = np.where(stages==3)[0]
    Num_N3 = len(N3_index)
    R_index = np.where(stages==4)[0]
    Num_R = len(R_index)
    #Total Sleep Time(TST) in min
    TST = round((Num_N1 + Num_N2 + Num_N3 + Num_R) * 30/60,2)
    # print('Total Sleep Time in minutes:',TST)
    #Percentage of each sleep stage
    N1_per = round((Num_N1*30/60)/TST,3)
    N2_per = round((Num_N2*30/60)/TST,3)
    D_per = round((Num_N3*30/60)/TST,3)
    R_per = round((Num_R*30/60)/TST,3)
    NREM_per = round(((Num_N1+Num_N2+Num_N3)*30/60)/TST,3)
    
    #Sleep Efficiency
    SE_per = TST/TRT
    # SE_per = SE_per*100
    SE = round(SE_per,3)
    # print('The normal sleep efficiency of adults should be bigger than 90%')
    # print('Sleep Efficiency:{}'.format(SE))
    #Deep Sleep Time(DST) in minutes
    
    # print('All Deep Sleep Time in minutes:',SWST)
    # print('Deep Sleep Percentage:',DSP)
    #REM Sleep Time(RST) in minutes
    
    # print('All REM Sleep Time in minutes:',RST)
    # print('REM Sleep Perentage:{}'.format(RSP))
    #Low Sleep Time(LST)
    LST = (Num_N1 + Num_N2) * 30/60
    # print('All Low Sleep Time in minutes:',LST)
    #Sleep Latency(SL) and REM Latency
    ##find index of the first sleep stage
    # print('The normal REM latency of adults should be between 90 and 120 minutes')
    
    lab_type,lab_count = np.unique(stages,return_counts=True)
    
    first_nonzero_ind = np.nonzero(stages)[0][0]
    last_nonzero_ind = np.nonzero(stages)[0][-1]
    if Num_R == 0:
        print('This subject does not have REM sleep stages')
        RL = 0
    else:
        firstRIndex = list(stages).index(4)
        RL = (firstRIndex-first_nonzero_ind)*30/60
        
    #Sleep Period Time(SPT). Note: it refers to the time from sleep onset to the last sleep epoch, including WASO
        
    SPT = (len(stages)+last_nonzero_ind-first_nonzero_ind)*30/60
    
    #Wake After Sleep Onset(WASO). Note: Sleep Onset refers to the first sleep epoch
    WASO_index = list(stages).index(0,first_nonzero_ind)#第一帧睡眠期到记录结束之间所有的清醒时间综合
    WASO = (WASO_index-first_nonzero_ind)*30/60#trt-st-lst
    # print('The normal WASO time of adults should be less than 20 minutes')
    # print('Wake After Sleep Onset(WASO) in minutes:',WASO)
       
    SL = first_nonzero_ind*30/60
    # print('The normal sleep latency of adults should be about 20 minutes')
    # print('Sleep Latency in Minutes:',SL)
    #REM Latency(RL).Note: it refers to the time from sleep onset to the first REM epoch
    # print('REM Latency in Minutes:',RL)    

  
    quality_index = [SL,RL,SE,D_per,R_per,NREM_per,SPT,TRT,WASO,TST,N1_per,N2_per,LST]
    return quality_index#,AHI
