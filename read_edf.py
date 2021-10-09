# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:35:22 2021

This code is used to extract singl-channel EEG, i.e., C4-A1, from PSG files.

@author: LI Fanfan, 3rd Ph.D in School of Biomedical Engineering of Dalian University of Technology. 
"""

import pyedflib
import numpy as np

def read_edfrecord(edffile): #the value of length is only related to respiratory signals

    EEG_channels = ['EEG(sec)','EEG(SEC)','EEG2','EEG 2','2']
    f = pyedflib.EdfReader(edffile)
    signal_labels = f.getSignalLabels()
    #selecting EEG channel name
    if len(list(set(EEG_channels)&set(signal_labels))) != 0:
         eeg_chan = list(set(EEG_channels)&set(signal_labels))   
         chan_index = signal_labels.index(eeg_chan[0])            
         eeg_sig = f.readSignal(chan_index)
         channel_sf = f.getSampleFrequency(chan_index)   
         epoch_len = channel_sf*30
         num_epochs = len(eeg_sig) // epoch_len
         sig_eff = eeg_sig[:int(num_epochs*epoch_len)]
         epoch_samples = np.array(np.split(sig_eff,num_epochs))
    
    return epoch_samples,channel_sf
