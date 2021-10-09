# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:37:07 2021

This code is used to read stag annotations from XML files.

@author: LI Fanfan, 3rd Ph.D in School of Biomedical Engineering of Dalian University of Technology.
"""

import re

def read_labels(xmlfile):
    
    with open(xmlfile,'r') as f:
       content = f.read()
       
    #read stages label
    patterns_stages = re.findall(
        r'<EventType>.+Stages</EventType>\n'+
        r'<EventConcept>.+</EventConcept>\n'+
        r'<Start>[0-9\.]+</Start>\n'+
        r'<Duration>[0-9\.]+</Duration>',content)

    stages = []
    starts = []
    durations = []
    for pattern in patterns_stages:
        lines = pattern.splitlines()
        #print(lines)
        stageline = lines[1]
        #print(stageline)
        stage = int(stageline[-16])
        startline = lines[2]
        start = float(startline[7:-8])
        durationline = lines[3]
        duration = float(durationline[10:-11])
        assert duration % 30 == 0
        epochs_duration = int(duration) // 30
        stages += [stage]*epochs_duration
        starts += [start]
        durations += [duration]
    assert int((start+duration)//30) == len(stages)
   
    return stages