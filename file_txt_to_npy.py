# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 01:47:47 2025

@author: artao
"""
import os
import numpy as np
folder = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\for report\MoreCs137_0,0"

def transform(folder, filename):
    data = np.loadtxt(os.path.join(folder, filename))
    np.save(os.path.join(folder, filename + ".npy"), data)
    return 
    

# 讀 txt
filename = ["OGS_H2DPI0", "OGS_H2DPI1", "OGS_H2DPI2", 
            "OGS_H2DPI0_All_Pulses", "OGS_H2DPI1_All_Pulses", "OGS_H2DPI2_All_Pulses" ]
for i in filename:
    transform(folder, i)


