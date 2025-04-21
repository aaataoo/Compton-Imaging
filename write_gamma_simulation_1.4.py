# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:12:18 2025

@author: artao
"""

import numpy as np
from numba import njit, prange
import pandas as pd
import os
import time

folder = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\L"
#folder = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\validation\CF252_90"
#%%


def Writing_Data(Doubles_Data):

    Bar_1 ='f0'
    Bar_2 ='f1'
    x1 = 'f2'
    y1 = 'f3'
    z1 = 'f4'
    x2 = 'f5'
    y2 = 'f6'
    z2 = 'f7'
    tof ='f8'
    Edep1_d ='f9'
    Edep2_d = 'f10'
    Etotal ='f11'
    #undeter_1 = 'f12'
    #undeter_2 = 'f13'
    outputStructure = np.zeros(len(Doubles_Data), 
                               dtype='int16, int16, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64')
    
    for coincidence in np.arange(0,len(Doubles_Data),1):
        outputStructure[coincidence][Bar_1] = Doubles_Data[coincidence][0]
        outputStructure[coincidence][Bar_2] = Doubles_Data[coincidence][1]
        outputStructure[coincidence][x1] = Doubles_Data[coincidence][2]
        outputStructure[coincidence][y1] = Doubles_Data[coincidence][3]
        outputStructure[coincidence][z1] = Doubles_Data[coincidence][4]
        outputStructure[coincidence][x2] = Doubles_Data[coincidence][5]
        outputStructure[coincidence][y2] = Doubles_Data[coincidence][6]
        outputStructure[coincidence][z2] = Doubles_Data[coincidence][7]    
        outputStructure[coincidence][tof] = Doubles_Data[coincidence][8]
        outputStructure[coincidence][Edep1_d] = Doubles_Data[coincidence][9]
        outputStructure[coincidence][Edep2_d] = Doubles_Data[coincidence][10]
        outputStructure[coincidence][Etotal] = Doubles_Data[coincidence][11]
        #outputStructure[coincidence][undeter_1] = Doubles_Data[coincidence][12]
        #outputStructure[coincidence][undeter_2] = Doubles_Data[coincidence][13]
    outputStructure.tofile(Out)
# np.random.seed(117)
#----------------------------------------------------------------------------1-
#%%
gen = np.random.default_rng(117)
Number_Bars = 12
Band_Separation = 400 
LO_Cutoff= 50
linspace_for_25_keVee = int((Band_Separation-LO_Cutoff)/25) + 1
slice_start = np.linspace(LO_Cutoff, Band_Separation, linspace_for_25_keVee, endpoint=True)

   
Min_Time_Window = 0 #ns 
Max_Time_Window = 10000   #ns 
nFiles = 1 #150
multiple_files = True #This should always be true even if they is only one file
total_counter = 0
good_counts = 0

total_no_psd_double_counts = 0
total_psd_double_counts = 0

Bar_Num_to_MCNP_Cell = np.array([
    [1,3],
    [2,4],
    [3,7],
    [4,8],
    [5,9],
    [6,10],
    [7,11],
    [8,12],
    [9,13],
    [10,14],
    [11,17],
    [12,18],
    [13,1],
    [14,2],
    [15,5],
    [16,6],
    [17,15],
    [18,16],
    [19,19],
    [20,20]
    ])

#---------------------------------------------------------------------------
'''
finds and returns the coincident events based on the events occuring within the time window
only return double events
'''
@njit
def Finding_Coincident_Events(All_Times, Min_Time_Window_Bar, Max_Time_Window_Bar):
    
    Coincidences = []
    Coincidence_Doubles = []
    Coincidence_Triples = []
    Coincidence_Four = []
    Data_Counter = 0
 
    while Data_Counter < len(All_Times)-3:
        First = All_Times[Data_Counter]
        Second = All_Times[Data_Counter+1]
        Third = All_Times[Data_Counter+2]
        Fourth = All_Times[Data_Counter+3]
        if First == 0:
            Data_Counter+=1
            continue
        
        First_Diff = Second - First 
        Second_Diff = Third - First
        Third_Diff = Fourth - First
        if Min_Time_Window_Bar < Third_Diff < Max_Time_Window_Bar:
            Coincidence_Four.append((First, Second, Third, Fourth))
            Data_Counter += 4
        elif Min_Time_Window_Bar < Second_Diff < Max_Time_Window_Bar:
            Coincidence_Triples.append((First, Second, Third))
            Data_Counter += 3
        elif Min_Time_Window_Bar < First_Diff < Max_Time_Window_Bar:
            Coincidence_Doubles.append((First, Second))
            Coincidences.append((First, Second))
            Data_Counter += 2
        else:
            Data_Counter += 1
            
   
            
    print("MCNP Gamma Event (After Coincidence Logic):")
    print("Number of Double Events: "+str(len(Coincidence_Doubles)))
    print("Number of Triple Events: "+str(len(Coincidence_Triples)))
    print("Number of Four Events: "+str(len(Coincidence_Four)))
    print("\n")
    return Coincidences


@njit(parallel=True)
def process_all_double_events(Coincident_Events_no_psd, data_g, data_raw, Bar_Num_to_MCNP_Cell, total_psd_double_counts):
    n = len(Coincident_Events_no_psd)
    Coincident_Data_Out = np.zeros((n, 12))
    recorded_history = np.zeros(n)
    valid_mask = np.zeros(n, dtype=np.bool_) # 建立 n array with bool value
        
    #for i in range(n):
    for i in prange(n):
        t0 = Coincident_Events_no_psd[i][0]
        t1 = Coincident_Events_no_psd[i][1]

        # 找出事件時間對應的 index
        idx_B1 = -1
        idx_B2 = -1
        for j in range(len(data_g)):
            if data_g[j, 5] == t0:
                idx_B1 = j
            elif data_g[j, 5] == t1:
                idx_B2 = j
            if idx_B1 != -1 and idx_B2 != -1:
                break
        if idx_B1 == -1 or idx_B2 == -1:
            continue

        First_Bar = data_g[idx_B1, 1] # bar number
        Second_Bar = data_g[idx_B2, 1] # bar number

        # Bar number convert to  MCNP cell number
        First_Bar_number = -1
        Second_Bar_number = -1
        for k in range(len(Bar_Num_to_MCNP_Cell)):
            if Bar_Num_to_MCNP_Cell[k, 1] == First_Bar:
                First_Bar_number = Bar_Num_to_MCNP_Cell[k, 0]
            if Bar_Num_to_MCNP_Cell[k, 1] == Second_Bar:
                Second_Bar_number = Bar_Num_to_MCNP_Cell[k, 0]
        if First_Bar_number == -1 or Second_Bar_number == -1:
            continue


        # 事件篩選條件
        if First_Bar_number == Second_Bar_number or not (First_Bar_number <= 12 and 13 <= Second_Bar_number <= 20):
            continue

        # 找出共同的 history id
        lookup_o = data_g[idx_B1, 0]
        recorded_history[i] = lookup_o

        # 找出所有對應該 history 的 raw 資料 index
        matched_idx = []
        for m in range(len(data_raw)):
            if data_raw[m, 0] == lookup_o:
                matched_idx.append(m)

        if len(matched_idx) == 0:
            continue

        d_file_B1_idx = -1
        d_file_B2_idx = -1
        for m in matched_idx:
            if data_raw[m, 5] == First_Bar:
                d_file_B1_idx = m
            elif data_raw[m, 5] == Second_Bar:
                d_file_B2_idx = m
            if d_file_B1_idx != -1 and d_file_B2_idx != -1:
                break
        if d_file_B1_idx == -1 or d_file_B2_idx == -1:
            continue
        #find out corresond history to data_g
        matched_idx_g = []
        for m in range(len(data_g)):
            if data_g[m, 0] == lookup_o:
                matched_idx_g.append(m)

        if len(matched_idx_g) == 0:
            continue

        d_file_B1_idx_g = -1
        d_file_B2_idx_g = -1
        for m in matched_idx_g:
            if data_g[m, 1] == First_Bar:
                d_file_B1_idx_g = m
            elif data_g[m, 1] == Second_Bar:
                d_file_B2_idx_g = m
            if d_file_B1_idx_g != -1 and d_file_B2_idx_g != -1:
                break
        if d_file_B1_idx_g == -1 or d_file_B2_idx_g == -1:
            continue

        # 寫入輸出陣列
        Coincident_Data_Out[i, 0] = First_Bar_number
        Coincident_Data_Out[i, 1] = Second_Bar_number
        Coincident_Data_Out[i, 2:5] = data_raw[d_file_B1_idx, 8:11]   # x, y, z Bar 1
        Coincident_Data_Out[i, 5:8] = data_raw[d_file_B2_idx, 8:11]   # x, y, z Bar 2
        Coincident_Data_Out[i, 8]   = t1 - t0                         # TOF
        if data_g[d_file_B1_idx_g,-2] > data_g[d_file_B2_idx_g,-2]:
            Coincident_Data_Out[i, 9]   = data_g[d_file_B2_idx, -1]      # Edep Bar 1
            Coincident_Data_Out[i,10]   = data_g[d_file_B1_idx, -1]      # Edep Bar 2
        elif data_g[d_file_B1_idx_g,-2] < data_g[d_file_B2_idx_g,-2]:
            Coincident_Data_Out[i, 9]   = data_g[d_file_B1_idx_g, -1]      # Edep Bar 1
            Coincident_Data_Out[i,10]   = data_g[d_file_B2_idx_g, -1]      # Edep Bar 2
        Coincident_Data_Out[i,11]   = Coincident_Data_Out[i,9] + Coincident_Data_Out[i,10]  # Edep sum

        valid_mask[i] = True

    return Coincident_Data_Out[valid_mask], recorded_history[valid_mask]







#--------------------------------------------------------------2
if multiple_files:
    Coincident_Data_Out = np.zeros((100000*nFiles, 12)) #why (10000*1) * 12?
    for j in range(nFiles):
    
        '''
        change the filename to gamma files
        '''
        filename = "dumn1_All_Pulses.npy"
        data = np.load(os.path.join(folder, filename))  #(datatype ndarray)
        filename = "dumn1.npy"
        data_raw = np.load(os.path.join(folder, filename))

        idx_g = np.where(data[:,3] == 2) #index of gammas
        # search from 4th, return the 'True' value, which row or column of data fullfil the condition from all_pulse
        data_g = data[idx_g] #from all_pulse
        #extract the whole row or column that match the condition 


        histories_g = data_g[:,0].tolist()
        #convert 'histories' from MCNP output file to list
        histories_count = pd.Series(histories_g).value_counts()
        #convert histories_g to pandas series and record the times of the events appear 
        # histories_count_index create by vin used to verify the time window double coincidence
        histories_count_index = histories_count[histories_count == 2].index.sort_values()
        
        singles_mask = histories_count == 1 
        doubles_mask = histories_count == 2
        triples_mask = histories_count == 3
        four_mask = histories_count == 4
        # return bool value
        print("File "+str(j+1)+'/'+str(nFiles))
        print("\n")
        print("MCNP gamma  Event Breakdown (True):")
        print("Total gamma: "+str(np.sum(histories_count[singles_mask])+np.sum(histories_count[doubles_mask])+np.sum(histories_count[triples_mask])+np.sum(histories_count[four_mask])))
        print("Number of Single Events: "+str(np.sum(histories_count[singles_mask])))
        print("Number of Double Events: "+str(int(np.sum(histories_count[doubles_mask])/2)))
        print("Number of Triple Events: "+str(int(np.sum(histories_count[triples_mask])/3)))
        print("Number of Four Events: "+str(int(np.sum(histories_count[four_mask])/4)))
        print("\n")
                     
        data_g[:,5] = data_g[:,5]*10 #Converting from shakes to ns
        
        All_Start_Times = data_g[:,5]
        time_add_hist = 0.0
        
        for i in range(len(All_Start_Times)):
            time_add = i*1000000000
            hist_num = data_g[i-1,0]
            if hist_num == data_g[i,0]: 
                All_Start_Times[i] = All_Start_Times[i] + time_add_hist   
            else:
                All_Start_Times[i] = All_Start_Times[i] + time_add
                time_add_hist = time_add
            #same history seems the time is the same
        
        Sorted_All_Start_Times = np.sort(All_Start_Times)
        
        Coincident_Events_no_psd = Finding_Coincident_Events(Sorted_All_Start_Times, Min_Time_Window, Max_Time_Window)
        Coincident_Events_no_psd = [list(data) for data in Coincident_Events_no_psd]
        '''
        above is discriminating the events, only double events are returned
        '''
       
        
        
        
        '''
        below is doing psd removal to seperate neutrons and gammas
        find the mesh to cell map
        '''
        
        recorded_history = [] # in the later data verification, this term is easy to lookup in raw data
        
        Coincident_Events_no_psd = np.array(Coincident_Events_no_psd)
        Bar_Num_to_MCNP_Cell = Bar_Num_to_MCNP_Cell.astype(np.int64)
        
        start_time = time.time()
        Coincident_Data_Out, recorded_history = process_all_double_events(
            Coincident_Events_no_psd,
            data_g.astype(np.float64),
            data_raw.astype(np.float64),
            Bar_Num_to_MCNP_Cell,
            total_psd_double_counts
        )

        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"執行時間: {elapsed_time:.4f} 秒")                   
           
    folder_psd_double_counts = len(Coincident_Data_Out)
    print("After determining the first event at OGS bar and second event at CeBr3:")
    print("MCNP gammas Event (After Coincidence Logic):")
    print("Number of Double Events: "+str(folder_psd_double_counts))
    print("\n")
    total_psd_double_counts = total_psd_double_counts + folder_psd_double_counts
    total_no_psd_double_counts = total_no_psd_double_counts + len(Coincident_Events_no_psd)
    
print('Total double scatter events found : '+str(total_no_psd_double_counts))
print('Total double scatter events written to file (After discriminate): '+str(total_psd_double_counts))
print("\n")

file_destination = "Simulation_gamma_Doubles_File_OGS_PyMPPost_Processed.dat"
Out = open(os.path.join(folder, file_destination),"wb")
Writing_Data(Coincident_Data_Out[:total_psd_double_counts])
Out.close()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"執行時間結束: {elapsed_time:.4f} 秒")
#%%

def Writing_Data_to_xlsx(Doubles_Data, file):
    columns = ['Bar_1', 'Bar_2', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'tof', 
               'Edep1_d', 'Edep2_d', 'Etotal']
    
    
    df = pd.DataFrame(Doubles_Data, columns=columns)
    df['history'] = recorded_history
    
    df.to_excel(file, index=False)
file_destination = "output_test.xlsx"
Writing_Data_to_xlsx(Coincident_Data_Out[:total_psd_double_counts], os.path.join(folder, file_destination))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"執行時間寫資料: {elapsed_time:.4f} 秒")