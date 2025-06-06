# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:12:18 2025

@author: artao
"""

import numpy as np
import pandas as pd
import os
import time
#folder = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\L"
folder = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\alright\Cf252_0,0"
start_time = time.time()


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
        else:
            First_Diff = Second - First 
            Second_Diff = Third - First
            Third_Diff = Fourth - First
            if Third_Diff < Max_Time_Window_Bar and Third_Diff > Min_Time_Window_Bar:
                Coincidence = [First, Second, Third, Fourth]
                Coincidence_Four.append(Coincidence)
                Data_Counter+=4
            elif Second_Diff < Max_Time_Window_Bar and Second_Diff > Min_Time_Window_Bar:
                Coincidence = [First, Second, Third]
                Coincidence_Triples.append(Coincidence)
                Data_Counter+=3
            elif First_Diff < Max_Time_Window_Bar and First_Diff > Min_Time_Window_Bar:
                Coincidence = [First, Second]
                Coincidence_Doubles.append(Coincidence)
                Coincidences.append(Coincidence)
                Data_Counter+=2
            else:
                Data_Counter+=1
                
    print("MCNP Gamma Event (After Coincidence Logic):")
    print("Number of Double Events: "+str(len(Coincidence_Doubles)))
    print("Number of Triple Events: "+str(len(Coincidence_Triples)))
    print("Number of Four Events: "+str(len(Coincidence_Four)))
    print("\n")
    return Coincidences



#--------------------------------------------------------------2
if multiple_files:
    Coincident_Data_Out = np.zeros((100000*nFiles, 12)) #why (10000*1) * 12?
    for j in range(nFiles):
    
        '''
        change the filename to gamma files
        '''
        filename = "dumn1_All_Pulses"
        data = np.loadtxt(os.path.join(folder, filename))  #(datatype ndarray)
        filename = "dumn1"
        data_raw = np.loadtxt(os.path.join(folder, filename))

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
        '''
        above is discriminating the events, only double events are returned
        '''
       
        
        
        
        '''
        below is doing psd removal to seperate neutrons and gammas
        find the mesh to cell map
        '''
        time_window_verify = [] #del once verify
        recorded_history = [] # in the later data verification, this term is easy to lookup in raw data
        folder_psd_double_counts = 0
        for coincidence in np.arange(0,len(Coincident_Events_no_psd),1):
        
            idx_B1    = np.where(data_g[:,5] == Coincident_Events_no_psd[coincidence][0]) #index
            First_Bar = data_g[idx_B1,1] #the cell number in mcnp
            
            First_Bar_number = Bar_Num_to_MCNP_Cell[np.where(Bar_Num_to_MCNP_Cell[:,1] == First_Bar)[1],0]

            idx_B2    = np.where(data_g[:,5] == Coincident_Events_no_psd[coincidence][1])
            Second_Bar = data_g[idx_B2,1]
            Second_Bar_number = Bar_Num_to_MCNP_Cell[np.where(Bar_Num_to_MCNP_Cell[:,1] == Second_Bar)[1],0]
            lookup_d = data_raw[:,0]      #histories
            lookup_o = data_g[idx_B1,0] #histories
            time_window_verify.append(lookup_o) #del once verify
            
            idx_B1_coord = np.where(lookup_d == lookup_o)
            
            
            if First_Bar_number != Second_Bar_number and First_Bar_number<=12 and 13<= Second_Bar_number <=20: 
                recorded_history.append(lookup_o)
                d_file_B1 = int(First_Bar)
                d_file_B2 = int(Second_Bar)
                
                d_file_bars_in_history = data_raw[idx_B1_coord[1],5]
                
                d_file_B1_idx = np.where(d_file_B1 == d_file_bars_in_history)[0]
                d_file_B2_idx = np.where(d_file_B2 == d_file_bars_in_history)[0]
                
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][0] =  First_Bar_number                                     # Bar 1 Nr.
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][1] =  Second_Bar_number                                    # Bar 2 Nr.
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][2] =  data_raw[idx_B1_coord[1][d_file_B1_idx],8][0]   # x Bar 1
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][3] =  data_raw[idx_B1_coord[1][d_file_B1_idx],9][0]   # y Bar 1
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][4] =  data_raw[idx_B1_coord[1][d_file_B1_idx],10][0]  # z Bar 1
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][5] =  data_raw[idx_B1_coord[1][d_file_B2_idx],8][0]   # x Bar 2
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][6] =  data_raw[idx_B1_coord[1][d_file_B2_idx],9][0]   # y Bar 2
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][7] =  data_raw[idx_B1_coord[1][d_file_B2_idx],10][0]  # z Bar 2
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][8] =  (data_g[idx_B2,5] - data_g[idx_B1,5])         # TOF Bar 1 to 2
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][9] =  data_raw[idx_B1_coord[1][d_file_B1_idx],6][0]   # Edep Bar 1 energy released 
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][10] = data_raw[idx_B1_coord[1][d_file_B2_idx],6][0]   # Edep Bar 2 energy released
                Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][11] =  data_raw[idx_B1_coord[1][d_file_B1_idx],6][0] + data_raw[idx_B1_coord[1][d_file_B2_idx],6][0] #E_dep bar1 +E_dep bar2  
                folder_psd_double_counts = folder_psd_double_counts + 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"執行時間: {elapsed_time:.4f} 秒")                    
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

import numpy as np
import pandas as pd

def Writing_Data_to_xlsx(Doubles_Data, file):
    columns = ['Bar_1', 'Bar_2', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'tof', 
               'Edep1_d', 'Edep2_d', 'Etotal']
    
    
    df = pd.DataFrame(Doubles_Data, columns=columns)
    df['history'] = [x[0][0] for x in recorded_history]
    
    df.to_excel(file, index=False)
file_destination = "output.xlsx"
Writing_Data_to_xlsx(Coincident_Data_Out[:total_psd_double_counts], os.path.join(folder, file_destination))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"執行時間寫資料: {elapsed_time:.4f} 秒")