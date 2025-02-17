# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:12:18 2025

@author: artao
"""

import numpy as np
import pandas as pd
from lmfit import  Model
import math
def Comptom_angle(E_dep_bar2, E_dep_total):
    E = E_dep_total
    E1 = E_dep_bar2
    print(E)
    print(E1)
    mcc = 0.511 #electron mass * light speed in eV
    cos_theta = 1 + mcc * ( 1/E - 1/E1 )                 
    print(cos_theta)
    theta = math.acos(cos_theta) 
    return theta


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
    Edep1_o ='f11'
    Edep2_o = 'f12'
    Etof = 'f13'
    outputStructure = np.zeros(len(Doubles_Data), dtype='int16, int16, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64')
    
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
        outputStructure[coincidence][Edep1_o] = Doubles_Data[coincidence][11]
        outputStructure[coincidence][Edep2_o] = Doubles_Data[coincidence][12]
        outputStructure[coincidence][Etof] = Doubles_Data[coincidence][13]
    outputStructure.tofile(Out)
# np.random.seed(117)
gen = np.random.default_rng(117)

'''
gamma decay = neutron decay??
'''
def decaydecay(t, N, tau):
    return N*np.exp(-tau**(t))
# where does the raw data come from
Raw_Percentage_Data = pd.read_excel(r'C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\For Vincnet\RL52_Raw_Neutron_Percent_Removed.xlsx',index_col=None, header=None)
Number_Bars = 12
Band_Separation = 400 
LO_Cutoff= 50
linspace_for_25_keVee = int((Band_Separation-LO_Cutoff)/25) + 1
slice_start = np.linspace(LO_Cutoff, Band_Separation, linspace_for_25_keVee, endpoint=True)
model = Model(decaydecay, independent_vars=['t'])
Fit_Bins = (slice_start[:-1]+slice_start[1:])/2
Percent_Removal_Coefficients = []

'''
find out removal coefficient in each bar
'''
for i in range(0,Number_Bars):
    Neutron_Removed_List = np.array(Raw_Percentage_Data.loc[i,:])
    model = Model(decaydecay, independent_vars=['t']) #(why do it again here)
    result = model.fit(Neutron_Removed_List, t=Fit_Bins, N=100, tau=1)
    fit_paramters = result.best_values.values()
    Percent_Removal_Coefficients.append(list(fit_paramters))
    
Min_Time_Window = 0.137 #ns 
Max_Time_Window = 9   #ns 
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
    [12,18]
    ])

#-----------------------------------------------------------------------
'''
finds and returns the coincident events based on the events occuring within the time window
only return double events
'''
def Finding_Coincident_Events(All_Times, Min_Time_Window_Bar, Max_Time_Window_Bar):
    
    Coincidences = []
    Coincidence_Doubles = []
    Coincidence_Triples = []
    Data_Counter = 0
 
    while Data_Counter < len(All_Times)-2:
        First = All_Times[Data_Counter]
        Second = All_Times[Data_Counter+1]
        Third = All_Times[Data_Counter+2]
        if First == 0:
            Data_Counter+=1
        else:
            First_Diff = Second - First 
            Second_Diff = Third - First
            if Second_Diff < Max_Time_Window_Bar and Second_Diff > Min_Time_Window_Bar:
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
                
    print("MCNP Neutron Event Breakdown (After Coincidence Logic):")
    print("Number of Double Events: "+str(len(Coincidence_Doubles)))
    print("Number of Triple Events: "+str(len(Coincidence_Triples)))
    print("\n")
    return Coincidences


def PSD_Removal(bar, LO):
    if LO > 0 and LO < 400:
        removal_percentage = decaydecay(LO, Percent_Removal_Coefficients[bar][0], Percent_Removal_Coefficients[bar][1])
        survival_percentage = removal_percentage/100
        if survival_percentage > 1.0:
            survival_percentage = 1.0
        elif survival_percentage < 0.0:
            survival_percentage = 0.0
        rand_val = gen.binomial(1,survival_percentage,size=1)[0]
        if rand_val == 1:
            return 1
        else:
            return 0
    else:
        return 1


if multiple_files:
    Coincident_Data_Out = np.zeros((10000*nFiles, 14)) #why (10000*1) * 14?
    for j in range(nFiles):
    
        # filename = base+exp+'pymppost_files/carbon_1_5/0_ns/RL52_Neg'+str(j)+'_All_pulses'    
        # data = np.loadtxt(filename)
        # filename = base+exp+'pymppost_files/carbon_1_5/Final_Time_Resolution_Analysis/RL52_Neg'+str(j)+'_All_pulses'    
        # data = np.loadtxt(filename)
        '''
        change the filename to gamma files
        '''
        filename = r'C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\For Vincnet\dumn1_out_All_Pulses'    
        data = np.loadtxt(filename)  #(datatype ndarray)
        # filename = base+exp+'pymppost_files/carbon_1_5/Final_LO_Broad_Analysis/RL52_Neg'+str(j)+'_All_pulses'    
        # data = np.loadtxt(filename)  
        filename = r'C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\For Vincnet\dumn1'   
        d_file = np.loadtxt(filename)

        idx_n = np.where(data[:,3] == 1) 
        # search from 4th, return the 'True' value, which row or column of data fullfil the condition
        data_n = data[idx_n]
        #extract the whole row or column that match the condition 


        histories_n = data_n[:,0].tolist()
        #convert 'histories' of MCNP to list
        histories_count = pd.Series(histories_n).value_counts()
        #convert hi_n to pandas series and record the times of the events appear 
        singles_mask = histories_count == 1 
        doubles_mask = histories_count == 2
        triples_mask = histories_count == 3
        # return bool value
        print("File "+str(j+1)+'/'+str(nFiles))
        print("\n")
        print("Before PSD:")
        print("MCNP Neutron\\\ gamma  Event Breakdown (True):")
        print("Total Neutrons\\\ gamma: "+str(np.sum(histories_count[singles_mask])+np.sum(histories_count[doubles_mask])+np.sum(histories_count[triples_mask])))
        print("Number of Single Events: "+str(np.sum(histories_count[singles_mask])))
        print("Number of Double Events: "+str(int(np.sum(histories_count[doubles_mask])/2)))
        print("Number of Triple Events: "+str(int(np.sum(histories_count[triples_mask])/3)))
        print("\n")
                     
        data_n[:,5] = data_n[:,5]*10 #Converting from shakes to ns
        
        All_Start_Times = data_n[:,5]
        time_add_hist = 0.0
        for i in range(len(All_Start_Times)):
            time_add = i*1000000000
            hist_num = data_n[i-1,0]
            if hist_num == data_n[i,0]: 
                All_Start_Times[i] = All_Start_Times[i] + time_add_hist   
            else:
                All_Start_Times[i] = All_Start_Times[i] + time_add
                time_add_hist = time_add
        
        Sorted_All_Start_Times = np.sort(All_Start_Times)
        Coincident_Events_no_psd = Finding_Coincident_Events(Sorted_All_Start_Times, Min_Time_Window, Max_Time_Window)
        
        
        
        
        
        '''
        above is discriminate the events, only double events 
        below is doing psd removal to seperate neutrons and gammas
        '''
        folder_psd_double_counts = 0
        for coincidence in np.arange(0,len(Coincident_Events_no_psd),1):
        
            idx_B1    = np.where(data_n[:,5] == Coincident_Events_no_psd[coincidence][0])
            First_Bar = data_n[idx_B1,1] #1-12, the bar where events occured
            First_Bar_number = Bar_Num_to_MCNP_Cell[np.where(Bar_Num_to_MCNP_Cell[:,1] == First_Bar)[1],0]
            idx_B2    = np.where(data_n[:,5] == Coincident_Events_no_psd[coincidence][1])
            Second_Bar = data_n[idx_B2,1]
            Second_Bar_number = Bar_Num_to_MCNP_Cell[np.where(Bar_Num_to_MCNP_Cell[:,1] == Second_Bar)[1],0]
            lookup_d = d_file[:,0]      #histories
            lookup_o = data_n[idx_B1,0] #histories
            idx_B1_coord = np.where(lookup_d == lookup_o)
            
            First_Lived = PSD_Removal(int(First_Bar_number - 1), float(data_n[idx_B1,6]*1000.0)) #0-11 for first paramter input
            Second_Lived = PSD_Removal(int(Second_Bar_number - 1), float(data_n[idx_B2,6]*1000.0))
            if First_Lived and Second_Lived:
                
                if First_Bar != Second_Bar: #making sure MPPost didn't do same bar twice
                    d_file_B1 = int(First_Bar)
                    d_file_B2 = int(Second_Bar)
                    d_file_bars_in_history = d_file[idx_B1_coord[1],5]
                    d_file_B1_idx = np.where(d_file_B1 == d_file_bars_in_history)[0]
                    d_file_B2_idx = np.where(d_file_B2 == d_file_bars_in_history)[0]
                    
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][0] =  First_Bar_number                                     # Bar 1 Nr.
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][1] =  Second_Bar_number                                    # Bar 2 Nr.
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][2] =  d_file[idx_B1_coord[1][d_file_B1_idx],8][0]   # x Bar 1
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][3] =  d_file[idx_B1_coord[1][d_file_B1_idx],9][0]   # y Bar 1
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][4] =  d_file[idx_B1_coord[1][d_file_B1_idx],10][0]  # z Bar 1
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][5] =  d_file[idx_B1_coord[1][d_file_B2_idx],8][0]   # x Bar 2
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][6] =  d_file[idx_B1_coord[1][d_file_B2_idx],9][0]   # y Bar 2
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][7] =  d_file[idx_B1_coord[1][d_file_B2_idx],10][0]  # z Bar 2
                    '''
                    change tof to comptom angles
                    if take off the comment the code will not run, there's energy loss more than half of the incident energy
                    i should eliminate 2 for gamma dataloader
                    '''
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][8] =  (data_n[idx_B2,5] - data_n[idx_B1,5])         # TOF Bar 1 to 2
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][9] =  d_file[idx_B1_coord[1][d_file_B1_idx],6][0]   # Edep Bar 1
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][10] = d_file[idx_B1_coord[1][d_file_B2_idx],6][0]   # Edep Bar 2
                    
                    #Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][8] =  Comptom_angle(d_file[idx_B1_coord[1][d_file_B2_idx],6][0] , d_file[idx_B1_coord[1][d_file_B1_idx],6][0]+d_file[idx_B1_coord[1][d_file_B2_idx],6][0]) 
                    
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][11] = data_n[idx_B1,6]                              # PI Bar 1 (MeVee)
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][12] = data_n[idx_B2,6]                              # PI Bar 2 (MeVee)
                    Coincident_Data_Out[folder_psd_double_counts+total_psd_double_counts][13] = 1 #placeholder, pymppost didnt rteurn this    # MCNP Energy before collision (E_TOF)
                    folder_psd_double_counts = folder_psd_double_counts + 1
                    
                else:    
                    print('Disqualified history!')
                
        print("After PSD:")
        print("MCNP Neutron Event Breakdown (After Coincidence Logic):")
        print("Number of Double Events: "+str(folder_psd_double_counts))
        print("\n")
        total_psd_double_counts = total_psd_double_counts + folder_psd_double_counts
        total_no_psd_double_counts = total_no_psd_double_counts + len(Coincident_Events_no_psd)
        
    print('Total double scatter events found (Before PSD): '+str(total_no_psd_double_counts))
    print('Total double scatter events written to file (After PSD): '+str(total_psd_double_counts))
    print("\n")

    Out = open(r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\For Vincnet\Simulation_gamma_Doubles_File_OGS_PyMPPost_Processed.dat","wb")
    Writing_Data(Coincident_Data_Out[:total_psd_double_counts])
    Out.close()







    