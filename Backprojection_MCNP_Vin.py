# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:28:13 2025

@author: artao
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
'''
the style of figures
'''
sns.set(rc={"figure.dpi":350, 'savefig.dpi':300})
sns.set_style("ticks")
sns.set_context("talk", font_scale=0.8)

#------------2----------2--------------2-------------2-----------2-----------2----------
def run_BP(Doubles_File_Name, Radius, scipy=None):
    # import numpy as np
    # import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib import cm
    from dataloader_Doubles_neutron_simulation_v2 import DataLoader_Doubles_Neutron_Simulation
    #from dataloader_Doubles_neutron_simulation_v2 import DataLoader_Doubles_Gammas_Simulation
    np.random.seed(117)
    ##############################################
    ############ Parameters to change ############
    ##############################################
    
    Relative_Uncertainty = 50 #100000  #Originally 50
    Number_Cones_to_Project = 600000 
    
    ##############################################
    
    LO_Cutoff = 50.0           #Originally 50 keVee
    Max_LO_Cutoff = 2725.0     #Originally 2725 keVee (max from Birks is 2732)
    Min_Time_Window = 0.137    #0.1   #0.137 ns [10 MeV]
    Max_Time_Window = 9        #9     #8.9 ns   
    Data_Structure = 180       #NxN matrix for outputing the waves
    z_broad = True
    
#-----------------------4-----------4------------------------4---------------4--    
    def Bar_Counts(Waves):
        Data_Out = np.zeros((len(Waves),14))
        good_counter = 0
        Bar_Counter = np.zeros(12)
    
        for event in np.arange(0, len(Waves), 1):
            Bar_1 = int(Waves[event][0]) - 1
            Bar_2 = int(Waves[event][1]) - 1
            Bar_Counter[Bar_1]+=1
            Bar_Counter[Bar_2]+=1
            Data_Out[good_counter][0] = Waves[event][0] - 1   # Bar 1
            Data_Out[good_counter][1] = Waves[event][1] - 1   # Bar 2
            Data_Out[good_counter][2] = Waves[event][2]       # X Position: Bar 1 [cm]
            Data_Out[good_counter][3] = Waves[event][3]       # Y Position: Bar 1 [cm]
            Data_Out[good_counter][4] = Waves[event][4]       # Z Position: Bar 1 [cm]
            Data_Out[good_counter][5] = Waves[event][5]       # X Position: Bar 2 [cm]
            Data_Out[good_counter][6] = Waves[event][6]       # Y Position: Bar 2 [cm]
            Data_Out[good_counter][7] = Waves[event][7]       # Z Position: Bar 2 [cm]
            Data_Out[good_counter][8] = Waves[event][8]       # TOF [ns]
            Data_Out[good_counter][9] = Waves[event][9]       # Edep Bar 1 [MeV]
            Data_Out[good_counter][10] = Waves[event][10]     # Edep Bar 2 [MeV]
            Data_Out[good_counter][11] = Waves[event][11]     # LO Bar 1 [MeVee]
            Data_Out[good_counter][12] = Waves[event][12]     # LO Bar 2 [MeVee]
            Data_Out[good_counter][13] = Waves[event][13]     # E_TOF [MeV]
            good_counter+=1
    
        print("\n")
        print("Initial Count of Events in the 12 Bars")
        print("\n")
        print(Bar_Counter)
        return Data_Out[:good_counter]
#--6-------6----------------------6-------------------6------6--------6------6-
    def Apply_E_Cuts(Doubles_Data):
        Data_Out = np.zeros((len(Doubles_Data),14))
        Edep_1_Counter = 0
        Edep_2_Counter = 0
        Edep_Both = 0
        
        # Commented out because avoiding using perfect Energy values from MCNP
        # Birks_Fit = Birks(a=0.518,b=2.392)
        # Birks_Fit = Birks(a=0.5366095577079871,b=2.6780735541073404)
        # E_Cutoff = Find_E(Birks_Fit, LO_Cutoff)
        # Max_E_Cutoff = Find_E(Birks_Fit, Max_LO_Cutoff)
        
        for double in Doubles_Data:
            E1_pass = False
            LO_1 = double[11]
            E2_pass = False
            LO_2 = double[12]
            if LO_1 > LO_Cutoff/1000 and LO_1 < Max_LO_Cutoff/1000:
                Edep_1_Counter+=1
                E1_pass = True
            if LO_2 > LO_Cutoff/1000 and LO_2 < Max_LO_Cutoff/1000:
                Edep_2_Counter+=1
                E2_pass = True
            if E1_pass and E2_pass:
                Data_Out[Edep_Both] = double
                Edep_Both+=1
    
        # for double in Doubles_Data:
        #     E1_pass = False
        #     E_1 = double[9]
        #     LO_1 = double[11]
        #     E2_pass = False
        #     E_2 = double[10]
        #     LO_2 = double[12]
        #     if E_1 > E_Cutoff and E_1 < Max_E_Cutoff and LO_1 > LO_Cutoff/1000 and LO_1 < Max_LO_Cutoff/1000:
        #         Edep_1_Counter+=1
        #         E1_pass = True
        #     if E_2 > E_Cutoff and E_2 < Max_E_Cutoff and LO_2 > LO_Cutoff/1000 and LO_2 < Max_LO_Cutoff/1000:
        #         Edep_2_Counter+=1
        #         E2_pass = True
        #     if E1_pass and E2_pass:
        #         Data_Out[Edep_Both] = double
        #         Edep_Both+=1
    
        print("\n")
        print("Number of interactions in first bar within LO thresholds/E thresholds: "+str(Edep_1_Counter))
        print("Number of interactions in second bar within LO thresholds/E thresholds: "+str(Edep_2_Counter))
        print("Number of interactions within LO thresholds/E thresholds in both: "+str(Edep_Both))
        print("\n")
        return Data_Out[:Edep_Both]
#-----9-------------------------9-------------------9------------9-----------9-    
    def Z_Uncertainty(Z, chan, LO):
        Path_to_Z_un = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\Cal_Stil_Data/"
        Min_LO = np.array([25,50,75,100,125,150,175,200,225,250,275,300,325,350,375])
        Chan_to_Bar = [2,3,5,6,7,8,9,10,11,12,14,15]
        Input_Data = open(Path_to_Z_un+"Bar_"+str(Chan_to_Bar[int(chan)])+"_un.txt","r")
        Temp = []
        
        for lines in Input_Data:
            line = lines.split()
            Temp.append(line)
        
        Data_Set = np.array(Temp)
        Column_Value = len(Min_LO[Min_LO<LO])
        Data = Data_Set[:,Column_Value]
        X_Set = Data_Set[:,0]
        if Z < float(X_Set[0]):
            Z_un = float(Data[0])
        elif Z > float(X_Set[-1]):
            Z_un = float(Data[-1])
        else:
            for val in X_Set:
                if float(val) > Z:
                    break
       
            Index_Val = list(X_Set).index(val)       
            y_1 = float(Data[Index_Val-1])
            y_2 = float(Data[Index_Val])
            x = Z
            x_1 = float(X_Set[Index_Val-1])
            x_2 = float(X_Set[Index_Val])
            Z_un = y_1 + (x - x_1)*(y_2-y_1)/(x_2-x_1)
        
        #print("Light Output: "+str(LO))
        #print("Z Position: "+str(Z))
        #print("Uncertainty in Z Position: "+str(Z_un))
        return Z_un

#----------8---------------8---------------8-------------8-------------------8-    
    def Convert_Channel_to_Position(E_Cut_Applied_Data):
        Data_Out = np.zeros((len(E_Cut_Applied_Data),14))
        good_counter = 0
        in_bar_Counter = 0    
        Center_Bars_Pos = np.array([
                            [-0.3165, 4.7555, 0.0],  # Bar # 0 - 2
                            [0.9495, 4.7555, 0.0],   # Bar # 1 - 3
                            #####
                            [-2.2155, 3.4895, 0.0],  # Bar # 2 - 6
                            [-0.9495, 3.4895, 0.0],  # Bar # 3 - 7 
                            [0.3165, 3.4895, 0.0],   # Bar # 4 - 8
                            [1.5825, 3.4895, 0.0],   # Bar # 5 - 9
                            #####
                            [-1.5825, 2.2235, 0.0], # Bar # 6 - 10
                            [-0.3165, 2.2235, 0.0], # Bar # 7 - 11
                            [0.9495, 2.2235, 0.0],  # Bar # 8 - 12
                            [2.2155, 2.2235, 0.0],  # Bar # 9 - 13
                            #####
                            [-0.9495, 0.9575, 0.0], # Bar # 10 - 16
                            [0.3165, 0.9575, 0.0]]) # Bar # 11 - 17
    
        for event in np.arange(0, len(E_Cut_Applied_Data), 1):
            Bar_1 = int(E_Cut_Applied_Data[event][0])
            Bar_2 = int(E_Cut_Applied_Data[event][1])
            ## Exact according to MCNP
            # X_1 = E_Cut_Applied_Data[event][2]
            # Y_1 = E_Cut_Applied_Data[event][3]
            # X_2 = E_Cut_Applied_Data[event][5]
            # Y_2 = E_Cut_Applied_Data[event][6]
            ## Center of bars like in experiment
            X_1 = Center_Bars_Pos[Bar_1][0]
            Y_1 = Center_Bars_Pos[Bar_1][1]
            X_2 = Center_Bars_Pos[Bar_2][0]
            Y_2 = Center_Bars_Pos[Bar_2][1]
            Z_1 = E_Cut_Applied_Data[event][4]
            Z_2 = E_Cut_Applied_Data[event][7]
            #define Z position uncertainty
            Z_1_un = Z_Uncertainty(Z_1, Bar_1, (E_Cut_Applied_Data[event][11]*1000))
            Z_2_un = Z_Uncertainty(Z_2, Bar_2, (E_Cut_Applied_Data[event][12]*1000))
            if z_broad:
                Z_1 = np.random.normal(Z_1, Z_1_un)
                Z_2 = np.random.normal(Z_2, Z_2_un)
                Z_1_un = Z_Uncertainty(Z_1, Bar_1, (E_Cut_Applied_Data[event][11]*1000))
                Z_2_un = Z_Uncertainty(Z_2, Bar_2, (E_Cut_Applied_Data[event][12]*1000))
            # if events happened in bar
            if Z_2 < 2.525 and Z_2 > -2.525 and Z_1 < 2.525 and Z_1 > -2.525:
                in_bar_Counter+=1    
            # Z position minus uncertain = event 100% happenned in bar
            if np.abs(Z_1) - Z_1_un < 2.525 and np.abs(Z_2) - Z_2_un < 2.525:
                Data_Out[good_counter][0] = Bar_1 # Bar 1
                Data_Out[good_counter][1] = Bar_2 # Bar 2
                Data_Out[good_counter][2] = X_1 # X Bar 1
                Data_Out[good_counter][3] = Y_1 # Y Bar 1
                Data_Out[good_counter][4] = Z_1 # Z Bar 1
                Data_Out[good_counter][5] = X_2 # X Bar 2
                Data_Out[good_counter][6] = Y_2 # Y Bar 2
                Data_Out[good_counter][7] = Z_2 # Z Bar 2
                Data_Out[good_counter][8] = E_Cut_Applied_Data[event][8]       # TOF [ns]
                Data_Out[good_counter][9] = E_Cut_Applied_Data[event][9]       # Edep Bar 1 [MeV]
                Data_Out[good_counter][10] = E_Cut_Applied_Data[event][10]     # Edep Bar 2 [MeV]
                Data_Out[good_counter][11] = E_Cut_Applied_Data[event][11]     # LO Bar 1 [MeVee]
                Data_Out[good_counter][12] = E_Cut_Applied_Data[event][12]     # LO Bar 2 [MeVee]
                Data_Out[good_counter][13] = E_Cut_Applied_Data[event][13]     # E_TOF [MeV]
                good_counter+=1
    
        print("\n")
        print("Number of events within the bars: "+str(in_bar_Counter))
        print("Number of events within uncertainty in the bars: "+str(good_counter))
        print("\n")
        return Data_Out[:good_counter]    
#----------10--------------10--------------10--------------10----------------10    
    def Time_Cuts(Waves, Min_Time_Window, Max_Time_Window):
        Data_Out = np.zeros((len(Waves),14))
        good_counter = 0
    
        for event in np.arange(0, len(Waves), 1):
            Delta_T = Waves[event][8]
            if Delta_T > Min_Time_Window and Delta_T < Max_Time_Window:
                Data_Out[good_counter] = Waves[event]
                good_counter+=1
                
        print("\n")
        print("Starting number of double events: "+str(len(Waves)))
        print("Starting number of double events in defined time range: "+str(good_counter))
        print("\n")
        return Data_Out[:good_counter]
    
    Data = DataLoader_Doubles_Neutron_Simulation((Doubles_File_Name))
    #Data = DataLoader_Doubles_Gammas_Simulation((Doubles_File_Name))

    Number_of_data_structures = Data.GetNumberOfWavesInFile()
    Waves = Data.LoadWaves(Number_of_data_structures)
#---------------------3---------------3------------------3-------------------3-    
    Doubles_Data = Bar_Counts(Waves)
#--------------------5-------------5----------5---------------5--------------5-    
    E_Cut_Applied = Apply_E_Cuts(Doubles_Data)
#-------------------7--------------7----------7------------------7-----------7-
    Position_Data = Convert_Channel_to_Position(E_Cut_Applied)
#------------------9----------------9---------------9----------------9-------9-
    Timing_Data = Time_Cuts(Position_Data, Min_Time_Window, Max_Time_Window)
    Energy_Data = E_TOF_LO(Timing_Data)
    
    print("\n")
    print("Number of cones before rel. uncert. cut: "+str(len(Energy_Data[0])))
    print("\n")
    
    z_data, E_out = Running_Back_Projection(Energy_Data[0])
    
    print("\n")
    print("Number of cones after rel. uncert. cut: "+str(len(E_out)))
    print("\n")
    
    plt.hist(E_out, bins=100, range=[0,10], histtype='step')
    plt.xlabel("Neutron Energy (MeV)")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.tight_layout()
    plt.title("Reconstructed Neutron Energy (After Imaging)")
    plt.show()
    plt.close()

    return z_data, E_out














#-------------1------------1--------1---1---------------1----------------------
file_path = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\For Vincnet\Simulation_Neutron_Doubles_File_OGS_PyMPPost_Processed.dat"
z_data, E_out = run_BP(file_path, 10)

'''
# Scratch with scripts needed to get plots from simulation backprojector
Binning = 180
Azimuth = np.linspace(-180,180,Binning)
Altitude = np.linspace(-90,90,Binning)
Theta,Phi = np.meshgrid(Azimuth,Altitude)
plt.pcolormesh(Theta,Phi,z_data, cmap='inferno')
plt.xlabel("Azimuthal Angle (θ)")
plt.ylabel("Altitude Angle (φ)")
plt.colorbar()
'''