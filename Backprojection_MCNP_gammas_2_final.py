"""
Notes: 
Due to the Write_Simulation_n_v2 script, time cuts are not necessary!
Due to PyMPPost input, time broadening should already be done on the data!
Are still performed for completeness though.
-rlopezle
"""
"""

-Vincent
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
sns.set(rc={"figure.dpi":350, 'savefig.dpi':300})
sns.set_style("ticks")
sns.set_context("talk", font_scale=0.8)
folder = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\for report\MoreCs137_90,0"
#folder = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\alright\Cf252_0,0"
#folder = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\for report\CF252_center"
"""
 ___            __  ___    __        __  
|__  |  | |\ | /  `  |  | /  \ |\ | /__` 
|    \__/ | \| \__,  |  | \__/ | \| .__/ 
                                         
"""
def run_BP(Doubles_File_Name, Radius, scipy=None):
    #------------------------------------------------------------------------2

    from matplotlib.colors import LogNorm
    from matplotlib import cm
    from dataloader_Doubles_neutron_simulation_v2 import DataLoader_Doubles_Gamma_Simulation
    np.random.seed(117)
    ##############################################
    ############ Parameters to change ############
    ##############################################
        
    Relative_Uncertainty = 50 #100000  #Originally 50
    Number_Cones_to_Project = 2057            #max of the cones
    Energy_gate = [0.6 ,200] # Energy_gate[0] +- Energy_gate[1] in MeV
    
    ##############################################
    
    
    E_Cutoff = 50.0           #Originally 50 keVee
    Max_E_Cutoff = 2725.0     #Originally 2725 keVee (max from Birks is 2732)
    Min_Time_Window = 0    #0.1   #0.137 ns [10 MeV]
    Max_Time_Window = 9        #9     #8.9 ns   
    Data_Structure = 180       #NxN matrix for outputing the waves, how to define
    z_broad = True
    
    ###############################################
    ###############################################
    ###############################################
    """
      __        __      __   __            ___  __  
    |__)  /\  |__)    /  ` /  \ |  | |\ |  |  /__` 
    |__) /~~\ |  \    \__, \__/ \__/ | \|  |  .__/ 
                                                  
    """
    #-------------------------------------------------------------------------4
    def Bar_Counts(Waves):
        Data_Out = np.zeros((len(Waves),12))
        good_counter = 0
        Bar_Counter = np.zeros(20) # 12 + 8 
    
        for event in np.arange(0, len(Waves), 1):
            Bar_1 = int(Waves[event][0]) - 1 #why (-1), because the counter starts at 0 but bars starts at 1
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
            Data_Out[good_counter][11] = Waves[event][11]     # Etotal 
            
            good_counter+=1
    
        print("\n")
        print("Initial Count of Events in the 12 Bars")
        print("\n")
        print(Bar_Counter)
        return Data_Out[:good_counter] #output data from write gamma simulations
    
    """
    ___          ___     __       ___  __  
     |  |  |\/| |__     /  ` |  |  |  /__` 
     |  |  |  | |___    \__, \__/  |  .__/ 
                                          
    """
    def Time_Cuts(Waves, Min_Time_Window, Max_Time_Window):
        Data_Out = np.zeros_like(Waves) 
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
    
    """
     ___     __       ___  __  
    |__     /  ` |  |  |  /__` 
    |___    \__, \__/  |  .__/
    """
    
    def Apply_E_Cuts(Doubles_Data): 
        Data_Out =  np.zeros_like(Doubles_Data) 
        Edep_1_Counter = 0
        Edep_2_Counter = 0
        Edep_Both = 0
        
             
        for double in Doubles_Data:
            E1_pass = False
            Edep_1 = double[9] 
            E2_pass = False
            Edep_2 = double[10]
            if Edep_1 > E_Cutoff/1000 and Edep_1 < Max_E_Cutoff/1000:
                Edep_1_Counter+=1
                E1_pass = True
            if Edep_2 > E_Cutoff/1000 and Edep_2 < Max_E_Cutoff/1000:
                Edep_2_Counter+=1
                E2_pass = True
            if E1_pass and E2_pass and Energy_gate[0]-Energy_gate[1] <=Edep_1 + Edep_2 <= Energy_gate[0]+Energy_gate[1]:
                Data_Out[Edep_Both] = double
                Edep_Both+=1
    
        
        print("\n")
        print("Number of interactions in first bar within LO thresholds/E thresholds: "+str(Edep_1_Counter))
        print("Number of interactions in second bar within LO thresholds/E thresholds: "+str(Edep_2_Counter))
        print("Interactions within LO/E limits and energy gate: "+str(Edep_Both))
        print("\n")
        return Data_Out[:Edep_Both]
    
    """
     __                    ___  __      __   __   __  
    /  ` |__|  /\  |\ |     |  /  \    |__) /  \ /__` 
    \__, |  | /~~\ | \|     |  \__/    |    \__/ .__/ 
                                                     
    """
    def Z_Uncertainty(Z, chan, LO):
        Path_to_Z_un = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\Cal_Stil_Data/"
        Min_LO = np.array([25,50,75,100,125,150,175,200,225,250,275,300,325,350,375])
        Chan_to_Bar = [2,3,5,6,7,8,9,10,11,12,14,15,  1,4,13,16,17,18,19,20]
        if chan in [1,4,13,16,17,18,19,20]: #uncertainty of CeBr
            Z_un = 0.5
            return Z_un 
        Input_Data = open(Path_to_Z_un+"Bar_"+str(Chan_to_Bar[int(chan)])+"_un.txt","r") #open uncertainty file
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
        
        return Z_un
    
    def Convert_Channel_to_Position(E_Cut_Applied_Data):
        Data_Out = np.zeros_like(E_Cut_Applied_Data) 
        good_counter = 0
        in_bar_Counter = 0    
        Center_Bars_Pos = np.array([[-0.3165, 2.2155, 0.0],  
                                [0.9495, 2.2155, 0.0],   
                                #####
                                [-2.2155, 0.9495, 0.0], 
                                [-0.9495, 0.9495, 0.0],  
                                [0.3165, 0.9495, 0.0],   
                                [1.5825, 0.9495, 0.0],  
                                #####
                                [-1.5825, -0.3165, 0.0], 
                                [-0.3165, -0.3165, 0.0], 
                                [0.9495, -0.3165, 0.0],  
                                [2.2155, -0.3165, 0.0],  
                                #####
                                [-0.9495, -1.5825, 0.0], 
                                [0.3165, -1.5825, 0.0], 
                                #####
                                [-1.5825, 2.2155, -2.2],  
                                [-1.5825, 2.2155, 2.2],  
                                [2.2155, 2.2155, -2.2],  
                                [2.2155, 2.2155, 2.2],   
                                [-2.2155, -1.5825, -2.2], 
                                [-2.2155, -1.5825, 2.2], 
                                [1.5825, -1.5825, -2.2],  
                                [1.5825, -1.5825, 2.2]]) 
    
        for event in np.arange(0, len(E_Cut_Applied_Data), 1):
            Bar_1 = int(E_Cut_Applied_Data[event][0])
            Bar_2 = int(E_Cut_Applied_Data[event][1])
            
            # Exact according to MCNP
            
            X_1 = E_Cut_Applied_Data[event][2]
            Y_1 = E_Cut_Applied_Data[event][3]
            Z_1 = E_Cut_Applied_Data[event][4]
            X_2 = E_Cut_Applied_Data[event][5]
            Y_2 = E_Cut_Applied_Data[event][6]
            Z_2 = E_Cut_Applied_Data[event][7]
            
            '''
            # Center of bars like in experiment
            X_1 = Center_Bars_Pos[Bar_1][0]
            Y_1 = Center_Bars_Pos[Bar_1][1]
            Z_1 = Center_Bars_Pos[Bar_1][2]
            X_2 = Center_Bars_Pos[Bar_2][0]
            Y_2 = Center_Bars_Pos[Bar_2][1]
            Z_2 = Center_Bars_Pos[Bar_2][2]
            '''
            '''
            X_1 = Center_Bars_Pos[Bar_1][0]
            Y_1 = Center_Bars_Pos[Bar_1][1]
            X_2 = Center_Bars_Pos[Bar_2][0]
            Y_2 = Center_Bars_Pos[Bar_2][1]
            Z_1 = E_Cut_Applied_Data[event][4]
            Z_2 = E_Cut_Applied_Data[event][7]
            '''
            '''
            X_1 = E_Cut_Applied_Data[event][2]
            Y_1 = E_Cut_Applied_Data[event][3]
            X_2 = E_Cut_Applied_Data[event][5]
            Y_2 = E_Cut_Applied_Data[event][6]
            Z_1 = Center_Bars_Pos[Bar_1][2]
            Z_2 = Center_Bars_Pos[Bar_2][2]
            '''
            Z_1_un = Z_Uncertainty(Z_1, Bar_1, (E_Cut_Applied_Data[event][9]*1000)) #E_dep1
            Z_2_un = Z_Uncertainty(Z_2, Bar_2, (E_Cut_Applied_Data[event][10]*1000)) #E_dep2
            
            ## Z_pos Broadening
            if z_broad:
                Z_1 = np.random.normal(Z_1, Z_1_un) # random value that is generated from normal distribution
                Z_2 = np.random.normal(Z_2, Z_2_un)
                Z_1_un = Z_Uncertainty(Z_1, Bar_1, (E_Cut_Applied_Data[event][9]*1000))
                Z_2_un = Z_Uncertainty(Z_2, Bar_2, (E_Cut_Applied_Data[event][10]*1000))
                
            
            if Z_1 < 2.525 and Z_1 > -2.525: #cebr3 condition should add here, changing z2
                in_bar_Counter+=1
    
            if np.abs(Z_1) - Z_1_un < 2.525: #cebr3 condition should add here, changing z2
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
                Data_Out[good_counter][11] = E_Cut_Applied_Data[event][11]     # E_total
                Data_Out[good_counter][12] = E_Cut_Applied_Data[event][12]     # History
                good_counter+=1
    
        print("\n")
        print("Number of events within the bars: "+str(in_bar_Counter))
        print("Number of events within uncertainty in the bars: "+str(good_counter))
        print("\n")
        return Data_Out[:good_counter] #return the data that happenned within the bars
    

    #change the e error to 10% of E instead of 1 kev!
    def E_Un_Plt(Position_Data):
        Data_Out = np.zeros((len(Position_Data),16))
        E1list_e = []
        E2list_e = []
        Etotlist_e = []
        
        Un_Etotlist_e = []
        Rel_Un_Etotlist_e = []
        
        Un_E1list_e = []
        Rel_Un_E1list_e = []
        
        Un_E2list_e = []
        Rel_Un_E2list_e = []
        
        good_counter = 0
        
        x1list = []
        y1list = []
        x2list = []
        y2list = []
    
        for event in np.arange(0, len(Position_Data), 1):
            x1list.append(Position_Data[event][2])
            y1list.append(Position_Data[event][3])
            x2list.append(Position_Data[event][5])
            y2list.append(Position_Data[event][6])
            Vector_1 = np.array([Position_Data[event][2], Position_Data[event][3], Position_Data[event][4]]) #bar 1 vector (x1, y1, z1)
            Vector_2 = np.array([Position_Data[event][5], Position_Data[event][6], Position_Data[event][7]])
            Diff_Vecs = Vector_1 - Vector_2
            
            E_1_e = Position_Data[event][9]  #Edep1 in terms of MeV
            Un_E_1_e = 0.1 * E_1_e
            E_2_e = Position_Data[event][10] #Edep2 in terms of MeV
            Un_E_2_e = 0.1 * E_2_e
            
            E_total_v = Position_Data[event][11]
            Un_E_total_v = np.sqrt(Un_E_1_e**2 + Un_E_2_e**2)
            
            E1list_e.append(E_1_e)
            E2list_e.append(E_2_e)
            Etotlist_e.append(E_total_v)
            
            Un_E1list_e.append(Un_E_1_e)
            Rel_Un_E1list_e.append(Un_E_1_e/E_1_e*100.0) #relative uncertainty
            
            Un_E2list_e.append(Un_E_2_e)
            Rel_Un_E2list_e.append(Un_E_2_e/E_2_e*100.0)
            
            Un_Etotlist_e.append(Un_E_total_v)
            Rel_Un_Etotlist_e.append(Un_E_total_v/E_total_v*100.0)
            
            
        
            Data_Out[good_counter][0] = Position_Data[event][0] # Bar 1 
            Data_Out[good_counter][1] = Position_Data[event][1] # Bar 2 
            Data_Out[good_counter][2] = Position_Data[event][2] # X1
            Data_Out[good_counter][3] = Position_Data[event][3] # Y1
            Data_Out[good_counter][4] = Position_Data[event][4] # Z1
            Data_Out[good_counter][5] = Position_Data[event][5] # X2
            Data_Out[good_counter][6] = Position_Data[event][6] # Y2
            Data_Out[good_counter][7] = Position_Data[event][7] # Z2
            Data_Out[good_counter][8] = Position_Data[event][8] # TOF [ns]
            Data_Out[good_counter][9] = E_1_e
            Data_Out[good_counter][10] = E_2_e
            Data_Out[good_counter][11] = E_total_v
            
            Data_Out[good_counter][12] = Un_E_1_e
            Data_Out[good_counter][13] = Un_E_2_e
            Data_Out[good_counter][14] = Un_E_total_v
            Data_Out[good_counter][15] = Position_Data[event][-1] #history
            
            good_counter+=1
            
        
        
        ## Particle travel dianogstic plots
        x1list = np.array(x1list)
        y1list = np.array(y1list)
        x2list = np.array(x2list)
        y2list = np.array(y2list)
        arrownum = len(x1list)
        arrownum =1000
        colors = np.sqrt(np.add(np.subtract(x2list[:arrownum],x1list[:arrownum])**2, np.subtract(y2list[:arrownum],y1list[:arrownum])**2))
        from matplotlib.colors import Normalize
        norm = Normalize()
        norm.autoscale(colors)
        colormap = cm.coolwarm
        plt.figure(dpi=300)
        plt.quiver(x1list[:arrownum], y1list[:arrownum],
                    np.subtract(x2list[:arrownum],x1list[:arrownum]),
                    np.subtract(y2list[:arrownum],y1list[:arrownum]),
                    angles='xy', scale_units='xy', scale=1,
                    color=colormap(norm(colors))
                    )
        plt.plot(x1list[:arrownum], y1list[:arrownum], 'k.')
        plt.plot(x2list[:arrownum], y2list[:arrownum], 'r.')
        plt.tight_layout()
        plt.title("Simulation Map")
        plt.show()
        plt.close()
        arrownum =30000
        colors2 = np.sqrt(np.add(np.subtract(x2list[:arrownum],x1list[:arrownum])**2, np.subtract(y2list[:arrownum],y1list[:arrownum])**2))
        sns.histplot(colors2, bins=100)
        plt.ylabel('Occurence')
        plt.xlabel('Distance traveled in xy (cm)')
        plt.tight_layout()
        plt.title("Distances")
        plt.show()
        plt.close()
    
        '''
        ## Uncertainty Plots  
        print("[Direct]     Average Neutron Energy Deposition in First Scatter: "+str(np.average(E1list_e))+" MeV")
        print("[Direct]     Average Neutron Energy Deposition in Second Scatter: "+str(np.average(E2list_e))+" MeV")
        print("\n")
        print("[Direct]     Average Neutron Energy Deposition (First Scatter) Uncertainty: "+str(np.average(Un_E1list_e))+" MeV")
        print("[Direct]     Average Neutron Energy Deposition (Second Scatter) Uncertainty: "+str(np.average(Un_E2list_e))+" MeV")
        print("\n")
        print("[Direct]     Average Neutron Energy Deposition (First Scatter) Relative Uncertainty: "+str(np.average(Rel_Un_E1list_e))+" %")
        print("[Direct]     Average Neutron Energy Deposition (Second Scatter) Relative Uncertainty: "+str(np.average(Rel_Un_E2list_e))+" %")
        print("\n")
        '''
        
        '''
        plt.hist(E1list_e, bins=100, range=[0,10], histtype='step')
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("E_1 Plot - MCNP")
        plt.show()
        plt.close()
        
        
        plt.hist(E2list_e, bins=100, range=[0,10], histtype='step')
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("E_2 Plot - MCNP")
        plt.show()
        plt.close()
        
        plt.hist(Un_E1list_e, bins=100, range=[0,1], label='Uncertainty in E$_{1}$', histtype='step')
        plt.legend()
        plt.xlabel("Uncertainty in Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - MCNP")
        plt.show()
        plt.close()
        
        
        plt.hist(Rel_Un_E1list_e, bins=200, range=[0,200], label='Uncertainty in E$_{1}$', histtype='step')
        plt.legend()
        plt.xlabel("Relative Uncertainty (%)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - MCNP")
        plt.show()
        plt.close()
        
        
        plt.hist(Un_E2list_e, bins=100, range=[0,1], label='Uncertainty in E$_{2}$', histtype='step')
        plt.legend()
        plt.xlabel("Uncertainty in Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - MCNP")
        plt.show()
        plt.close()
        
        
        plt.hist(Rel_Un_E2list_e, bins=200, range=[0,200], label='Uncertainty in E$_{2}$', histtype='step')
        plt.legend()
        plt.xlabel("Relative Uncertainty (%)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - MCNP")
        plt.show()
        plt.close()
        '''
        return Data_Out[:good_counter], Etotlist_e
    
    """
     __        __           __   __   __        ___  __  ___    __       
    |__)  /\  /  ` |__/    |__) |__) /  \    | |__  /  `  |  | /  \ |\ | 
    |__) /~~\ \__, |  \    |    |  \ \__/ \__/ |___ \__,  |  | \__/ | \| 
                                                                         
    """
    def Running_Back_Projection(Energy_Data):
        weird_alpha_data = np.empty((0, len(Energy_Data[0,:]+1)))
        accept_alpha_data = np.empty((0, len(Energy_Data[0,:]+1)))
        alpha_list = []
        alpha_acc_list = []
        Binning = Data_Structure
        Data_Range = np.arange(0,int(Binning*Binning),1)
        F_Types = []
    
        for dat in Data_Range:     #create headers
            val = "f" + str(dat)
            F_Types.append(val)
    
        # Defining Coordinates
        Azimuth = np.linspace(-180,180,Binning)
        Altitude = np.linspace(-90,90,Binning)
        Theta,Phi = np.meshgrid(Azimuth,Altitude)
        Azimuth_Radians = np.linspace(0,2.0*np.pi,Binning)  #0~2pi rad
        Altitude_Radians = np.linspace(0,np.pi,Binning)     #0~pi, rad
        Theta_Rad,Phi_Rad = np.meshgrid(Azimuth_Radians,Altitude_Radians)
        Z = (Binning,Binning)
        Z = np.zeros(Z)
        counter = 0
        Alpha_Uncer = []
        Energies_Not_Extracted = []
        Lever_Arms = np.zeros((Number_Cones_to_Project,3))               #x,y,z
        Lever_Arms_Angles = np.zeros((Number_Cones_to_Project,2))        #azimuth, altitude
        raw_count = 0
        for Event in np.arange(0,len(Energy_Data),1):
            if counter < Number_Cones_to_Project:
                if Energy_Data[Event][14] < Relative_Uncertainty:
                    Energies_Not_Extracted.append(Energy_Data[Event][11]) #E_total
                    Z_Data = Creating_Real_Projections(Energy_Data[Event], Radius, Theta_Rad, Phi_Rad)  #Z, Cone_Vector, (Alpha_Var/Alpha)
                    # make sure there is no scatter angle greater than one
                    if abs(Z_Data[-1])>1:
                        weird_data = Energy_Data[Event].reshape(1,-1)
                        weird_alpha_data = np.append(weird_alpha_data, weird_data , axis =0)
                        alpha_list.append(Z_Data[-1])
                    else:
                        accept_data = Energy_Data[Event].reshape(1,-1)
                        accept_alpha_data = np.append(accept_alpha_data, accept_data , axis =0)
                        alpha_acc_list.append(Z_Data[-1])
                    
                    Normalized_Cone_Vector = Z_Data[1]                         
                    Lever_Arms[counter] = Normalized_Cone_Vector
                    Alpha_Uncer.append(Z_Data[2])

                    if Normalized_Cone_Vector[1] <0:
                        Lever_Arms_Angles[counter][0] = -((np.arccos(Normalized_Cone_Vector[0]/(np.sqrt(Normalized_Cone_Vector[0]**2+Normalized_Cone_Vector[1]**2)))*180.0/np.pi)-90.0)
                    else:
                        Lever_Arms_Angles[counter][0] = ((np.arccos(Normalized_Cone_Vector[0]/(np.sqrt(Normalized_Cone_Vector[0]**2+Normalized_Cone_Vector[1]**2)))*180.0/np.pi)-90.0)
                    if Normalized_Cone_Vector[2] <0:
                        Lever_Arms_Angles[counter][1] = -np.arccos((np.sqrt(Normalized_Cone_Vector[0]**2+Normalized_Cone_Vector[1]**2))/(np.sqrt(Normalized_Cone_Vector[0]**2+Normalized_Cone_Vector[1]**2+Normalized_Cone_Vector[2]**2)))*180.0/np.pi
                    else:
                        Lever_Arms_Angles[counter][1] = np.arccos((np.sqrt(Normalized_Cone_Vector[0]**2+Normalized_Cone_Vector[1]**2))/(np.sqrt(Normalized_Cone_Vector[0]**2+Normalized_Cone_Vector[1]**2+Normalized_Cone_Vector[2]**2)))*180.0/np.pi
    
                    Z_Data = Z_Data[0]
                    Z = Z + (Z_Data/(sum(sum(Z_Data))))
                    counter+=1
                    
                    if counter%100000 == 0:
                        print("Cone "+str(counter)+' / '+str(len(Energy_Data))+" being projected.")
        
        alpha_array = np.array(alpha_list).reshape(-1,1)
        alpha_array_acc = np.array(alpha_acc_list).reshape(-1,1)
        weird_data_out = np.concatenate([weird_alpha_data, alpha_array], axis=1)
        acc_data_out = np.concatenate([accept_alpha_data, alpha_array_acc], axis=1)
        Z = np.roll(Z, int(Binning/4.0), axis=1)   
        Z = Z[::-1] 
    
        for row in np.arange(0,len(Z), 1):
            Z[row] = Z[row][::-1]
        print(raw_count)
        return Z, Energies_Not_Extracted, [weird_data_out, acc_data_out]
    
    """
     __   ___               __   __   __        ___  __  ___    __       
    |__) |__   /\  |       |__) |__) /  \    | |__  /  `  |  | /  \ |\ | 
    |  \ |___ /~~\ |___    |    |  \ \__/ \__/ |___ \__,  |  | \__/ | \| 
    """

    def Creating_Real_Projections(Event, Radius, Theta_Rad, Phi_Rad):
        x_1 = np.array([float(Event[2]), float(Event[3]), float(Event[4])])
        x_2 = np.array([float(Event[5]), float(Event[6]), float(Event[7])])
        Cone_Vector = x_1 - x_2
        Normalized_Cone_Vector = np.linalg.norm(Cone_Vector)
        Cone_Vector = Cone_Vector/Normalized_Cone_Vector
        x = Radius*np.cos(Theta_Rad)*np.sin(Phi_Rad) - x_1[0]
        y = Radius*np.sin(Theta_Rad)*np.sin(Phi_Rad) - x_1[1]
        z = Radius*np.cos(Phi_Rad) - x_1[2]
        dot_product = (Cone_Vector[0]*x + Cone_Vector[1]*y +Cone_Vector[2]*z)    
        Scattering_Angle = np.arccos((1+0.511*(1/Event[11]-1/(Event[10]))))*180.0/np.pi 
        if Scattering_Angle > 0.0 and Scattering_Angle < 90.0:
            dot_product[dot_product<0.0] = 0.0
            Alpha = float((1+0.511*(1/Event[11]-1/(Event[10])))**2)
            beta = dot_product*dot_product/(x*x + y*y + z*z)
            Variances = Uncertainty(x,y,z,Event)
            Alpha_Var = Variances[0]
            Beta_Var = Variances[1]/(x*x + y*y + z*z)      
            Z = np.exp(-1.0*((beta - Alpha)*(beta - Alpha))/(2*(Alpha_Var**2+Beta_Var**2)))
            
            return Z, Cone_Vector, (Alpha_Var/Alpha), Alpha
        elif Scattering_Angle > 90.0:
            dot_product[dot_product>0.0] = 0.0
            Alpha = float((1+0.511*(1/(Event[11])-1/(Event[10])))**2)
            beta = dot_product*dot_product/(x*x + y*y + z*z)
            Variances = Uncertainty(x,y,z,Event)
            Alpha_Var = Variances[0]
            Beta_Var = Variances[1]/(x*x + y*y + z*z)
            Z = np.exp(-1.0*((beta - Alpha)*(beta - Alpha))/(2*(Alpha_Var**2+Beta_Var**2)))
            
            return Z, Cone_Vector, (Alpha_Var/Alpha), Alpha
        else:
            print('Undefined Scattering Angle!')
            return 0, -1    
    """
               __   ___  __  ___             ___     
    |  | |\ | /  ` |__  |__)  |   /\  | |\ |  |  \ / 
    \__/ | \| \__, |___ |  \  |  /~~\ | | \|  |   |  
                                                    
    """
    def Uncertainty(xdel,ydel,zdel,Event): #See the Uncertainty_Analysis.py script for where the 2 uncertainty functions come from   
        m_e = 0.511 #MeV*c**2
        x_1 = Event[2]
        y_1 = Event[3]
        z_1 = Event[4]
        x_2 = Event[5]
        y_2 = Event[6]
        z_2 = Event[7]
        E_1 = Event[9]
        E_2 = Event[10]
        
        un_E_1 = Event[12]
        un_E_2 = Event[13]
        un_Z_1 = 1#Z_1_un    extract from above data
        un_Z_2 = 1#Z_2_un
        
        
        Alpha_Uncer = (4*m_e**2*un_E_1**2*(m_e*(1/(E_1 + E_2) - 1/E_2) + 1.0)**2/(E_1 + E_2)**4 + 
                      4*m_e**2*un_E_2**2*(m_e*(1/(E_1 + E_2) - 1/E_2) + 1.0)**2*(-1/(E_1 + E_2)**2 + 
                      E_2**(-2))**2)**0.5
        Beta_Uncertainty = (un_Z_1**2*(xdel*(-x_1 + x_2)*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5) + ydel*(-y_1 + y_2)*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5) + zdel*(-z_1 + z_2)*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5))**2*(2*xdel*(-x_1 + x_2)*(-1.0*z_1 + 1.0*z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*ydel*(-y_1 + y_2)*(-1.0*z_1 + 1.0*z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*zdel*(-z_1 + z_2)*(-1.0*z_1 + 1.0*z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) - 2*zdel*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5))**2 + un_Z_2**2*(xdel*(-x_1 + x_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-0.5) + ydel*(-y_1 + y_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-0.5) + zdel*(-z_1 + z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-0.5))**2*(2*xdel*(-x_1 + x_2)*(1.0*z_1 - 1.0*z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*ydel*(-y_1 + y_2)*(1.0*z_1 - 1.0*z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*zdel*(-z_1 + z_2)*(1.0*z_1 - 1.0*z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*zdel*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5))**2 + 0.09*(xdel*(-x_1 + x_2)*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5) + ydel*(-y_1 + y_2)*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5) + zdel*(-z_1 + z_2)*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5))**2*(2*xdel*(-x_1 + x_2)*(-1.0*x_1 + 1.0*x_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) - 2*xdel*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5) + 2*ydel*(-1.0*x_1 + 1.0*x_2)*(-y_1 + y_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*zdel*(-1.0*x_1 + 1.0*x_2)*(-z_1 + z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5))**2 + 0.09*(xdel*(-x_1 + x_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-0.5) + ydel*(-y_1 + y_2)*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5) + zdel*(-z_1 + z_2)*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5))**2*(2*xdel*(-x_1 + x_2)*(1.0*x_1 - 1.0*x_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*xdel*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5) + 2*ydel*(1.0*x_1 - 1.0*x_2)*(-y_1 + y_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*zdel*(1.0*x_1 - 1.0*x_2)*(-z_1 + z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5))**2 + 0.09*(xdel*(-x_1 + x_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-0.5) + ydel*(-y_1 + y_2)*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5) + zdel*(-z_1 + z_2)*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5))**2*(2*xdel*(-x_1 + x_2)*(-1.0*y_1 + 1.0*y_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*ydel*(-y_1 + y_2)*(-1.0*y_1 + 1.0*y_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) - 2*ydel*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5) + 2*zdel*(-1.0*y_1 + 1.0*y_2)*(-z_1 + z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5))**2 + 0.09*(xdel*(-x_1 + x_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-0.5) + ydel*(-y_1 + y_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-0.5) + zdel*(-z_1 + z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-0.5))**2*(2*xdel*(-x_1 + x_2)*(1.0*y_1 - 1.0*y_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*ydel*(-y_1 + y_2)*(1.0*y_1 - 1.0*y_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5) + 2*ydel*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + 
                           (-z_1 + z_2)**2)**(-0.5) + 2*zdel*(1.0*y_1 - 1.0*y_2)*(-z_1 + z_2)*((-x_1 + x_2)**2 + 
                           (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**(-1.5))**2)**0.5  
        return [Alpha_Uncer, Beta_Uncertainty]
        
    

    
    def Real_projection(Energy_Data):
        Data_Out = np.zeros((len(Position_Data),16))
        nan_count = 0
        real_count = 0

        
        for event in np.arange(0,len(Energy_Data),1):
            Scattering_Angle = np.arccos((1+0.511*(1/(Energy_Data[event][11])-1/(Energy_Data[event][10]))))*180.0/np.pi
            if np.isnan(Scattering_Angle):
                nan_count+=1     
            else:
                Data_Out[real_count][:] = Energy_Data[event][:]
                real_count+=1
        print(f'real counts:{real_count}')
        print(f'filtered counts:{nan_count}')        
        return Data_Out[:real_count,:]
    
    #%%
    
    def attach_history(data):
        filename = "output_data.xlsx"
        
        df = pd.read_excel(os.path.join(folder, filename))
        array = df.to_numpy()
        attached = array[:,-1]
        attached = attached.reshape(-1, 1)
        data = np.concatenate((data, attached), axis=1)
        return data 
    
    
    #------------------------------------------------------------------------3
    #load data
    Data = DataLoader_Doubles_Gamma_Simulation((Doubles_File_Name))#load data from other program
    Number_of_data_structures = Data.GetNumberOfWavesInFile()#same size of gammas event #load data from other program
    Waves = Data.LoadWaves(Number_of_data_structures) #all datas (events, 13)
    Doubles_Data = Bar_Counts(Waves)              ## return output data from write gamma simulations
    Doubles_Data = attach_history(Doubles_Data)
    #-------------------------------------------------------------------------5
    E_Cut_Applied = Apply_E_Cuts(Doubles_Data)                                 ## #filtered the double events in our ROI
    Position_Data = Convert_Channel_to_Position(E_Cut_Applied)
    Timing_Data = Time_Cuts(Position_Data, Min_Time_Window, Max_Time_Window)
    Energy_Data = E_Un_Plt(Timing_Data)
    Filtered_data = Real_projection(Energy_Data[0])
    
    print("\n")
    print("Number of cones before rel. uncert. cut: "+str(len(Filtered_data)))
    print("\n")
    
    z_data, E_out, weird_data = Running_Back_Projection(Filtered_data)
    
    print("\n")
    print("Number of cones after rel. uncert. cut: "+str(len(E_out)))
    print("\n")
    
    '''
    plt.hist(E_out, bins=100, range=[0,10], histtype='step')
    plt.xlabel("Neutron Energy (MeV)")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.tight_layout()
    plt.title("Reconstructed Neutron Energy (After Imaging)")
    plt.show()
    plt.close()
    '''
    return z_data, E_out, weird_data

# %%
#-----------------------------------------------------------------------------1
filename =  "Simulation_gamma_Doubles_File_OGS_PyMPPost_Processed.dat"
z_data, E_out, weird_data_out = run_BP(os.path.join(folder, filename), Radius=10) 


# Scratch with scripts needed to get plots from simulation backprojector
Binning = 180
Azimuth = np.linspace(-180,180,Binning)
Altitude = np.linspace(-90,90,Binning)
Theta,Phi = np.meshgrid(Azimuth,Altitude)

max_idx = np.unravel_index(np.argmax(z_data), z_data.shape)
max_theta = Theta[max_idx]
max_phi = Phi[max_idx]




plt.pcolormesh(Theta,Phi,z_data, cmap='inferno')
plt.xlabel("Azimuthal Angle (θ)")
plt.ylabel("Altitude Angle (φ)")
plt.title(f"Cones: {len(E_out)}")
plt.colorbar()

# Mark the lightest spot
plt.scatter(max_theta, max_phi, facecolors='none', edgecolors='red', s=20, marker='o')



plt.savefig(os.path.join(folder,"result.png"), dpi=300)
plt.show()
plt.close()


#%% export weird data used for dubugging

'''
# this part is leave for debugging, export the data in excel form
filename = "dumn1"
data_raw = np.loadtxt(os.path.join(folder, filename))
dest = "data_compare.xlsx"

cols_post =  ['Bar_1', 'Bar_2', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'tof', 
           'Edep1_d', 'Edep2_d', 'Esum', "13","14","15", "history", "alpha value"]

df_w = pd.DataFrame(weird_data_out[0], columns = cols_post)
df_acc = pd.DataFrame(weird_data_out[1], columns = cols_post)

filtered_w_r = data_raw[np.isin(data_raw[:, 0], weird_data_out[0][:,-2])]
filtered_acc_r = data_raw[np.isin(data_raw[:, 0], weird_data_out[1][:,-2])]



cols_raw = ['history', 'particle type ', 'cell', 'energy released', 'Energy prior to collision']

df_w_r = pd.DataFrame(filtered_w_r[:,[0,2,5,6,15]], columns = cols_raw)
df_acc_r = pd.DataFrame(filtered_acc_r[:,[0,2,5,6,15]], columns = cols_raw)


with pd.ExcelWriter(os.path.join(folder, dest)) as writer:
    df_w.to_excel(writer, sheet_name='wierd_data', index=False)
    df_w_r.to_excel(writer, sheet_name='wierd_raw', index=False)
    df_acc.to_excel(writer, sheet_name='acc_data', index=False)
    df_acc_r.to_excel(writer, sheet_name='acc_raw', index=False)
    '''