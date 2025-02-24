"""
Notes: 
Due to the Write_Simulation_n_v2 script, time cuts are not necessary!
Due to PyMPPost input, time broadening should already be done on the data!
Are still performed for completeness though.
-rlopezle
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.dpi":350, 'savefig.dpi':300})
sns.set_style("ticks")
sns.set_context("talk", font_scale=0.8)
"""
 ___            __  ___    __        __  
|__  |  | |\ | /  `  |  | /  \ |\ | /__` 
|    \__/ | \| \__,  |  | \__/ | \| .__/ 
                                         
"""
def run_BP(Doubles_File_Name, Radius, scipy=None):
    # import numpy as np
    # import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib import cm
    from dataloader_Doubles_neutron_simulation_v2 import DataLoader_Doubles_Neutron_Simulation
    np.random.seed(117)
    ##############################################
    ############ Parameters to change ############
    ##############################################
    
    Relative_Uncertainty = 50 #100000  #Originally 50
    Number_Cones_to_Project = 600000                #?????????????????????????????????
    
    ##############################################
    
    LO_Cutoff = 50.0           #Originally 50 keVee
    Max_LO_Cutoff = 2725.0     #Originally 2725 keVee (max from Birks is 2732)
    Min_Time_Window = 0.137    #0.1   #0.137 ns [10 MeV]
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
    def Bar_Counts(Waves):
        Data_Out = np.zeros((len(Waves),14))
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
            Data_Out[good_counter][11] = Waves[event][11]     # LO Bar 1 [MeVee]
            Data_Out[good_counter][12] = Waves[event][12]     # LO Bar 2 [MeVee]
            Data_Out[good_counter][13] = Waves[event][13]     # E_TOF [MeV]
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
    
    """
     ___     __       ___  __  
    |__     /  ` |  |  |  /__` 
    |___    \__, \__/  |  .__/
    """
    def BuildTable(a,b,Edep,dEdx): #for birks law
        integrand = a/(1 + b*dEdx)
        nPoints = len(Edep)
        LTable = np.zeros(nPoints)
        for i in range(nPoints):
            LTable[i] = np.trapz(y=integrand[0:i+1], x=Edep[0:i+1])*1000
        return LTable
        
    def Get_E_Edep(Stilbene_dEdx_Path, Stilbene_dEdx_Name):
        E = []
        dEdx = []
        In = open(Stilbene_dEdx_Path+Stilbene_dEdx_Name)
        for lines in In:
            line=lines.split()
            E.append(float(line[0]))
            dEdx.append(float(line[1]))
        return np.array(E), np.array(dEdx)
    
    def Birks(a,b):
        Stilbene_dEdx_Path = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN/"
        Stilbene_dEdx_Name = "OGS_dEdX_1eV_25MeV.txt"
        Read_In_File = Get_E_Edep(Stilbene_dEdx_Path, Stilbene_dEdx_Name)
        Edep = Read_In_File[0]
        dEdx = Read_In_File[1]
        Table = BuildTable(a,b,Edep,dEdx)
        return Edep, Table
    
    def Find_E(Birks, LO): # Takes a light output value and returns energy deposited
        for lightout in np.arange(0,len(Birks[1]),1):
            if LO < Birks[1][lightout]:
                break
        y_2 = Birks[0][lightout]
        y_1 = Birks[0][lightout-1]
        x = LO
        x_1 = Birks[1][lightout-1]
        x_2 = Birks[1][lightout]     
        y = y_1 + (x - x_1)*(y_2-y_1)/(x_2-x_1)   
        return y
    
    def Find_LO(Birks, E): # This takes an energy deposited and returns a light output
        for edep in np.arange(0,len(Birks[0]),1):
            if E < Birks[0][edep]:
                break    
        y_2 = Birks[1][edep]
        y_1 = Birks[1][edep-1]
        x = E
        x_1 = Birks[0][edep-1]
        x_2 = Birks[0][edep]           
        y = y_1 + (x - x_1)*(y_2-y_1)/(x_2-x_1)     
        return y
    
    def Find_dE_dL(Birks, LO):
        for lightout in np.arange(0,len(Birks[1]),1):
            if LO < Birks[1][lightout]:
                break
        y_2 = Birks[0][lightout]
        y_1 = Birks[0][lightout-1]
        x_1 = Birks[1][lightout-1]
        x_2 = Birks[1][lightout]
        dE_dL = (y_2-y_1)/(x_2-x_1)
        return dE_dL
    
    def LO_Uncertainty(Event, E_1, E_2): # Light Output in terms of MeVee, #This is wrong!
        GEB_Factors = np.array([[0,    0.075419184, 0],
                                [0,    0.079770291, 0],
                                [0,    0.083430745, 0],
                                [0,    0.080253747, 0],
                                [0,    0.075281053, 0],
                                [0,    0.075281053, 0],
                                [0,    0.074106945, 0],
                                [0,    0.075419184, 0],
                                [0,    0.079770291, 0],
                                [0,    0.083430745, 0],
                                [0,    0.080253747, 0],
                                [0,    0.075281053, 0],
                                #above OGS bar, below CeBr3
                                [0,    0.075281053, 0],
                                [0,    0.075281053, 0],
                                [0,    0.074106945, 0],
                                [0,    0.075419184, 0],
                                [0,    0.079770291, 0],
                                [0,    0.083430745, 0],
                                [0,    0.080253747, 0],
                                [0,    0.075281053, 0]])
        First = int(Event[0])
        Second = int(Event[1])
        a_1 = GEB_Factors[First][0]
        a_2 = GEB_Factors[Second][0]
        b_1 = GEB_Factors[First][1]
        b_2 = GEB_Factors[Second][1]
        c_1 = GEB_Factors[First][2]
        c_2 = GEB_Factors[Second][2]
        Un_E_1 = (a_1 + b_1*np.sqrt(E_1 + c_1*E_1**2))/2.355 # In terms of MeVee
        Un_E_2 = (a_2 + b_2*np.sqrt(E_2 + c_2*E_2**2))/2.355 # In terms of MeVee
        return Un_E_1, Un_E_2
    
    def Apply_E_Cuts(Doubles_Data): #filtered the double events in our ROI, 做能量切片?
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
            LO_1 = double[9] #modified
            E2_pass = False
            LO_2 = double[10] #modified
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
    
    """
      __                    ___  __      __   __   __  
    /  ` |__|  /\  |\ |     |  /  \    |__) /  \ /__` 
    \__, |  | /~~\ | \|     |  \__/    |    \__/ .__/ 
                                                     
    """
    def Z_Uncertainty(Z, chan, LO):
        Path_to_Z_un = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\Cal_Stil_Data/"
        Min_LO = np.array([25,50,75,100,125,150,175,200,225,250,275,300,325,350,375])
        Chan_to_Bar = [2,3,5,6,7,8,9,10,11,12,14,15,  1,4,13,16,17,18,19,20]             #where these from? it's different from gamma simulation 1.2
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
            Z_un = y_1 + (x - x_1)*(y_2-y_1)/(x_2-x_1)# UNCERTAINTY TO BE CHANGED
        
        #print("Light Output: "+str(LO))
        #print("Z Position: "+str(Z))
        #print("Uncertainty in Z Position: "+str(Z_un))
        return Z_un
    
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
                            [0.3165, 0.9575, 0.0],  # Bar # 11 - 17
                            #above is OGS bar, below is CeBr3 
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
            Z_1_un = Z_Uncertainty(Z_1, Bar_1, (E_Cut_Applied_Data[event][9]*1000))
            Z_2_un = Z_Uncertainty(Z_2, Bar_2, (E_Cut_Applied_Data[event][9]*1000))
            
            ## Z_pos Broadening
            if z_broad:
                Z_1 = np.random.normal(Z_1, Z_1_un) # random value that is generated from normal distribution
                Z_2 = np.random.normal(Z_2, Z_2_un)
                Z_1_un = Z_Uncertainty(Z_1, Bar_1, (E_Cut_Applied_Data[event][9]*1000))
                Z_2_un = Z_Uncertainty(Z_2, Bar_2, (E_Cut_Applied_Data[event][9]*1000))
                
            
            if Z_2 < 2.525 and Z_2 > -2.525 and Z_1 < 2.525 and Z_1 > -2.525: #cebr3 condition should add here, changing z2
                in_bar_Counter+=1
    
            if np.abs(Z_1) - Z_1_un < 2.525 and np.abs(Z_2) - Z_2_un < 2.525: #cebr3 condition should add here, changing z2
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
        return Data_Out[:good_counter] #return the data that happenned within the bars
    
    """
    ___  __   ___ 
      |  /  \ |__  
      |  \__/ |   
    """
    #change the e error to 10% of E instead of 1 kev!
    def E_TOF_LO(Position_Data):
        Data_Out = np.zeros((len(Position_Data),21)) #+7 new entries? why seven more entries
        # Birks_Fit = Birks(a=0.518,b=2.392)
        Birks_Fit = Birks(a=0.5366095577079871,b=2.6780735541073404)
        
        E1list_e = []
        E2list_e = []
        E1list_lo = []
        E2list_lo = []
        Etoflist_mcnp = []
        Etoflist_calc = []
        Un_E1list_e = []
        Rel_Un_E1list_e = []
        Un_E1list_lo = []
        Rel_Un_E1list_lo = []
        Un_E2list_e = []
        Rel_Un_E2list_e = []
        Un_E2list_lo = []
        Rel_Un_E2list_lo = []
        Un_Etoflist_mcnp = []
        Rel_Un_Etoflist_mcnp = []
        Un_Etoflist_calc = []
        Rel_Un_Etoflist_calc = []
    
        Total_Energies_no_TOF_Check = []
        Total_Energies_yes_TOF_Check = []  
        Un_Totlist = []
        Rel_Un_Totlist = []
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
            Mag = np.linalg.norm(Diff_Vecs) #vector length
            velocity = Mag/Position_Data[event][8] #cm/ns  neutron velocity
            Etof_calc = 0.5*939.56542052*(velocity/29.9792458)**2   #neutron energy based on TOF
            ##print("Calculated E_TOF: "+str(Etof_calc))
            E_1_e = Position_Data[event][9]  #Edep1 in terms of MeV
            Un_E_1_e = 0.001
            E_2_e = Position_Data[event][10] #Edep2 in terms of MeV
            Un_E_2_e = 0.001
            E_1_lo = Find_E(Birks_Fit, (Position_Data[event][11]*1000)) #LO Bar 1 [MeVee] converted to MeV,,,,deposit energy
            Un_Light_Outputs = LO_Uncertainty(Position_Data[event], Position_Data[event][11], Position_Data[event][12]) # returns un_LO1 and 2 in MeVee
            Un_LO_1 = Un_Light_Outputs[0]*1000.0
            Un_LO_2 = Un_Light_Outputs[1]*1000.0  
            Un_E_1_lo = Find_dE_dL(Birks_Fit, (Position_Data[event][11]*1000))*Un_LO_1  # Returns derivative MeV/keVee
            E_2_lo = Find_E(Birks_Fit, (Position_Data[event][12]*1000)) 
            Un_E_2_lo = Find_dE_dL(Birks_Fit, (Position_Data[event][12]*1000))*Un_LO_2  # Returns derivative MeV/keVee
            
            Etof_mcnp = Position_Data[event][13]
            ##print("MCNP Direct E_TOF: "+str(Etof))
            TOF_Uncertainty = Uncertainty_TOF(Position_Data[event])
            
            # ## Option A       
            # E_Tot_A = E_1_e + Etof_mcnp
            # E_Tot = E_Tot_A
            # Un_Totlist.append(np.sqrt(TOF_Uncertainty**2+Un_E_1_e**2))
            # Rel_Un_Totlist.append(np.sqrt(TOF_Uncertainty**2+Un_E_1_e**2)/E_Tot*100.0)
            # E_2 = E_2_e
            # Etof = Etof_mcnp
            # E_1 = E_1_e
            # Un_E_1 = Un_E_1_e
            # ## Option B   
            # E_Tot_B = E_1_e + Etof_calc
            # E_Tot = E_Tot_B
            # Un_Totlist.append(np.sqrt(TOF_Uncertainty**2+Un_E_1_e**2))
            # Rel_Un_Totlist.append(np.sqrt(TOF_Uncertainty**2+Un_E_1_e**2)/E_Tot*100.0)
            # E_2 = E_2_e
            # Etof = Etof_calc
            # E_1 = E_1_e
            # Un_E_1 = Un_E_1_e
            # ## Option C 
            # E_Tot_C = E_1_lo + Etof_mcnp
            # E_Tot = E_Tot_C
            # Un_Totlist.append(np.sqrt(TOF_Uncertainty**2+Un_E_1_lo**2))
            # Rel_Un_Totlist.append(np.sqrt(TOF_Uncertainty**2+Un_E_1_lo**2)/E_Tot*100.0)
            # E_2 = E_2_lo
            # Etof = Etof_mcnp
            # E_1 = E_1_lo
            # Un_E_1 = Un_E_1_lo
            ## Option D  
            E_Tot_D = E_1_lo + Etof_calc
            E_Tot = E_Tot_D
            Un_Totlist.append(np.sqrt(TOF_Uncertainty**2+Un_E_1_lo**2))
            Rel_Un_Totlist.append(np.sqrt(TOF_Uncertainty**2+Un_E_1_lo**2)/E_Tot*100.0)
            E_2 = E_2_lo
            Etof = Etof_calc
            E_1 = E_1_lo
            Un_E_1 = Un_E_1_lo
            
            E1list_e.append(E_1_e)
            E2list_e.append(E_2_e)
            E1list_lo.append(E_1_lo)
            E2list_lo.append(E_2_lo)
            Etoflist_mcnp.append(Etof_mcnp)
            Etoflist_calc.append(Etof_calc)
            Total_Energies_no_TOF_Check.append(E_Tot)
            
            Un_E1list_e.append(Un_E_1_e)
            Rel_Un_E1list_e.append(Un_E_1_e/E_1_e*100.0) #relative uncertainty
            Un_E1list_lo.append(Un_E_1_lo)
            Rel_Un_E1list_lo.append(Un_E_1_lo/E_1_lo*100.0)
            Un_E2list_e.append(Un_E_2_e)
            Rel_Un_E2list_e.append(Un_E_2_e/E_2_e*100.0)
            Un_E2list_lo.append(Un_E_2_lo)
            Rel_Un_E2list_lo.append(Un_E_2_lo/E_2_lo*100.0)
            
            Un_Etoflist_mcnp.append(TOF_Uncertainty)
            Rel_Un_Etoflist_mcnp.append(TOF_Uncertainty/Etof_mcnp*100.0)
            Un_Etoflist_calc.append(TOF_Uncertainty)
            Rel_Un_Etoflist_calc.append(TOF_Uncertainty/Etof_calc*100.0)
    
            if E_2 < Etof:
                #Maybe append everything here ater the cut?
                Total_Energies_yes_TOF_Check.append(E_Tot)
                Data_Out[good_counter][0] = Position_Data[event][0] # Bar 1 
                Data_Out[good_counter][1] = Position_Data[event][1] # Bar 2 
                Data_Out[good_counter][2] = Position_Data[event][2] # X1
                Data_Out[good_counter][3] = Position_Data[event][3] # Y1
                Data_Out[good_counter][4] = Position_Data[event][4] # Z1
                Data_Out[good_counter][5] = Position_Data[event][5] # X2
                Data_Out[good_counter][6] = Position_Data[event][6] # Y2
                Data_Out[good_counter][7] = Position_Data[event][7] # Z2
                Data_Out[good_counter][8] = Position_Data[event][8] # TOF [ns]
                Data_Out[good_counter][9] = Etof
                Data_Out[good_counter][10] = E_1
                Data_Out[good_counter][11] = E_2
                Data_Out[good_counter][12] = E_Tot
                Data_Out[good_counter][13] = Position_Data[event][11]*1000 #E_1 actually in LO [keVee]
                Data_Out[good_counter][14] = Position_Data[event][12]*1000 #E_2 actually in LO [keVee]
                Data_Out[good_counter][15] = 0
                Data_Out[good_counter][16] = 0
                Data_Out[good_counter][17] = Un_E_1
                Data_Out[good_counter][18] = 0 #Un_E_2
                Data_Out[good_counter][19] = TOF_Uncertainty
                Data_Out[good_counter][20] = np.sqrt(TOF_Uncertainty**2+Un_E_1**2)/E_Tot*100.0
                good_counter+=1
    
        print("\n")
        print("Number of events before E_TOF filtering: "+str(len(Position_Data)))
        print("Number of events after E_TOF filtering: "+str(good_counter))
        print("\n")
    
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
    
        # plot that pairs distance and energy?
    
        ## Uncertainty Plots  
        print("[Direct]     Average Neutron Energy Deposition in First Scatter: "+str(np.average(E1list_e))+" MeV")
        print("[Calculated] Average Neutron Energy Deposition in First Scatter: "+str(np.average(E1list_lo))+" MeV")
        print("[Direct]     Average Neutron Energy Deposition in Second Scatter: "+str(np.average(E2list_e))+" MeV")
        print("[Calculated] Average Neutron Energy Deposition in Second Scatter: "+str(np.average(E2list_lo))+" MeV")
        print("\n")
        print("[Direct]     Average Neutron Energy Deposition (First Scatter) Uncertainty: "+str(np.average(Un_E1list_e))+" MeV")
        print("[Calculated] Average Neutron Energy Deposition (First Scatter) Uncertainty: "+str(np.average(Un_E1list_lo))+" MeV")
        print("[Direct]     Average Neutron Energy Deposition (Second Scatter) Uncertainty: "+str(np.average(Un_E2list_e))+" MeV")
        print("[Calculated] Average Neutron Energy Deposition (Second Scatter) Uncertainty: "+str(np.average(Un_E2list_lo))+" MeV")
        print("\n")
        print("[Direct]     Average Neutron Energy Deposition (First Scatter) Relative Uncertainty: "+str(np.average(Rel_Un_E1list_e))+" %")
        print("[Calculated] Average Neutron Energy Deposition (First Scatter) Relative Uncertainty: "+str(np.average(Rel_Un_E1list_lo))+" %")
        print("[Direct]     Average Neutron Energy Deposition (Second Scatter) Relative Uncertainty: "+str(np.average(Rel_Un_E2list_e))+" %")
        print("[Calculated] Average Neutron Energy Deposition (Second Scatter) Relative Uncertainty: "+str(np.average(Rel_Un_E2list_lo))+" %")
        print("\n")
        print("[Direct]     Average Neutron TOF Uncertainty: "+str(np.average(Un_Etoflist_mcnp))+" MeV")
        print("[Calculated] Average Neutron TOF Uncertainty: "+str(np.average(Un_Etoflist_calc))+" MeV")
        print("\n")
        print("[Direct]     Average Neutron TOF Relative Uncertainty: "+str(np.average(Rel_Un_Etoflist_mcnp))+" %")
        print("[Calculated] Average Neutron TOF Relative Uncertainty: "+str(np.average(Rel_Un_Etoflist_calc))+" %")
        print("\n")
        print("Average Reconstructed Neutron Energy (before TOF check): "+str(np.average(Total_Energies_no_TOF_Check))+" MeV")
        print("Average Reconstructed Neutron Energy (after TOF check): "+str(np.average(Total_Energies_yes_TOF_Check))+" MeV")
        print("\n")
        print("Average Reconstructed Neutron Energy Uncertainty: "+str(np.average(Un_Totlist))+" MeV")
        print("Average Reconstructed Neutron Energy Relative Uncertainty: "+str(np.average(Rel_Un_Totlist))+" %")
        print("\n")
    
        plt.hist(E1list_e, bins=100, range=[0,10], histtype='step')
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("E_1 Plot - MCNP")
        plt.show()
        plt.close()
        
        plt.hist(E1list_lo, bins=100, range=[0,10], histtype='step')
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("E_1 Plot - MPPost/Birks")
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
        
        plt.hist(E2list_lo, bins=100, range=[0,10], histtype='step')
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("E_2 Plot - MPPost/Birks")
        plt.show()
        plt.close()
        
        plt.hist(Etoflist_mcnp, bins=100, range=[0,10], histtype='step')
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("E_TOF - MCNP")
        plt.show()
        plt.close()
        
        plt.hist(Etoflist_calc, bins=100, range=[0,10], histtype='step')
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("E_TOF - Calculated")
        plt.show()
        plt.close()
    
        plt.hist(Total_Energies_yes_TOF_Check, bins=100, range=[0,10], histtype='step')
        plt.xlabel("Neutron Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Reconstructed Neutron Energy (Before Imaging)")
        plt.show()
        plt.close()
    
        # plt.hist(Data_A, bins=100, range=[0,10], label='Option A', histtype='step')
        # plt.hist(Data_B, bins=100, range=[0,10], label='Option B', histtype='step')
        # plt.hist(Data_C, bins=100, range=[0,10], label='Option C', histtype='step')
        # plt.hist(Data_D, bins=100, range=[0,10], label='Option D', histtype='step')
        # plt.legend()
        # plt.xlabel("Neutron Energy (MeV)")
        # plt.ylabel("Counts")
        # plt.yscale('log')
        # plt.tight_layout()
        # plt.title("Reconstructed Neutron Energy (Before Imaging)")
        # plt.show()  
        
        plt.hist(Un_E1list_e, bins=100, range=[0,1], label='Uncertainty in E$_{1}$', histtype='step')
        plt.legend()
        plt.xlabel("Uncertainty in Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - MCNP")
        plt.show()
        plt.close()
        
        plt.hist(Un_E1list_lo, bins=100, range=[0,1], label='Uncertainty in E$_{1}$', histtype='step')
        plt.legend()
        plt.xlabel("Uncertainty in Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - Birks")
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
        
        plt.hist(Rel_Un_E1list_lo, bins=200, range=[0,200], label='Uncertainty in E$_{1}$', histtype='step')
        plt.legend()
        plt.xlabel("Relative Uncertainty (%)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - Birks")
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
        
        plt.hist(Un_E2list_lo, bins=100, range=[0,1], label='Uncertainty in E$_{2}$', histtype='step')
        plt.legend()
        plt.xlabel("Uncertainty in Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - Birks")
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
        
        plt.hist(Rel_Un_E2list_lo, bins=200, range=[0,200], label='Uncertainty in E$_{2}$', histtype='step')
        plt.legend()
        plt.xlabel("Relative Uncertainty (%)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - Birks")
        plt.show()
        plt.close()
        
        plt.hist(Un_Etoflist_mcnp, bins=1000, range=[0,10], label='Uncertainty in MCNP E$_{tof}$', histtype='step')
        plt.legend()
        plt.xlabel("Uncertainty in Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - MCNP")
        plt.show()
        plt.close()
        
        plt.hist(Rel_Un_Etoflist_mcnp, bins=200, range=[0,200], label='Uncertainty in MCNP E$_{tof}$', histtype='step')
        plt.legend()
        plt.xlabel("Relative Uncertainty (%)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - Birks")
        plt.show()
        plt.close()
        
        plt.hist(Un_Etoflist_calc, bins=1000, range=[0,10], label='Uncertainty in calc E$_{tof}$', histtype='step')
        plt.legend()
        plt.xlabel("Uncertainty in Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - MCNP")
        plt.show()
        plt.close()
        
        plt.hist(Rel_Un_Etoflist_calc, bins=200, range=[0,200], label='Uncertainty in calc E$_{tof}$', histtype='step')
        plt.legend()
        plt.xlabel("Relative Uncertainty (%)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot - Birks")
        plt.show()
        plt.close()
        
        plt.hist(Un_Totlist, bins=1000, range=[0,10], histtype='step')
        plt.xlabel("Uncertainty in Energy (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title('Uncertainty in E$_{total}$')
        plt.show()
        plt.close()
        
        plt.hist(Rel_Un_Totlist, bins=200, range=[0,200], histtype='step')
        plt.xlabel("Relative Uncertainty (%)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title('Uncertainty in E$_{total}$')
        plt.show()
        plt.close()
        
        plt.hist(Un_E1list_lo, bins=1000, range=[0,10], label='Uncertainty in E$_{1}$', histtype='step')
        plt.hist(Un_Etoflist_calc, bins=1000, range=[0,10], label='Uncertainty in E$_{TOF}$', histtype='step')
        plt.hist(Un_Totlist, bins=1000, range=[0,10], label='Uncertainty in E$_{total}$', histtype='step')
        plt.legend()
        plt.xlim(0,2)
        plt.xlabel("Uncertainty (MeV)")
        plt.ylabel("Counts")
        plt.tight_layout()
        plt.title("Uncertainty Plot")
        plt.show()
        plt.close()
        
        plt.hist(Un_E1list_lo, bins=1000, range=[0,10], label='Uncertainty in E$_{1}$', histtype='step')
        plt.hist(Un_Etoflist_calc, bins=1000, range=[0,10], label='Uncertainty in E$_{TOF}$', histtype='step')
        plt.hist(Un_Totlist, bins=1000, range=[0,10], label='Uncertainty in E$_{total}$', histtype='step')
        plt.legend()
        plt.xlim(0,2)
        plt.xlabel("Uncertainty (MeV)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Uncertainty Plot")
        plt.show()
        plt.close()
        
        plt.hist(Rel_Un_E1list_lo, bins=200, range=[0,200], label='Uncertainty in E$_{1}$', histtype='step')
        plt.hist(Rel_Un_Etoflist_calc, bins=200, range=[0,200], label='Uncertainty in E$_{TOF}$', histtype='step')
        plt.hist(Rel_Un_Totlist, bins=200, range=[0,200], label='Uncertainty in E$_{total}$', histtype='step')
        plt.xlim(0,200)
        plt.legend()
        plt.xlabel("Relative Uncertainty (%)")
        plt.ylabel("Counts")
        plt.tight_layout()
        plt.title("Relative Uncertainty Plot")
        plt.show()
        plt.close()
        
        plt.hist(Rel_Un_E1list_lo, bins=200, range=[0,200], label='Uncertainty in E$_{1}$', histtype='step')
        plt.hist(Rel_Un_Etoflist_calc, bins=200, range=[0,200], label='Uncertainty in E$_{TOF}$', histtype='step')
        plt.hist(Rel_Un_Totlist, bins=200, range=[0,200], label='Uncertainty in E$_{total}$', histtype='step')
        plt.xlim(0,200)
        plt.legend()
        plt.xlabel("Relative Uncertainty (%)")
        plt.ylabel("Counts")
        plt.yscale('log')
        plt.tight_layout()
        plt.title("Relative Uncertainty Plot")
        plt.show()
        plt.close()
    
        return Data_Out[:good_counter], Total_Energies_yes_TOF_Check
    
    """
      __        __           __   __   __        ___  __  ___    __       
    |__)  /\  /  ` |__/    |__) |__) /  \    | |__  /  `  |  | /  \ |\ | 
    |__) /~~\ \__, |  \    |    |  \ \__/ \__/ |___ \__,  |  | \__/ | \| 
                                                                         
    """
    def Running_Back_Projection(Energy_Data):
        Binning = Data_Structure
        Data_Range = np.arange(0,int(Binning*Binning),1)
        F_Types = []
    
        for dat in Data_Range:     #create headers?
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
    
        for Event in np.arange(0,len(Energy_Data),1):
            if counter < Number_Cones_to_Project:
                if Energy_Data[Event][20] < Relative_Uncertainty:
                    Energies_Not_Extracted.append(Energy_Data[Event][12]) #E_total
                    Z_Data = Creating_Real_Projections(Energy_Data[Event], Radius, Theta_Rad, Phi_Rad)  #Z, Cone_Vector, (Alpha_Var/Alpha)
                    Normalized_Cone_Vector = Z_Data[1]                         
                    Lever_Arms[counter] = Normalized_Cone_Vector
                    Alpha_Uncer.append(Z_Data[2])
    ############################??????????????????????????????????
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
                    #if counter%1000 == 0:
                    if counter%100000 == 0:
                        print("Cone "+str(counter)+' / '+str(len(Energy_Data))+" being projected.")
    
        Z = np.roll(Z, int(Binning/4.0), axis=1)   #移動元素 沿著列方向右移 (binninb/4.0)
        Z = Z[::-1] 
    
        for row in np.arange(0,len(Z), 1):
            Z[row] = Z[row][::-1]
    
        return Z, Energies_Not_Extracted
    
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
        dot_product[dot_product<0.0] = 0.0
        beta = dot_product*dot_product/(x*x + y*y + z*z)
        Variances = Uncertainty(x,y,z,Event)
        Alpha_Var = Variances[0]
        Beta_Var = Variances[1]/(x*x + y*y + z*z)
        Alpha = float(Event[9]/Event[12]) #E_tof/E_tot
        Z = np.exp(-1.0*((beta - Alpha)*(beta - Alpha))/(2*(Alpha_Var**2+Beta_Var**2))) #C(d,b) in paper
        return Z, Cone_Vector, (Alpha_Var/Alpha)
    
    """
                __   ___  __  ___             ___     
    |  | |\ | /  ` |__  |__)  |   /\  | |\ |  |  \ / 
    \__/ | \| \__, |___ |  \  |  /~~\ | | \|  |   |  
                                                    
    """
    def Uncertainty(xdel,ydel,zdel,Event): #See the Uncertainty_Analysis.py script for where the 2 uncertainty functions come from
        m_n = 939.56542052 #MeV*c**2
        speed_light = 29.9792458 #cm/ns
        Bar_1 = int(Event[0])
        Bar_2 = int(Event[1])
        x_1 = Event[2]
        y_1 = Event[3]
        z_1 = Event[4]
        un_X_1 = 0.6/np.sqrt(12)
        un_Y_1 = 0.6/np.sqrt(12)
        un_Z_1 = Z_Uncertainty(z_1, Bar_1, (Event[11]))
        x_2 = Event[5]
        y_2 = Event[6]
        z_2 = Event[7]
        un_X_2 = 0.6/np.sqrt(12)
        un_Y_2 = 0.6/np.sqrt(12)
        un_Z_2 = Z_Uncertainty(z_2, Bar_2, (Event[12]))
        t_diff = Event[8]
        uncer_t_diff = 0.217 #ns
        E_dep = Event[10] #E1
        uncer_E_dep = Event[17] #Un_E_1
        Alpha_Uncer = (E_dep**2*(0.25*m_n**2*un_X_1**2*(2*x_1 - 2*x_2)**2/(speed_light**4*t_diff**4) +
                      0.25*m_n**2*un_X_2**2*(-2*x_1 + 2*x_2)**2/(speed_light**4*t_diff**4) +
                      0.25*m_n**2*un_Y_1**2*(2*y_1 - 2*y_2)**2/(speed_light**4*t_diff**4) +
                      0.25*m_n**2*un_Y_2**2*(-2*y_1 + 2*y_2)**2/(speed_light**4*t_diff**4) +
                      0.25*m_n**2*un_Z_1**2*(2*z_1 - 2*z_2)**2/(speed_light**4*t_diff**4) +
                      0.25*m_n**2*un_Z_2**2*(-2*z_1 + 2*z_2)**2/(speed_light**4*t_diff**4) +
                      1.0*m_n**2*uncer_t_diff**2*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 +
                      (-z_1 + z_2)**2)**2/(speed_light**4*t_diff**6))**1.0/(E_dep + 0.5*m_n*((-x_1 + x_2)**2 +
                      (-y_1 + y_2)**2 + (-z_1 + z_2)**2)/(speed_light**2*t_diff**2))**4 +
                      0.25*m_n**2*uncer_E_dep**2*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 +
                      (-z_1 + z_2)**2)**2/(speed_light**4*t_diff**4*(E_dep + 0.5*m_n*((-x_1 + x_2)**2 +
                      (-y_1 + y_2)**2 + (-z_1 + z_2)**2)/(speed_light**2*t_diff**2))**4))**0.5
        Beta_Uncertainty = (un_X_1**2*((-2*x_1 + 2*xdel)*((-x_1 + xdel)*(x_1 - x_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5) + (-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5) + (-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5))**2/((-x_1 + xdel)**2 + (-y_1 + ydel)**2 + (-z_1 + zdel)**2)**2 +
                            ((-x_1 + xdel)*(x_1 - x_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) +
                            (-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) +
                            (-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5))*(2*(-1.0*x_1 + 1.0*x_2)*(-x_1 + xdel)*(x_1 - x_2)*((x_1 - x_2)**2 +
                            (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-1.5) + 2*(-1.0*x_1 + 1.0*x_2)*(-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 +
                            (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-1.5) + 2*(-1.0*x_1 + 1.0*x_2)*(-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 +
                            (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-1.5) + 2*(-x_1 + xdel)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) -
                            2*(x_1 - x_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5))/((-x_1 + xdel)**2 + (-y_1 + ydel)**2 +
                            (-z_1 + zdel)**2))**2 + un_X_2**2*((-x_1 + xdel)*(x_1 - x_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) +
                            (-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) +
                            (-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5))**2*(2*(-x_1 + xdel)*(x_1 - x_2)*(1.0*x_1 - 1.0*x_2)*((x_1 - x_2)**2 +
                            (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-1.5) - 2*(-x_1 + xdel)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5) + 2*(1.0*x_1 - 1.0*x_2)*(-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 +
                            (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-1.5) + 2*(1.0*x_1 - 1.0*x_2)*(-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 +
                            (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-1.5))**2/((-x_1 + xdel)**2 + (-y_1 + ydel)**2 + (-z_1 + zdel)**2)**2 +
                            un_Y_1**2*((-2*y_1 + 2*ydel)*((-x_1 + xdel)*(x_1 - x_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) +
                            (-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) + (-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 +
                            (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5))**2/((-x_1 + xdel)**2 + (-y_1 + ydel)**2 + (-z_1 + zdel)**2)**2 +
                            ((-x_1 + xdel)*(x_1 - x_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) + (-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 +
                            (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) + (-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5))*(2*(-x_1 + xdel)*(x_1 - x_2)*(-1.0*y_1 + 1.0*y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-1.5) + 2*(-1.0*y_1 + 1.0*y_2)*(-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-1.5) + 2*(-1.0*y_1 + 1.0*y_2)*(-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-1.5) + 2*(-y_1 + ydel)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) -
                            2*(y_1 - y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5))/((-x_1 + xdel)**2 + (-y_1 + ydel)**2 +
                            (-z_1 + zdel)**2))**2 + un_Y_2**2*((-x_1 + xdel)*(x_1 - x_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) +
                            (-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) + (-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 +
                            (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5))**2*(2*(-x_1 + xdel)*(x_1 - x_2)*(1.0*y_1 - 1.0*y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-1.5) + 2*(-y_1 + ydel)*(y_1 - y_2)*(1.0*y_1 - 1.0*y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-1.5) -
                            2*(-y_1 + ydel)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) +
                            2*(1.0*y_1 - 1.0*y_2)*(-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-1.5))**2/((-x_1 + xdel)**2 + (-y_1 + ydel)**2 + (-z_1 + zdel)**2)**2 +
                            un_Z_1**2*((-2*z_1 + 2*zdel)*((-x_1 + xdel)*(x_1 - x_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5) + (-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5) + (-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5))**2/((-x_1 + xdel)**2 + (-y_1 + ydel)**2 + (-z_1 + zdel)**2)**2 +
                            ((-x_1 + xdel)*(x_1 - x_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) +
                            (-y_1 + ydel)*(y_1 - y_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) +
                            (-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5))*(2*(-x_1 + xdel)*(x_1 -
                            x_2)*(-1.0*z_1 + 1.0*z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-1.5) + 2*(-y_1 + ydel)*(y_1 -
                            y_2)*(-1.0*z_1 + 1.0*z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-1.5) + 2*(-1.0*z_1 + 1.0*z_2)*(-z_1 +
                            zdel)*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-1.5) + 2*(-z_1 + zdel)*((x_1 - x_2)**2 +
                            (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) - 2*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5))/((-x_1 + xdel)**2 + (-y_1 + ydel)**2 + (-z_1 + zdel)**2))**2 + un_Z_2**2*((-x_1 +
                            xdel)*(x_1 - x_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) + (-y_1 + ydel)*(y_1 - y_2)*((x_1 -
                            x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5) + (-z_1 + zdel)*(z_1 - z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-0.5))**2*(2*(-x_1 + xdel)*(x_1 - x_2)*(1.0*z_1 - 1.0*z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-1.5) + 2*(-y_1 + ydel)*(y_1 - y_2)*(1.0*z_1 - 1.0*z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-1.5) + 2*(-z_1 + zdel)*(z_1 - z_2)*(1.0*z_1 - 1.0*z_2)*((x_1 - x_2)**2 + (y_1 - y_2)**2 +
                            (z_1 - z_2)**2)**(-1.5) - 2*(-z_1 + zdel)*((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)**(-0.5))**2/((-x_1 + xdel)**2 +
                            (-y_1 + ydel)**2 + (-z_1 + zdel)**2)**2)**0.5
        return [Alpha_Uncer, Beta_Uncertainty]
    
    def Uncertainty_TOF(Event):
        m_n = 939.56542052 #MeV*c**2
        speed_light = 29.9792458 #cm/ns
        Bar_1 = int(Event[0])
        Bar_2 = int(Event[1])
        x_1 = Event[2]
        y_1 = Event[3]
        z_1 = Event[4]
        un_X_1 = 0.6/np.sqrt(12)
        un_Y_1 = 0.6/np.sqrt(12)
        un_Z_1 = Z_Uncertainty(z_1, Bar_1, (Event[11]*1000))
        x_2 = Event[5]
        y_2 = Event[6]
        z_2 = Event[7]
        un_X_2 = 0.6/np.sqrt(12)
        un_Y_2 = 0.6/np.sqrt(12)
        un_Z_2 = Z_Uncertainty(z_2, Bar_2, (Event[12]*1000))
        t_diff = Event[8]
        uncer_t_diff = 0.217 #ns
        Uncer_TOF = (0.25*m_n**2*un_X_1**2*(2*x_1 - 2*x_2)**2/(speed_light**4*t_diff**4) + 0.25*m_n**2*un_X_2**2*(-2*x_1 +
                    2*x_2)**2/(speed_light**4*t_diff**4) + 0.25*m_n**2*un_Y_1**2*(2*y_1 - 2*y_2)**2/(speed_light**4*t_diff**4)
                    + 0.25*m_n**2*un_Y_2**2*(-2*y_1 + 2*y_2)**2/(speed_light**4*t_diff**4) + 0.25*m_n**2*un_Z_1**2*(2*z_1 -
                    2*z_2)**2/(speed_light**4*t_diff**4) + 0.25*m_n**2*un_Z_2**2*(-2*z_1 + 2*z_2)**2/(speed_light**4*t_diff**4) +
                    1.0*m_n**2*uncer_t_diff**2*((-x_1 + x_2)**2 + (-y_1 + y_2)**2 + (-z_1 + z_2)**2)**2/(speed_light**4*t_diff**6))**0.5
        return Uncer_TOF
    
    #%%
    Data = DataLoader_Doubles_Neutron_Simulation((Doubles_File_Name))
    #print(Data.shape)
    #print(type(Data))
    Number_of_data_structures = Data.GetNumberOfWavesInFile()#same size of gammas event
    Waves = Data.LoadWaves(Number_of_data_structures) #all datas (events, 13)
    Doubles_Data = Bar_Counts(Waves)                                           ## return output data from write gamma simulations
    E_Cut_Applied = Apply_E_Cuts(Doubles_Data)                                 ## #filtered the double events in our ROI
    Position_Data = Convert_Channel_to_Position(E_Cut_Applied)
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

# %%

file_path = r"C:\Users\artao\Desktop\Master\NERS599 independent reseach 25WN\project\Simulation_gamma_Doubles_File_OGS_PyMPPost_Processed.dat"
z_data, E_out = run_BP(file_path, Radius=10) #radius self=define?


# Scratch with scripts needed to get plots from simulation backprojector
Binning = 180
Azimuth = np.linspace(-180,180,Binning)
Altitude = np.linspace(-90,90,Binning)
Theta,Phi = np.meshgrid(Azimuth,Altitude)
plt.pcolormesh(Theta,Phi,z_data, cmap='inferno')
plt.xlabel("Azimuthal Angle (θ)")
plt.ylabel("Altitude Angle (φ)")
plt.colorbar()



















