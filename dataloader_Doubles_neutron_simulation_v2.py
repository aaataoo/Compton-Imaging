"""
Defines the class for loading in data from an MCNP simulation file

@author: pakari, bllrdk
"""

"""
         __   __   __  ___  __  
|  |\/| |__) /  \ |__)  |  /__` 
|  |  | |    \__/ |  \  |  .__/ 

"""
import numpy as np
import os

"""
 __       ___               __        __   ___  __  
|  \  /\   |   /\     |    /  \  /\  |  \ |__  |__) 
|__/ /~~\  |  /~~\    |___ \__/ /~~\ |__/ |___ |  \
"""
class DataLoader_Doubles_Neutron_Simulation:
    
    def __init__(self, fileName):
        self.fileName = fileName
        self.blockType = np.dtype([('BAR_1',(np.int16,1)),
                                   ('BAR_2',(np.int16,1)),
								   ('X_1',(np.float64,1)),
								   ('Y_1',(np.float64,1)),
								   ('Z_1',(np.float64,1)),
								   ('X_2',(np.float64,1)),
								   ('Y_2',(np.float64,1)),
								   ('Z_2',(np.float64,1)),
								   ('tof',(np.float64,1)),
								   ('Edep1_d',(np.float64,1)),
                                   ('Edep2_d',(np.float64,1)),
                                   ('Edep1_o',(np.float64,1)),
                                   ('Edep2_o',(np.float64,1)),
								   ('Etof',(np.float64,1))])
        self.location = 0

## functions
    # gets the number of waves in a given file
    def GetNumberOfWavesInFile(self):
        return int(os.path.getsize(self.fileName) / self.blockType.itemsize)

    #Loads numWaves waveforms. If numWaves == -1, loads all waveforms in the file
    def LoadWaves(self, numWaves):
        fid = open(self.fileName, "rb")
        fid.seek(self.location, os.SEEK_SET)
        self.location += self.blockType.itemsize * numWaves
        return np.fromfile(fid, dtype = self.blockType, count=numWaves)

    # resets the location of the given parameter
    def Rewind(self):
        self.location = 0
        
class DataLoader_Doubles_Gamma_Simulation:
    
    def __init__(self, fileName):
        self.fileName = fileName
        self.blockType = np.dtype([('BAR_1',(np.int16,1)),
                                   ('BAR_2',(np.int16,1)),
								   ('X_1',(np.float64,1)),
								   ('Y_1',(np.float64,1)),
								   ('Z_1',(np.float64,1)),
								   ('X_2',(np.float64,1)),
								   ('Y_2',(np.float64,1)),
								   ('Z_2',(np.float64,1)),
								   ('nTOF',(np.float64,1)),
								   ('Edep1_d',(np.float64,1)),
								   ('Edep2_d',(np.float64,1)),
                                   ('Etotal',(np.float64,1))
                                   ])
        self.location = 0

## functions
    # gets the number of waves in a given file
    def GetNumberOfWavesInFile(self):
        return int(os.path.getsize(self.fileName) / self.blockType.itemsize)

    #Loads numWaves waveforms. If numWaves == -1, loads all waveforms in the file
    def LoadWaves(self, numWaves):
        fid = open(self.fileName, "rb")
        fid.seek(self.location, os.SEEK_SET)
        self.location += self.blockType.itemsize * numWaves
        return np.fromfile(fid, dtype = self.blockType, count=numWaves)

    # resets the location of the given parameter
    def Rewind(self):
        self.location = 0