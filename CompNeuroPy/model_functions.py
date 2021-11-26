from ANNarchy import compile, get_population, Monitor, dt
import os
import numpy as np
from CompNeuroPy.system_functions import create_dir


def compile_in_folder(folder_name):
    """
        creates the compilation folder in annarchy_folders/
        or uses existing one
        compiles the current network
    """
    create_dir('annarchy_folders/'+folder_name, print_info=1)
    compile('annarchy_folders/'+folder_name)
    if os.getcwd().split('/')[-1]=='annarchy_folders': os.chdir('../')
    
    
def addMonitors(monDict):
    """
        generate monitors defined by monDict
        
        monDict form:
            {'pop;popName':list with variables to record,
             ...}
        currently only pop as compartments
    """
    mon={}
    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        ### check if compartment is pop
        if compartmentType=='pop':
            mon[compartment] = Monitor(get_population(compartment),val, start=False)
    return mon
    
    
def startMonitors(monDict,mon):
    """
        start monitores defined by monDict
    """
    started={}
    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        if compartmentType=='pop':
            started[compartment]=False

    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        if compartmentType=='pop' and started[compartment]==False:
            mon[compartment].start()
            started[compartment]=True
            
            
def getMonitors(monDict,mon):
    """
        get recorded values from monitors
        
        monitors and recorded values defined by monDict
    """
    recordings = {}
    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        for val_val in val:
            temp = mon[compartment].get(val_val)
            ### check if it's data of only one neuron --> remove unnecessary dimension
            if isinstance(temp, np.ndarray): # only if temp is an numpy array
                if len(temp.shape) == 2:
                    if temp.shape[1]==1:
                        temp = temp[:,0]
            recordings[compartment+';'+val_val] = temp
    recordings['dt'] = dt()
    return recordings
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
