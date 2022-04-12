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
        starts or resumes monitores defined by monDict
    """
    ### for each compartment generate started variable (because compartments can ocure multiple times if multiple variables of them are recorded --> do not start same monitor multiple times)
    started={}
    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        if compartmentType=='pop':
            started[compartment]=False

    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        if compartmentType=='pop' and started[compartment]==False:
            if len(vars(mon[compartment])['_recorded_variables'][val[0]]['stop'])>len(vars(mon[compartment])['_recorded_variables'][val[0]]['start']):
                ### monitor is currently paused --> resume TODO: doesnt wokr with new times function
                mon[compartment].resume()
                print('resume', compartment)
            else:
                mon[compartment].start()
                print('start', compartment)
            started[compartment]=True
            
            
def pauseMonitors(monDict,mon):
    """
        pause monitores defined by monDict
    """
    ### for each compartment generate paused variable (because compartments can ocure multiple times if multiple variables of them are recorded --> do not pause same monitor multiple times)
    paused={}
    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        if compartmentType=='pop':
            paused[compartment]=False

    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        if compartmentType=='pop' and paused[compartment]==False:
            mon[compartment].pause()
            paused[compartment]=True
            
          
            
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
    
    
def get_monitor_times(monDict,mon):
    """
        get recording times of monitors in ms
        
        monitors and recorded values defined by monDict
    """
    times = {}
    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        for val_val in val:
            times['start'] = np.array(mon[compartment].times()[val_val]['start'])*dt() # ANNarchy returns times for each recorded variable of Monitor, in CompNeuroPy they are usually startet and ended all at the same time... only return single start/end times
            times['stop']   = np.array(mon[compartment].times()[val_val]['stop'])*dt()
    return times
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
