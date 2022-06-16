from ANNarchy import compile, get_population, Monitor, dt, get_time, populations, projections
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
    
    
def startMonitors(compartment_list,mon,timings=None):
    """
        starts or resumes monitores defined by monDict
        compartment_list: list with model compartments
        mon: dict with the corresponding monitors
        currently_paused: dict with key=compartment+variable name and val=if currently paused
    """
    ### for each compartment generate started variable (because compartments can ocure multiple times if multiple variables of them are recorded --> do not start same monitor multiple times)
    started={}
    for key in compartment_list:
        compartmentType, compartment = key.split(';')
        if compartmentType=='pop':
            started[compartment]=False

    if timings==None:
        ### information about pauses not available, just start
        for key in compartment_list:
            compartmentType, compartment = key.split(';')
            if compartmentType=='pop' and started[compartment]==False:
                mon[compartment].start()
                print('start', compartment)
                started[compartment]=True
        return None
    else:
        ### information about pauses available, start if not paused, resume if paused
        for key in compartment_list:
            compartmentType, compartment = key.split(';')
            if compartmentType=='pop' and started[compartment]==False:
                if timings[compartment]['currently_paused']:
                    ### monitor is currently paused --> resume
                    mon[compartment].resume()
                else:
                    mon[compartment].start()
                started[compartment]=True
                ### update currently_paused
                timings[compartment]['currently_paused']=False
                ### never make start longer than stop+1!... this can be caused if start is called multiple times without pause in between
                if len(timings[compartment]['start']) <= len(timings[compartment]['stop']):
                    timings[compartment]['start'].append(get_time())
        return timings
            
            
def pauseMonitors(compartment_list,mon,timings=None):
    """
        pause monitores defined by compartment_list
    """
    ### for each compartment generate paused variable (because compartments can ocure multiple times if multiple variables of them are recorded --> do not pause same monitor multiple times)
    paused={}
    for key in compartment_list:
        compartmentType, compartment = key.split(';')
        if compartmentType=='pop':
            paused[compartment]=False

    for key in compartment_list:
        compartmentType, compartment = key.split(';')
        if compartmentType=='pop' and paused[compartment]==False:
            mon[compartment].pause()
            paused[compartment]=True
            
    if timings!=None:
        ### information about pauses is available, update it
        for key,val in paused.items():
            timings[key]['currently_paused'] = True
            ### never make pause longer than start, this can be caused if pause is called multiple times without start in between
            if len(timings[key]['stop']) < len(timings[key]['start']):
                timings[key]['stop'].append(get_time())
            ### if pause is directly called after start --> start == stop --> remove these entries, this is no actual period
            if len(timings[key]['stop']) == len(timings[key]['start']) and timings[key]['stop'][-1]==timings[key]['start'][-1]:
                timings[key]['stop'] = timings[key]['stop'][:-1]
                timings[key]['start'] = timings[key]['start'][:-1]
        return timings
    else:
        return None
        
          
            
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
            recordings[compartment+';'+val_val] = temp
    recordings['dt'] = dt()
    return recordings
    
    
def get_monitor_times(monDict,mon):# TODO: currently not used, object Monitors uses self-defined timings
    """
        get recording times of monitors in ms
        
        monitors and recorded values defined by monDict
    """
    times = {}
    for key, val in monDict.items():
        compartmentType, compartment = key.split(';')
        print(mon[compartment].times())
        for val_val in val:
            times['start'] = np.array(mon[compartment].times()[val_val]['start'])*dt() # ANNarchy returns times for each recorded variable of Monitor, in CompNeuroPy they are usually startet and ended all at the same time... only return single start/end times
            times['stop']  = np.array(mon[compartment].times()[val_val]['stop'])*dt()
    return times
    
    
def get_full_model():
    """
        return all current population and projection names
    """
    return {'populations':[ pop.name for pop in populations() ], 'projections':[ proj.name for proj in projections() ]}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
