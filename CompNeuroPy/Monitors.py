import CompNeuroPy.neuron_models as nm
import CompNeuroPy.model_functions as mf
import CompNeuroPy.simulation_functions as sif
import CompNeuroPy.system_functions as syf
from ANNarchy import get_time, dt, reset
import numpy as np

class Monitors:

    
    def __init__(self, monDict={}):
    
        self.mon     = mf.addMonitors(monDict)
        self.monDict = monDict

        timings = {}
        for key, val in monDict.items():
            compartmentType, compartment = key.split(';')
            timings[compartment] = {'currently_paused':False, 'start':[], 'stop':[]}
        self.timings = timings
        
        self.recordings=[]
        self.recording_times=[]
        
    def start(self, monDict=[]):
        if isinstance(monDict, list):
            monDict = self.monDict
        
        self.timings = mf.startMonitors(monDict,self.mon,self.timings)
        
    def pause(self, monDict=[]):
        if isinstance(monDict, list):
            monDict = self.monDict
        
        self.timings = mf.pauseMonitors(monDict,self.mon,self.timings)
        
    def get_recordings(self, monDict=[]):
        if isinstance(monDict, list):
            monDict = self.monDict
        
        self.recordings.append(mf.getMonitors(monDict,self.mon))
        return self.recordings
        
    def get_recording_times(self, monDict=[]):
        if isinstance(monDict, list):
            monDict = self.monDict
            
        temp_timings={}
        for key, val in monDict.items():
            compartmentType, compartment = key.split(';')
            if len(self.timings[compartment]['start']) > len(self.timings[compartment]['stop']):
                ### was started/resumed but never stoped after --> use curretn time for stop time
                self.timings[compartment]['stop'].append(get_time())
            ### calculate the idx of the recorded arrays which correspond to the timings and remove 'currently_paused'
            diff_timings = (np.array(self.timings[compartment]['stop']) - np.array(self.timings[compartment]['start']))/dt() ## the length of the periods defines the idx of the arrays + dt
            start_idx = [np.sum(diff_timings[0:i]).astype(int) for i in range(diff_timings.size)]
            stop_idx  = [np.sum(diff_timings[0:i+1]).astype(int) for i in range(diff_timings.size)]
            temp_timings[compartment]={'start':{'ms':self.timings[compartment]['start'], 'idx':start_idx}, 'stop':{'ms':self.timings[compartment]['stop'], 'idx':stop_idx}}

        self.recording_times.append(temp_timings)
        
        ### generate a object from recording_times and return this instead of the dict
        recording_times_ob = recording_times_cl(self.recording_times)
        
        return recording_times_ob
        
    def reset(self, populations=True, projections=False, synapses=False, monitors=True, model=True, net_id=0):
        """
            TODO: get recordings before emptiing the monitors by reset
        """
        self.get_recordings()
        self.get_recording_times()
        ### reset timings, after reset, add a zero to start if the monitor is still running (this is not resetted by reset())
        for key in self.timings.keys():
            self.timings[key]['start']=[]
            self.timings[key]['stop']=[]
            if self.timings[key]['currently_paused']==False: self.timings[key]['start'].append(0)
        if model:
            reset(populations, projections, synapses, monitors, net_id=0)
        
        
        
class recording_times_cl:

    def __init__(self, recording_times_list):
        self.recording_times_list = recording_times_list
        
    def time_lims(self, chunk=0, compartment=None, period=0):
        if compartment==None:
            ### by default just use the first compartment
            compartment=list(self.recording_times_list[chunk].keys())[0]
        time_lims = [self.recording_times_list[chunk][compartment]['start']['ms'][period], self.recording_times_list[chunk][compartment]['stop']['ms'][period]]
        return time_lims
        
    def idx_lims(self, chunk=0, compartment=None, period=0):
        if compartment==None:
            ### by default just use the first compartment
            compartment=list(self.recording_times_list[chunk].keys())[0]
        idx_lims = [self.recording_times_list[chunk][compartment]['start']['idx'][period], self.recording_times_list[chunk][compartment]['stop']['idx'][period]]
        return idx_lims
        
    def all(self):
        return self.recording_times_list
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
