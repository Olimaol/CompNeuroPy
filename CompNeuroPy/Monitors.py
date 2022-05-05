import CompNeuroPy.neuron_models as nm
import CompNeuroPy.model_functions as mf
import CompNeuroPy.simulation_functions as sif
import CompNeuroPy.system_functions as syf
from ANNarchy import get_time

class Monitors:

    
    def __init__(self, monDict={}):
    
        self.mon     = mf.addMonitors(monDict)
        self.monDict = monDict

        timings = {}
        for key, val in monDict.items():
            compartmentType, compartment = key.split(';')
            timings[compartment] = {'currently_paused':False, 'start':[], 'stop':[]}
        self.timings = timings
        
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
        
        return mf.getMonitors(monDict,self.mon)
        
    def get_times(self, monDict=[]):
        if isinstance(monDict, list):
            monDict = self.monDict
            
        for key, val in monDict.items():
            compartmentType, compartment = key.split(';')
            if len(self.timings[compartment]['start']) > len(self.timings[compartment]['stop']):
                ### was started/resumed but never stoped after --> use curretn time for stop time
                self.timings[compartment]['stop'].append(get_time())
            ### remove 'currently_paused'
            self.timings[compartment]={'start':self.timings[compartment]['start'], 'stop':self.timings[compartment]['stop']}

        return self.timings
