from CompNeuroPy import neuron_models as nm
from CompNeuroPy import model_functions as mf
from CompNeuroPy import simulation_functions as sim
from CompNeuroPy import system_functions as sf

class Monitors:

    
    def __init__(self, monDict):
    
        self.mon     = mf.addMonitors(monDict)
        self.monDict = monDict
        
    def start(self, monDict=[]):
        if isinstance(monDict, list):
            monDict = self.monDict
        
        mf.startMonitors(monDict,self.mon)
        
    def get_recordings(self, monDict=[]):
        if isinstance(monDict, list):
            monDict = self.monDict
        
        return mf.getMonitors(monDict,self.mon)
