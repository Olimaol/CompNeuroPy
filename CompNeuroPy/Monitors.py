import CompNeuroPy.neuron_models as nm
import CompNeuroPy.model_functions as mf
import CompNeuroPy.simulation_functions as sif
import CompNeuroPy.system_functions as syf

class Monitors:

    
    def __init__(self, monDict={}):
    
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
