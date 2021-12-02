class Experiment():
    
    import ANNarchy as ann
    import CompNeuroPy as cnp
    
    def __init__(self, reset_function=0, reset_kwargs={}):
        self.recordings={}
        self.mon=self.cnp.Monitors()
        self.sim=[]
        
        ### check function to reset network
        if isinstance(reset_function, int) == True:
            self.reset_function = self.ann.reset
        else:
            self.reset_function = reset_function
        self.reset_kwargs = reset_kwargs
        
    def results(self):
        return {'recordings':self.recordings, 'monDict':self.mon.monDict, 'sim':self.sim}
