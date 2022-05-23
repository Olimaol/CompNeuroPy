class Experiment():
    
    import ANNarchy as ann
    import CompNeuroPy as cnp
    
    def __init__(self, reset_function=None, reset_kwargs={}):
        self.recordings={} # save dict for monitor recordings
        self.mon=self.cnp.Monitors() # dict for monitors
        self.sim=[] # list for simulations
        self.data={} # dict for optional data
        
        ### check function to reset network
        if reset_function == None:
            self.reset_function = self.ann.reset
        else:
            self.reset_function = reset_function
        self.reset_kwargs = reset_kwargs
        
    def results(self):
        return {'recordings':self.recordings, 'monDict':self.mon.monDict, 'sim':self.sim, 'data':self.data}
