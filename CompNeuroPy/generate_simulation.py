from gc import get_objects
from ANNarchy import get_time
from .extra_functions import remove_key
import numpy as np

class generate_simulation:

    def __init__(self, simulation_function, simulation_kwargs=None, name='simulation', description='', requirements=None, kwargs_warning=True):
    
        # set simulaiton function
        self.name = name
        if name=='simulation': self.name = name+str(self.__nr_simulation__())
        self.description = description
        self.simulation_function = simulation_function
        self.simulation_kwargs = simulation_kwargs
        if requirements==None:
            self.requirements=[]
        else:
            self.requirements = requirements
        self.start = []
        self.end = []
        self.info = []
        self.kwargs = []
        if kwargs_warning:
            self.warned = False
        else:
            self.warned = True

        ### test initial requirements
        self.__test_req__(simulation_kwargs=simulation_kwargs)
        
    def run(self, simulation_kwargs=None):
        """
            runs simulation function
            with each run extend start, end and info list
            simulaiton_kwargs: optionally define new temporary simulation kwargs which override the initialized simulation kwargs
        """
        
        ### define the current simulation kwargs
        if simulation_kwargs!=None:
            tmp_kwargs=self.simulation_kwargs.copy()
            ### not replace initialized kwargs completely but only the kwargs which are given
            for key, val in simulation_kwargs.items(): tmp_kwargs[key] = val
            if not(self.warned):
                print('\nWARNING! run',self.name,'changed simulation kwargs, initial requirements may no longer be fulfilled!\n')
                self.warned=True
        else:
            tmp_kwargs=self.simulation_kwargs
            
        ### before each run, test requirements
        self.__test_req__(simulation_kwargs=tmp_kwargs)
        
        ### and append current simulation kwargs to the kwargs variable
        self.kwargs.append(tmp_kwargs)
        
        ### run the simulation, store start and end simulation time
        self.start.append(get_time())
        if tmp_kwargs!=None:
            self.info.append(self.simulation_function(**tmp_kwargs))
        else:
            self.info.append(self.simulation_function())
        self.end.append(get_time())
        
    def __nr_simulations__(self):
        """
            returns the current number of initialized CompNeuroPy simulations
        """
        
        sim_list = []
        object_list = get_objects()
        for obj in object_list:
            test=str(obj)
            compare='<CompNeuroPy.generate_simulation.generate_simulation object'
            if len(test)>=len(compare):
                if compare == test[:len(compare)]:
                    sim_list.append(vars(obj)['name'])
        del(object_list)
        return len(sim_list)
        
    def __test_req__(self, simulation_kwargs=None):
        """
            tests the initialized requirements with the current simulation_kwargs
        """
        
        if simulation_kwargs==None:#--> use the initial simulation_kwargs
            simulation_kwargs=self.simulation_kwargs
        
        for req in self.requirements:
            if len(list(req.keys()))>1:#--> requirement and arguments
                req_kwargs=remove_key(req, 'req')
                ### check if req_kwargs reference to sim_kwargs, if yes, use the corresponding current sim_kwarg as req_kwarg, if not do not update the initialized requirements kwargs
                for key, val in req_kwargs.items():
                    if isinstance(val,str):
                        val_split=val.split('.')
                        if val_split[0]=='simulation_kwargs':
                            if len(val_split)==1:
                                ### val is only simulation_kwargs
                                req_kwargs = simulation_kwargs
                            elif len(val_split)==2:
                                ### val is simulation_kwargs.something
                                req_kwargs[key] = simulation_kwargs[val_split[1]]
                            else:
                                ### val is simulation_kwargs.something.something... e.g. key='pops' and val= 'simulation_kwargs.model.populations'
                                req_kwargs[key] = eval('simulation_kwargs["'+val_split[1]+'"].'+'.'.join(val_split[2:]))
                                
                req['req'](**req_kwargs).run()
                
            else: #--> only requirement
                req['req']().run()
                
    def get_current_arr(self, dt, flat=False):
        """
            function for current_step simulations
            gets the current array (value for each time step) of all runs
            it returns a list of arrays (len of list = nr of runs)
            if flat --> it returns a flattened array --> assumes that all runs are run consecutively without brakes
        """
        assert self.simulation_function. __name__ == 'current_step', 'ERROR get_current_arr: Simulation has to be "current_step"!'
        
        current_arr = []
        for run in range(len(self.kwargs)):
            t1 = self.kwargs[run]['t1']
            t2 = self.kwargs[run]['t2']
            a1 = self.kwargs[run]['a1']
            a2 = self.kwargs[run]['a2']
            
            if t1>0 and t2>0:
                current_arr.append(np.concatenate([np.ones(int(t1/dt))*a1, np.ones(int(t2/dt))*a2]))
            elif t2>0:
                current_arr.append(np.ones(int(t2/dt))*a2)
            else:
                current_arr.append(np.ones(int(t1/dt))*a1)
                
        if flat:
            return np.concatenate(current_arr)
        else:
            return current_arr
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
