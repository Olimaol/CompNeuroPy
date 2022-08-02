from gc import get_objects
from ANNarchy import get_time
from .extra_functions import remove_key
import numpy as np

class generate_simulation:

    def __init__(self, simulation_function, simulation_kwargs=None, name='simulation', description='', requirements=None, kwargs_warning=True, monitor_object=None):
    
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
        self.monitor_object = monitor_object
        if monitor_object != None:
            self.monitor_chunk = []
        else:
            self.monitor_chunk = None
            

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
            if self.simulation_kwargs!=None:
                ### not replace initialized kwargs completely but only the kwargs which are given
                tmp_kwargs=self.simulation_kwargs.copy()
                for key, val in simulation_kwargs.items(): tmp_kwargs[key] = val
            else:
                ### there are no initial kwargs --> only use the kwargs which are given
                tmp_kwargs=simulation_kwargs
            if not(self.warned) and len(self.requirements)>0:
                print('\nWARNING! run',self.name,'changed simulation kwargs, initial requirements may no longer be fulfilled!\n')
                self.warned=True
        else:
            tmp_kwargs=self.simulation_kwargs
            
        ### before each run, test requirements
        self.__test_req__(simulation_kwargs=tmp_kwargs)
        
        ### and append current simulation kwargs to the kwargs variable
        self.kwargs.append(tmp_kwargs)
        
        ### and append the current chunk of the monitors object to the chunk variable
        if self.monitor_object != None:
            self.monitor_chunk.append(self.monitor_object.__current_chunk__())
        
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
        print('WARNING get_current_arr function will only be available in simulation_info_cl soon.')
        current_arr = []
        for run in range(len(self.kwargs)):
            t1 = self.kwargs[run]['t1']
            t2 = self.kwargs[run]['t2']
            a1 = self.kwargs[run]['a1']
            a2 = self.kwargs[run]['a2']
            
            if t1>0 and t2>0:
                current_arr.append(np.concatenate([np.ones(int(round(t1/dt)))*a1, np.ones(int(round(t2/dt)))*a2]))
            elif t2>0:
                current_arr.append(np.ones(int(round(t2/dt)))*a2)
            else:
                current_arr.append(np.ones(int(round(t1/dt)))*a1)
                
        if flat:
            return np.concatenate(current_arr)
        else:
            return current_arr
            
            
    def simulation_info(self):
        
        simulation_info_obj = simulation_info_cl(self.name, self.description, self.simulation_function.__name__, self.start, self.end, self.info, self.kwargs, self.monitor_chunk)
        
        return simulation_info_obj
        
        
class simulation_info_cl:

    def __init__(self, name, description, simulation_function, start, end, info, kwargs, monitor_chunk):
        self.name = name
        self.description = description
        self.simulation_function = simulation_function
        self.start = start
        self.end = end
        self.info = info
        self.kwargs = kwargs
        self.monitor_chunk = monitor_chunk
        
        
    def get_current_arr(self, dt, flat=False):
        """
            function for current_step simulations
            gets the current array (value for each time step) of all runs
            it returns a list of arrays (len of list = nr of runs)
            if flat --> it returns a flattened array --> assumes that all runs are run consecutively without brakes
        """
        assert self.simulation_function == 'current_step' or self.simulation_function == 'current_stim' or self.simulation_function == 'current_ramp', 'ERROR get_current_arr: Simulation has to be "current_step", "current_stim" or "current_ramp"!'
        
        if self.simulation_function == 'current_step':
            current_arr = []
            for run in range(len(self.kwargs)):
                t1 = self.kwargs[run]['t1']
                t2 = self.kwargs[run]['t2']
                a1 = self.kwargs[run]['a1']
                a2 = self.kwargs[run]['a2']
                
                if t1>0 and t2>0:
                    current_arr.append(np.concatenate([np.ones(int(round(t1/dt)))*a1, np.ones(int(round(t2/dt)))*a2]))
                elif t2>0:
                    current_arr.append(np.ones(int(round(t2/dt)))*a2)
                else:
                    current_arr.append(np.ones(int(round(t1/dt)))*a1)
                    
            if flat:
                return np.concatenate(current_arr)
            else:
                return current_arr
                
        elif self.simulation_function == 'current_stim':
            current_arr = []
            for run in range(len(self.kwargs)):
                t = self.kwargs[run]['t']
                a = self.kwargs[run]['a']
                
                if t>0:
                    current_arr.append(np.ones(int(round(t/dt)))*a)
                    
            if flat:
                return np.concatenate(current_arr)
            else:
                return current_arr
                
        elif self.simulation_function == 'current_ramp':
            current_arr = []
            for run in range(len(self.kwargs)):
            
                amp = self.kwargs[run]['a0']
                current_arr_ramp = []
                for stim_idx in range(self.kwargs[run]['n']):
                    t = self.info[run]['dur_stim']
                    a = amp
                    current_arr_ramp.append(np.ones(int(round(t/dt)))*a)
                    amp = amp + self.info[run]['da']
                current_arr.append(list(np.concatenate(current_arr_ramp)))
                    
            if flat:
                return np.concatenate(current_arr)
            else:
                return current_arr
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
