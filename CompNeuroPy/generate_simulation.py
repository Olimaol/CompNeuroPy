from gc import get_objects
from ANNarchy import get_time
from .extra_functions import remove_key

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
