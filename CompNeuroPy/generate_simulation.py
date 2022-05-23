from gc import get_objects
from ANNarchy import get_time

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

        ### test requirements
        for req in requirements:
            req.run()
        
    def run(self, simulation_kwargs=None):
        """
            runs simulation function
            with each run extend start, end and info list
            simulaiton_kwargs: optionally define new temporary simulation kwargs which override the initialized simulation kwargs
        """
        if simulation_kwargs!=None:
            tmp_kwargs=self.simulation_kwargs.copy()
            ### not replace initialized kwargs completely but only the kwargs which are given
            for key, val in simulation_kwargs.items(): tmp_kwargs[key] = val
            if not(self.warned):
                print('\nWARNING! run',self.name,'changed simulation kwargs, initial requirements may no longer be fulfilled!\n')
                self.warned=True
        else:
            tmp_kwargs=self.simulation_kwargs
        self.kwargs.append(tmp_kwargs)
        
        ### run the simulation, store start and end simualtion time
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
        for obj in get_objects():
            test=str(obj)
            compare='<CompNeuroPy.generate_simulation.generate_simulation object'
            if len(test)>=len(compare):
                if compare == test[:len(compare)]:
                    sim_list.append(vars(obj)['name'])
                    
        return len(sim_list)
