from ANNarchy import setup, Population, get_population, reset
import numpy as np
import traceback
import CompNeuroPy.neuron_models as nm
import CompNeuroPy.model_functions as mf
import CompNeuroPy.simulation_functions as sif
import CompNeuroPy.system_functions as syf
from .Experiment import Experiment

# hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope

# multiprocessing
from multiprocessing import Process
import multiprocessing


class opt_neuron:

    opt_created = []
    
    def __init__(self, results_soll, experiment, get_loss_function, fitting_variables_space, const_params, compile_folder_name='annarchy_opt_neuron', num_rep_loss=1, neuron_model=None):
    
        if len(self.opt_created)>0:
            print('opt_neuron: Error: Already another opt_neuron created. Only create one per python session!')
            quit()
        else:
            print('opt_neuron: Initialize opt_neuron... better not create anything with ANNarchy before!')
            
            ### set object variables
            self.opt_created.append(1)
            self.results_soll = results_soll
            self.experiment = experiment
            self.fv_space = fitting_variables_space
            self.const_params = const_params
            self.num_rep_loss = num_rep_loss
            self.neuron_model = neuron_model
            
            ### check get_loss function compatibility with experiment
            self.__get_loss__ = self.__check_get_loss_function__(get_loss_function)
        
            ### setup ANNarchy
            setup(dt=results_soll['recordings'][0]['dt'])

            ### create and compile model
            model = self.__raw_neuron__(do_compile=True, compile_folder_name=compile_folder_name)
            self.iz = model[0]

            ### check variables
            self.__test_variables__()
            
            
    def __check_get_loss_function__(self, function):
        """
            function: function whith arguments (results_ist, results_soll) which computes and returns a loss value
            
            Checks if function is compatible to the experiment.
            To test, function is run with results_soll (also for results_ist argument).
        """
        
        try:
            function(self.results_soll, self.results_soll)
        except:
            print('\nThe get_loss_function is not compatible to the specified experiment:\n')
            traceback.print_exc()
            quit()
            
        return function
            
            
    def __raw_neuron__(self, do_compile=False, compile_folder_name='annarchy_opt_neuron'):
        """
            generates one neuron, default=Izhikevich neuron model, and optionally compiles the network
            
            returns a list of the names of the populations (for later access)
        """
        if isinstance(self.neuron_model, type(nm.Izhikevich2007)):
            pop   = Population(1, neuron=self.neuron_model, name='user_defined_neuron')
            ret   = ['user_defined_neuron']
        else:
            pop   = Population(1, neuron=nm.Izhikevich2007, name='Iz_neuron')
            ret   = ['Iz_neuron']

        if do_compile:
            mf.compile_in_folder(compile_folder_name)
        
        return ret
        
        
    def __test_variables__(self):
        """
            self.iz: name of created neuron population
            self.fv_space: hyperopt variable space list
            self.const_params: dictionary with constant variables
            
            Checks if self.iz contains all variables of self.fv_space and self.const_params and if self.iz has more variables.
        """
        ### collect all names
        fitting_vars_names = [self.fv_space[i].pos_args[0].pos_args[0]._obj for i in range(len(self.fv_space))]
        all_vars_names = np.concatenate([np.array(list(self.const_params.keys())), np.array(fitting_vars_names)]).tolist()
        ### check if pop has these parameters
        pop_parameter_names = get_population(self.iz).parameters
        for name in pop_parameter_names.copy():
            if name in all_vars_names:
                all_vars_names.remove(name)
                pop_parameter_names.remove(name)
        if len(pop_parameter_names)>0:
            print('opt_neuron: WARNING: parameters',pop_parameter_names,'of population',self.iz,'are not used.')
        if len(all_vars_names)>0:
            print('opt_neuron: Error: Population',self.iz,'does not contain parameters',all_vars_names,'!')
            quit()
            
            
    def __run_simulator__(self, fitparams):
        """
            runs the function simulator with the multiprocessing manager (if function is called multiple times this saves memory, otherwise same as calling simulator directly)
            
            fitparams: list, for description see function simulator
            return: returns dictionary needed for optimization with hyperopt
        """
        ### initialize manager and generate m_list = dictionary to save data
        manager = multiprocessing.Manager()
        m_list = manager.dict()

        ### in case of noisy models, here optionally run multiple simulations, to mean the loss
        lossAr = np.zeros(self.num_rep_loss)
        for nr_run in range(self.num_rep_loss):
            ### initialize for each run a new rng (--> not always have same noise in case of noisy models/simulations)
            rng = np.random.default_rng()
            ### run simulator with multiprocessign manager
            proc = Process(target=self.__simulator__,args=(fitparams,rng,m_list))
            proc.start()
            proc.join()
            ### get simulation results/loss
            lossAr[nr_run]=m_list[0]
            results_ist=m_list[1]
        
        ### calculate mean and std of loss
        if self.num_rep_loss > 1:
            loss = np.mean(lossAr)
            std  = np.std(lossAr)
        else:
            loss = lossAr[0]
            std  = None
        
        ### return loss and other things for optimization
        if self.num_rep_loss > 1:
            return {
                'status': STATUS_OK,
                'loss': loss,
                'loss_variance': std,
                'std': std,
                'results': results_ist
                }
        else:
            return {
                'status': STATUS_OK,
                'loss': loss,
                'std': std,
                'results': results_ist
                }
              
    
    class myexp(Experiment):
        """
            cnp Experiment class
            The method "run" is given as an argument during object instantiation
        """
        def __init__(self, reset_function, reset_kwargs, experiment_function):
            """
                runs the init of the standard (parent) Experiment class, which defines the reset function of the Experiment object
                additionally stores the experiment funciton which can later be used in the run() function
                the experiment function is given, this enables to use a loaded run function from a previously defined Experiment Object
            """
            super().__init__(reset_function, reset_kwargs)
            self.experiment_function = experiment_function
        def run(self, experiment_kwargs):
            return(self.experiment_function(self, **experiment_kwargs))
             
                
    def __simulator__(self, fitparams, rng, m_list=[0,0,0]):
        """
            fitparams: from hyperopt
                
            m_list: variable to store results, from multiprocessing
        """
        
        ### reset model and set parameters which should not be optimized and parameters which should be optimized
        self.__set_fitting_parameters__(fitparams)
        
        ### conduct loaded experiment
        experiment_function = self.experiment
        experiment_kwargs = {'population':self.iz}
        reset_function = self.__set_fitting_parameters__
        reset_kwargs = {'fitparams':fitparams}
        
        exp_obj = self.myexp(reset_function=reset_function, reset_kwargs=reset_kwargs, experiment_function=experiment_function)
        results = exp_obj.run(experiment_kwargs)
                
        ### compute loss
        loss = self.__get_loss__(results, self.results_soll)

        ### "return" loss and other optional things
        m_list[0]=loss
        m_list[1]=results
        
        
    def __set_fitting_parameters__(self, fitparams):
        """
            self.iz: ANNarchy population name
            fitparams: list with values for fitting parameters
            self.fv_space: hyperopt variable space list
            self.const_params: dictionary with constant variables
        
            Sets all given parameters for the population self.iz.
        """
        ### reset model to compilation state
        reset()
        
        ### set fitting parameters
        fitting_vars_names = [self.fv_space[i].pos_args[0].pos_args[0]._obj for i in range(len(self.fv_space))]
        for idx in range(len(fitparams)):
            setattr(get_population(self.iz), fitting_vars_names[idx], fitparams[idx])
            if fitting_vars_names[idx]=='v_r':
                get_population(self.iz).v = fitparams[idx]
        ### set constant parameters
        for key, val in self.const_params.items():
            setattr(get_population(self.iz), key, val)
            if key=='v_r':
                get_population(self.iz).v = val
            
        
    def __test_fit__(self, fitparamsDict):
        """
            fitparamsDict: dictionary with parameters, format = as hyperopt returns fit results
            
            Thus, this function can be used to run the simulator function directly with fitted parameters obtained with hyperopt
            
            Returns the loss computed in simulator function.
        """
        
        fitting_vars_names = [self.fv_space[i].pos_args[0].pos_args[0]._obj for i in range(len(self.fv_space))]
        
        return self.__run_simulator__([fitparamsDict[name] for name in fitting_vars_names])
        
     
    def run(self, max_evals, results_file_name='best.npy'):
        ### start optimization run
        best = fmin(
                    fn=self.__run_simulator__,
                    space=self.fv_space,
                    algo=tpe.suggest,
                    max_evals=max_evals
                    )
        fit=self.__test_fit__(best)
        best['loss'] = fit['loss']
        best['std'] = fit['std']
        best['results'] = fit['results']
        self.results=best
        
        ### SAVE OPTIMIZED PARAMS AND LOSS
        syf.save_data([best], ['parameter_fit/'+results_file_name])

