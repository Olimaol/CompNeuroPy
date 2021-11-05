from ANNarchy import setup, Population, get_population, reset
from CompNeuroPy.neuron_models import Izhikevich2007
from CompNeuroPy.model_functions import compile_in_folder, addMonitors, startMonitors, getMonitors
from CompNeuroPy.system_functions import save_data
from CompNeuroPy.simulation_functions import current_step
import numpy as np

# hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope

# multiprocessing
from multiprocessing import Process
import multiprocessing


class opt_Izh:

    opt_created = []
    
    def __init__(self, recordings_soll, sp_soll, fitting_variables_space, const_params, compile_folder_name='annarchy_raw_Izhikevich', num_rep_loss=1):
    
        if len(self.opt_created)>0:
            print('opt_Izh: Error: Already another opt_Izh created. Only create one per python session!')
            quit()
        else:
            self.opt_created.append(1)
            
            print('opt_Izh: Initialize opt_Izh... better not create anything with ANNarchy before!')
            
            ### check if recordings fit to simulation protocol
            if recordings_soll['sp'] != sp_soll['name']:
                print('opt_Izh: Error: Loaded simulation protocol does not fit loeaded recordings!')
                quit()
            self.recordings_soll = recordings_soll
            self.sp_soll = sp_soll
            self.fv_space = fitting_variables_space
            self.const_params = const_params
            self.num_rep_loss = num_rep_loss
        
            ### setup ANNarchy
            setup(dt=recordings_soll['dt'])

            ### create and compile model
            model = self.__raw_Izhikevich__(do_compile=True, compile_folder_name=compile_folder_name)
            self.iz = model[0]

            ### check variables
            self.__test_variables__()
            
            
    def __raw_Izhikevich__(self, do_compile=False, compile_folder_name='annarchy_raw_Izhikevich'):
        """
            generates one neuron of the Izhikevich neuron model and optionally compiles the network
            
            returns a list of the names of the populations (for later access)
        """
        pop   = Population(1, neuron=Izhikevich2007, name='Iz_neuron')

        if do_compile:
            compile_in_folder(compile_folder_name)
        
        return ['Iz_neuron']
        
        
    def __test_variables__(self):
        """
            self.iz: name of created Izhikevich neuron population
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
            print('opt_Izh: WARNING: parameters',pop_parameter_names,'of population',self.iz,'are not used.')
        if len(all_vars_names)>0:
            print('opt_Izh: Error: Population',self.iz,'does not contain parameters',all_vars_names,'!')
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
            simulation_protocol_ist=m_list[1]
            recordings_ist=m_list[2]
        
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
                'simulation_protocol': simulation_protocol_ist,
                'recordings': recordings_ist
                }
        else:
            return {
                'status': STATUS_OK,
                'loss': loss,
                'std': std,
                'simulation_protocol': simulation_protocol_ist,
                'recordings': recordings_ist
                }
                
                
    def __simulator__(self, fitparams, rng, m_list=[0,0,0]):
        """
            fitparams: from hyperopt
                
            m_list: variable to store results, from multiprocessing
        """

        ### reset model to compilation state
        reset()

        ### create monitors, iz is a global variable
        monDict={'pop;'+self.iz:['v', 'spike']}
        mon=addMonitors(monDict)
        
        ### set parameters which should not be optimized and parameters which should be optimized
        self.__set_fitting_parameters__(fitparams)
        
        ### simulate, use the same simulation as for the soll-data
        startMonitors(monDict,mon)
        sim1 = current_step(self.iz, t1=self.sp_soll['t1'], t2=self.sp_soll['t2'], a1=self.sp_soll['a1'], a2=self.sp_soll['a2'])
        
        ### get monitors
        recordings=getMonitors(monDict,mon)
        
        ### get data for comparison, recordings_soll and sp_soll are global variables
        ist, soll = self.__get_data_for_comparison__(recordings)
        
        ### compute loss
        loss = np.sqrt(np.mean((soll-ist)**2))

        ### "return" loss and other optional things
        m_list[0]=loss
        m_list[1]=sim1
        m_list[2]=recordings
        
        
    def __set_fitting_parameters__(self, fitparams):
        """
            self.iz: ANNarchy population name
            fitparams: list with values for fitting parameters
            self.fv_space: hyperopt variable space list
            self.const_params: dictionary with constant variables
        
            Sets all given parameters for the population self.iz.
        """
        ### set fitting parameters
        fitting_vars_names = [self.fv_space[i].pos_args[0].pos_args[0]._obj for i in range(len(self.fv_space))]
        for idx in range(len(fitparams)):
            setattr(get_population(self.iz), fitting_vars_names[idx], fitparams[idx])
        ### set constant parameters
        for key, val in self.const_params.items():
            setattr(get_population(self.iz), key, val)
            
            
    def __get_data_for_comparison__(self, recordings):
        """
            recordings: recordings dictionary of simulation, has to contain spike recordings (of one neuron)
            
            transforms spike recordings of ist and soll data into interspike intervals (unequal number of spikes --> fill up with zeros)
            
            global variables:
                self.iz : name of population
                self.recordings_soll, self.sp_soll : loaded recordings and protocol of soll data
        """

        ### get spike times from recordings
        spike_times_ist = np.array(recordings[self.iz+';spike'][0])
        spike_times_soll = np.array(self.recordings_soll[self.sp_soll['pop']+';spike'][0])
        
        ### only use spike times from second half
        spike_times_ist = spike_times_ist[spike_times_ist>int(self.sp_soll['t1']/self.recordings_soll['dt'])]
        spike_times_soll = spike_times_soll[spike_times_soll>int(self.sp_soll['t1']/self.recordings_soll['dt'])]
        
        ### calculate inter spike intervals
        if len(spike_times_ist)>0:
            isi_arr_ist  = np.diff(np.concatenate([np.zeros(1),np.array(spike_times_ist)]))
        else:
            isi_arr_ist = np.zeros(1)
        if len(spike_times_soll)>0:
            isi_arr_soll = np.diff(np.concatenate([np.zeros(1),np.array(spike_times_soll)]))
        else:
            isi_arr_soll = np.zeros(1)
        
        ### append entries so that arrays have same length, use values from other array but multiplied by 100
        if len(isi_arr_ist) > len(isi_arr_soll):
            ### append entries to isi_arr_soll
            nr_diff=len(isi_arr_ist)-len(isi_arr_soll)
            isi_arr_soll = np.concatenate([isi_arr_soll, isi_arr_ist[-nr_diff:]*100])        
        elif len(isi_arr_ist) < len(isi_arr_soll):
            ### append entries to isi_arr_ist
            nr_diff=len(isi_arr_soll)-len(isi_arr_ist)
            isi_arr_ist = np.concatenate([isi_arr_ist, isi_arr_soll[-nr_diff:]*100])

        return [isi_arr_ist, isi_arr_soll]
        
        
    def __test_fit__(self, fitparamsDict):
        """
            fitparamsDict: dictionary with parameters, format = as hyperopt returns fit results
            
            Thus, this function can be used to run the simulator function directly with fitted parameters obtained with hyperopt
            
            Returns the loss computed in simulator function.
        """
        
        fitting_vars_names = [self.fv_space[i].pos_args[0].pos_args[0]._obj for i in range(len(self.fv_space))]
        
        return self.__run_simulator__([fitparamsDict[name] for name in fitting_vars_names])
        
     
    def run(self, max_evals, results_file_name='best'):
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
        best['simulation_protocol'] = fit['simulation_protocol']
        best['recordings'] = fit['recordings']
        self.results=best
        
        ### SAVE OPTIMIZED PARAMS AND LOSS
        save_data([best], ['parameter_fit/'+results_file_name+'.npy'])

