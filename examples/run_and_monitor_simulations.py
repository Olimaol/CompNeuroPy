import numpy as np
from CompNeuroPy import Monitors, save_data, generate_simulation, req_pop_attr
from ANNarchy import simulate, get_population
### we can import our model initialized in create_model.py
from create_model import my_model

### create and compile the model and save the populaiton names in some shorter variables
my_model.create()


### Next we define what should be recorded, i.e., create monitors with the Monitors class from CompNeuroPy
### the Monitor object helps to create multiple monitors at once
### Monitors takes a monitor_dictionary as argument
### format: monitor_dictionary = {'what;name':['variables', 'to', 'record']}
### the Monitors object currently only supports populations, thus, 'what;' has to be 'pop;'
### here we record from each population the variable p and spikes
monitor_dictionary = {'pop;'+my_model.populations[0]:['p', 'spike'], 'pop;'+my_model.populations[1]:['p', 'spike']}
mon = Monitors(monitor_dictionary)


### next we define some simulations with the generate_simulations object of CompNeuroPy
### similar to the generate_model object with the model_creation_function we also need a simulation_function here, in which the actual simulation is defined
### here are two examples in which we consider our model with multiple Poisson populations which contain the parameter 'rates'
### use the ANNarchy functions get_population and get_projection to access populations ad projections using their names which should be unique and obtainable from the CompNeuroPy model

def set_rates(pop, rates=0, duration=0):
    """
        sets the rates variable of one population and simulates duration in ms
    """
    ### set rates and simulate
    get_population(pop).rates = rates
    simulate(duration)

### the set_rates function is already a perfectly fine simulation_function, but let's create something more complex using it
def increase_rates(pop, rate_step=0, time_step=0, nr_steps=0):
    """
        increase rates variable of pop, if pop == list --> increase rates variable of multiple populations
        rate_step: increase of rate with each step, initial step = current rates of pop
        time_step: duration of each step in ms
    """
    
    ### convert single pop into list
    if not(isinstance(pop, list)): pop = [pop]

    ### define initial value for rates for each pop
    start_rate = np.array([get_population(pop_name).rates[0] for pop_name in pop])
    
    ### simulate all steps
    for step in range(nr_steps):
        ### calculate rates for each pop
        rates=step*rate_step+start_rate
        ### set rates variable of all populations
        for pop_idx, pop_name in enumerate(pop):
            set_rates(pop_name, rates=rates[pop_idx], duration=0) ### use already defined simulation set_rates
        ### then simulate step
        set_rates(pop[0], rates=rates[0], duration=time_step)
        
    ### simulation_functions can return some information which may be helpful later
    ### the simulation arguments do not need to be returned, since they are accessible through the generate_model object anyway
    return {'duration':time_step*nr_steps, 'd_rates':rate_step*nr_steps}

### Now use the simulation_function and add a clear framework with the generate_simulation object
### as arguments provide the simulation_function, its arguments (as kwargs dictionary), the name and description of the simualtion and a requirements list
### the requirements list contains requirements objects (here req_pop_attr)
### with requirements one can determine things that the model should fulfill
### req_pop_attr checks if a population (or multiple) contain a specific attribut (or multiple)
### for our defined simulation_funciton we test if the model populations contain the variable 'rates'
increase_rates_pop1 = generate_simulation(increase_rates,
                                          simulation_kwargs={'pop':my_model.populations[0], 'rate_step':10, 'time_step':100, 'nr_steps':15},
                                          name='increase_rates_pop1',
                                          description='increase rates variable of pop1',
                                          requirements=[req_pop_attr(my_model.populations[0],'rates')]) 
                                          
increase_rates_all_pops = generate_simulation(increase_rates,
                                              simulation_kwargs={'pop':my_model.populations, 'rate_step':10, 'time_step':100, 'nr_steps':15},
                                              name='increase_rates_all_pops',
                                              description='increase rates variable of all pops',
                                              requirements=[req_pop_attr(my_model.populations,'rates')]) 


### Now let's use these simulations
### in the following lines various different use cases for the simualtions and Monitor object functions are demonstrated
### the Monitors object automatically structures the recordings based on recording pauses and model resets
### the recordings are split into chunks by resets
### in this script we will have two chunks
### further, chunks are subdivided into periods which are separeted by pauses (a reset automatically closes a period)
### in this script the first chunk will only contain one period, the second chunk will contain two periods
### first start the monitors with the Monitors object after a 500 ms resting-state simulation
simulate(500)
mon.start()

### run the simulation increase_rates_pop1
increase_rates_pop1.run()

### run the simulation increase_rates_all_pops
increase_rates_all_pops.run()

### by resetting the model one can start again from scratch (time=0 again and model in its compile state)
### if one uses the Monitors object one should reset the model with the Monitors object function reset (thus, the recordings are automatically structured)
mon.reset()

### let's simualte again a resting-state simulation
### monitors work the same as before a reset, therefore the monitors must be paused here if one does not want to record the resting-state simulation
mon.pause()
simulate(700)
mon.start()

### run again the simulation increase_rates_all_pops
increase_rates_all_pops.run()

### simulate 1000 ms but do not record
mon.pause()
simulate(1000)
mon.start()

### run again the simulation increase_rates_all_pops, but this time with different simulation kwargs
increase_rates_all_pops.run({'rate_step':50, 'time_step':500, 'nr_steps':2})

### simulate another 1000 ms, again do not record
mon.pause()
simulate(1000)

### get recordings and recording times from the Monitors object
recordings = mon.get_recordings()
recording_times = mon.get_recording_times()


### TODO demonstrate what is included in the simulaton objects
### TODO what is included in recordings and recording times


### one could directly analyze/plot recordings here but we first save them with CompNeuroPy save_data function
### one can save different things, given in a list + for each thing the corresponding save folder + name, also given in a list
### all things are saved in the directory dataRaw/
### we her save, for example, the two simulation objects which contain usefull information for later analyses and the recordings and recording_times
folder='run_and_monitor_simulations/'
save_data([increase_rates_pop1, increase_rates_all_pops, recordings, recording_times],[folder+'increase_rates_pop1.npy', folder+'increase_rates_all_pops.npy', folder+'recordings.npy', folder+'recording_times.npy'])



