from ANNarchy import simulate, get_population, get_time
from CompNeuroPy import simulation_requirements as req
import numpy as np

"""
    Here we define some simulation fucntions.
    In simulation functions 
"""
                     
def set_rates(pop, rates=0, duration=0):
    """
        sets the rates variable of one population and simulates duration in ms
    """
    ### set rates and simulate
    get_population(pop).rates = rates
    simulate(duration)
                             
                             
def increase_rates(pop, rate_step=0, time_step=0, nr_steps=0):
    """
        increase rates variable of pop, if pop == list --> increase rates variable of multiple populations
        rate_step: increase of rate with each step, initial step = current rates of pop
        time_step: how long each step in ms
    """
    
    if not(isinstance(pop, list)): pop = [pop]

    start_rate = np.array([get_population(pop_name).rates[0] for pop_name in pop])
    
    for step in range(nr_steps):
        rates=step*rate_step+start_rate
        
        ### use already defined simulations to create more complex ones
        ### set rates variable of multiple populations
        for pop_idx, pop_name in enumerate(pop):
            set_rates(pop_name, rates=rates[pop_idx], duration=0)
        ### then simulate step
        set_rates(pop[0], rates=rates[0], duration=time_step)
        
    return {'duration':time_step*nr_steps, 'd_rates':rate_step*nr_steps}
                             
                             
            
                
                             
    
    
    
    

