from ANNarchy import simulate, get_population, dt
import numpy as np
import CompNeuroPy as cnp

    
def current_step(pop, t1=500, t2=500, a1=0, a2=100):
    """
        stimulates a given population in two periods with two input currents
        
        pop: population name of population, which should be stimulated with input current
             neuron model of population has to contain "I_app" as input current
        t1/t2: times in ms before/after current step
        a1/a2: current amplitudes before/after current step
    """
    
    ### save prev input current
    I_prev = get_population(pop).I_app
    
    ### first/pre current step simulation
    get_population(pop).I_app = a1
    simulate(t1)
    
    ### second/post current step simulation
    get_population(pop).I_app = a2
    simulate(t2)
    
    ### reset input current to previous value
    get_population(pop).I_app = I_prev
    
    ### return some additional information which could be usefull
    return {'duration':t1+t2}
    
    
def current_stim(pop, t=500, a=100):
    """
        stimulates a given population during specified period 't' with input current with amplitude 'a', after this stimulation the current is reset to initial value (before stimulation)
        
        pop: population name of population, which should be stimulated with input current
             neuron model of population has to contain "I_app" as input current
        t: duration in ms
        a: current amplitude
    """
    
    return current_step(pop, t1=t, t2=0, a1=a, a2=0)
    
    
def current_ramp(pop, a0, a1, dur, n):
    """
        Conducts multiple current stimulations with constantly changing current inputs.
        After this current_ramp stimulation the current amplitude is reset to the initial value (before current ramp).
        
        a0: initial current amplitude (of first stimulation)
        a1: final current amplitude (of last stimulation)
        dur: duration of the complete current ramp (all stimulaiton)
        n:  number of stimulations 
        
        resulting duration of one stimulation = dur/n, should be divisible by the simulation time step without remainder
    """
    
    assert (dur/n)/dt() % 1 == 0, 'ERROR current_ramp: dur/n should result in a duration (for a single stimulation) which is divisible by the simulation time step (without remainder)\ncurrent duration = '+str(dur/n)+', timestep = '+str(dt())+'!\n'
    
    da = (a1-a0)/(n-1)# for n stimulations only n-1 steps occur
    dur_stim = dur/n
    amp = a0
    for stim_idx in range(n):
        current_stim(pop, t=dur_stim, a=amp)
        amp = amp + da
        
    return {'da':da, 'dur_stim':dur_stim}
        

def increasing_current(pop,I1,step,nr_steps,durationI2):
    """
        step : step size with which the external current is increased
        I1,I2 : current amplitudes before/after the step increase 
        durationI : duration in which the external current is inserted
    """
    current_list = []
    for i in range(nr_steps):
       
        I2 = I1 + step
        current_list.append(I1)
        current_list.append(I2)
        current_step(pop,500,durationI2,I1,I2)
        I1 = I2
 

    return {'duration':500+ durationI2, 'current_list':current_list}

    
    


   


