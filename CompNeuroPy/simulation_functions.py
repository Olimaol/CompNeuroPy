from ANNarchy import simulate, get_population

def current_step(pop, t1=500, t2=500, a1=0, a2=100):
    """
        stimulates a given population in two periods with two input currents
        
        pop: population name of population, which should be stimulated with input current
             neuron model of population has to contain "I_app" as input current in pA
        t1/t2: times in ms before/after current step
        a1/a2: current amplitudes before/after current step in pA
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
