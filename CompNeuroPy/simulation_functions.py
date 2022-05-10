from ANNarchy import simulate, get_population, get_time

def current_step(pop, t1=500, t2=500, a1=0, a2=100):
    """
        simulates the model
        
        pop: population name of population, which should be stimulated with input current
             neuron model of population has to contain "I_app" as input current in pA
        t1/t2: times in ms before/after current step
        a1/a2: current amplitudes before/after current step in pA
    """
    
    ### start = current time
    start = get_time()
    
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
    
    return {'name':'current_step', 'pop':pop, 't1':t1, 't2':t2, 'a1':a1, 'a2':a2, 'start':start, 'end':get_time()}
