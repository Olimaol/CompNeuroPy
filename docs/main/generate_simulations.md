## Introduction
A CompNeuroPy-simulation can be created using the [`CompNeuroSim`](#CompNeuroPy.generate_simulation.CompNeuroSim) class. Similar to the [`CompNeuroModel`](generate_models.md#CompNeuroPy.generate_model.CompNeuroModel) class, a function must be defined that contains the actual simulation (the _simulation_function_) and the [`CompNeuroSim`](#CompNeuroPy.generate_simulation.CompNeuroSim) object adds a clear framework. A [`CompNeuroSim`](#CompNeuroPy.generate_simulation.CompNeuroSim) is first initialized and can then be run multiple times.

## Example:
```python
from CompNeuroPy import CompNeuroSim
my_simulation = CompNeuroSim(simulation_function=some_simulation,           ### the most important part, this function defines the simulation
                            simulation_kwargs={'pop':pop1, 'duration':100}, ### define the two arguments pop and duration of simulation_function
                            name='my_simulation',                           ### you can give the simulation a name
                            description='my simple example simulation',     ### you can give the simulation a description
                            requirements=[req],                             ### a list of requirements for the simulation (here only a single requirement)
                            kwargs_warning=True,                            ### should a warning be printed if simulation kwargs change in future runs
                            monitor_object = mon)                           ### the Monitors object which is used to record variables                   
```

A possible _simulation_function_ could be:
```python
def some_simulation(pop, duration=1):
    get_population(pop).a = 5  ### adjust paramter a of pop
    get_population(pop).b = 5  ### adjust paramter b of pop
    simulate(duration)         ### simulate the duration in ms

    ### return some info
    ### will later be accessible for each run
    return {'paramter a': a, 'paramter b': b, 'a_x_duration': a*duration} 
```

And a corresponding requirement could be:
```python
from CompNeuroPy import ReqPopHasAttr
req = {'req':ReqPopHasAttr, 'pop':pop1, 'attr':['a', 'b']}
```
Here, one checks if the population _pop1_ contains the attributes _a_ and _b_. The [`ReqPopHasAttr`](../additional/simulation_requirements.md#CompNeuroPy.simulation_requirements.ReqPopHasAttr) is a built-in requirements-class of CompNeuroPy (see below).

A more detailed example is available in the [Examples](../examples/run_and_monitor_simulations.md).

## Simulation information
The function _simulation_info()_ returns a [`SimInfo`](#CompNeuroPy.generate_simulation.SimInfo) object which contains usefull information about the simulation runs (see below). The [`SimInfo`](#CompNeuroPy.generate_simulation.SimInfo) object also provides usefull analysis functions associated with specific simulation functions. Currently it provides the _get_current_arr()_ which returns arrays containing the input current for each time step of the built-in simulation functions _current_step()_, _current_stim()_, and _current_ramp()_.

## Simulation functions
Just define a classic ANNarchy simulation in a function. Within the functions, the ANNarchy functions _get_population()_ and _get_projection()_ can be used to access the populations and projections using the population and projection names provided by a [`CompNeuroModel`](generate_models.md#CompNeuroPy.generate_model.CompNeuroModel). The return value of the simulation function can later be retrieved from the [`SimInfo`](#CompNeuroPy.generate_simulation.SimInfo) object (the _info_ attribute) in a list containing the return value for each run of the simulation.

### Example:
```python
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
```

## Requirements
In order to perform simulations with models, the models must almost always fulfill certain requirements. For example, if the input current of a population is to be set, this population (or the neuron model) must of course have the corresponding variable. Such preconditions can be tested in advance with the `simulation_requirements` classes. They only need to contain a function _run()_ to test the requirements (if requirements are not met, cause an error). In CompNeuroPy predefined [`simulation_requirements`](../additional/simulation_requirements.md) classes are available (CompNeuroPy.simulation_requirements; currently only [`ReqPopHasAttr`](../additional/simulation_requirements.md#CompNeuroPy.simulation_requirements.ReqPopHasAttr)). In the [`CompNeuroSim`](#CompNeuroPy.generate_simulation.CompNeuroSim) class, the requirements are passed as arguments in a list (see above). Each requirement (list entry) must be defined as a dictionary with keys _req_ (the requirement class) and the arguments of the requirement class (e.g., _pop_ and _attr_ for the [`ReqPopHasAttr`](../additional/simulation_requirements.md#CompNeuroPy.simulation_requirements.ReqPopHasAttr)).

Here two requirements are defined (both [`ReqPopHasAttr`](../additional/simulation_requirements.md#CompNeuroPy.simulation_requirements.ReqPopHasAttr)). All populations of _my_model_ should contain the attribute (variable or parameter) _'I'_ and all populations of _my_other_model_ should contain the attribute _'v'_:

```python
req1 = {'req':ReqPopHasAttr, 'pop':my_model.populations, 'attr':'I'}
req2 = {'req':ReqPopHasAttr, 'pop':my_other_model.populations, 'attr':'v'}
my_two_model_simulation = CompNeuroSim(..., requirements=[req1, req2])
```

As described above, new simulation_kwargs can be passed to the _run()_ function of a [`CompNeuroSim`](#CompNeuroPy.generate_simulation.CompNeuroSim) object. Thus, one could initially pass a particular model as simulation_kwargs and for a later run pass a different model. If the requirements are defined as shown above, it is not tested again whether the new model (e.g. _my_third_model_) also fulfills the requirements (because the requirements were defined for _my_model_ and _my_other_model_). To work around this, an argument for a `simulation_requirements` class can also be linked to a simulation_kwargs entry. Thus, if new simulation_kwargs are used, also the simulation_requirements arguments adapt. This can be done using a string with the syntax "simulation_kwargs.<kwarg_name\>.<optional_attribute_of_kwarg\>", as shown in this example:

```python
req1 = {'req':ReqPopHasAttr, 'pop':"simulation_kwargs.model1.populations", 'attr':'I'}
req2 = {'req':ReqPopHasAttr, 'pop':"simulation_kwargs.model2.populations", 'attr':'v'}
my_two_model_simulation = CompNeuroSim(simulation_kwargs={'model1':my_model, 'model2':my_other_model, 'parameter':5},
                                        ...,
                                        requirements=[req1, req2])
...
my_two_model_simulation.run({'model1':my_third_model})
```

Due to the string "simulation_kwargs.model1.populations" the _pop_ argument of _req1_ is now linked to _model1_ (defined in the simulation_kwargs). Thus, in the run where a different model (_my_third_model_) is used for _model1_, _req1_ is automatically tested for the new _model1_.

::: CompNeuroPy.generate_simulation.CompNeuroSim
::: CompNeuroPy.generate_simulation.SimInfo