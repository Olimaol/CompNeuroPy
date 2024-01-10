"""
This example demonstrates how to use the CompNeuroSim class to define simulations.
It is shown how to define the simulation functions, requirements and how to use the
simulation information object.

This example imports the "my_model" from other example "create_model.py" and saves
recorded data used in other example "plot_recordings.py".
"""
import numpy as np
from CompNeuroPy import (
    CompNeuroMonitors,
    CompNeuroSim,
    ReqPopHasAttr,
    save_variables,
    CompNeuroModel,
)
from ANNarchy import (
    simulate,
    get_population,
    Population,
    Neuron,
    Projection,
    Synapse,
    Uniform,
)
from CompNeuroPy.examples.create_model import my_model


### CompNeuroSim is a class to define simulations
### It requires a simulation function, which we will define here:
def set_rates(pop_name: str, rates: float = 0.0, duration: float = 0.0):
    """
    Sets the rates variable of a population given by pop_name and simulates duration ms.

    Args:
        pop_name (str):
            name of the population
        rates (float, optional):
            rates variable of the population
        duration (float, optional):
            duration of the simulation in ms
    """
    ### set rates and simulate
    get_population(pop_name).rates = rates
    simulate(duration)


### Also create a second more complex simulation function
def increase_rates(
    pop_name: str | list[str],
    rate_step: float = 0.0,
    time_step: float = 0.0,
    nr_steps: int = 0,
):
    """
    Increase rates variable of population(s).

    Args:
        pop_name (str or list of str):
            name of population(s)
        rate_step (float, optional):
            increase of rate with each step, initial step = current rates of pop
        time_step (float, optional):
            duration of each step in ms
        nr_steps (int, optional):
            number of steps
    """

    ### convert single pop into list
    pop_name_list = pop_name
    if not (isinstance(pop_name_list, list)):
        pop_name_list = [pop_name_list]

    ### define initial value for rates for each pop (assume all neurons have same rates)
    start_rate_arr = np.array(
        [get_population(pop_name).rates[0] for pop_name in pop_name_list]
    )

    ### simulate all steps
    for step in range(nr_steps):
        ### calculate rates for each pop
        rates_arr = step * rate_step + start_rate_arr
        ### set rates variable of all populations
        for pop_idx, pop_name in enumerate(pop_name_list):
            set_rates(
                pop_name, rates=rates_arr[pop_idx], duration=0
            )  # use already defined simulation set_rates
        ### then simulate step
        set_rates(pop_name_list[0], rates=rates_arr[0], duration=time_step)

    ### simulation_functions can return some information which may be helpful later
    ### the simulation arguments do not need to be returned, since they are accessible
    ### through the CompNeuroSim object anyway (see below)
    return {"duration": time_step * nr_steps, "d_rates": rate_step * nr_steps}


### see below why we need this function
def extend_model(my_model: CompNeuroModel):
    """
    Create a simple projections and a projection with decaying weights.

    Args:
        my_model (CompNeuroModel):
            model to which the projection should be added
    """

    ### create a simple population for later use
    Population(1, neuron=Neuron(equations="r=0"), name="simple_pop")

    ### create a projection with decaying weights to demonstrate recording of projection
    proj = Projection(
        pre=my_model.populations[0],
        post=my_model.populations[1],
        target="ampa",
        synapse=Synapse(parameters="tau=500", equations="dw/dt=-w/tau"),
        name="ampa_proj",
    )
    proj.connect_all_to_all(weights=Uniform(1.0, 2.0))


def main():
    ### create and compile the model from other example "create_model.py"
    my_model.create(do_compile=False)

    ### extend the model to demonstrate the functionality of CompNeuroSim requirements
    ### (see below) and the recording of projections (recorded data will be used in
    ### other example "plot_recordings.py")
    extend_model(my_model)
    my_model.compile()

    ### Define Monitors, recording p and spike from both model populations with periods
    ### of 10 ms and 15 ms and the weights of the ampa projection with period of 10 ms
    monitor_dictionary = {
        f"{my_model.populations[0]};10": ["p", "spike"],
        f"{my_model.populations[1]};15": ["p", "spike"],
        "ampa_proj;10": ["w"],
    }
    mon = CompNeuroMonitors(monitor_dictionary)

    ### Now use CompNeuroSim to define a simulation. Use the previously defined
    ### simulation functions and define their arguments as kwargs dictionary. Give the
    ### simulation a name and description and you can also define requirements for the
    ### simulation. Here, for example, we require that the populations contain the
    ### attribute 'rates'. One can define multiple requirements in a list of
    ### dictionaries. The arguments of the requirements can be inherited from the
    ### simulation kwargs by using the syntax 'simulation_kwargs.<kwarg_name>'.
    ### The monitor object is also given to the simulation, so that the simulation
    ### runs can be automatically associated with the monitor recording chunks.
    increase_rates_pop = CompNeuroSim(
        simulation_function=increase_rates,
        simulation_kwargs={
            "pop_name": my_model.populations[0],
            "rate_step": 10,
            "time_step": 100,
            "nr_steps": 15,
        },
        name="increase_rates_pop",
        description="increase rates variable of pop",
        requirements=[
            {"req": ReqPopHasAttr, "pop": "simulation_kwargs.pop_name", "attr": "rates"}
        ],
        monitor_object=mon,
    )

    ### Now let's use this simulation
    ### Simulate 500 ms without recordings and then run the simulation
    simulate(500)
    mon.start()
    increase_rates_pop.run()

    ### resetting monitors and model, creating new recording chunk
    mon.reset()

    ### again simulate 700 ms without recording
    ### then run the simulation with different simulation kwargs (for all populations)
    mon.pause()
    simulate(700)
    mon.start()
    increase_rates_pop.run({"pop_name": my_model.populations})
    simulate(500)

    ### now again change the pop_name kwarg but use the simple_pop population without
    ### the required attribute 'rates'
    ### this will raise an error
    try:
        increase_rates_pop.run({"pop_name": "simple_pop"})
    except Exception as e:
        print("\n###############################################")
        print(
            "Running simulation with population not containing attribute 'rates' causes the following error:"
        )
        print(e)
        print("###############################################\n")

    ### get recordings and recording times from the CompNeuroMonitors object
    recordings = mon.get_recordings()
    recording_times = mon.get_recording_times()

    ### get the simulation information object from the CompNeuroSim object
    increase_rates_pop_info = increase_rates_pop.simulation_info()

    ### save the recordings, recording times and simulation information
    save_variables(
        variable_list=[recordings, recording_times, increase_rates_pop_info],
        name_list=["recordings", "recording_times", "increase_rates_pop_info"],
        path="run_and_monitor_simulations",
    )

    ### print the information contained in the simulation information object
    print("\nA simulation object contains:")
    print("name\n", increase_rates_pop_info.name)
    print("\ndescription\n", increase_rates_pop_info.description)
    print("\nstart (for each run)\n", increase_rates_pop_info.start)
    print("\nend (for each run)\n", increase_rates_pop_info.end)
    print("\ninfo (for each run)\n", increase_rates_pop_info.info)
    print("\nkwargs (for each run)\n", increase_rates_pop_info.kwargs)
    print("\nmonitor chunk (for each run)\n", increase_rates_pop_info.monitor_chunk)

    return 1


if __name__ == "__main__":
    main()
