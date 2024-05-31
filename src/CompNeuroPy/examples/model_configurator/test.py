from ANNarchy import (
    Neuron,
    Population,
    compile,
    get_time,
    setup,
    dt,
    Projection,
    Synapse,
    Binomial,
    get_projection,
    get_population,
    CurrentInjection,
    simulate,
    projections,
    populations,
)
from CompNeuroPy import (
    CompNeuroMonitors,
    PlotRecordings,
    interactive_plot,
    timing_decorator,
    annarchy_compiled,
)
from CompNeuroPy.neuron_models import PoissonNeuron
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.stats import binom
from functools import wraps
import time
from collections.abc import Iterable
from tqdm import tqdm
from math import ceil

setup(dt=0.1)


neuron_izh = Neuron(
    parameters="""
        C = 100.0 : population
        k = 0.7 : population
        v_r = -60.0 : population
        v_t = -40.0 : population
        a = 0.03 : population
        b = -2.0 : population
        c = -50.0 : population
        d = 100.0 : population
        v_peak = 0.0 : population
        I_app = 0.0
        E_ampa = 0.0 : population
        tau_ampa = 10.0 : population
    """,
    equations="""
        ### synaptic current
        tau_ampa * dg_ampa/dt = -g_ampa
        I_ampa = -neg(g_ampa*(v - E_ampa))
        ### Izhikevich spiking
        I_v        = I_app + I_ampa
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_v
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
)


neuron_izh_aux = Neuron(
    parameters="""
        C = 100.0 : population
        k = 0.7 : population
        v_r = -60.0 : population
        v_t = -40.0 : population
        a = 0.03 : population
        b = -2.0 : population
        c = -50.0 : population
        d = 100.0 : population
        v_peak = 0.0 : population
        I_app = 0.0
        E_ampa = 0.0 : population
        tau_ampa = 10.0 : population
    """,
    equations="""
        ### synaptic current
        tau_ampa * dg_ampa/dt = -g_ampa + tau_ampa*g_ampaaux/dt
        I_ampa = -neg(g_ampa*(v - E_ampa))
        ### Izhikevich spiking
        I_v        = I_app + I_ampa
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_v
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
)


class SpikeProbCalcNeuron(Neuron):
    def __init__(self, reduced_size=1):
        parameters = f"""
            reduced_size = {reduced_size} : population
            tau= 1.0 : population
        """
        equations = """
            tau*dr/dt = g_ampa/reduced_size - r
            g_ampa = 0
        """
        super().__init__(parameters=parameters, equations=equations)


class InputCalcNeuron(Neuron):
    def __init__(self, projection_dict):
        """
        This neurons get the spike probabilities of the pre neurons and calculates the
        incoming spikes for each projection. It accumulates the incoming spikes of all
        projections (of the same target type) and calculates the conductance increase
        for the post neuron.

        Args:
            projection_dict (dict):
                keys: names of afferent projections (of the same target type)
                values: dict with keys "weights", "pre_name"
        """

        ### create parameters
        parameters = [
            f"""
            number_synapses_{proj_name} = 0
            weights_{proj_name} = {vals['weights']}
        """
            for proj_name, vals in projection_dict.items()
        ]
        parameters = "\n".join(parameters)

        ### create equations
        equations = [
            f"""
            incoming_spikes_{proj_name} = number_synapses_{proj_name} * sum(spikeprob_{vals['pre_name']}) + Normal(0, 1)*sqrt(number_synapses_{proj_name} * sum(spikeprob_{vals['pre_name']}) * (1 - sum(spikeprob_{vals['pre_name']}))) : min=0, max=number_synapses_{proj_name}
        """
            for proj_name, vals in projection_dict.items()
        ]
        equations = "\n".join(equations)
        sum_of_conductance_increase = (
            "r = "
            + "".join(
                [
                    f"incoming_spikes_{proj_name} * weights_{proj_name} + "
                    for proj_name in projection_dict.keys()
                ]
            )[:-3]
        )
        equations = equations + "\n" + sum_of_conductance_increase

        super().__init__(parameters=parameters, equations=equations)


CONNECTION_PROB = 0.1
WEIGHTS = 0.1
POP_PRE_SIZE = 50
POP_POST_SIZE = 50
POP_REDUCED_SIZE = 100
REDUCED_MODEL = True


### create not reduced model
### pre
pop_pre1 = Population(POP_PRE_SIZE, neuron=PoissonNeuron(rates=10.0), name="pop_pre1")
pop_pre2 = Population(POP_PRE_SIZE, neuron=PoissonNeuron(rates=10.0), name="pop_pre2")
### post
pop_post = Population(POP_POST_SIZE, neuron=neuron_izh, name="pop_post")
### pre to post
proj_pre1__post = Projection(
    pre=pop_pre1, post=pop_post, target="ampa", name="proj_pre1__post"
)
proj_pre1__post.connect_fixed_probability(weights=WEIGHTS, probability=CONNECTION_PROB)
proj_pre2__post = Projection(
    pre=pop_pre2, post=pop_post, target="ampa", name="proj_pre2__post"
)
proj_pre2__post.connect_fixed_probability(weights=WEIGHTS, probability=CONNECTION_PROB)


def create_reduced_pop(pop: Population):
    ### TODO in ReduceModel class the initial arguments of the populaitons can be
    ### obtained with the population names, here just use size, neuron and name
    print(f"create_reduced_pop for {pop.name}")
    if pop.name == "pop_post":
        Population(
            min([POP_REDUCED_SIZE, pop.size]),
            neuron=neuron_izh_aux,
            name=pop.name + "_reduced",
        )
    else:
        Population(
            min([POP_REDUCED_SIZE, pop.size]),
            neuron=pop.neuron_type,  # TODO neuron type in reuced model has to be different
            name=pop.name + "_reduced",
        )


def create_spike_collecting_aux_pop(pop: Population, projection_list: list[Projection]):
    ### get all efferent projections
    efferent_projection_list = [proj for proj in projection_list if proj.pre == pop]
    ### check if pop has efferent projections
    if len(efferent_projection_list) == 0:
        return
    print(f"create_spike_collecting_aux_pop for {pop.name}")
    ### create the spike collecting population
    pop_aux = Population(
        1,
        neuron=SpikeProbCalcNeuron(reduced_size=min([POP_REDUCED_SIZE, pop.size])),
        name=f"{pop.name}_spike_collecting_aux",
    )
    ### create the projection from reduced pop to spike collecting aux pop
    proj = Projection(
        pre=get_population(pop.name + "_reduced"),
        post=pop_aux,
        target="ampa",
        name=f"proj_{pop.name}_spike_collecting_aux",
    )
    proj.connect_all_to_all(weights=1)


def create_conductance_aux_pop(
    pop: Population, projection_list: list[Projection], target: str
):
    ### get all afferent projections
    afferent_projection_list = [proj for proj in projection_list if proj.post == pop]
    ### check if pop has afferent projections
    if len(afferent_projection_list) == 0:
        return
    ### get all afferent projections with target type
    afferent_target_projection_list = [
        proj for proj in afferent_projection_list if proj.target == target
    ]
    ### check if there are afferent projections with target type
    if len(afferent_target_projection_list) == 0:
        return
    print(f"create_conductance_aux_pop for {pop.name} target {target}")
    ### get projection informations TODO in ReduceModel class weights and probs not global constants
    projection_dict = {
        proj.name: {
            "pre_size": proj.pre.size,
            "connection_prob": CONNECTION_PROB,
            "weights": WEIGHTS,
            "pre_name": proj.pre.name,
        }
        for proj in afferent_target_projection_list
    }
    ### create the conductance calculating population
    pop_aux = Population(
        pop.size,
        neuron=InputCalcNeuron(projection_dict=projection_dict),
        name=f"{pop.name}_{target}_aux",
    )
    ### set number of synapses parameter for each projection
    for proj_name, vals in projection_dict.items():
        number_synapses = Binomial(
            n=vals["pre_size"], p=vals["connection_prob"]
        ).get_values(pop.size)
        setattr(pop_aux, f"number_synapses_{proj_name}", number_synapses)
    ### create the "current injection" projection from conductance calculating
    ### population to the reduced post population
    proj = CurrentInjection(
        pre=pop_aux,
        post=get_population(f"{pop.name}_reduced"),
        target=f"{target}aux",
        name=f"proj_{pop.name}_{target}_aux",
    )
    proj.connect_current()
    ### create projection from spike_prob calculating aux neurons of presynaptic
    ### populations to conductance calculating aux population
    for proj in afferent_target_projection_list:
        pre_pop = proj.pre
        pre_pop_spike_collecting_aux = get_population(
            f"{pre_pop.name}_spike_collecting_aux"
        )
        proj = Projection(
            pre=pre_pop_spike_collecting_aux,
            post=pop_aux,
            target=f"spikeprob_{pre_pop.name}",
            name=f"{proj.name}_spike_collecting_to_conductance",
        )
        proj.connect_all_to_all(weights=1)


if REDUCED_MODEL:
    ### create reduced model
    population_list = populations().copy()
    projection_list = projections().copy()
    ### for each population create a reduced population
    for pop in population_list:
        create_reduced_pop(pop)
    ### for each population which is a presynaptic population, create a spikes collecting aux population
    for pop in population_list:
        create_spike_collecting_aux_pop(pop, projection_list)
    ## for each population which has afferents create a population for incoming spikes for each target type
    for pop in population_list:
        create_conductance_aux_pop(pop, projection_list, target="ampa")
        create_conductance_aux_pop(pop, projection_list, target="gaba")

if REDUCED_MODEL:
    mon_dict = {
        pop_pre1.name: ["spike"],
        pop_pre2.name: ["spike"],
        pop_post.name: ["spike", "g_ampa"],
        f"{pop_pre1.name}_reduced": ["spike"],
        f"{pop_pre2.name}_reduced": ["spike"],
        f"{pop_post.name}_reduced": ["spike", "g_ampa"],
        f"{pop_pre1.name}_spike_collecting_aux": ["r"],
        f"{pop_pre2.name}_spike_collecting_aux": ["r"],
        f"{pop_post.name}_ampa_aux": [
            "incoming_spikes_proj_pre1__post",
            "incoming_spikes_proj_pre2__post",
            "r",
        ],
    }
else:
    mon_dict = {
        pop_pre1.name: ["spike"],
        pop_pre2.name: ["spike"],
        pop_post.name: ["spike", "g_ampa"],
    }

monitors = CompNeuroMonitors(
    mon_dict=mon_dict,
)

compile()

monitors.start()

start = time.time()
simulate(50.0)
pop_pre1.rates = 1000.0
if REDUCED_MODEL:
    get_population(f"{pop_pre1.name}_reduced").rates = 1000.0
simulate(50.0)
pop_pre2.rates = 1000.0
if REDUCED_MODEL:
    get_population(f"{pop_pre2.name}_reduced").rates = 1000.0
simulate(100.0)
print("simulate time:", time.time() - start)

recordings = monitors.get_recordings()
recording_times = monitors.get_recording_times()

if REDUCED_MODEL:
    PlotRecordings(
        figname="test.png",
        recordings=recordings,
        recording_times=recording_times,
        shape=(4, 3),
        plan={
            "position": [1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12],
            "compartment": [
                pop_pre1.name,
                pop_pre2.name,
                pop_post.name,
                pop_post.name,
                f"{pop_pre1.name}_reduced",
                f"{pop_pre2.name}_reduced",
                f"{pop_post.name}_reduced",
                f"{pop_post.name}_reduced",
                f"{pop_pre1.name}_spike_collecting_aux",
                f"{pop_pre2.name}_spike_collecting_aux",
                f"{pop_post.name}_ampa_aux",
                f"{pop_post.name}_ampa_aux",
            ],
            "variable": [
                "spike",
                "spike",
                "spike",
                "g_ampa",
                "spike",
                "spike",
                "spike",
                "g_ampa",
                "r",
                "r",
                "incoming_spikes_proj_pre1__post",
                "incoming_spikes_proj_pre2__post",
            ],
            "format": [
                "hybrid",
                "hybrid",
                "hybrid",
                "line_mean",
                "hybrid",
                "hybrid",
                "hybrid",
                "line_mean",
                "line_mean",
                "line_mean",
                "line_mean",
                "line_mean",
            ],
        },
    )
else:
    PlotRecordings(
        figname="test.png",
        recordings=recordings,
        recording_times=recording_times,
        shape=(4, 2),
        plan={
            "position": [1, 3, 5, 7],
            "compartment": [
                pop_pre1.name,
                pop_pre2.name,
                pop_post.name,
                pop_post.name,
            ],
            "variable": [
                "spike",
                "spike",
                "spike",
                "g_ampa",
            ],
            "format": [
                "hybrid",
                "hybrid",
                "hybrid",
                "line_mean",
            ],
        },
    )
