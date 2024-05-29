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
)
from CompNeuroPy import (
    CompNeuroMonitors,
    PlotRecordings,
    interactive_plot,
    timing_decorator,
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
        tau_ampa * dg_ampa/dt = -g_ampa + tau_ampa*g_exc/dt
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

neuron_aux1 = Neuron(
    parameters="""
        pre_size = 1 : population
    """,
    equations="""
        r = g_ampa/pre_size
        g_ampa = 0
    """,
)

neuron_aux2 = Neuron(
    parameters="""
        number_synapses = 0
    """,
    equations="""
        ### r = incoming spikes
        r = Normal(0, 1) * sqrt(number_synapses * sum(spikeprob) * (1 - sum(spikeprob))) + number_synapses * sum(spikeprob) : min=0, max=number_synapses
    """,
)


CONNECTION_PROB = 0.1
WEIGHTS = 1.0
POP_PRE_SIZE = 1000
POP_POST_SIZE = 100
POP_REDUCED_SIZE = 100

### create not reduced model
### pre
pop_pre = Population(POP_PRE_SIZE, neuron=PoissonNeuron(rates=10.0), name="pre")
### post
pop_post = Population(POP_POST_SIZE, neuron=neuron_izh, name="post")
### pre to post
proj = Projection(pre=pop_pre, post=pop_post, target="ampa", name="proj")
proj.connect_fixed_probability(weights=WEIGHTS, probability=CONNECTION_PROB)

### create reduced model
### pre
pop_pre2 = Population(POP_REDUCED_SIZE, neuron=PoissonNeuron(rates=10.0), name="pre2")
### post
pop_post2 = Population(POP_REDUCED_SIZE, neuron=neuron_izh_aux, name="post2")
### aux
pop_aux1 = Population(1, neuron=neuron_aux1, name="aux1")
pop_aux1.pre_size = POP_REDUCED_SIZE
pop_aux2 = Population(
    POP_REDUCED_SIZE,
    neuron=neuron_aux2,
    name="aux2",
)
pop_aux2.number_synapses = Binomial(n=POP_PRE_SIZE, p=CONNECTION_PROB).get_values(
    POP_REDUCED_SIZE
)
### pre to aux
proj_pre__aux = Projection(
    pre=pop_pre2, post=pop_aux1, target="ampa", name="proj_pre__aux"
)
proj_pre__aux.connect_all_to_all(weights=1)
### aux2 to aux2
proj_aux__aux = Projection(
    pre=pop_aux1, post=pop_aux2, target="spikeprob", name="proj_aux__aux"
)
proj_aux__aux.connect_all_to_all(weights=1)
### aux to post
proj_aux__pre = CurrentInjection(pop_aux2, pop_post2, "exc")
proj_aux__pre.connect_current()


monitors = CompNeuroMonitors(
    mon_dict={
        "pre": ["spike"],
        "post": ["v", "spike", "I_ampa", "g_ampa"],
        "pre2": ["spike"],
        "post2": ["v", "spike", "I_ampa", "g_ampa", "g_exc"],
        "aux1": ["r"],
        "aux2": ["r"],
    }
)

compile()

monitors.start()

start = time.time()
simulate(100.0)
pop_pre.rates = 1000.0
pop_pre2.rates = 1000.0
simulate(100.0)
print("simulate time:", time.time() - start)

recordings = monitors.get_recordings()
recording_times = monitors.get_recording_times()

PlotRecordings(
    figname="test.png",
    recordings=recordings,
    recording_times=recording_times,
    shape=(4, 2),
    plan={
        "position": [1, 3, 5, 2, 4, 6, 8],
        "compartment": ["pre", "post", "post", "pre2", "post2", "post2", "aux1"],
        "variable": ["spike", "spike", "g_ampa", "spike", "spike", "g_ampa", "r"],
        "format": [
            "hybrid",
            "hybrid",
            "line_mean",
            "hybrid",
            "hybrid",
            "line_mean",
            "line",
        ],
    },
)

### compare incoming spikes, i.e. the r of aux 2 and the incoming spikes of post
### idx: [neuron][nr_spikes]
pre_spikes = recordings[0]["pre;spike"]
incoming_spikes_dict = {}

### loop over post neuron dendrites (only first post neuron)
for post_dendrite in proj:
    incoming_spikes_dict[post_dendrite.post_rank] = []
    if post_dendrite is None:
        continue
    ### if post neuron has incoming synapses, loop over pre neurons
    for pre_neuron in post_dendrite:
        incoming_spikes_dict[post_dendrite.post_rank].extend(
            pre_spikes[pre_neuron.rank]
        )
    ### sort incoming spikes
    incoming_spikes_dict[post_dendrite.post_rank] = np.sort(
        incoming_spikes_dict[post_dendrite.post_rank]
    )
    break

### get histogram of incoming spikes to get the sum of incoming spikes for each timestep
incoming_spikes_sum, time_step_arr = np.histogram(
    incoming_spikes_dict[0], bins=np.arange(0, 2000, 1)
)

plt.figure()
plt.plot(
    time_step_arr[:-1], incoming_spikes_sum, label="incoming spikes real", alpha=0.5
)
plt.plot(
    time_step_arr,
    recordings[0]["aux2;r"][:, 0],
    label="incoming spikes aux2",
    alpha=0.5,
)
plt.legend()
plt.tight_layout()
plt.show()
