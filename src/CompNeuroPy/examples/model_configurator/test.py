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
    timing_decorator,
    annarchy_compiled,
    CompNeuroModel,
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
from CompNeuroPy.examples.model_configurator.reduce_model import _CreateReducedModel

setup(dt=0.1)


CONNECTION_PROB = 0.6
WEIGHTS = 0.1
POP_PRE_SIZE = 100
POP_POST_SIZE = 100
POP_REDUCED_SIZE = 100


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


def create_model():
    ### create not reduced model
    ### pre
    pop_pre1 = Population(
        POP_PRE_SIZE, neuron=PoissonNeuron(rates=10.0), name="pop_pre1"
    )
    pop_pre2 = Population(
        POP_PRE_SIZE, neuron=PoissonNeuron(rates=10.0), name="pop_pre2"
    )
    ### post
    pop_post = Population(POP_POST_SIZE, neuron=neuron_izh, name="pop_post")
    ### pre to post
    proj_pre1__post = Projection(
        pre=pop_pre1, post=pop_post, target="ampa", name="proj_pre1__post"
    )
    proj_pre1__post.connect_fixed_probability(
        weights=WEIGHTS, probability=CONNECTION_PROB
    )
    proj_pre2__post = Projection(
        pre=pop_pre2, post=pop_post, target="ampa", name="proj_pre2__post"
    )
    proj_pre2__post.connect_fixed_probability(
        weights=WEIGHTS, probability=CONNECTION_PROB
    )


if __name__ == "__main__":
    ### run normal model
    print("normal model")
    ### create model
    model = CompNeuroModel(model_creation_function=create_model)
    ### create monitors
    mon_dict = {
        "pop_pre1": ["spike"],
        "pop_pre2": ["spike"],
        "pop_post": ["spike", "g_ampa"],
    }
    monitors = CompNeuroMonitors(
        mon_dict=mon_dict,
    )
    monitors.start()
    ### run simulation
    start = time.time()
    simulate(50.0)
    get_population("pop_pre1").rates = 30.0
    simulate(50.0)
    get_population("pop_pre2").rates = 30.0
    simulate(100.0)
    print("simulate time:", time.time() - start)
    recordings_normal = monitors.get_recordings()
    recording_times_normal = monitors.get_recording_times()

    ### run reduced model
    print("reduced model")
    ### create model
    model = _CreateReducedModel(
        model=model, reduced_size=POP_REDUCED_SIZE, do_create=True, do_compile=True
    ).model_reduced
    ### create monitors
    mon_dict = {
        "pop_pre1_reduced": ["spike"],
        "pop_pre2_reduced": ["spike"],
        "pop_post_reduced": ["spike", "g_ampa"],
        "pop_pre1_spike_collecting_aux": ["r"],
        "pop_pre2_spike_collecting_aux": ["r"],
        "pop_post_ampa_aux": [
            "incoming_spikes_proj_pre1__post",
            "incoming_spikes_proj_pre2__post",
            "r",
        ],
    }
    monitors = CompNeuroMonitors(
        mon_dict=mon_dict,
    )
    monitors.start()
    ### run simulation
    start = time.time()
    simulate(50.0)
    get_population("pop_pre1_reduced").rates = 30.0
    simulate(50.0)
    get_population("pop_pre2_reduced").rates = 30.0
    simulate(100.0)
    print("simulate time:", time.time() - start)
    recordings_reduced = monitors.get_recordings()
    recording_times_reduced = monitors.get_recording_times()

    ### plot
    PlotRecordings(
        figname="test_normal.png",
        recordings=recordings_normal,
        recording_times=recording_times_normal,
        shape=(4, 3),
        plan={
            "position": [1, 4, 7, 10],
            "compartment": [
                "pop_pre1",
                "pop_pre2",
                "pop_post",
                "pop_post",
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
                "line",
            ],
        },
    )

    PlotRecordings(
        figname="test_reduced.png",
        recordings=recordings_reduced,
        recording_times=recording_times_reduced,
        shape=(4, 3),
        plan={
            "position": [2, 5, 8, 11, 3, 6, 9, 12],
            "compartment": [
                "pop_pre1_reduced",
                "pop_pre2_reduced",
                "pop_post_reduced",
                "pop_post_reduced",
                "pop_pre1_spike_collecting_aux",
                "pop_pre2_spike_collecting_aux",
                "pop_post_ampa_aux",
                "pop_post_ampa_aux",
            ],
            "variable": [
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
                "line",
                "line_mean",
                "line_mean",
                "line_mean",
                "line_mean",
            ],
        },
    )
