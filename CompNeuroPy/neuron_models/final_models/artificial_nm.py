from ANNarchy import Neuron


### artificial neuron models


integrator_neuron = Neuron(
    parameters="""
        tau       = 1 : population
        threshold = 0 : population
        neuron_id = 0
    """,
    equations="""
        dg_ampa/dt = - g_ampa / tau
        ddecision/dt = 0
    """,
    spike="""
        g_ampa >= threshold
    """,
    reset="""
        decision = neuron_id
    """,
    name="integrator_neuron",
    description="Integrator Neuron, which integrates incoming spikes with value g_ampa and emits a spike when reaching a threshold. After spike decision changes, which can be used as a stop condition",
)


integrator_neuron_simple = Neuron(
    parameters="""
        tau       = 1 : population
        neuron_id = 0
    """,
    equations="""
        dg_ampa/dt = - g_ampa / tau
        r = 0
    """,
    name="integrator_neuron_simple",
    description="Integrator Neuron, which integrates incoming spikes with value g_ampa, which can be used as a stop condition",
)


poisson_neuron = Neuron(
    parameters="""
        rates   = 0
    """,
    equations="""
        p       = Uniform(0.0, 1.0) * 1000.0 / dt
    """,
    spike="""
        p <= rates
    """,
    reset="""
        p = 0.0
    """,
    name="poisson_neuron",
    description="Poisson neuron whose rate can be specified and is reached instantaneous.",
)


poisson_neuron_up_down = Neuron(
    parameters="""
        rates   = 0
        tau_up   = 1 : population
        tau_down = 1 : population
    """,
    equations="""
        p       = Uniform(0.0, 1.0) * 1000.0 / dt
        dact/dt = if (rates - act) > 0:
                      (rates - act) / tau_up
                  else:
                      (rates - act) / tau_down
    """,
    spike="""
        p <= act
    """,
    reset="""
        p = 0.0
    """,
    name="poisson_neuron_up_down",
    description="Poisson neuron whose rate can be specified and is reached with time constants tau_up and tau_down.",
)


poisson_neuron_sin = Neuron(
    parameters="""
        amplitude = 0 # in Hz
        base = 0 # in Hz
        frequency = 0 # in Hz
        phase = 0 # in sec
    """,
    equations="""
        rates = amplitude * sin(((2*pi)/frequency)*(t/1000-phase)) + base
        p     = Uniform(0.0, 1.0) * 1000.0 / dt
    """,
    spike="""
        p <= rates
    """,
    reset="""
        p = 0.0
    """,
    name="poisson_neuron_sin",
    description="Poisson neuron whose rate varies with a sinus function.",
)
