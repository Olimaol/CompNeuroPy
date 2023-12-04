from ANNarchy import Neuron

### Izhikevich (2003)-like neuron model templates
### based on: Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6), 1569–1572. https://doi.org/10.1109/TNN.2003.820440


Izhikevich2003_noisy_AMPA = Neuron(
    parameters="""
        a              = 0 : population
        b              = 0 : population
        c              = 0 : population
        d              = 0 : population
        tau_ampa       = 1 : population
        tau_gaba       = 1 : population
        E_ampa         = 0 : population
        E_gaba         = 0 : population
        I_app          = 0
        increase_noise = 0 : population
        rates_noise    = 0
    """,
    equations="""
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba / tau_gaba
        dv/dt      = 0.04 * v * v + 5 * v + 140 - u + I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))
        du/dt      = a * (b * v - u)
    """,
    spike="""
        v >= 30
    """,
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2003_noisy_AMPA",
    description="Standard neuron model from Izhikevich (2003) with additional conductance based synapses for AMPA and GABA currents with noise in AMPA conductance.",
)


Izhikevich2003_flexible_noisy_AMPA = Neuron(
    parameters="""
        a              = 0 : population
        b              = 0 : population
        c              = 0 : population
        d              = 0 : population
        n2             = 0 : population
        n1             = 0 : population
        n0             = 0 : population
        tau_ampa       = 1 : population
        tau_gaba       = 1 : population
        E_ampa         = 0 : population
        E_gaba         = 0 : population
        I_app          = 0
        increase_noise = 0 : population
        rates_noise    = 0
    """,
    equations="""
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba / tau_gaba
        dv/dt      = n2 * v * v + n1 * v + n0 - u + I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))
        du/dt      = a * (b * v - u)
    """,
    spike="""
        v >= 30
    """,
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2003_flexible_noisy_AMPA",
    description="Flexible neuron model from Izhikevich (2003). Flexible means, the 3 factors of the quadratic equation can be changed. With additional conductance based synapses for AMPA and GABA currents with noise in AMPA conductance.",
)

Izhikevich2003_flexible_noisy_AMPA_nonlin = Neuron(
    parameters="""
        a              = 0 : population
        b              = 0 : population
        c              = 0 : population
        d              = 0 : population
        n2             = 0 : population
        n1             = 0 : population
        n0             = 0 : population
        tau_ampa       = 1 : population
        tau_gaba       = 1 : population
        E_ampa         = 0 : population
        E_gaba         = 0 : population
        I_app          = 0
        increase_noise = 0 : population
        rates_noise    = 0
        nonlin         = 1 : population
    """,
    equations="""
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba / tau_gaba
        I = I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))
        dv/dt      = n2 * v * v + n1 * v + n0 - u + f(I,nonlin)
        du/dt      = a * (b * v - u)
    """,
    spike="""
        v >= 30
    """,
    reset="""
        v = c
        u = u + d
    """,
    functions="""
        f(x,y)=((abs(x))**(1/y))/((x+1e-20)/(abs(x)+ 1e-20))
    """,
    name="Izhikevich2003_flexible_noisy_AMPA_nonlin",
    description="Flexible neuron model from Izhikevich (2003). Flexible means, the 3 factors of the quadratic equation can be changed. With additional conductance based synapses for AMPA and GABA currents with noise in AMPA conductance. With nonlinear function for external current.",
)

Izhikevich2003_flexible_noisy_I_nonlin = Neuron(
    parameters="""
        a               = 0 : population
        b               = 0 : population
        c               = 0 : population
        d               = 0 : population
        n2              = 0 : population
        n1              = 0 : population
        n0              = 0 : population
        tau_ampa        = 1 : population
        tau_gaba        = 1 : population
        E_ampa          = 0 : population
        E_gaba          = 0 : population
        I_app           = 0
        base_mean       = 0
        base_noise      = 0
        rate_base_noise = 0
        nonlin          = 1 : population
    """,
    equations="""
        dg_ampa/dt  = -g_ampa/tau_ampa
        dg_gaba/dt  = -g_gaba / tau_gaba
        offset_base = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rate_base_noise, offset_base, Normal(0, 1) * base_noise)
        I_base      = base_mean + offset_base
        I           = I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))
        dv/dt       = n2 * v * v + n1 * v + n0 - u + f(I,nonlin) + I_base
        du/dt       = a * (b * v - u)
    """,
    spike="""
        v >= 30
    """,
    reset="""
        v = c
        u = u + d
    """,
    functions="""
        f(x,y)=((abs(x))**(1/y))/((x+1e-20)/(abs(x)+ 1e-20))
    """,
    name="Izhikevich2003_flexible_noisy_I_nonlin",
    description="Flexible neuron model from Izhikevich (2003). Flexible means, the 3 factors of the quadratic equation can be changed. With additional conductance based synapses for AMPA and GABA currents and a noisy baseline current. With nonlinear function for external current.",
)

Izhikevich2003_flexible_noisy_I = Neuron(
    parameters="""
        a               = 0 : population
        b               = 0 : population
        c               = 0 : population
        d               = 0 : population
        n2              = 0 : population
        n1              = 0 : population
        n0              = 0 : population
        tau_ampa        = 1 : population
        tau_gaba        = 1 : population
        E_ampa          = 0 : population
        E_gaba          = 0 : population
        I_app           = 0
        base_mean       = 0
        base_noise      = 0
        rate_base_noise = 0
    """,
    equations="""
        dg_ampa/dt  = -g_ampa/tau_ampa
        dg_gaba/dt  = -g_gaba / tau_gaba
        offset_base = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rate_base_noise, offset_base, Normal(0, 1) * base_noise)
        I_base      = base_mean + offset_base
        I           = I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba)) + I_base
        dv/dt       = n2 * v * v + n1 * v + n0 - u + I
        du/dt       = a * (b * v - u)
    """,
    spike="""
        v >= 30
    """,
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2003_flexible_noisy_I",
    description="Flexible neuron model from Izhikevich (2003). Flexible means, the 3 factors of the quadratic equation can be changed. With additional conductance based synapses for AMPA and GABA currents and a noisy baseline current.",
)

Izhikevich2003_flexible_noisy_AMPA_oscillating = Neuron(
    parameters="""
        a              = 0 : population
        b              = 0 : population
        c              = 0 : population
        d              = 0 : population
        n2             = 0 : population
        n1             = 0 : population
        n0             = 0 : population
        tau_ampa       = 1 : population
        tau_gaba       = 1 : population
        E_ampa         = 0 : population
        E_gaba         = 0 : population
        I_app          = 0
        increase_noise = 0 : population
        rates_noise    = 0
        freq           = 0
        amp            = 6
    """,
    equations="""
        osc        = amp * sin(t * 2 * pi * (freq / 1000))
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba / tau_gaba
        dv/dt      = n2 * v * v + n1 * v + n0 - u + I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba)) + osc
        du/dt      = a * (b * v - u)
    """,
    spike="""
        v >= 30
    """,
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2003_flexible_noisy_AMPA_oscillating",
    description="Flexible neuron model from Izhikevich (2003). Flexible means, the 3 factors of the quadratic equation can be changed. With additional conductance based synapses for AMPA and GABA currents with noise in AMPA conductance. With additional oscillation term",
)
