from ANNarchy import Neuron

### Izhikevich (2007)-like neuron model templates
### based on: Izhikevich, E. M. (2007). Dynamical Systems in Neuroscience. MIT Press. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.725.8216&rep=rep1&type=pdf

Izhikevich2007 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population # pS * mV**-1
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population # nS
        c      = 0 : population # mV
        d      = 0 : population # pA
        v_peak = 0 : population # mV
        I_app  = 0 # pA
    """,
    equations="""
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_app
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2007",
    description="Simple neuron model equations from Izhikevich (2007).",
)


Izhikevich2007_record_currents = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population # pS * mV**-1
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population # nS
        c      = 0 : population # mV
        d      = 0 : population # pA
        v_peak = 0 : population # mV
        I_app  = 0 # pA
    """,
    equations="""
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_app
        du/dt      = a*(b*(v - v_r) - u)
        I_u = -u
        I_k = k*(v - v_r)*(v - v_t)
        I_a = I_app
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2007_record_currents",
    description="Simple neuron model equations from Izhikevich (2007). The individual currents are separate variable which can be recorded.",
)


Izhikevich2007_voltage_clamp = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population # pS * mV**-1
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population # nS
        c      = 0 : population # mV
        d      = 0 : population # pA
        v_peak = 0 : population # mV
        I_app  = 0 # pA
    """,
    equations="""
        C * dv/dt  = 0
        du/dt      = a*(b*(v - v_r) - u)
        I_inf      = k*(v - v_r)*(v - v_t) - u + I_app
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2007_voltage_clamp",
    description="Simple neuron model equations from Izhikevich (2007). With voltage clamp (variable v can be set and will not change) to record I_inf. I_inf also contains the slow u-current.",
)

Izhikevich2007_syn = Neuron(
    parameters="""
        C      = 0     : population # pF
        k      = 0     : population # pS * mV**-1
        v_r    = 0     : population # mV
        v_t    = 0     : population # mV
        a      = 0     : population # ms**-1
        b      = 0     : population # nS
        c      = 0     : population # mV
        d      = 0     : population # pA
        v_peak = 0     : population # mV
        I_app  = 0     # pA
        tau_ampa = 10  : population # ms
        tau_gaba = 10  : population # ms
        E_ampa   = 0   : population # mV
        E_gaba   = -90 : population # mV
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2007_syn",
    description="Simple neuron model equations from Izhikevich (2007) with conductance-based AMPA and GABA synapses/currents.",
)

Izhikevich2007_noisy_AMPA = Neuron(
    parameters="""
        C              = 0 : population
        k              = 0 : population
        v_r            = 0 : population
        v_t            = 0 : population
        a              = 0 : population
        b              = 0 : population
        c              = 0 : population
        d              = 0 : population
        v_peak         = 0 : population
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
        dg_gaba/dt = -g_gaba/tau_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2007_noisy_AMPA",
    description="Standard neuron model from Izhikevich (2007) with additional conductance based synapses for AMPA and GABA currents with noise in AMPA conductance.",
)

Izhikevich2007_noisy_I = Neuron(
    parameters="""
        C              = 0 : population
        k              = 0 : population
        v_r            = 0 : population
        v_t            = 0 : population
        a              = 0 : population
        b              = 0 : population
        c              = 0 : population
        d              = 0 : population
        v_peak         = 0 : population
        tau_ampa       = 1 : population
        tau_gaba       = 1 : population
        E_ampa         = 0 : population
        E_gaba         = 0 : population
        I_app          = 0
        base_mean       = 0
        base_noise      = 0
        rate_base_noise = 0
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        offset_base = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rate_base_noise, offset_base, Normal(0, 1) * base_noise)
        I_base      = base_mean + offset_base
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba)) + I_base
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2007_noisy_I",
    description="Standard neuron model from Izhikevich (2007) with additional conductance based synapses for AMPA and GABA currents and noisy baseline current.",
)


Izhikevich2007_fsi_noisy_AMPA = Neuron(
    parameters="""
        C              = 0 : population
        k              = 0 : population
        v_r            = 0 : population
        v_t            = 0 : population
        v_b            = 0 : population
        a              = 0 : population
        b              = 0 : population
        c              = 0 : population
        d              = 0 : population
        v_peak         = 0 : population
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
        dg_gaba/dt = -g_gaba/tau_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))
        du/dt      = if v<v_b: -a * u else: a * (b * (v - v_b)**3 - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2007_fsi_noisy_AMPA",
    description="Fast spiking cortical interneuron model from Izhikevich (2007) with additional conductance based synapses with noise in AMPA conductance.",
)


Izhikevich2007_Corbit_FSI_noisy_AMPA = Neuron(
    parameters="""
        C              = 0 : population # myF/cm^2
        k              = 0 : population #
        b_n            = 0 : population #
        a_s            = 0 : population #
        a_n            = 0 : population #
        v_r            = 0 : population # mV
        v_t            = 0 : population # mV
        a              = 0 : population # ms**-1
        b              = 0 : population #
        d              = 0 : population
        c              = 0 : population # mV
        v_peak         = 0 : population # mV
        x              = 0 : population
        tau_ampa       = 1 : population
        tau_gaba       = 1 : population
        E_ampa         = 0 : population
        E_gaba         = 0 : population
        increase_noise = 0 : population
        rates_noise    = 0
        I_app          = 0 # yA/cm^2
    """,
    equations="""
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba/tau_gaba
        I = I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))
      
        C * dv/dt = k*(v - v_r)*(v - v_t) - u - n + ((abs(I))**(1/x))/((I+1e-20)/(abs(I)+ 1e-20))

        du/dt     = a*(b*(v - v_r) - u)
        ds/dt     = a_s*(pos(u)**0.1 - s)
        dn/dt     = a_n*(b_n*(pos(u)**0.1-s) - n)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2007_Corbit_FSI_noisy_AMPA",
    description="Simple neuron model equations from Izhikevich (2007) adjusted version to fit the striatal FSI neuron model from Corbit et al. (2016) should be able to produce late spiking.",
)
# Corbit, V. L., Whalen, T. C., Zitelli, K. T., Crilly, S. Y., Rubin, J. E., & Gittis, A. H. (2016). Pallidostriatal projections promote β oscillations in a dopamine-depleted biophysical network model. Journal of Neuroscience, 36(20), 5556-5571.

Izhikevich2007_Corbit_FSI_noisy_I = Neuron(
    parameters="""
        C               = 0 : population # myF/cm^2
        k               = 0 : population #
        b_n             = 0 : population #
        a_s             = 0 : population #
        a_n             = 0 : population #
        v_r             = 0 : population # mV
        v_t             = 0 : population # mV
        a               = 0 : population # ms**-1
        b               = 0 : population #
        d               = 0 : population
        c               = 0 : population # mV
        v_peak          = 0 : population # mV
        x               = 0 : population
        tau_ampa        = 1 : population
        tau_gaba        = 1 : population
        E_ampa          = 0 : population
        E_gaba          = 0 : population
        I_app           = 0 # yA/cm^2
        base_mean       = 0
        base_noise      = 0
        rate_base_noise = 0
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        offset_base = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rate_base_noise, offset_base, Normal(0, 1) * base_noise)
        I_base      = base_mean + offset_base
        I = I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))
        C * dv/dt = k*(v - v_r)*(v - v_t) - u - n + ((abs(I))**(1/x))/((I+1e-20)/(abs(I)+ 1e-20)) + I_base

        du/dt     = a*(b*(v - v_r) - u)
        ds/dt     = a_s*(pos(u)**0.1 - s)
        dn/dt     = a_n*(b_n*(pos(u)**0.1-s) - n)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2007_Corbit_FSI_noisy_I",
    description="Simple neuron model equations from Izhikevich (2007) adjusted version to fit the striatal FSI neuron model from Corbit et al. (2016) should be able to produce late spiking.",
)

Izhikevich2007_noisy_AMPA_oscillating = Neuron(
    parameters="""
        C              = 0 : population
        k              = 0 : population
        v_r            = 0 : population
        v_t            = 0 : population
        a              = 0 : population
        b              = 0 : population
        c              = 0 : population
        d              = 0 : population
        v_peak         = 0 : population
        tau_ampa       = 1 : population
        tau_gaba       = 1 : population
        E_ampa         = 0 : population
        E_gaba         = 0 : population
        I_app          = 0
        increase_noise = 0 : population
        rates_noise    = 0
        freq           = 0
        amp            = 300
    """,
    equations="""
        osc        = amp * sin(t * 2 * pi * (freq  /1000))
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba/tau_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba)) + osc
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="Izhikevich2007_noisy_AMPA_oscillating",
    description="Standard neuron model from Izhikevich (2007) with additional conductance based synapses for AMPA and GABA currents with noise in AMPA conductance.",
)