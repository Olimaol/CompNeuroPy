from ANNarchy import Neuron

_fit_Bogacz = Neuron(
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
        x              = 1 : population
        R_input_megOhm = 1 : population
    """,
    equations="""
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba/tau_gaba
        I = I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))

        dv/dt      = n2 * v * v + n1 * v + n0 - u + R_input_megOhm*((abs(I))**(1/x))/((I+1e-20)/(abs(I)+ 1e-20))
        du/dt      = a * (b * v - u)
    """,
    spike="""
        v >= 0
    """,
    reset="""
        v = c
        u = u + d
    """,
    name="_fit_Bogacz",
    description="Model based on Izhikevich (2003). To fit GPe Proto and Arky from Bogacz et al. (2016).",
)


_fit_Bogacz_2 = Neuron(
    parameters="""
        C        = 0 : population # pF
        k        = 0 : population # pS * mV**-1
        v_r      = 0 : population # mV
        v_t      = 0 : population # mV
        a        = 0 : population # ms**-1
        b        = 0 : population # nS
        c        = 0 : population # mV
        d        = 0 : population # pA
        v_peak   = 0 : population # mV
        E_ampa   = 0 : population # mV
        E_gaba   = 0 : population # mV
        tau_ampa = 1 : population # ms
        tau_gaba = 1 : population # ms
        I_app           = 0 # pA
        base_mean       = 0 # pA
        base_noise      = 0 # pA
        rate_base_noise = 0 # s**-1
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        offset_base = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rate_base_noise, offset_base, Normal(0, 1) * base_noise)
        I_base      = base_mean + offset_base
        I           = I_app - neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba)) + I_base
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u*pos(v_peak - v) + I
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="_fit_Bogacz_2",
    description="Model based on Izhikevich (2007). To fit GPe Proto and Arky from Bogacz et al. (2016).",
)
