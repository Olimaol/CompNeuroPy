from CompNeuroPy import ann

"""
Presented neuron models for SPNs and FSI with dopamine modlulation from https://doi.org/10.1016/j.neunet.2009.07.018
"""

parameters_spn_dict = {
    "C": 50,  # nF, paper mentions pF but that does not fit to the other units
    "b": -20,  # nS
    "c": -55,  # mV
    "v_r": -80,  # mV
    "v_peak": 40,  # mV
    "k": 1.14,  # pA/mV**2
    "v_t": -33.8,  # mV
    "a": 0.05,  # ms**-1
    "d": 377,  # pA
    "alpha": 0.03,
    "g_da": 22.7,  # nS
    "E_da": -68.4,  # mV
    "beta_1": 3.75,
    "beta_2": 0.156,
    "E_ampa": 0,  # mV
    "E_nmda": 0,  # mV
    "E_gaba": -60,  # mV
    "tau_ampa": 6,  # ms
    "tau_nmda": 160,  # ms
    "tau_gaba": 4,  # ms
}

parameters_fsi_dict = {
    "a": 0.2,  # ms**-1
    "b": 0.025,  # nS
    "d": 0,  # pA
    "k": 1,  # pA/mV**2
    "v_peak": 25,  # mV
    "v_b": -55,  # mV
    "C": 80,  # nF, paper mentions pF but that does not fit to the other units
    "c": -60,  # mV
    "v_r": -70,  # mV
    "v_t": -50,  # mV
    "eta": 0.1,
    "epsilon": 0.625,
    "E_ampa": 0,  # mV
    "E_gaba": -60,  # mV
    "tau_ampa": 6,  # ms
    "tau_gaba": 4,  # ms
}

current_spn_equations = """
    tau_ampa * dg_ampa/dt = -g_ampa + g_glut
    tau_nmda * dg_nmda/dt = -g_nmda + g_glut
    tau_gaba * dg_gaba/dt = -g_gaba
    B = 1 / (1 + 0.28 * exp(-0.062 * v)) # MG2+/3.57 --> 0.28
"""

current_D1_equations = """
    I_v = g_ampa * (v - E_ampa) + g_nmda * B * (v - E_nmda) * (1 + beta_1 * phi_1) + g_gaba * (v - E_gaba) + I_base
"""

current_D2_equations = """
    I_v = g_ampa * (v - E_ampa) * (1 - beta_2 * phi_2) + g_nmda * B * (v - E_nmda) + g_gaba * (v - E_gaba) + I_base
"""

current_fsi_equations = """
    tau_ampa * dg_ampa/dt = -g_ampa
    tau_gaba * dg_gaba/dt = -g_gaba
    I_v = g_ampa * (v - E_ampa) + g_gaba * (v - E_gaba) * (1 - epsilon * phi_2) + I_base
"""

_izhikevich2007_humphries_2009_spn_d1 = ann.Neuron(
    parameters="""
        # synaptic current parameters
        tau_ampa = 'tau_ampa'    : population
        tau_nmda = 'tau_nmda'    : population
        tau_gaba = 'tau_gaba'    : population
        E_ampa   = 'E_ampa'    : population
        E_nmda   = 'E_nmda'    : population
        E_gaba   = 'E_gaba'  : population
        I_base  = 0 : population

        # neuron model parameters
        C      = 'C'    : population
        k      = 'k'    : population
        v_r    = 'v_r'   : population
        v_t    = 'v_t'   : population
        c_da   = 'g_da'  : population
        E_da   = 'E_da'  : population
        a      = 'a'     : population
        b      = 'b'     : population
        c      = 'c'     : population
        d      = 'd'     : population
        v_peak = 'v_peak' : population

        # dopamine modulation parameter
        phi_1    = 0     : population # D1 receptor activation level
        beta_1     = 'beta_1' : population
    """,
    equations=current_spn_equations
    + current_D1_equations
    + """
        C * dv/dt = k * (v - v_r) * (v - v_t) - u + I_v + phi_1 * c_da * (v - E_da)
        du/dt     = a * (b * (v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="_Izhikevich2007_Humphries_2009_SPN_D1",
    description="Based on Izhikevich2007 and Humphries2009 model of D1 SPN neurons.",
    extra_values=parameters_spn_dict,
)

_izhikevich2007_humphries_2009_spn_d2 = ann.Neuron(
    parameters="""
        # synaptic current parameters
        tau_ampa = 'tau_ampa'    : population
        tau_nmda = 'tau_nmda'    : population
        tau_gaba = 'tau_gaba'    : population
        E_ampa   = 'E_ampa'    : population
        E_nmda   = 'E_nmda'    : population
        E_gaba   = 'E_gaba'  : population
        I_base  = 0 : population

        # neuron model parameters
        C      = 'C'    : population
        k      = 'k'    : population
        v_r    = 'v_r'   : population
        v_t    = 'v_t'   : population
        a      = 'a'     : population
        b      = 'b'     : population
        c      = 'c'     : population
        d      = 'd'     : population
        v_peak = 'v_peak' : population

        # dopamine modulation parameter
        phi_2    = 0     : population # D2 receptor activation level
        alpha   = 'alpha' : population
        beta_2     = 'beta_2' : population
    """,
    equations=current_spn_equations
    + current_D2_equations
    + """
        C * dv/dt = k * (1 - alpha * phi_2) * (v - v_r) * (v - v_t) - u + I_v
        du/dt     = a * (b * (v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="_Izhikevich2007_Humphries_2009_SPN_D2",
    description="Based on Izhikevich2007 and Humphries2009 model of D2 SPN neurons.",
    extra_values=parameters_spn_dict,
)

_izhikevich2007_humphries_2009_fsi = ann.Neuron(
    parameters="""
        # synaptic current parameters
        tau_ampa = 'tau_ampa'    : population
        tau_gaba = 'tau_gaba'    : population
        E_ampa   = 'E_ampa'    : population
        E_gaba   = 'E_gaba'  : population
        I_base  = 0 : population

        # neuron model parameters
        C      = 'C'    : population
        k      = 'k'    : population
        v_r    = 'v_r'   : population
        v_t    = 'v_t'   : population
        v_b    = 'v_b'   : population
        a      = 'a'     : population
        b      = 'b'     : population
        c      = 'c'     : population
        v_peak = 'v_peak' : population

        # dopamine modulation parameter
        phi_1    = 0     : population # D1 receptor activation level
        phi_2    = 0     : population # D2 receptor activation level
        eta     = 'eta' : population
        epsilon  = 'epsilon' : population
    """,
    equations=current_fsi_equations
    + """
        C * dv/dt = k * (v - v_r * (1 - eta * phi_1)) * (v - v_t) - u + I_v
        du/dt     = if v < v_b:
                        -a * u
                    else:
                        a * (b * (v - v_b)**3 - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u
    """,
    name="_Izhikevich2007_Humphries_2009_FSI",
    description="Based on Izhikevich2007 and Humphries2009 model of FSI neurons.",
    extra_values=parameters_fsi_dict,
)
