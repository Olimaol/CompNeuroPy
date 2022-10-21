from ANNarchy import Neuron

### Fit neuron model for FSI neurons from Corbit et al. (2016)
### Corbit, V. L., Whalen, T. C., Zitelli, K. T., Crilly, S. Y., Rubin, J. E., & Gittis, A. H. (2016). Pallidostriatal Projections Promote β Oscillations in a Dopamine-Depleted Biophysical Network Model. Journal of Neuroscience, 36(20), 5556–5571. https://doi.org/10.1523/JNEUROSCI.0339-16.2016


_Izhikevich2007_Corbit = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        p1     = 0 : population #
        p2     = 0 : population #
        p3     = 0 : population #
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
        I_inst    = k*(v - v_r)*(v - v_t)*(1+p2*pos(v - v_r)) + p1*(v_d - v)
        C * dv/dt = k*(v - v_r)*(v - v_t)*(1+p2*pos(v - v_r)) - u + p1*(v_d - v) + I_app
        du/dt     = a*(b*pos(v - v_r)**4 - u)
        dv_d/dt   = p3*(v - v_d)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="_Izhikevich2007_Corbit",
    description="Based on Izhikevich2007 adjusted to fit Corbit FSI neuron model.",
)


_Izhikevich2007_Corbit2 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        k_d    = 0 : population #
        a_d    = 0 : population #
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
        C * dv/dt = k*(v - v_r)*(v - v_t) - u + k_d*(v_d - v) + I_app
        du/dt     = a*(b*(v - v_r) - u)
        dv_d/dt   = a_d*(v - v_d)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="_Izhikevich2007_Corbit2",
    description="Based on Izhikevich2007 adjusted to fit Corbit FSI neuron model. Adjusted version should be able to produce late spiking.",
)


_Izhikevich2007_Corbit3 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        p1     = 0 : population #
        p2     = 0 : population #
        p3     = 0 : population #
        k_t    = 0 : population #
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
        C * dv/dt = k*(v - v_r) + k_t*pos(v - v_t)**2 - u + s*p1*(v_d - v) + I_app
        du/dt     = a*(b*pos(v - v_t)**2 - u)
        dv_d/dt   = p2*(v - v_d)
        ds/dt     = p3*(-s) : init=0
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
        s = 1
    """,
    name="_Izhikevich2007_Corbit3",
    description="Based on Izhikevich2007 adjusted to fit Corbit FSI neuron model. Adjusted version should be able to produce late spiking.",
)


_Izhikevich2007_Corbit4 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        k_d    = 0 : population #
        a_d    = 0 : population #
        k_t    = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population # nS
        c      = 0 : population # mV
        v_peak = 0 : population # mV
        I_app  = 0 # pA
    """,
    equations="""
        C * dv/dt = k*(v - v_r) + k_t*pos(v - v_t)**2 - u + k_d*(v_d - v) + I_app
        du/dt     = a*(b*pos(v - v_t)**2 - u)
        dv_d/dt   = a_d*(v - v_d)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
    """,
    name="_Izhikevich2007_Corbit4",
    description="Based on Izhikevich2007 adjusted to fit Corbit FSI neuron model. Adjusted version should be able to produce late spiking.",
)


_Izhikevich2007_Corbit5 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        k_s    = 0 : population #
        a_s    = 0 : population #
        a_n    = 0 : population #
        #a_d    = 0 : population #
        k_t    = 0 : population #
        #k_d    = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population # nS
        c      = 0 : population # mV
        v_peak = 0 : population # mV
        I_app  = 0 # pA
    """,
    equations="""
        C * dv/dt = k*(v - v_r) + k_t*pos(v - v_t)**2 - u - n + I_app#+ k_d*(v_d - v) + I_app
        du/dt     = a*(b*pos(v - v_t)**2 - u)
        #dv_d/dt   = a_d*(v - v_d)
        ds/dt     = a_s*(u**0.1 - s)
        #dn/dt     = a_n*(u**0.1 - n)
        dn/dt     = a_n*(k_s*(u**0.1-s) - n)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
    """,
    name="_Izhikevich2007_Corbit5",
    description="Based on Izhikevich2007 adjusted to fit Corbit FSI neuron model. Adjusted version should be able to produce late spiking.",
)


_Izhikevich2007_Corbit6 = Neuron(
    parameters="""
        C      = 0 : population # ATTENTION! H&H model is myF/cm^2 --> here also myF/cm^2 and not pF --> current also myA/cm^2 and not pA
        k      = 0 : population #
        b_n    = 0 : population #
        a_s    = 0 : population #
        a_n    = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population #
        c      = 0 : population # mV
        v_peak = 0 : population # mV
        I_app  = 0
    """,
    equations="""
        C * dv/dt = k*(v - v_r)*(v - v_t) - u - n + I_app
        du/dt     = a*(b*(v - v_r) - u)
        ds/dt     = a_s*(pos(u)**0.1 - s)
        dn/dt     = a_n*(b_n*(pos(u)**0.1-s) - n)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
    """,
    name="_Izhikevich2007_Corbit6",
    description="Based on Izhikevich2007 adjusted to fit Corbit FSI neuron model. Adjusted version should be able to produce late spiking.",
)


_Izhikevich2007_Corbit7 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        b_n    = 0 : population #
        a_s    = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population # nS
        c      = 0 : population # mV
        v_peak = 0 : population # mV
        I_app  = 0 # pA
    """,
    equations="""
        C * dv/dt = k*(v - v_r)*(v - v_t) - u - b_n*(pos(u)**0.1-s) + I_app
        du/dt     = a*(b*(v - v_r) - u)
        ds/dt     = a_s*(pos(u)**0.1 - s)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
    """,
    name="_Izhikevich2007_Corbit7",
    description="Based on Izhikevich2007 adjusted to fit Corbit FSI neuron model. Adjusted version should be able to produce late spiking.",
)


_Izhikevich2007_Corbit8 = Neuron(
    parameters="""
        C      = 0 : population # ATTENTION! H&H model is myF/cm^2 --> here also myF/cm^2 and not pF --> current also myF/cm^2 and not pA
        k      = 0 : population #
        b_n    = 0 : population #
        a_s    = 0 : population #
        a_n    = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population #
        d      = 0 : population
        c      = 0 : population # mV
        v_peak = 0 : population # mV
        I_app  = 0
        x      = 0                     # exponent = nth root
    """,
    equations="""
        C * dv/dt = k*(v - v_r)*(v - v_t) - u - n + ((abs(I_app))**(1/x))/((I_app+1e-20)/(abs(I_app)+ 1e-20))

        du/dt     = a*(b*(v - v_r) - u)
        ds/dt     = a_s*(pos(u)**0.1 - s)
        dn/dt     = a_n*(b_n*(pos(u)**0.1-s) - n)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d                           ### new ###
    """,
    name="_Izhikevich2007_Corbit8",
    description="Based on Izhikevich2007 adjusted to fit Corbit FSI neuron model. Adjusted version should be able to produce late spiking and non-linear f-I curve.",
)


_Izhikevich2007_Corbit9 = Neuron(
    parameters="""
        C      = 0 : population # ATTENTION! H&H model is myF/cm^2 --> here also myF/cm^2 and not pF --> current also myF/cm^2 and not pA
        k      = 0 : population #
        b_n    = 0 : population #
        a_s    = 0 : population #
        a_n    = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population #
        c      = 0 : population # mV
        v_peak = 0 : population # mV
        I_app  = 0
        x      = 0                     # exponent = nth root
    """,
    equations="""
         C * dv/dt = k*(v - v_r)*(v - v_t) - u - n + ((abs(I_app))**(1/x))/((I_app+1e-20)/(abs(I_app)+ 1e-20))

        du/dt     = a*(b*(v - v_r) - u)
        ds/dt     = a_s*(pos(u)**0.1 - s)
        dn/dt     = a_n*(b_n*(pos(u)**0.1-s) - n)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
    """,
    name="_Izhikevich2007_Corbit9",
    description="Based on Izhikevich2007 adjusted to fit Corbit FSI neuron model. Adjusted version should be able to produce late spiking and non-linear f-I curve. Combination of Corbit6 and Corbit8 without parameter d, but x ",
)
