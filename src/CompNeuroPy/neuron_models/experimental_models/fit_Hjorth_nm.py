from ANNarchy import Neuron

### Fit neuron model for ChIN based on electrophysiological recordings from Hjorth et al. (2020)
### Hjorth, J. J. J., Kozlov, A., Carannante, I., Frost Nylén, J., Lindroos, R., Johansson, Y., Tokarska, A., Dorst, M. C., Suryanarayana, S. M., Silberberg, G., Hellgren Kotaleski, J., & Grillner, S. (2020). The microcircuits of striatum in silico. Proceedings of the National Academy of Sciences, 117(17), 9554–9565. https://doi.org/10.1073/pnas.2000671117


_Izhikevich2007_Hjorth_2020_ChIN1 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        k_d    = 0 : population #
        a_d    = 0 : population #
        v_r    = 0 : population # mV
        v_t_0    = 0 : population # mV
        a_v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population # nS
        c     = 0 : population # mV
        d      = 0 : population # pA
        v_peak = 0 : population # mV
        I_app  = 0 # pA
    """,
    equations="""
        v_t  = pos(a_v_t*I_app)+v_t_0
        C * dv/dt = k*(v - v_r)*(v - v_t) - u  + I_app + k_d*(v_d - v)
        du/dt     = a*(b*(v - v_r) - u)
        dv_d/dt   = a_d*(v - v_d)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="_Izhikevich2007_Hjorth_2020_ChIN1",
    description="Based on Izhikevich2007 adjusted to fit Hjorth data",
)


_Izhikevich2007_Hjorth_2020_ChIN2 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        k_d    = 0 : population #
        a_d    = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        b      = 0 : population # nS
        c_0     = 0 : population # mV
        a_c     = 0 : population #
        d_c     = 0 : population #
        d      = 0 : population # pA
        v_peak = 0 : population # mV
        I_app  = 0 # pA
    """,
    equations="""
        C * dv/dt = k*(v - v_r)*(v - v_t) - u  + I_app + k_d*(v_d - v)
        du/dt     = a*(b*(v - v_r) - u)
        dv_d/dt   = a_d*(v - v_d)
        dc/dt     = a_c*(clip((6.0/360.0)*I_app-65.5,-63,-40) - c)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
        c = c + d_c
    """,
    name="_Izhikevich2007_Hjorth_2020_ChIN2",
    description="Based on Izhikevich2007 adjusted to fit Hjorth data",
)


_Izhikevich2007_Hjorth_2020_ChIN3 = Neuron(
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
        u_t = 0 : population
        p_1 = 0 : population
        p_2 = 0 : population
        f_t = 0 : population
    """,
    equations="""
        du/dt     = a*(b*(v - v_r) - u)
        dv_d/dt   = a_d*(v - v_d)

        C * dv/dt = k*(v - v_r)*(v - v_t) - foo(u, u_t, p_1, p_2) + I_app + k_d*(v_d - v)
        #C * dv/dt = k*(v - v_r)*(v - v_t) - u  + I_app + k_d*(v_d - v)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="_Izhikevich2007_Hjorth_2020_ChIN3",
    description="Based on Izhikevich2007 adjusted to fit Hjorth data",
    functions="""
        foo(x, x_t, p_1, p_2) = x*(p_1 * pos(x - x_t)**p_2 + 1)
    """,
)


_Izhikevich2007_Hjorth_2020_ChIN4 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        a_spike      = 0 : population
        b      = 0 : population # nS
        c      = 0 : population # mV
        d      = 0 : population # pA
        v_peak = 0 : population # mV
        I_app  = 0 # pA
        f_t = 0 : population
        u_t = 0 : population
        p_1 = 0 : population
    """,
    equations="""
        du_v/dt     = a*(b*(v - v_r) - u_v)
        du_spike/dt = -a_spike*u_spike
        C * dv/dt = k*(v - v_r)*(v - v_t) - u_v - u_spike + foo(I_app, u_spike, u_t, p_1) ### TODO future synaptic current also in foo
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u_spike = u_spike + d
    """,
    name="_Izhikevich2007_Hjorth_2020_ChIN4",
    description="Based on Izhikevich2007 adjusted to fit Hjorth data",
    functions="""
        foo(y, x, u_t, p_1) = y * (1.0 - (1.0 / (1.0 + exp(-p_1*(x-u_t)))))
    """,
)


_Izhikevich2007_Hjorth_2020_ChIN5 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a      = 0 : population # ms**-1
        a_spike      = 0 : population
        b_spike      = 0 : population
        c_spike      = 0 : population
        b      = 0 : population # nS
        c      = 0 : population # mV
        v_peak = 0 : population # mV
        I_app  = 0 # pA
        a_n = 0 : population
        n_t = 0 : population
        b_n = 1 : population
    """,
    equations="""
        du_v/dt     = a*(b*(v - v_r) - u_v) : max=0
        du_spike/dt = -a_spike*u_spike
        C * dv/dt = k*(v - v_r)*(v - v_t) - u_v - u_spike + I_app
        dn/dt=-a_n*n
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        n = n + u_spike/(b_spike*(I_app))
        u_spike = u_spike + (1+b_n*pos(n-n_t))*c_spike*(b_spike*(I_app) - u_spike)### TODO future synaptic current also here
    """,
    name="_Izhikevich2007_Hjorth_2020_ChIN5",
    description="Based on Izhikevich2007 adjusted to fit Hjorth data",
)


_Izhikevich2007_Hjorth_2020_ChIN6 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a_slow      = 0 : population # ms**-1
        a_fast      = 0 : population # ms**-1
        a_brake      = 0 : population # ms**-1
        b_slow      = 0 : population # nS
        b_fast      = 0 : population # nS
        brake_0      = 0 : population # mV
        d_brake      = 0 : population # mV
        c      = 0 : population # mV
        d      = 0 : population # pA
        d_fast      = 0 : population # pA
        v_peak = 0 : population # mV
        I_app  = 0 # pA
        base   = 0 # pA
        k_slow=0
        th_slow=0
    """,
    equations="""
        du_v_slow/dt     = a_slow*(b_slow*(v - v_r) - u_v_slow)  # b_slow positive!
        du_v_fast/dt     = a_fast*(b_fast*(v - v_r) - u_v_fast)  # b_fast negative!
        dbrake/dt        = a_brake*(brake_0-brake)
        
        C * dv/dt = k*((v - v_r)*(v - v_t) - brake*pos(v-v_r)) - u_v_slow - u_v_fast  + I_app + base # - k_slow*pos(u_v_slow-th_slow)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        brake = brake*d + d_brake
        u_v_fast = u_v_fast + d_fast
    """,
    name="_Izhikevich2007_Hjorth_2020_ChIN6",
    description="Based on Izhikevich2007 adjusted to fit Hjorth data",
)


_Izhikevich2007_Hjorth_2020_ChIN7 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        v_t    = 0 : population # mV
        a_slow      = 0 : population # ms**-1
        b_slow      = 0 : population # nS
        c      = 0 : population # mV
        d_v_r      = 0 : population # pA
        delta_v_r      = 0 : population # pA
        a_v_r      = 0 : population # pA
        v_r_0      = 0 : population # pA
        a_d = 0 : population
        k_d = 0 : population
        v_peak = 0 : population # mV
        I_app  = 0 # pA
        base   = 0 # pA
        k_slow=0
        th_slow=0
    """,
    equations="""
    
        dv_d/dt       = a_d*(v - v_d)
        du_v_slow/dt  = a_slow*(b_slow*(v - v_r) - u_v_slow)  # b_slow positive!
        dv_r/dt       = a_v_r*(v_r_0-v_r)
        
        C * dv/dt = k*(v - v_r)*(v - v_t) - u_v_slow  + I_app + base + k_d*(v_d - v) # - k_slow*pos(u_v_slow-th_slow)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        v_r = v_r * delta_v_r - d_v_r
    """,
    name="_Izhikevich2007_Hjorth_2020_ChIN7",
    description="Based on Izhikevich2007 adjusted to fit Hjorth data",
)


_Izhikevich2007_Hjorth_2020_ChIN8 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        v_t    = 0 : population # mV
        v_r    = 0 : population # mV
        a_slow      = 0 : population # ms**-1
        b_slow      = 0 : population # nS
        c      = 0 : population # mV
        a_d = 0 : population
        k_d = 0 : population
        v_peak = 0 : population # mV
        I_app  = 0 # pA
        base   = 0 : population # pA
        k_slow=0 : population
        th_slow=0 : population
        a_brake = 0 : population
        brake_0 = 0 : population
        d_rel = 0 : population
        d_abs = 0 : population
        
    """,
    equations="""
    
        dv_d/dt       = a_d*(v - v_d)
        du_v_slow/dt  = a_slow*(b_slow*(v - v_r) - u_v_slow)  # b_slow positive!
        dbrake/dt     = a_brake*(brake_0-brake)
        
        f = pos((brake/(v_t-c))*neg(v-v_t)+brake)
        
        C * dv/dt = k*((v - v_r)*(v - v_t) - f) - u_v_slow  + I_app + base + k_d*(v_d - v) # - k_slow*pos(u_v_slow-th_slow)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        brake = brake*d_rel + d_abs
    """,
    name="_Izhikevich2007_Hjorth_2020_ChIN8",
    description="Based on Izhikevich2007 adjusted to fit Hjorth data",
)


_Izhikevich2007_Hjorth_2020_ChIN9 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a_slow      = 0 : population # ms**-1
        a_fast      = 0 : population # ms**-1
        b_slow      = 0 : population # nS
        b_fast      = 0 : population # nS
        a_brake      = 0 : population # ms**-1
        brake_0      = 0 : population # mV
        brake_max      = 0 : population # mV
        d_brake      = 0 : population # mV
        c      = 0 : population # mV
        d_fast      = 0 : population # pA
        v_peak = 0 : population # mV
        I_app  = 0 # pA
        base   = 0 # pA
        k_slow=0
        th_slow=0
    """,
    equations="""
        du_v_slow/dt     = a_slow*(b_slow*(v - v_r) - u_v_slow)  # b_slow positive!
        du_v_fast/dt     = a_fast*(b_fast*(v - v_r) - u_v_fast)  # b_fast negative!
        dbrake/dt        = a_brake*(brake_0-brake)
        
        f = brake*neg((v - v_r)*(v - v_t))
        
        C * dv/dt = k*((v - v_r)*(v - v_t) + f) - u_v_slow - u_v_fast  + I_app + base # - k_slow*pos(u_v_slow-th_slow)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        brake = brake+(brake_max-brake)*d_brake
        u_v_fast = u_v_fast + d_fast
    """,
    name="_Izhikevich2007_Hjorth_2020_ChIN9",
    description="Based on Izhikevich2007 adjusted to fit Hjorth data",
)


_Izhikevich2007_Hjorth_2020_ChIN10 = Neuron(
    parameters="""
        C      = 0 : population # pF
        k      = 0 : population #
        v_r    = 0 : population # mV
        v_t    = 0 : population # mV
        a_slow      = 0 : population # ms**-1
        b_slow      = 0 : population # nS
        a_spike      = 0 : population # ms**-1
        d_spike      = 0 : population # mV
        c      = 0 : population # mV
        v_peak = 0 : population # mV
        I_app  = 0 # pA
        base   = 0 : population # pA
        k_slow=0 : population
        th_slow=0 : population
        a_d=0 : population
        k_d=0 : population
    """,
    equations="""
        du_v_slow/dt  = a_slow*(b_slow*(v - v_r) - u_v_slow)
        dv_d/dt       = a_d*(v - v_d)
        du_spike/dt   = - a_spike*u_spike
        
        C * dv/dt = k*(v - v_r)*(v - v_t) - u_v_slow - u_spike  + I_app + base + k_d*(v_d - v) # - k_slow*pos(u_v_slow-th_slow)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u_spike = u_spike + d_spike
    """,
    name="_Izhikevich2007_Hjorth_2020_ChIN10",
    description="Based on Izhikevich2007 adjusted to fit Hjorth data",
)
