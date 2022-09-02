from ANNarchy import Neuron
import math


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
    description="Integrator Neuron, which integrates incoming spikes with value g_ampa and emits a spike when reaching a threshold. After spike decision changes, which can be used as stop condition",
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
    description="Integrator Neuron, which integrates incoming spikes with value g_ampa, which can be used as stop condition",
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
    description="Poisson neuron whose rate can be specified and is reached instanteneous.",
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
        I_add          = 0
        increase_noise = 0 : population
        rates_noise    = 0
    """,
    equations="""
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba / tau_gaba     
        dv/dt      = 0.04 * v * v + 5 * v + 140 - u + I_add - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
        du/dt      = a * (b * v - u)
    """,
    spike="""
        v >= 30
    """,
    reset="""
        v = c
        u = u + d
    """,
    name="izhikevich2003",
    description="Standard neuron model from Izhikevich (2007) with additional conductance based synapses for AMPA and GABA currents with noise in AMPA conductance.",
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
        I_add          = 0
        increase_noise = 0 : population
        rates_noise    = 0
    """,
    equations="""
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba / tau_gaba
        dv/dt      = n2 * v * v + n1 * v + n0 - u + I_add - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
        du/dt      = a * (b * v - u)
    """,
    spike="""
        v >= 30
    """,
    reset="""
        v = c
        u = u + d
    """,
    name="izhikevich2003_modified",
    description="Flexible neuron model from Izhikevich (2003) with additional conductance based synapses for AMPA and GABA currents with noise in AMPA conductance.",
)


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
    name="Izhikevich2007",
    description="Simple neuron model equations from Izhikevich (2007).",
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
    name="Izhikevich2007_vc",
    description="Simple neuron model equations from Izhikevich (2007). With voltage clamp to record I_inf.",
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
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_app - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
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

Izhikevich2007_Corbit = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) adjusted to fit Corbit et al. (2016) FSI neuron model.",
)

Izhikevich2007_Corbit2 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) adjusted version should be able to produce late spiking.",
)

Izhikevich2007_Hjorth_2020_ChIN1 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) for fitting tests.",
)


Izhikevich2007_Hjorth_2020_ChIN2 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) for fitting tests.",
)


Izhikevich2007_Hjorth_2020_ChIN3 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) for fitting tests.",
    functions="""
        foo(x, x_t, p_1, p_2) = x*(p_1 * pos(x - x_t)**p_2 + 1)
    """,
)


Izhikevich2007_Hjorth_2020_ChIN4 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) for fitting tests.",
    functions="""
        foo(y, x, u_t, p_1) = y * (1.0 - (1.0 / (1.0 + exp(-p_1*(x-u_t)))))
    """,
)


Izhikevich2007_Hjorth_2020_ChIN5 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) for fitting tests.",
)


Izhikevich2007_Hjorth_2020_ChIN6 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) for fitting tests.",
)


Izhikevich2007_Hjorth_2020_ChIN7 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) for fitting tests.",
)


Izhikevich2007_Hjorth_2020_ChIN8 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) for fitting tests.",
)


Izhikevich2007_Hjorth_2020_ChIN9 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) for fitting tests.",
)


Izhikevich2007_Hjorth_2020_ChIN10 = Neuron(
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
        du_spike/dt   = 0#- a_spike*u_spike
        
        C * dv/dt = k*(v - v_r)*(v - v_t) - u_v_slow - u_spike  + I_app + base + k_d*(v_d - v) # - k_slow*pos(u_v_slow-th_slow)
    """,
    spike="v >= v_peak",
    reset="""
        v = -100#c
        u_spike = u_spike #+ d_spike
    """,
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) for fitting tests.",
)


Izhikevich2007_Corbit3 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) adjusted version should be able to produce late spiking and adjusted to fit Corbit et al. (2016) FSI neuron model.",
)

Izhikevich2007_Corbit4 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) adjusted version should be able to produce late spiking and adjusted to fit Corbit et al. (2016) FSI neuron model.",
)


Izhikevich2007_Corbit5 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) adjusted version should be able to produce late spiking and adjusted to fit Corbit et al. (2016) FSI neuron model.",
)


Izhikevich2007_Corbit6 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) adjusted version should be able to produce late spiking.",
)


Izhikevich2007_Corbit7 = Neuron(
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
    name="Izhikevich2007_Corbit",
    description="Simple neuron model equations from Izhikevich (2007) adjusted version should be able to produce late spiking.",
)

Izhikevich2007_Corbit8 = Neuron(
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
    name="Izhikevich2007_Corbit8",
    description="Simple neuron model equations from Izhikevich (2007) adjusted version should be able to produce late spiking.",
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
        I_add          = 0
        increase_noise = 0 : population
        rates_noise    = 0
    """,
    equations="""
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba/tau_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_add - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="izhikevich2007_standard",
    description="Standard neuron model from Izhikevich (2007) with additional conductance based synapses for AMPA and GABA currents with noise in AMPA conductance.",
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
        I_add          = 0
        increase_noise = 0 : population
        rates_noise    = 0
    """,
    equations="""
        dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
        dg_gaba/dt = -g_gaba/tau_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_add - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba)
        du/dt      = if v<v_b: -a * u else: a * (b * (v - v_b)**3 - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
    name="izhikevich2007_fsi",
    description="Fast spiking cortical interneuron model from Izhikevich (2007) with additional conductance based synapses with noise in AMPA conductance.",
)


H_and_H_Bischop = Neuron(
    parameters="""
        C_m      = 30 # pF
        
        gg_L     = 2.5 # nS
        E_L      = -68 # mV
        
        gg_Na    = 700 # nS
        E_Na     = 74 # mV
        
        gg_Kv1   = 2 # nS
        gg_Kv3   = 300 # nS
        gg_SK    = 2 # nS
        E_K      = -90 # mV
        
        K_SK     = 0.5 # yM --> strange units for k
        F        = 96485.332 # (As)/M --> strange units for I_Ca / (2 * F * A * d)
        A        = 3000 # ym**2
        d        = 0.2 # ym
        gamma    = 1 # ms**-1
        Ca_rest  = 0.07 # yM
        k_on_ca  = 0.1 # yM**-1 * ms**-1
        k_off_ca = 0.001 # ms**-1
        k_on_mg  = 0.0008 # yM**-1 * ms**-1
        k_off_mg = 0.025 # ms**-1
        Mg       = 500 # yM
        PV_T     = 63.5 # yM, from Lee et al. (2000), The Journal of physiology, Kinetics
                        # in Bischop et al. (2012) varied from 50 - 1000 yM
                        
        gg_Ca    = 30 # nS
        E_Ca     = 80 # mV
        
        I_app    = 0 # pA
        
        vt       = 30 # mV
    """,
    equations="""
        prev_v = v
        
        I_L          = gg_L * (v - E_L)
        
        alpha_h      = 0.0035 * exp(-v / 24.186)
        beta_h       = 0.017 * (-51.25 - v) / (exp((-51.25 - v) / 5.2) - 1)
        h_inf        = alpha_h / (alpha_h + beta_h)
        tau_h        = 1 / (alpha_h + beta_h)
        dh/dt        = (h_inf - h) / tau_h
        alpha_m      = 40 * (75.5 - v) / (exp((75.5 - v) / 13.5) - 1)
        beta_m       = 1.2262 * exp(-v/42.248)
        m_inf        = alpha_m / (alpha_m + beta_m)
        m            = m_inf        
        I_Na         = gg_Na * m**3 * h * (v - E_Na)
        
        alpha_n1     = 0.014 * (-44 - v) / (exp((-44 - v) / 2.3) - 1)
        beta_n1      = 0.0043 * exp((44 + v) / 34)
        n1_inf       = alpha_n1 / (alpha_n1 + beta_n1)
        tau_n1       = 1 / (alpha_n1 + beta_n1)
        dn1/dt       = (n1_inf - n1) / tau_n1
        I_Kv1        = gg_Kv1 * n1**4 * (v - E_K)
        
        alpha_n3     = (95 - v) / (exp((95 - v) / 11.8) - 1)
        beta_n3      = 0.025 * exp(-v / 22.222)
        n3_inf       = alpha_n3 / (alpha_n3 + beta_n3)
        tau_n3       = 1 / (alpha_n3 + beta_n3)
        dn3/dt       = (n3_inf - n3) / tau_n3
        I_Kv3        = gg_Kv3 * n3**2 * (v - E_K)
        
        PV           = PV_T - PV_Ca - PV_Mg
        dPV_Mg/dt    = k_on_mg * Mg * PV - k_off_mg * PV_Mg
        dPV_Ca_dt    = k_on_ca * Ca * PV - k_off_ca * PV_Ca
        dPV_Ca/dt    = dPV_Ca_dt
        dCa/dt       = I_Ca / (2 * F * A * d) - gamma * (Ca - Ca_rest) - dPV_Ca_dt
        k_inf        = Ca / (K_SK + Ca)
        tau_k        = 1 / (K_SK + Ca)
        dk/dt        = (k_inf - k) / tau_k        
        I_SK         = gg_SK * k**2 * (v - E_K)
        
        a_inf        = 1 / (1 + exp((-6-v) / 7.775))
        a            = a_inf
        I_Ca         = gg_Ca * a**2 * (v - E_Ca)
        
        C_m * dv/dt  = -I_L - I_Na - I_Kv1 - I_Kv3 - I_SK - I_Ca + I_app : init=-68
    """,
    spike="(v > vt) and (prev_v <= vt)",
    reset="",
    name="H_and_H_Bischop",
    description="H & H model of Bischop et al. (2012).",
)


H_and_H_Bischop_syn = Neuron(
    parameters="""
        C_m      = 30 # pF
        
        gg_L     = 2.5 # nS
        E_L      = -68 # mV
        
        gg_Na    = 700 # nS
        E_Na     = 74 # mV
        
        gg_Kv1   = 2 # nS
        gg_Kv3   = 300 # nS
        gg_SK    = 2 # nS
        E_K      = -90 # mV
        
        K_SK     = 0.5 # yM --> strange units for k
        F        = 96485.332 # (As)/M --> strange units for I_Ca / (2 * F * A * d)
        A        = 3000 # ym**2
        d        = 0.2 # ym
        gamma    = 1 # ms**-1
        Ca_rest  = 0.07 # yM
        k_on_ca  = 0.1 # yM**-1 * ms**-1
        k_off_ca = 0.001 # ms**-1
        k_on_mg  = 0.0008 # yM**-1 * ms**-1
        k_off_mg = 0.025 # ms**-1
        Mg       = 500 # yM
        PV_T     = 63.5 # yM, from Lee et al. (2000), The Journal of physiology, Kinetics
                        # in Bischop et al. (2012) varied from 50 - 1000 yM
                        
        gg_Ca    = 30 # nS
        E_Ca     = 80 # mV
        
        I_app    = 0 # pA
        
        vt       = 30 # mV
        
        tau_ampa = 10  : population # ms
        tau_gaba = 10  : population # ms
        E_ampa   = 0   : population # mV
        E_gaba   = -90 : population # mV
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        
        prev_v = v
        
        I_L          = gg_L * (v - E_L)
        
        alpha_h      = 0.0035 * exp(-v / 24.186)
        beta_h       = 0.017 * (-51.25 - v) / (exp((-51.25 - v) / 5.2) - 1)
        h_inf        = alpha_h / (alpha_h + beta_h)
        tau_h        = 1 / (alpha_h + beta_h)
        dh/dt        = (h_inf - h) / tau_h
        alpha_m      = 40 * (75.5 - v) / (exp((75.5 - v) / 13.5) - 1)
        beta_m       = 1.2262 * exp(-v/42.248)
        m_inf        = alpha_m / (alpha_m + beta_m)
        m            = m_inf        
        I_Na         = gg_Na * m**3 * h * (v - E_Na)
        
        alpha_n1     = 0.014 * (-44 - v) / (exp((-44 - v) / 2.3) - 1)
        beta_n1      = 0.0043 * exp((44 + v) / 34)
        n1_inf       = alpha_n1 / (alpha_n1 + beta_n1)
        tau_n1       = 1 / (alpha_n1 + beta_n1)
        dn1/dt       = (n1_inf - n1) / tau_n1        
        I_Kv1        = gg_Kv1 * n1**4 * (v - E_K)
        
        alpha_n3     = (95 - v) / (exp((95 - v) / 11.8) - 1)
        beta_n3      = 0.025 * exp(-v / 22.222)
        n3_inf       = alpha_n3 / (alpha_n3 + beta_n3)
        tau_n3       = 1 / (alpha_n3 + beta_n3)
        dn3/dt       = (n3_inf - n3) / tau_n3
        I_Kv3        = gg_Kv3 * n3**2 * (v - E_K)
        
        PV           = PV_T - PV_Ca - PV_Mg
        dPV_Mg/dt    = k_on_mg * Mg * PV - k_off_mg * PV_Mg
        dPV_Ca_dt    = k_on_ca * Ca * PV - k_off_ca * PV_Ca
        dPV_Ca/dt    = dPV_Ca_dt
        dCa/dt       = I_Ca / (2 * F * A * d) - gamma * (Ca - Ca_rest) - dPV_Ca_dt
        k_inf        = Ca / (K_SK + Ca)
        tau_k        = 1 / (K_SK + Ca)
        dk/dt        = (k_inf - k) / tau_k        
        I_SK         = gg_SK * k**2 * (v - E_K)
        
        a_inf        = 1 / (1 + exp((-6-v) / 7.775))
        a            = a_inf
        I_Ca         = gg_Ca * a**2 * (v - E_Ca)
        
        C_m * dv/dt  = -I_L - I_Na - I_Kv1 - I_Kv3 - I_SK - I_Ca + I_app - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba) : init=-68
    """,
    spike="(v > vt) and (prev_v <= vt)",
    reset="",
    name="H_and_H_Bischop",
    description="H & H model of Bischop et al. (2012).",
)


H_and_H_Corbit = Neuron(
    parameters="""
        v_t = 0             # :population #mV (see Corbit et al.,2016 under "Model analysis")
        C_m =  1.0          # :population #myF/cm^2
        v_L = -70           # :population # mV
        gg_L = 0.25         # :population # mS/cm^2
    
        E_Na     = 50       # :population # mV
        gg_Na     = 112.5   # :population # mS/cm^2
        th_m_Na  = -24.0    # :population # mV
        k_m_Na   = 11.5     # :population # mV
        tau_0_m_Na = 0      # :population # ms default value (Corbit et al., 2016 description table 1)
        tau_1_m_Na = 0      # :population # ms default value (Corbit et al., 2016 description table 1)
        sigma_0_m_Na = 1    # :population # mV default value (Corbit et al., 2016 description table 1)
        th_h_Na  = -58.3    # :population # mV
        k_h_Na   = -6.7      # :population # mV
        phi_h_Na = -60      # :population # mV
        sigma_0_h_Na = -12.0# :population # mV
        tau_0_h_Na = 0.5    # :population # ms
        tau_1_h_Na = 13.5   # :population # ms

        E_Kv3 = -90         # :population # mV
        gg_Kv3 = 225.0      # :population # mS/cm^2
        th_n_Kv3 = -12.4    # :population # mV
        k_n_Kv3 = 6.8       # :population # mV
        tau_0_n_Kv3 = 0.087 # :population # ms
        tau_1_n_Kv3 = 11.313# :population # ms
        phi_a_n_Kv3 = -14.6 # :population # mV
        phi_b_n_Kv3 = 1.3   # :population # mV
        sigma_0_a_n_Kv3 = -8.6  # :population #mV
        sigma_0_b_n_Kv3 = 18.7  # :population #mV
        
        E_Kv1 = -90         # :population #mV
        gg_Kv1 = 0.39       # :population #mS/cm^2 in Corbit et al. (2016) they use 0.1, but they say they use delayed-tonic firing which is 0.39 in Golomb (2007), 0.39 fits also better to Figure in Corbit et al. (2016)
        th_m_Kv1 = -50      # :population #mV
        k_m_Kv1 =  20       # :population #mV
        tau_m_Kv1 = 2       # :population #ms
        th_h_Kv1 = -70      # :population #mV
        k_h_Kv1 =  -6       # :population #mV
        tau_h_Kv1 = 150     # :population #ms
        
        #I_app  = 0     : population
    
    """,
    equations="""
        prev_v = v
        
        I_L = gg_L * (v - v_L) #CHECK
        
  # Na current with h particle and instantaneous m:
        m_Na = 1.0 / (1.0 + exp((th_m_Na - v) / k_m_Na))
        dh_Na/dt = (1.0 / (1.0 + exp((th_h_Na - v) / k_h_Na)) - h_Na) / (tau_0_h_Na + (tau_1_h_Na + tau_0_h_Na) / (1.0 + exp((phi_h_Na - v) / sigma_0_h_Na))) : init=0.8521993432343666 #CHECK, in paper: tau_1-tau_0 in numerator, in code: tau_1+tau_0 (= 14) in numerator, in paper: exp + exp in denominator, in code: exp + 1 in denominator, for MSNs they set the term to 0... maybe for FSIs they set it to 1                                                      
        I_Na = (v - E_Na) * gg_Na * pow(m_Na,3) * h_Na #CHECK
      
  # Kv3 current with n particle (but in Corbit et al., 2016 paper version h): 
        n_Kv3_inf = 1.0 / (1.0 + exp((th_n_Kv3 - v) / k_n_Kv3)) #CHECK
        tau_n_Kv3_inf = (tau_0_n_Kv3 + (tau_0_n_Kv3 + tau_1_n_Kv3) / (1.0 + exp((phi_a_n_Kv3 - v) / sigma_0_a_n_Kv3)))*(tau_0_n_Kv3 + (tau_0_n_Kv3 + tau_1_n_Kv3) / (1.0 + exp((phi_b_n_Kv3 - v) / sigma_0_b_n_Kv3))) # CHECK
        dn_Kv3/dt = (n_Kv3_inf - n_Kv3) / tau_n_Kv3_inf : init=0.00020832726857419512
        I_Kv3 = (v - E_Kv3) * gg_Kv3 * pow(n_Kv3,2) #CHECK
    
  # Kv1 current with m, h particle : 
        dm_Kv1/dt = (1.0 / (1.0 + exp((th_m_Kv1 - v) / k_m_Kv1)) - m_Kv1) / tau_m_Kv1 : init=0.2685669882630486
        dh_Kv1/dt = (1.0 / (1.0 + exp((th_h_Kv1 - v) / k_h_Kv1)) - h_Kv1) / tau_h_Kv1 : init=0.5015877164262258
        I_Kv1 = (v - E_Kv1)  * gg_Kv1 * pow(m_Kv1, 3) * h_Kv1 #CHECK
        
        dI_app/dt = 0
        C_m * dv/dt  = -I_L - I_Na - I_Kv3 - I_Kv1 + I_app : init=-70.03810532250634
        
    """,
    spike="(v > v_t) and (prev_v <= v_t)",
    reset="",
    name="H_and_H_Corbit",
    description="H & H model of Corbit et al. (2016).",
)


H_and_H_Corbit_voltage_clamp = Neuron(
    parameters="""
        v_t = 0             # :population #mV (see Corbit et al.,2016 under "Model analysis")
        C_m =  1.0          # :population #myF/cm^2
        v_L = -70           # :population # mV
        gg_L = 0.25         # :population # mS/cm^2
    
        E_Na     = 50       # :population # mV
        gg_Na     = 112.5   # :population # mS/cm^2
        th_m_Na  = -24.0    # :population # mV
        k_m_Na   = 11.5     # :population # mV
        tau_0_m_Na = 0      # :population # ms default value (Corbit et al., 2016 description table 1)
        tau_1_m_Na = 0      # :population # ms default value (Corbit et al., 2016 description table 1)
        sigma_0_m_Na = 1    # :population # mV default value (Corbit et al., 2016 description table 1)
        th_h_Na  = -58.3    # :population # mV
        k_h_Na   = -6.7      # :population # mV
        phi_h_Na = -60      # :population # mV
        sigma_0_h_Na = -12.0# :population # mV
        tau_0_h_Na = 0.5    # :population # ms
        tau_1_h_Na = 13.5   # :population # ms

        E_Kv3 = -90         # :population # mV
        gg_Kv3 = 225.0      # :population # mS/cm^2
        th_n_Kv3 = -12.4    # :population # mV
        k_n_Kv3 = 6.8       # :population # mV
        tau_0_n_Kv3 = 0.087 # :population # ms
        tau_1_n_Kv3 = 11.313# :population # ms
        phi_a_n_Kv3 = -14.6 # :population # mV
        phi_b_n_Kv3 = 1.3   # :population # mV
        sigma_0_a_n_Kv3 = -8.6  # :population #mV
        sigma_0_b_n_Kv3 = 18.7  # :population #mV
        
        E_Kv1 = -90         # :population #mV
        gg_Kv1 = 0.39       # :population #mS/cm^2 in Corbit et al. (2016) they use 0.1, but they say they use delayed-tonic firing which is 0.39 in Golomb (2007), 0.39 fits also better to Figure in Corbit et al. (2016)
        th_m_Kv1 = -50      # :population #mV
        k_m_Kv1 =  20       # :population #mV
        tau_m_Kv1 = 2       # :population #ms
        th_h_Kv1 = -70      # :population #mV
        k_h_Kv1 =  -6       # :population #mV
        tau_h_Kv1 = 150     # :population #ms
        
        I_app  = 0     : population
    
    """,
    equations="""
        prev_v = v
        
        I_L = gg_L * (v - v_L) #CHECK
        
  # Na current with h particle and instantaneous m:
        m_Na = 1.0 / (1.0 + exp((th_m_Na - v) / k_m_Na))
        dh_Na/dt = (1.0 / (1.0 + exp((th_h_Na - v) / k_h_Na)) - h_Na) / (tau_0_h_Na + (tau_1_h_Na + tau_0_h_Na) / (1.0 + exp((phi_h_Na - v) / sigma_0_h_Na))) : init=0.8521993432343666 #CHECK, in paper: tau_1-tau_0 in numerator, in code: tau_1+tau_0 (= 14) in numerator, in paper: exp + exp in denominator, in code: exp + 1 in denominator, for MSNs they set the term to 0... maybe for FSIs they set it to 1                                                      
        I_Na = (v - E_Na) * gg_Na * pow(m_Na,3) * h_Na #CHECK
      
  # Kv3 current with n particle (but in Corbit et al., 2016 paper version h): 
        n_Kv3_inf = 1.0 / (1.0 + exp((th_n_Kv3 - v) / k_n_Kv3)) #CHECK
        tau_n_Kv3_inf = (tau_0_n_Kv3 + (tau_0_n_Kv3 + tau_1_n_Kv3) / (1.0 + exp((phi_a_n_Kv3 - v) / sigma_0_a_n_Kv3)))*(tau_0_n_Kv3 + (tau_0_n_Kv3 + tau_1_n_Kv3) / (1.0 + exp((phi_b_n_Kv3 - v) / sigma_0_b_n_Kv3))) # CHECK
        dn_Kv3/dt = (n_Kv3_inf - n_Kv3) / tau_n_Kv3_inf : init=0.00020832726857419512
        I_Kv3 = (v - E_Kv3) * gg_Kv3 * pow(n_Kv3,2) #CHECK
    
  # Kv1 current with m, h particle : 
        dm_Kv1/dt = (1.0 / (1.0 + exp((th_m_Kv1 - v) / k_m_Kv1)) - m_Kv1) / tau_m_Kv1 : init=0.2685669882630486
        dh_Kv1/dt = (1.0 / (1.0 + exp((th_h_Kv1 - v) / k_h_Kv1)) - h_Kv1) / tau_h_Kv1 : init=0.5015877164262258
        I_Kv1 = (v - E_Kv1)  * gg_Kv1 * pow(m_Kv1, 3) * h_Kv1 #CHECK
        

        C_m * dv/dt  = 0 : init=-70.03810532250634
        I_inf = -I_L - I_Na - I_Kv3 - I_Kv1 + I_app
        
    """,
    spike="(v > v_t) and (prev_v <= v_t)",
    reset="",
    name="H_and_H_Corbit",
    description="H & H model of Corbit et al. (2016).",
)


H_and_H_Corbit_syn = Neuron(
    parameters="""
        v_t = 0             # :population #mV (see Corbit et al.,2016 under "Model analysis")
        C_m =  1.0          # :population #myF/cm^2
        v_L = -70           # :population # mV
        gg_L = 0.25         # :population # mS/cm^2
    
        E_Na     = 50       # :population # mV
        gg_Na     = 112.5   # :population # mS/cm^2
        th_m_Na  = -24.0    # :population # mV
        k_m_Na   = 11.5     # :population # mV
        tau_0_m_Na = 0      # :population # ms default value (Corbit et al., 2016 description table 1)
        tau_1_m_Na = 0      # :population # ms default value (Corbit et al., 2016 description table 1)
        sigma_0_m_Na = 1    # :population # mV default value (Corbit et al., 2016 description table 1)
        th_h_Na  = -58.3    # :population # mV
        k_h_Na   = -6.7      # :population # mV
        phi_h_Na = -60      # :population # mV
        sigma_0_h_Na = -12.0# :population # mV
        tau_0_h_Na = 0.5    # :population # ms
        tau_1_h_Na = 13.5   # :population # ms

        E_Kv3 = -90         # :population # mV
        gg_Kv3 = 225.0      # :population # mS/cm^2
        th_n_Kv3 = -12.4    # :population # mV
        k_n_Kv3 = 6.8       # :population # mV
        tau_0_n_Kv3 = 0.087 # :population # ms
        tau_1_n_Kv3 = 11.313# :population # ms
        phi_a_n_Kv3 = -14.6 # :population # mV
        phi_b_n_Kv3 = 1.3   # :population # mV
        sigma_0_a_n_Kv3 = -8.6  # :population #mV
        sigma_0_b_n_Kv3 = 18.7  # :population #mV
        
        E_Kv1 = -90         # :population #mV
        gg_Kv1 = 0.39       # :population #mS/cm^2 in Corbit et al. (2016) they use 0.1, but they say they use delayed-tonic firing which is 0.39 in Golomb (2007), 0.39 fits also better to Figure in Corbit et al. (2016)
        th_m_Kv1 = -50      # :population #mV
        k_m_Kv1 =  20       # :population #mV
        tau_m_Kv1 = 2       # :population #ms
        th_h_Kv1 = -70      # :population #mV
        k_h_Kv1 =  -6       # :population #mV
        tau_h_Kv1 = 150     # :population #ms
        
        I_app  = 0     : population
        
        tau_ampa = 10  : population # ms
        tau_gaba = 10  : population # ms
        E_ampa   = 0   : population # mV
        E_gaba   = -90 : population # mV
    
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa
        dg_gaba/dt = -g_gaba/tau_gaba
        
        prev_v = v
        
        I_L = gg_L * (v - v_L) #CHECK
        
  # Na current with h particle and instantaneous m:
        m_Na = 1.0 / (1.0 + exp((th_m_Na - v) / k_m_Na))
        dh_Na/dt = (1.0 / (1.0 + exp((th_h_Na - v) / k_h_Na)) - h_Na) / (tau_0_h_Na + (tau_1_h_Na + tau_0_h_Na) / (1.0 + exp((phi_h_Na - v) / sigma_0_h_Na))) : init=0.8521993432343666 #CHECK, in paper: tau_1-tau_0 in numerator, in code: tau_1+tau_0 (= 14) in numerator, in paper: exp + exp in denominator, in code: exp + 1 in denominator, for MSNs they set the term to 0... maybe for FSIs they set it to 1                                                      
        I_Na = (v - E_Na) * gg_Na * pow(m_Na,3) * h_Na #CHECK
      
  # Kv3 current with n particle (but in Corbit et al., 2016 paper version h): 
        n_Kv3_inf = 1.0 / (1.0 + exp((th_n_Kv3 - v) / k_n_Kv3)) #CHECK
        tau_n_Kv3_inf = (tau_0_n_Kv3 + (tau_0_n_Kv3 + tau_1_n_Kv3) / (1.0 + exp((phi_a_n_Kv3 - v) / sigma_0_a_n_Kv3)))*(tau_0_n_Kv3 + (tau_0_n_Kv3 + tau_1_n_Kv3) / (1.0 + exp((phi_b_n_Kv3 - v) / sigma_0_b_n_Kv3))) # CHECK
        dn_Kv3/dt = (n_Kv3_inf - n_Kv3) / tau_n_Kv3_inf : init=0.00020832726857419512
        I_Kv3 = (v - E_Kv3) * gg_Kv3 * pow(n_Kv3,2) #CHECK
    
  # Kv1 current with m, h particle : 
        dm_Kv1/dt = (1.0 / (1.0 + exp((th_m_Kv1 - v) / k_m_Kv1)) - m_Kv1) / tau_m_Kv1 : init=0.2685669882630486
        dh_Kv1/dt = (1.0 / (1.0 + exp((th_h_Kv1 - v) / k_h_Kv1)) - h_Kv1) / tau_h_Kv1 : init=0.5015877164262258
        I_Kv1 = (v - E_Kv1)  * gg_Kv1 * pow(m_Kv1, 3) * h_Kv1 #CHECK
        

        C_m * dv/dt  = -I_L - I_Na - I_Kv3 - I_Kv1 + I_app - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba) : init=-70.03810532250634
        
    """,
    spike="(v > v_t) and (prev_v <= v_t)",
    reset="",
    name="H_and_H_Corbit",
    description="H & H model of Corbit et al. (2016).",
)
