from ANNarchy import Neuron
from CompNeuroPy import extra_functions as ef

### Hodgkin Huxley like neuron models


### striatal FSI neuron of Bischop et al. (2012)
### Bischop, D. P., Orduz, D., Lambot, L., Schiffmann, S., & Gall, D. (2012). Control of Neuronal Excitability by Calcium Binding Proteins: A New Mathematical Model for Striatal Fast-Spiking Interneurons. Frontiers in Molecular Neuroscience, 5. https://www.frontiersin.org/articles/10.3389/fnmol.2012.00078
HHB = ef.data_obj()

HHB.parameters.base = """
    C_m      = 30 : population # pF
    
    gg_L     = 2.5 : population # nS
    E_L      = -68 : population # mV
    
    gg_Na    = 700 : population # nS
    E_Na     = 74  : population # mV
    
    gg_Kv1   = 2   : population # nS
    gg_Kv3   = 300 : population # nS
    gg_SK    = 2   : population # nS
    E_K      = -90 : population # mV
    
    K_SK     = 0.5       : population # yM --> strange units for k
    F        = 96485.332 : population # (As)/M --> strange units for I_Ca / (2 * F * A * d)
    A        = 3000      : population # ym**2
    d        = 0.2       : population # ym
    gamma    = 1         : population # ms**-1
    Ca_rest  = 0.07      : population # yM
    k_on_ca  = 0.1       : population # yM**-1 * ms**-1
    k_off_ca = 0.001     : population # ms**-1
    k_on_mg  = 0.0008    : population # yM**-1 * ms**-1
    k_off_mg = 0.025     : population # ms**-1
    Mg       = 500       : population # yM
    PV_T     = 63.5      : population # yM, from Lee et al. (2000), The Journal of physiology, Kinetics
                                        # in Bischop et al. (2012) varied from 50 - 1000 yM
                    
    gg_Ca    = 30 : population # nS
    E_Ca     = 80 : population # mV
    
    vt       = 30 : population # mV
    
    I_app    = 0 # pA
"""

HHB.parameters.conductance = """
    tau_ampa = 10  : population # ms
    tau_gaba = 10  : population # ms
    E_ampa   = 0   : population # mV
    E_gaba   = -90 : population # mV
"""

HHB.equations.base = """
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
"""

HHB.equations.conductance = """
dg_ampa/dt = -g_ampa/tau_ampa
dg_gaba/dt = -g_gaba/tau_gaba
"""

HHB.equations.membrane.base = """
C_m * dv/dt  = -I_L - I_Na - I_Kv1 - I_Kv3 - I_SK - I_Ca + I_app : init=-68
"""

HHB.equations.membrane.conductance = """
C_m * dv/dt  = -I_L - I_Na - I_Kv1 - I_Kv3 - I_SK - I_Ca + I_app - g_ampa*(v - E_ampa) - g_gaba*(v - E_gaba) : init=-68
"""


H_and_H_Bischop = Neuron(
    parameters=HHB.parameters.base,
    equations=HHB.equations.base + HHB.equations.membrane.base,
    spike="(v > vt) and (prev_v <= vt)",
    reset="",
    name="H_and_H_Bischop",
    description="Hodgkin Huxley neuron model for striatal FSI from Bischop et al. (2012).",
)


H_and_H_Bischop_syn = Neuron(
    parameters=HHB.parameters.base + HHB.parameters.conductance,
    equations=HHB.equations.conductance
    + HHB.equations.base
    + HHB.equations.membrane.conductance,
    spike="(v > vt) and (prev_v <= vt)",
    reset="",
    name="H_and_H_Bischop_syn",
    description="Hodgkin Huxley neuron model for striatal FSI from Bischop et al. (2012) with conductance-based synapses/currents for AMPA and GABA.",
)


### striatal FSI neuron of Corbit et al. (2016)
### Corbit, V. L., Whalen, T. C., Zitelli, K. T., Crilly, S. Y., Rubin, J. E., & Gittis, A. H. (2016). Pallidostriatal Projections Promote β Oscillations in a Dopamine-Depleted Biophysical Network Model. Journal of Neuroscience, 36(20), 5556–5571. https://doi.org/10.1523/JNEUROSCI.0339-16.2016
### based on: Golomb, D., Donner, K., Shacham, L., Shlosberg, D., Amitai, Y., & Hansel, D. (2007). Mechanisms of Firing Patterns in Fast-Spiking Cortical Interneurons. PLOS Computational Biology, 3(8), e156. https://doi.org/10.1371/journal.pcbi.0030156

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
