from ANNarchy import Neuron


### Hodgkin Huxley like neuron models
class _ConductanceStrings:
    """
    Class for storing the parameters and equations of the conductance-based
    synapses/currents for AMPA and GABA which need to be added to the
    base equations.

    Attributes:
        parameters (str):
            Parameters of the model.
        equations (str):
            Equations of the model.
        membrane (str):
            Equations of the membrane potential of the model.
    """

    def __init__(self) -> None:
        self.parameters = """
            tau_ampa = 10  : population # ms
            tau_gaba = 10  : population # ms
            E_ampa   = 0   : population # mV
            E_gaba   = -90 : population # mV
        """
        self.equations = """
            dg_ampa/dt   = -g_ampa/tau_ampa
            dg_gaba/dt   = -g_gaba/tau_gaba
        """
        self.membrane = "- neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))"


class _BischopStrings:
    """
    Class for storing the parameters and equations of the Hodgkin Huxley like
    neuron model for striatal FSI from Bischop et al. (2012).

    Attributes:
        parameters_base (str):
            Base parameters of the model.
        equations_base (str):
            Base equations of the model.
        membrane_base (str):
            Base equations of the membrane potential of the model.
        parameters_conductance (str):
            Parameters of the model with conductance-based synapses/currents for
            AMPA and GABA.
        equations_conductance (str):
            Equations of the model with conductance-based synapses/currents for
            AMPA and GABA.
        membrane_conductance (str):
            Equations of the membrane potential of the model with
            conductance-based synapses/currents for AMPA and GABA.
    """

    def __init__(self) -> None:
        # base
        self.parameters_base = """
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
            
            v_t       = 30 : population # mV
            
            I_app    = 0 # pA
        """

        self.equations_base = """
            prev_v       = v
            
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

        self.membrane_base = """
            C_m * dv/dt  = -I_L - I_Na - I_Kv1 - I_Kv3 - I_SK - I_Ca + I_app : init=-68
        """

        # with conductance-based synapses/currents for AMPA and GABA
        conductance = _ConductanceStrings()

        self.parameters_conductance = self.parameters_base + conductance.parameters

        self.equations_conductance = conductance.equations + self.equations_base

        self.membrane_conductance = (
            self.membrane_base.split(":")[0]
            + conductance.membrane
            + ":"
            + self.membrane_base.split(":")[1]
        )


class _CorbitStrings:
    """
    Class for storing the parameters and equations of the Hodgkin Huxley like
    neuron model for striatal FSI from Corbit et al. (2016).

    Attributes:
        parameters_base (str):
            Base parameters of the model.
        equations_base (str):
            Base equations of the model.
        membrane_base (str):
            Base equations of the membrane potential of the model.
        parameters_conductance (str):
            Parameters of the model with conductance-based synapses/currents for
            AMPA and GABA.
        equations_conductance (str):
            Equations of the model with conductance-based synapses/currents for
            AMPA and GABA.
        membrane_conductance (str):
            Equations of the membrane potential of the model with
            conductance-based synapses/currents for AMPA and GABA.
        membrane_voltage_clamp (str):
            Equations of the membrane potential of the model with voltage clamp.
    """

    def __init__(self) -> None:
        # base
        self.parameters_base = """
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
        """

        self.equations_base = """
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
            
            # Kv1 current with m, h particle:
            dm_Kv1/dt = (1.0 / (1.0 + exp((th_m_Kv1 - v) / k_m_Kv1)) - m_Kv1) / tau_m_Kv1 : init=0.2685669882630486
            dh_Kv1/dt = (1.0 / (1.0 + exp((th_h_Kv1 - v) / k_h_Kv1)) - h_Kv1) / tau_h_Kv1 : init=0.5015877164262258
            I_Kv1 = (v - E_Kv1)  * gg_Kv1 * pow(m_Kv1, 3) * h_Kv1 #CHECK
        """

        self.membrane_base = """
            C_m * dv/dt  = -I_L - I_Na - I_Kv3 - I_Kv1 + I_app : init=-70.03810532250634
        """

        # with conductance-based synapses/currents for AMPA and GABA
        conductance = _ConductanceStrings()

        self.parameters_conductance = conductance.parameters + self.parameters_base

        self.equations_conductance = conductance.equations + self.equations_base

        self.membrane_conductance = (
            self.membrane_base.split(":")[0]
            + conductance.membrane
            + ":"
            + self.membrane_base.split(":")[1]
        )

        # with voltage clamp
        self.membrane_voltage_clamp = """
            C_m * dv/dt  = 0 : init=-70.03810532250634
            I_inf = -I_L - I_Na - I_Kv3 - I_Kv1 + I_app
        """


class _HHneuron(Neuron):
    """
    PREDEFINED

    Class for building Hodgkin Huxley like neuron models.
    """

    # For reporting
    _instantiated = []

    def __init__(self):
        super().__init__(
            parameters=self._get_parameters(),
            equations=self._get_equations(),
            spike="(v > v_t) and (prev_v <= v_t)",
            reset="",
            name=self._get_name(),
            description=self._get_description(),
        )

        # For reporting
        self._instantiated.append(True)

    def _get_parameters(self):
        raise NotImplementedError(
            """
            To build a Hodgkin Huxley like neuron model, you need to define the
            _get_parameters() method.
            """
        )

    def _get_equations(self):
        raise NotImplementedError(
            """
            To build a Hodgkin Huxley like neuron model, you need to define the
            _get_equations() method.
            """
        )

    def _get_name(self):
        raise NotImplementedError(
            """
            To build a Hodgkin Huxley like neuron model, you need to define the
            _get_name() method.
            """
        )

    def _get_description(self):
        raise NotImplementedError(
            """
            To build a Hodgkin Huxley like neuron model, you need to define the
            _get_description() method.
            """
        )


class HHneuronBischop(_HHneuron):
    """
    PREDEFINED

    Hodgkin Huxley neuron model for striatal FSI from
    [Bischop et al. (2012)](https://doi.org/10.3389/fnmol.2012.00078).

    Variables to record:
        - prev_v
        - I_L
        - alpha_h
        - beta_h
        - h_inf
        - tau_h
        - h
        - alpha_m
        - beta_m
        - m_inf
        - m
        - I_Na
        - alpha_n1
        - beta_n1
        - n1_inf
        - tau_n1
        - n1
        - I_Kv1
        - alpha_n3
        - beta_n3
        - n3_inf
        - tau_n3
        - n3
        - I_Kv3
        - PV
        - PV_Mg
        - dPV_Ca_dt
        - PV_Ca
        - Ca
        - k_inf
        - tau_k
        - k
        - I_SK
        - a_inf
        - a
        - I_Ca
        - v
        - r
    """

    def __init__(self):
        self.bischop = _BischopStrings()

        super().__init__()

    def _get_parameters(self):
        return self.bischop.parameters_base

    def _get_equations(self):
        return self.bischop.equations_base + self.bischop.membrane_base

    def _get_name(self):
        return "H_and_H_Bischop"

    def _get_description(self):
        return (
            "Hodgkin Huxley neuron model for striatal FSI from Bischop et al. (2012)."
        )


class HHneuronBischopSyn(_HHneuron):
    """
    PREDEFINED

    Hodgkin Huxley neuron model for striatal FSI from
    [Bischop et al. (2012)](https://doi.org/10.3389/fnmol.2012.00078) with
    conductance-based synapses/currents for AMPA and GABA.

    Variables to record:
        - g_ampa
        - g_gaba
        - prev_v
        - I_L
        - alpha_h
        - beta_h
        - h_inf
        - tau_h
        - h
        - alpha_m
        - beta_m
        - m_inf
        - m
        - I_Na
        - alpha_n1
        - beta_n1
        - n1_inf
        - tau_n1
        - n1
        - I_Kv1
        - alpha_n3
        - beta_n3
        - n3_inf
        - tau_n3
        - n3
        - I_Kv3
        - PV
        - PV_Mg
        - dPV_Ca_dt
        - PV_Ca
        - Ca
        - k_inf
        - tau_k
        - k
        - I_SK
        - a_inf
        - a
        - I_Ca
        - v
        - r
    """

    def __init__(self):
        self.bischop = _BischopStrings()

        super().__init__()

    def _get_parameters(self):
        return self.bischop.parameters_conductance

    def _get_equations(self):
        return self.bischop.equations_conductance + self.bischop.membrane_conductance

    def _get_name(self):
        return "H_and_H_Bischop_syn"

    def _get_description(self):
        return """
                Hodgkin Huxley neuron model for striatal FSI from Bischop et al. (2012)
                with conductance-based synapses/currents for AMPA and GABA.
            """


class HHneuronCorbit(_HHneuron):
    """
    PREDEFINED

    Hodgkin Huxley neuron model for striatal FSI from
    [Corbit et al. (2016)](https://doi.org/10.1523/JNEUROSCI.0339-16.2016).

    Variables to record:
        - prev_v
        - I_L
        - m_Na
        - h_Na
        - I_Na
        - n_Kv3_inf
        - tau_n_Kv3_inf
        - n_Kv3
        - I_Kv3
        - m_Kv1
        - h_Kv1
        - I_Kv1
        - v
        - r
    """

    def __init__(self):
        self.corbit = _CorbitStrings()

        super().__init__()

    def _get_parameters(self):
        return self.corbit.parameters_base

    def _get_equations(self):
        return self.corbit.equations_base + self.corbit.membrane_base

    def _get_name(self):
        return "H_and_H_Corbit"

    def _get_description(self):
        return "Hodgkin Huxley neuron model for striatal FSI from Corbit et al. (2016)."


class HHneuronCorbitVoltageClamp(_HHneuron):
    """
    PREDEFINED

    Hodgkin Huxley neuron model for striatal FSI from
    [Corbit et al. (2016)](https://doi.org/10.1523/JNEUROSCI.0339-16.2016) with
    voltage clamp. Membrane potential v is clamped and I_inf can be recorded.

    Variables to record:
        - prev_v
        - I_L
        - m_Na
        - h_Na
        - I_Na
        - n_Kv3_inf
        - tau_n_Kv3_inf
        - n_Kv3
        - I_Kv3
        - m_Kv1
        - h_Kv1
        - I_Kv1
        - v
        - I_inf
        - r
    """

    def __init__(self):
        self.corbit = _CorbitStrings()

        super().__init__()

    def _get_parameters(self):
        return self.corbit.parameters_base

    def _get_equations(self):
        return self.corbit.equations_base + self.corbit.membrane_voltage_clamp

    def _get_name(self):
        return "H_and_H_Corbit_voltage_clamp"

    def _get_description(self):
        return """
                Hodgkin Huxley neuron model for striatal FSI from Corbit et al. (2016)
                with voltage clamp.
            """


class HHneuronCorbitSyn(_HHneuron):
    """
    PREDEFINED

    Hodgkin Huxley neuron model for striatal FSI from
    [Corbit et al. (2016)](https://doi.org/10.1523/JNEUROSCI.0339-16.2016) with
    conductance-based synapses/currents for AMPA and GABA.

    Variables to record:
        - g_ampa
        - g_gaba
        - prev_v
        - I_L
        - m_Na
        - h_Na
        - I_Na
        - n_Kv3_inf
        - tau_n_Kv3_inf
        - n_Kv3
        - I_Kv3
        - m_Kv1
        - h_Kv1
        - I_Kv1
        - v
        - r
    """

    def __init__(self):
        self.corbit = _CorbitStrings()

        super().__init__()

    def _get_parameters(self):
        return self.corbit.parameters_conductance

    def _get_equations(self):
        return self.corbit.equations_conductance + self.corbit.membrane_conductance

    def _get_name(self):
        return "H_and_H_Corbit_syn"

    def _get_description(self):
        return """
                Hodgkin Huxley neuron model for striatal FSI from Corbit et al. (2016)
                with conductance-based synapses/currents for AMPA and GABA.
            """


### create objects for backward compatibility
H_and_H_Bischop = HHneuronBischop()
H_and_H_Bischop_syn = HHneuronBischopSyn()
H_and_H_Corbit = HHneuronCorbit()
H_and_H_Corbit_syn = HHneuronCorbitSyn()
H_and_H_Corbit_voltage_clamp = HHneuronCorbitVoltageClamp()
