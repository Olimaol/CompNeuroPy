from ANNarchy import Neuron

### Izhikevich (2003)-like neuron model templates
### based on: Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6), 1569–1572. https://doi.org/10.1109/TNN.2003.820440


class Izhikevich2003FixedNoisyAmpa(Neuron):
    """
    TEMPLATE

    [Izhikevich (2003)](https://doi.org/10.1109/TNN.2003.820440)-like neuron model with
    additional conductance based synapses for AMPA and GABA currents with noise in AMPA
    conductance. Fixed means, the 3 factors of the quadratic equation cannot be changed.

    Parameters:
        a (float, optional):
            Time constant of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential v.
        d (float, optional):
            After-spike change of the recovery variable u.
        tau_ampa (float, optional):
            Time constant of the AMPA conductance.
        tau_gaba (float, optional):
            Time constant of the GABA conductance.
        E_ampa (float, optional):
            Reversal potential of the AMPA conductance.
        E_gaba (float, optional):
            Reversal potential of the GABA conductance.
        I_app (float, optional):
            External applied current.
        increase_noise (float, optional):
            Increase of the Poisson distributed (equivalent to a Poisson distributed
            spike train as input) noise in the AMPA conductance.
        rates_noise (float, optional):
            Rate of the Poisson distributed noise in the AMPA conductance.

    Variables to record:
        - g_ampa
        - g_gaba
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        a: float = 0,
        b: float = 0,
        c: float = 0,
        d: float = 0,
        tau_ampa: float = 1,
        tau_gaba: float = 1,
        E_ampa: float = 0,
        E_gaba: float = 0,
        I_app: float = 0,
        increase_noise: float = 0,
        rates_noise: float = 0,
    ):
        # Create the arguments
        parameters = f"""
            a              = {a} : population
            b              = {b} : population
            c              = {c} : population
            d              = {d} : population
            tau_ampa       = {tau_ampa} : population
            tau_gaba       = {tau_gaba} : population
            E_ampa         = {E_ampa} : population
            E_gaba         = {E_gaba} : population
            I_app          = {I_app}
            increase_noise = {increase_noise} : population
            rates_noise    = {rates_noise}
        """

        super().__init__(
            parameters=parameters,
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
            name="Izhikevich2003_fixed_noisy_AMPA",
            description="""
                Standard neuron model from Izhikevich (2003) with additional
                conductance-based synapses for AMPA and GABA currents with noise in AMPA
                conductance.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2003NoisyAmpa(Neuron):
    """
    TEMPLATE

    [Izhikevich (2003)](https://doi.org/10.1109/TNN.2003.820440)-like neuron model with
    additional conductance based synapses for AMPA and GABA currents with noise in AMPA
    conductance.

    Parameters:
        a (float, optional):
            Time constant of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential v.
        d (float, optional):
            After-spike change of the recovery variable u.
        n2 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        n1 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        n0 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        tau_ampa (float, optional):
            Time constant of the AMPA conductance.
        tau_gaba (float, optional):
            Time constant of the GABA conductance.
        E_ampa (float, optional):
            Reversal potential of the AMPA conductance.
        E_gaba (float, optional):
            Reversal potential of the GABA conductance.
        I_app (float, optional):
            External applied current.
        increase_noise (float, optional):
            Increase of the Poisson distributed (equivalent to a Poisson distributed
            spike train as input) noise in the AMPA conductance.
        rates_noise (float, optional):
            Rate of the Poisson distributed noise in the AMPA conductance.

    Variables to record:
        - g_ampa
        - g_gaba
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        a: float = 0,
        b: float = 0,
        c: float = 0,
        d: float = 0,
        n2: float = 0,
        n1: float = 0,
        n0: float = 0,
        tau_ampa: float = 1,
        tau_gaba: float = 1,
        E_ampa: float = 0,
        E_gaba: float = 0,
        I_app: float = 0,
        increase_noise: float = 0,
        rates_noise: float = 0,
    ):
        # Create the arguments
        parameters = f"""
            a              = {a} : population
            b              = {b} : population
            c              = {c} : population
            d              = {d} : population
            n2             = {n2} : population
            n1             = {n1} : population
            n0             = {n0} : population
            tau_ampa       = {tau_ampa} : population
            tau_gaba       = {tau_gaba} : population
            E_ampa         = {E_ampa} : population
            E_gaba         = {E_gaba} : population
            I_app          = {I_app}
            increase_noise = {increase_noise} : population
            rates_noise    = {rates_noise}
        """

        super().__init__(
            parameters=parameters,
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
            name="Izhikevich2003_noisy_AMPA",
            description="""
                Neuron model from Izhikevich (2003). With additional conductance based
                synapses for AMPA and GABA currents with noise in AMPA conductance.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2003NoisyAmpaNonlin(Neuron):
    """
    TEMPLATE

    [Izhikevich (2003)](https://doi.org/10.1109/TNN.2003.820440)-like neuron model with
    additional conductance based synapses for AMPA and GABA currents with noise in AMPA
    conductance. With nonlinear function for external current.

    Parameters:
        a (float, optional):
            Time constant of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential v.
        d (float, optional):
            After-spike change of the recovery variable u.
        n2 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        n1 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        n0 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        tau_ampa (float, optional):
            Time constant of the AMPA conductance.
        tau_gaba (float, optional):
            Time constant of the GABA conductance.
        E_ampa (float, optional):
            Reversal potential of the AMPA conductance.
        E_gaba (float, optional):
            Reversal potential of the GABA conductance.
        I_app (float, optional):
            External applied current.
        increase_noise (float, optional):
            Increase of the Poisson distributed (equivalent to a Poisson distributed
            spike train as input) noise in the AMPA conductance.
        rates_noise (float, optional):
            Rate of the Poisson distributed noise in the AMPA conductance.
        nonlin (float, optional):
            Exponent of the nonlinear function for the external current.

    Variables to record:
        - g_ampa
        - g_gaba
        - I
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        a: float = 0,
        b: float = 0,
        c: float = 0,
        d: float = 0,
        n2: float = 0,
        n1: float = 0,
        n0: float = 0,
        tau_ampa: float = 1,
        tau_gaba: float = 1,
        E_ampa: float = 0,
        E_gaba: float = 0,
        I_app: float = 0,
        increase_noise: float = 0,
        rates_noise: float = 0,
        nonlin: float = 1,
    ):
        # Create the arguments
        parameters = f"""
            a              = {a} : population
            b              = {b} : population
            c              = {c} : population
            d              = {d} : population
            n2             = {n2} : population
            n1             = {n1} : population
            n0             = {n0} : population
            tau_ampa       = {tau_ampa} : population
            tau_gaba       = {tau_gaba} : population
            E_ampa         = {E_ampa} : population
            E_gaba         = {E_gaba} : population
            I_app          = {I_app}
            increase_noise = {increase_noise} : population
            rates_noise    = {rates_noise}
            nonlin         = {nonlin} : population
        """

        super().__init__(
            parameters=parameters,
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
            name="Izhikevich2003_noisy_AMPA_nonlin",
            description="""
                Neuron model from Izhikevich (2003). With additional conductance based
                synapses for AMPA and GABA currents with noise in AMPA conductance.
                With nonlinear function for external current.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2003NoisyAmpaOscillating(Neuron):
    """
    TEMPLATE

    [Izhikevich (2003)](https://doi.org/10.1109/TNN.2003.820440)-like neuron model with
    additional conductance based synapses for AMPA and GABA currents with noise in AMPA
    conductance. With additional oscillation term.

    Parameters:
        a (float, optional):
            Time constant of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential v.
        d (float, optional):
            After-spike change of the recovery variable u.
        n2 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        n1 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        n0 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        tau_ampa (float, optional):
            Time constant of the AMPA conductance.
        tau_gaba (float, optional):
            Time constant of the GABA conductance.
        E_ampa (float, optional):
            Reversal potential of the AMPA conductance.
        E_gaba (float, optional):
            Reversal potential of the GABA conductance.
        I_app (float, optional):
            External applied current.
        increase_noise (float, optional):
            Increase of the Poisson distributed (equivalent to a Poisson distributed
            spike train as input) noise in the AMPA conductance.
        rates_noise (float, optional):
            Rate of the Poisson distributed noise in the AMPA conductance.
        freq (float, optional):
            Frequency of the oscillation term.
        amp (float, optional):
            Amplitude of the oscillation term.

    Variables to record:
        - osc
        - g_ampa
        - g_gaba
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        a: float = 0,
        b: float = 0,
        c: float = 0,
        d: float = 0,
        n2: float = 0,
        n1: float = 0,
        n0: float = 0,
        tau_ampa: float = 1,
        tau_gaba: float = 1,
        E_ampa: float = 0,
        E_gaba: float = 0,
        I_app: float = 0,
        increase_noise: float = 0,
        rates_noise: float = 0,
        freq: float = 0,
        amp: float = 6,
    ):
        # Create the arguments
        parameters = f"""
            a              = {a} : population
            b              = {b} : population
            c              = {c} : population
            d              = {d} : population
            n2             = {n2} : population
            n1             = {n1} : population
            n0             = {n0} : population
            tau_ampa       = {tau_ampa} : population
            tau_gaba       = {tau_gaba} : population
            E_ampa         = {E_ampa} : population
            E_gaba         = {E_gaba} : population
            I_app          = {I_app}
            increase_noise = {increase_noise} : population
            rates_noise    = {rates_noise}
            freq           = {freq}
            amp            = {amp}
        """

        super().__init__(
            parameters=parameters,
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
            name="Izhikevich2003_noisy_AMPA_oscillating",
            description="""
                Neuron model from Izhikevich (2003). With additional conductance based
                synapses for AMPA and GABA currents with noise in AMPA conductance.
                With additional oscillation term.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2003NoisyBase(Neuron):
    """
    TEMPLATE

    [Izhikevich (2003)](https://doi.org/10.1109/TNN.2003.820440)-like neuron model with
    additional conductance based synapses for AMPA and GABA currents and a noisy baseline
    current.

    Parameters:
        a (float, optional):
            Time constant of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential v.
        d (float, optional):
            After-spike change of the recovery variable u.
        n2 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        n1 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        n0 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        tau_ampa (float, optional):
            Time constant of the AMPA conductance.
        tau_gaba (float, optional):
            Time constant of the GABA conductance.
        E_ampa (float, optional):
            Reversal potential of the AMPA conductance.
        E_gaba (float, optional):
            Reversal potential of the GABA conductance.
        I_app (float, optional):
            External applied current.
        base_mean (float, optional):
            Mean of the baseline current.
        base_noise (float, optional):
            Standard deviation of the baseline current.
        rate_base_noise (float, optional):
            Rate of the Poisson distributed noise in the baseline current, i.e. how
            often the baseline current is changed randomly.

    Variables to record:
        - g_ampa
        - g_gaba
        - offset_base
        - I_base
        - I
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        a: float = 0,
        b: float = 0,
        c: float = 0,
        d: float = 0,
        n2: float = 0,
        n1: float = 0,
        n0: float = 0,
        tau_ampa: float = 1,
        tau_gaba: float = 1,
        E_ampa: float = 0,
        E_gaba: float = 0,
        I_app: float = 0,
        base_mean: float = 0,
        base_noise: float = 0,
        rate_base_noise: float = 0,
    ):
        # Create the arguments
        parameters = f"""
            a               = {a} : population
            b               = {b} : population
            c               = {c} : population
            d               = {d} : population
            n2              = {n2} : population
            n1              = {n1} : population
            n0              = {n0} : population
            tau_ampa        = {tau_ampa} : population
            tau_gaba        = {tau_gaba} : population
            E_ampa          = {E_ampa} : population
            E_gaba          = {E_gaba} : population
            I_app           = {I_app}
            base_mean       = {base_mean}
            base_noise      = {base_noise}
            rate_base_noise = {rate_base_noise}
        """

        super().__init__(
            parameters=parameters,
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
            name="Izhikevich2003_noisy_I",
            description="""
                Neuron model from Izhikevich (2003). With additional conductance based
                synapses for AMPA and GABA currents and a noisy baseline current.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2003NoisyBaseNonlin(Neuron):
    """
    TEMPLATE

    [Izhikevich (2003)](https://doi.org/10.1109/TNN.2003.820440)-like neuron model with
    additional conductance based synapses for AMPA and GABA currents and a noisy baseline
    current. With nonlinear function for external current.

    Parameters:
        a (float, optional):
            Time constant of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential v.
        d (float, optional):
            After-spike change of the recovery variable u.
        n2 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        n1 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        n0 (float, optional):
            Factor of the quadratic equation of the membrane potential v.
        tau_ampa (float, optional):
            Time constant of the AMPA conductance.
        tau_gaba (float, optional):
            Time constant of the GABA conductance.
        E_ampa (float, optional):
            Reversal potential of the AMPA conductance.
        E_gaba (float, optional):
            Reversal potential of the GABA conductance.
        I_app (float, optional):
            External applied current.
        base_mean (float, optional):
            Mean of the baseline current.
        base_noise (float, optional):
            Standard deviation of the baseline current.
        rate_base_noise (float, optional):
            Rate of the Poisson distributed noise in the baseline current, i.e. how
            often the baseline current is changed randomly.
        nonlin (float, optional):
            Exponent of the nonlinear function for the external current.

    Variables to record:
        - g_ampa
        - g_gaba
        - offset_base
        - I_base
        - I
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        a: float = 0,
        b: float = 0,
        c: float = 0,
        d: float = 0,
        n2: float = 0,
        n1: float = 0,
        n0: float = 0,
        tau_ampa: float = 1,
        tau_gaba: float = 1,
        E_ampa: float = 0,
        E_gaba: float = 0,
        I_app: float = 0,
        base_mean: float = 0,
        base_noise: float = 0,
        rate_base_noise: float = 0,
        nonlin: float = 1,
    ):
        # Create the arguments
        parameters = f"""
            a               = {a} : population
            b               = {b} : population
            c               = {c} : population
            d               = {d} : population
            n2              = {n2} : population
            n1              = {n1} : population
            n0              = {n0} : population
            tau_ampa        = {tau_ampa} : population
            tau_gaba        = {tau_gaba} : population
            E_ampa          = {E_ampa} : population
            E_gaba          = {E_gaba} : population
            I_app           = {I_app}
            base_mean       = {base_mean}
            base_noise      = {base_noise}
            rate_base_noise = {rate_base_noise}
            nonlin          = {nonlin} : population
        """

        super().__init__(
            parameters=parameters,
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
            name="Izhikevich2003_noisy_I_nonlin",
            description="""
                Neuron model from Izhikevich (2003). With additional conductance based
                synapses for AMPA and GABA currents and a noisy baseline current.
                With nonlinear function for external current.
            """,
        )

        # For reporting
        self._instantiated.append(True)


### create objects for backward compatibility
Izhikevich2003_noisy_AMPA = Izhikevich2003FixedNoisyAmpa()
Izhikevich2003_flexible_noisy_AMPA = Izhikevich2003NoisyAmpa()
Izhikevich2003_flexible_noisy_AMPA_nonlin = Izhikevich2003NoisyAmpaNonlin()
Izhikevich2003_flexible_noisy_AMPA_oscillating = Izhikevich2003NoisyAmpaOscillating()
Izhikevich2003_flexible_noisy_I = Izhikevich2003NoisyBase()
Izhikevich2003_flexible_noisy_I_nonlin = Izhikevich2003NoisyBaseNonlin()
