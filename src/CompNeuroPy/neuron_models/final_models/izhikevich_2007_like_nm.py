from ANNarchy import Neuron
import re

### Izhikevich (2007)-like neuron model templates
### based on: Izhikevich, E. M. (2007). Dynamical Systems in Neuroscience. MIT Press.


_dv_default = "k*(v - v_r)*(v - v_t) - u + I_v"
_du_default = "a*(b*(v - v_r) - u)"
_syn_default = """
    dg_ampa/dt = -g_ampa/tau_ampa
    dg_gaba/dt = -g_gaba/tau_gaba
"""
_syn_noisy = """
    dg_ampa/dt = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rates_noise, -g_ampa/tau_ampa, -g_ampa/tau_ampa + increase_noise/dt)
    dg_gaba/dt = -g_gaba/tau_gaba
"""
_I_syn = "- neg(g_ampa*(v - E_ampa)) - pos(g_gaba*(v - E_gaba))"
_I_base_noise = """
    offset_base = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rate_base_noise, offset_base, Normal(0, 1) * base_noise)
    I_base      = base_mean + offset_base
"""


def _get_equation_izhikevich_2007(
    syn="", i_v="I_app", dv=_dv_default, du=_du_default, prefix="", affix=""
):
    return f"""
        {prefix}
        {syn}
        I_v        = {i_v}
        C * dv/dt  = {dv}
        du/dt      = {du}
        {affix}
    """


def _set_init(equations, init):
    """
    Set initial values for variables in the equations.

    Args:
        equations (str):
            Equations of the neuron model.
        init (dict):
            Dictionary with variable names as keys and initial values as values.

    Returns:
        equations (str):
            Equations of the neuron model with initial values set.
    """
    if len(init) == 0:
        return equations
    # go through each line of the equations and check if an initial value should be
    # set
    variable_name_list = []
    equations_line_split = equations.split("\n")
    for eqation_line_idx, equation_line in enumerate(equations_line_split):
        # remove all whitespace
        equation_line = equation_line.replace(" ", "")
        # check which variable is left of the equation
        if "/dt" in equation_line.split("=")[0]:
            # it's a differential equation
            # find the variable name using regular expression
            # the syntax is: "*d<variable_name>/dt*=*"
            variable_name = re.findall(".*?(d[_a-zA-Z]+)/dt.*=.*", equation_line)[0][1:]
        else:
            # it's not a differential equation
            # variable name is left of the equal sign
            variable_name = equation_line.split("=")[0]
        # check if the variable name is in the init dict
        if variable_name in init:
            # set the initial value
            equations_line_split[eqation_line_idx] = (
                f"{equations_line_split[eqation_line_idx]} : init={init[variable_name]}"
            )
        # add the variable name to the list
        variable_name_list.append(variable_name)
    # join the lines back together
    equations = "\n".join(equations_line_split)

    # print a warning if a variable in the init dict is not in the equations
    for key in init:
        if key not in variable_name_list:
            print(
                f"Warning: Variable {key} in init dict is not in the equations of the Izhikevich2007 model. It will be ignored."
            )

    return equations


############################################################################################
############################################################################################


class Izhikevich2007(Neuron):
    """
    TEMPLATE

    [Izhikevich (2007)](https://isbnsearch.org/isbn/9780262090438)-like neuron model.

    Parameters:
        C (float, optional):
            Membrane capacitance.
        k (float, optional):
            Scaling factor for the quadratic term in the membrane potential.
        v_r (float, optional):
            Resting membrane potential.
        v_t (float, optional):
            Instantaneous activation threshold potential.
        a (float, optional):
            Time scale of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential.
        d (float, optional):
            After-spike change of the recovery variable u.
        v_peak (float, optional):
            Spike cut-off value for the membrane potential.
        I_app (float, optional):
            External applied input current.
        params_for_pop (bool, optional):
            If True, the parameters are population-wide and not neuron-specific.
        init (dict, optional):
            Initial values for the variables.

    Variables to record:
        - I_v
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        C: float = 100.0,
        k: float = 0.7,
        v_r: float = -60.0,
        v_t: float = -40.0,
        a: float = 0.03,
        b: float = -2.0,
        c: float = -50.0,
        d: float = 100.0,
        v_peak: float = 35.0,
        I_app: float = 0.0,
        params_for_pop: bool = False,
        init: dict = {},
    ):
        # Create the arguments
        parameters = f"""
            C      = {C} {': population' if params_for_pop else ''} # pF
            k      = {k} {': population' if params_for_pop else ''} # pS * mV**-1
            v_r    = {v_r} {': population' if params_for_pop else ''} # mV
            v_t    = {v_t} {': population' if params_for_pop else ''} # mV
            a      = {a} {': population' if params_for_pop else ''} # ms**-1
            b      = {b} {': population' if params_for_pop else ''} # nS
            c      = {c} {': population' if params_for_pop else ''} # mV
            d      = {d} {': population' if params_for_pop else ''} # pA
            v_peak = {v_peak} {': population' if params_for_pop else ''} # mV
            I_app  = {I_app} # pA
        """

        # get equations
        equations = _get_equation_izhikevich_2007()

        # set initial values
        equations = _set_init(equations, init)

        # create the neuron
        super().__init__(
            parameters=parameters,
            equations=equations,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="Izhikevich2007",
            description="Neuron model equations from Izhikevich (2007).",
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2007RecCur(Neuron):
    """
    TEMPLATE

    [Izhikevich (2007)](https://isbnsearch.org/isbn/9780262090438)-like neuron model
    with separate currents to record.

    Parameters:
        C (float, optional):
            Membrane capacitance.
        k (float, optional):
            Scaling factor for the quadratic term in the membrane potential.
        v_r (float, optional):
            Resting membrane potential.
        v_t (float, optional):
            Instantaneous activation threshold potential.
        a (float, optional):
            Time scale of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential.
        d (float, optional):
            After-spike change of the recovery variable u.
        v_peak (float, optional):
            Spike cut-off value for the membrane potential.
        I_app (float, optional):
            External applied input current.
        params_for_pop (bool, optional):
            If True, the parameters are population-wide and not neuron-specific.
        init (dict, optional):
            Initial values for the variables.

    Variables to record:
        - I_v
        - v
        - u
        - I_u
        - I_k
        - I_a
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        C: float = 100.0,
        k: float = 0.7,
        v_r: float = -60.0,
        v_t: float = -40.0,
        a: float = 0.03,
        b: float = -2.0,
        c: float = -50.0,
        d: float = 100.0,
        v_peak: float = 35.0,
        I_app: float = 0.0,
        params_for_pop: bool = False,
        init: dict = {},
    ):
        # Create the arguments
        parameters = f"""
            C      = {C} {': population' if params_for_pop else ''} # pF
            k      = {k} {': population' if params_for_pop else ''} # pS * mV**-1
            v_r    = {v_r} {': population' if params_for_pop else ''} # mV
            v_t    = {v_t} {': population' if params_for_pop else ''} # mV
            a      = {a} {': population' if params_for_pop else ''} # ms**-1
            b      = {b} {': population' if params_for_pop else ''} # nS
            c      = {c} {': population' if params_for_pop else ''} # mV
            d      = {d} {': population' if params_for_pop else ''} # pA
            v_peak = {v_peak} {': population' if params_for_pop else ''} # mV
            I_app  = {I_app} # pA
        """

        affix = """
            I_u = -u
            I_k = k*(v - v_r)*(v - v_t)
            I_a = I_app
        """

        # get equations
        equations = _get_equation_izhikevich_2007(affix=affix)

        # set initial values
        equations = _set_init(equations, init)

        super().__init__(
            parameters=parameters,
            equations=equations,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="Izhikevich2007_record_currents",
            description="""
                Neuron model equations from Izhikevich (2007) with separate
                currents to record.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2007VoltageClamp(Neuron):
    """
    TEMPLATE

    [Izhikevich (2007)](https://isbnsearch.org/isbn/9780262090438)-like neuron model
    with voltage clamp to record I_inf.

    Parameters:
        C (float, optional):
            Membrane capacitance.
        k (float, optional):
            Scaling factor for the quadratic term in the membrane potential.
        v_r (float, optional):
            Resting membrane potential.
        v_t (float, optional):
            Instantaneous activation threshold potential.
        a (float, optional):
            Time scale of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential.
        d (float, optional):
            After-spike change of the recovery variable u.
        v_peak (float, optional):
            Spike cut-off value for the membrane potential.
        I_app (float, optional):
            External applied input current.
        params_for_pop (bool, optional):
            If True, the parameters are population-wide and not neuron-specific.
        init (dict, optional):
            Initial values for the variables.

    Variables to record:
        - I_v
        - v
        - u
        - I_inf
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        C: float = 100.0,
        k: float = 0.7,
        v_r: float = -60.0,
        v_t: float = -40.0,
        a: float = 0.03,
        b: float = -2.0,
        c: float = -50.0,
        d: float = 100.0,
        v_peak: float = 35.0,
        I_app: float = 0.0,
        params_for_pop: bool = False,
        init: dict = {},
    ):
        # Create the arguments
        parameters = f"""
            C      = {C} {': population' if params_for_pop else ''} # pF
            k      = {k} {': population' if params_for_pop else ''} # pS * mV**-1
            v_r    = {v_r} {': population' if params_for_pop else ''} # mV
            v_t    = {v_t} {': population' if params_for_pop else ''} # mV
            a      = {a} {': population' if params_for_pop else ''} # ms**-1
            b      = {b} {': population' if params_for_pop else ''} # nS
            c      = {c} {': population' if params_for_pop else ''} # mV
            d      = {d} {': population' if params_for_pop else ''} # pA
            v_peak = {v_peak} {': population' if params_for_pop else ''} # mV
            I_app  = {I_app} # pA
        """

        dv = "0"
        affix = f"I_inf = {_dv_default}"

        # get equations
        equations = _get_equation_izhikevich_2007(dv=dv, affix=affix)

        # set initial values
        equations = _set_init(equations, init)

        super().__init__(
            parameters=parameters,
            equations=equations,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="Izhikevich2007_voltage_clamp",
            description="""
                Neuron model equations from Izhikevich (2007) with voltage clamp
                to record I_inf.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2007Syn(Neuron):
    """
    TEMPLATE

    [Izhikevich (2007)](https://isbnsearch.org/isbn/9780262090438)-like neuron model
    with conductance-based synapses.

    Parameters:
        C (float, optional):
            Membrane capacitance.
        k (float, optional):
            Scaling factor for the quadratic term in the membrane potential.
        v_r (float, optional):
            Resting membrane potential.
        v_t (float, optional):
            Instantaneous activation threshold potential.
        a (float, optional):
            Time scale of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential.
        d (float, optional):
            After-spike change of the recovery variable u.
        v_peak (float, optional):
            Spike cut-off value for the membrane potential.
        I_app (float, optional):
            External applied input current.
        tau_ampa (float, optional):
            Time constant of the AMPA synapse.
        tau_gaba (float, optional):
            Time constant of the GABA synapse.
        E_ampa (float, optional):
            Reversal potential of the AMPA synapse.
        E_gaba (float, optional):
            Reversal potential of the GABA synapse.
        params_for_pop (bool, optional):
            If True, the parameters are population-wide and not neuron-specific.
        init (dict, optional):
            Initial values for the variables.

    Variables to record:
        - g_ampa
        - g_gaba
        - I_v
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        C: float = 100.0,
        k: float = 0.7,
        v_r: float = -60.0,
        v_t: float = -40.0,
        a: float = 0.03,
        b: float = -2.0,
        c: float = -50.0,
        d: float = 100.0,
        v_peak: float = 35.0,
        I_app: float = 0.0,
        tau_ampa: float = 10.0,
        tau_gaba: float = 10.0,
        E_ampa: float = 0.0,
        E_gaba: float = -90.0,
        params_for_pop: bool = False,
        init: dict = {},
    ):
        # Create the arguments
        parameters = f"""
            C      = {C} {': population' if params_for_pop else ''}
            k      = {k} {': population' if params_for_pop else ''}
            v_r    = {v_r} {': population' if params_for_pop else ''}
            v_t    = {v_t} {': population' if params_for_pop else ''}
            a      = {a} {': population' if params_for_pop else ''}
            b      = {b} {': population' if params_for_pop else ''}
            c      = {c} {': population' if params_for_pop else ''}
            d      = {d} {': population' if params_for_pop else ''}
            v_peak = {v_peak} {': population' if params_for_pop else ''}
            I_app  = {I_app} # pA
            tau_ampa = {tau_ampa} {': population' if params_for_pop else ''}
            tau_gaba = {tau_gaba} {': population' if params_for_pop else ''}
            E_ampa   = {E_ampa} {': population' if params_for_pop else ''}
            E_gaba   = {E_gaba} {': population' if params_for_pop else ''}
        """

        syn = _syn_default
        i_v = f"I_app {_I_syn}"

        # get equations
        equations = _get_equation_izhikevich_2007(syn=syn, i_v=i_v)

        # set initial values
        equations = _set_init(equations, init)

        super().__init__(
            parameters=parameters,
            equations=equations,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="Izhikevich2007_syn",
            description="""
                Neuron model equations from Izhikevich (2007) with conductance-based
                AMPA and GABA synapses/currents.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2007NoisyAmpa(Neuron):
    """
    TEMPLATE

    [Izhikevich (2007)](https://isbnsearch.org/isbn/9780262090438)-like neuron model
    with conductance-based AMPA and GABA synapses with noise in the AMPA conductance.

    Parameters:
        C (float, optional):
            Membrane capacitance.
        k (float, optional):
            Scaling factor for the quadratic term in the membrane potential.
        v_r (float, optional):
            Resting membrane potential.
        v_t (float, optional):
            Instantaneous activation threshold potential.
        a (float, optional):
            Time scale of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential.
        d (float, optional):
            After-spike change of the recovery variable u.
        v_peak (float, optional):
            Spike cut-off value for the membrane potential.
        I_app (float, optional):
            External applied input current.
        tau_ampa (float, optional):
            Time constant of the AMPA synapse.
        tau_gaba (float, optional):
            Time constant of the GABA synapse.
        E_ampa (float, optional):
            Reversal potential of the AMPA synapse.
        E_gaba (float, optional):
            Reversal potential of the GABA synapse.
        increase_noise (float, optional):
            Increase of AMPA conductance due to noise (equal to a Poisson distributed
            spike train as input).
        rates_noise (float, optional):
            Rate of the noise in the AMPA conductance.
        params_for_pop (bool, optional):
            If True, the parameters are population-wide and not neuron-specific.
        init (dict, optional):
            Initial values for the variables.

    Variables to record:
        - g_ampa
        - g_gaba
        - I_v
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        C: float = 100.0,
        k: float = 0.7,
        v_r: float = -60.0,
        v_t: float = -40.0,
        a: float = 0.03,
        b: float = -2.0,
        c: float = -50.0,
        d: float = 100.0,
        v_peak: float = 35.0,
        I_app: float = 0.0,
        tau_ampa: float = 10.0,
        tau_gaba: float = 10.0,
        E_ampa: float = 0.0,
        E_gaba: float = -90.0,
        increase_noise: float = 0.0,
        rates_noise: float = 0.0,
        params_for_pop: bool = False,
        init: dict = {},
    ):
        # Create the arguments
        parameters = f"""
            C              = {C} {': population' if params_for_pop else ''}
            k              = {k} {': population' if params_for_pop else ''}
            v_r            = {v_r} {': population' if params_for_pop else ''}
            v_t            = {v_t} {': population' if params_for_pop else ''}
            a              = {a} {': population' if params_for_pop else ''}
            b              = {b} {': population' if params_for_pop else ''}
            c              = {c} {': population' if params_for_pop else ''}
            d              = {d} {': population' if params_for_pop else ''}
            v_peak         = {v_peak} {': population' if params_for_pop else ''}
            tau_ampa       = {tau_ampa} {': population' if params_for_pop else ''}
            tau_gaba       = {tau_gaba} {': population' if params_for_pop else ''}
            E_ampa         = {E_ampa} {': population' if params_for_pop else ''}
            E_gaba         = {E_gaba} {': population' if params_for_pop else ''}
            I_app          = {I_app} # pA
            increase_noise = {increase_noise} {': population' if params_for_pop else ''}
            rates_noise    = {rates_noise}
        """

        syn = _syn_noisy
        i_v = f"I_app {_I_syn}"

        # get equations
        equations = _get_equation_izhikevich_2007(syn=syn, i_v=i_v)

        # set initial values
        equations = _set_init(equations, init)

        super().__init__(
            parameters=parameters,
            equations=equations,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="Izhikevich2007_noisy_AMPA",
            description="""
                Standard neuron model from Izhikevich (2007) with additional
                conductance based synapses for AMPA and GABA currents with noise
                in AMPA conductance.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2007NoisyBase(Neuron):
    """
    TEMPLATE

    [Izhikevich (2007)](https://isbnsearch.org/isbn/9780262090438)-like neuron model
    with conductance-based AMPA and GABA synapses with noise in the baseline current.

    Parameters:
        C (float, optional):
            Membrane capacitance.
        k (float, optional):
            Scaling factor for the quadratic term in the membrane potential.
        v_r (float, optional):
            Resting membrane potential.
        v_t (float, optional):
            Instantaneous activation threshold potential.
        a (float, optional):
            Time scale of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential.
        d (float, optional):
            After-spike change of the recovery variable u.
        v_peak (float, optional):
            Spike cut-off value for the membrane potential.
        I_app (float, optional):
            External applied input current.
        tau_ampa (float, optional):
            Time constant of the AMPA synapse.
        tau_gaba (float, optional):
            Time constant of the GABA synapse.
        E_ampa (float, optional):
            Reversal potential of the AMPA synapse.
        E_gaba (float, optional):
            Reversal potential of the GABA synapse.
        base_mean (float, optional):
            Mean of the baseline current.
        base_noise (float, optional):
            Standard deviation of the baseline current noise.
        rate_base_noise (float, optional):
            Rate of the noise update (Poisson distributed) in the baseline current.
        params_for_pop (bool, optional):
            If True, the parameters are population-wide and not neuron-specific.
        init (dict, optional):
            Initial values for the variables.

    Variables to record:
        - offset_base
        - I_base
        - g_ampa
        - g_gaba
        - I_v
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        C: float = 100.0,
        k: float = 0.7,
        v_r: float = -60.0,
        v_t: float = -40.0,
        a: float = 0.03,
        b: float = -2.0,
        c: float = -50.0,
        d: float = 100.0,
        v_peak: float = 35.0,
        I_app: float = 0.0,
        tau_ampa: float = 10.0,
        tau_gaba: float = 10.0,
        E_ampa: float = 0.0,
        E_gaba: float = -90.0,
        base_mean: float = 0.0,
        base_noise: float = 0.0,
        rate_base_noise: float = 0.0,
        params_for_pop: bool = False,
        init: dict = {},
    ):
        # Create the arguments
        parameters = f"""
            C              = {C} {': population' if params_for_pop else ''}
            k              = {k} {': population' if params_for_pop else ''}
            v_r            = {v_r} {': population' if params_for_pop else ''}
            v_t            = {v_t} {': population' if params_for_pop else ''}
            a              = {a} {': population' if params_for_pop else ''}
            b              = {b} {': population' if params_for_pop else ''}
            c              = {c} {': population' if params_for_pop else ''}
            d              = {d} {': population' if params_for_pop else ''}
            v_peak         = {v_peak} {': population' if params_for_pop else ''}
            tau_ampa       = {tau_ampa} {': population' if params_for_pop else ''}
            tau_gaba       = {tau_gaba} {': population' if params_for_pop else ''}
            E_ampa         = {E_ampa} {': population' if params_for_pop else ''}
            E_gaba         = {E_gaba} {': population' if params_for_pop else ''}
            I_app          = {I_app} # pA
            base_mean      = {base_mean}
            base_noise     = {base_noise}
            rate_base_noise = {rate_base_noise}
        """

        syn = _syn_default
        i_v = f"I_app {_I_syn} + I_base"
        prefix = _I_base_noise

        # get equations
        equations = _get_equation_izhikevich_2007(syn=syn, i_v=i_v, prefix=prefix)

        # set initial values
        equations = _set_init(equations, init)

        super().__init__(
            parameters=parameters,
            equations=equations,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="Izhikevich2007_noisy_base",
            description="""
                Standard neuron model from Izhikevich (2007) with additional
                conductance based synapses for AMPA and GABA currents and noisy
                baseline current.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2007FsiNoisyAmpa(Neuron):
    """
    TEMPLATE

    [Izhikevich (2007)](https://isbnsearch.org/isbn/9780262090438)-like neuron model
    for fast-spiking neurons, with conductance-based AMPA and GABA synapses with noise
    in the AMPA conductance.

    Parameters:
        C (float, optional):
            Membrane capacitance.
        k (float, optional):
            Scaling factor for the quadratic term in the membrane potential.
        v_r (float, optional):
            Resting membrane potential.
        v_t (float, optional):
            Instantaneous activation threshold potential.
        v_b (float, optional):
            Instantaneous activation threshold potential for the recovery variable u.
        a (float, optional):
            Time scale of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential.
        d (float, optional):
            After-spike change of the recovery variable u.
        v_peak (float, optional):
            Spike cut-off value for the membrane potential.
        I_app (float, optional):
            External applied input current.
        tau_ampa (float, optional):
            Time constant of the AMPA synapse.
        tau_gaba (float, optional):
            Time constant of the GABA synapse.
        E_ampa (float, optional):
            Reversal potential of the AMPA synapse.
        E_gaba (float, optional):
            Reversal potential of the GABA synapse.
        increase_noise (float, optional):
            Increase of AMPA conductance due to noise (equal to a Poisson distributed
            spike train as input).
        rates_noise (float, optional):
            Rate of the noise in the AMPA conductance.
        params_for_pop (bool, optional):
            If True, the parameters are population-wide and not neuron-specific.
        init (dict, optional):
            Initial values for the variables.

    Variables to record:
        - g_ampa
        - g_gaba
        - I_v
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        C: float = 20.0,
        k: float = 1.0,
        v_r: float = -55.0,
        v_t: float = -40.0,
        v_b: float = -55.0,
        a: float = 0.1,
        b: float = -2.0,
        c: float = -50.0,
        d: float = 2.0,
        v_peak: float = 25.0,
        I_app: float = 0.0,
        tau_ampa: float = 2.0,
        tau_gaba: float = 5.0,
        E_ampa: float = 0.0,
        E_gaba: float = -80.0,
        increase_noise: float = 0.0,
        rates_noise: float = 0.0,
        params_for_pop: bool = False,
        init: dict = {},
    ):
        # Create the arguments
        parameters = f"""
            C              = {C} {': population' if params_for_pop else ''}
            k              = {k} {': population' if params_for_pop else ''}
            v_r            = {v_r} {': population' if params_for_pop else ''}
            v_t            = {v_t} {': population' if params_for_pop else ''}
            v_b            = {v_b} {': population' if params_for_pop else ''}
            a              = {a} {': population' if params_for_pop else ''}
            b              = {b} {': population' if params_for_pop else ''}
            c              = {c} {': population' if params_for_pop else ''}
            d              = {d} {': population' if params_for_pop else ''}
            v_peak         = {v_peak} {': population' if params_for_pop else ''}
            tau_ampa       = {tau_ampa} {': population' if params_for_pop else ''}
            tau_gaba       = {tau_gaba} {': population' if params_for_pop else ''}
            E_ampa         = {E_ampa} {': population' if params_for_pop else ''}
            E_gaba         = {E_gaba} {': population' if params_for_pop else ''}
            I_app          = {I_app} # pA
            increase_noise = {increase_noise} {': population' if params_for_pop else ''}
            rates_noise    = {rates_noise}
        """

        syn = _syn_noisy
        i_v = f"I_app {_I_syn}"
        du = "if v<v_b: -a * u else: a * (b * (v - v_b)**3 - u)"

        # get equations
        equations = _get_equation_izhikevich_2007(syn=syn, i_v=i_v, du=du)

        # set initial values
        equations = _set_init(equations, init)

        super().__init__(
            parameters=parameters,
            equations=equations,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="Izhikevich2007_FSI_noisy_AMPA",
            description="""
                Standard neuron model from Izhikevich (2007) with additional
                conductance based synapses for AMPA and GABA currents with noise
                in AMPA conductance.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2007CorbitFsiNoisyAmpa(Neuron):
    """
    TEMPLATE

    [Izhikevich (2007)](https://isbnsearch.org/isbn/9780262090438)-like neuron model
    with conductance-based AMPA and GABA synapses with noise in the AMPA conductance.
    Additional slow currents were added to fit the striatal FSI neuron model from
    [Corbit et al. (2016)](https://doi.org/10.1523/JNEUROSCI.0339-16.2016). The
    additional currents should allow the neuron to produce late spiking.

    Parameters:
        C (float, optional):
            Membrane capacitance.
        k (float, optional):
            Scaling factor for the quadratic term in the membrane potential.
        b_n (float, optional):
            Sensitivity of the slow current n to the difference between the slow current
            s and the recovery variable u.
        a_s (float, optional):
            Time scale of the slow current s.
        a_n (float, optional):
            Time scale of the slow current n.
        v_r (float, optional):
            Resting membrane potential.
        v_t (float, optional):
            Instantaneous activation threshold potential.
        a (float, optional):
            Time scale of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential.
        d (float, optional):
            After-spike change of the recovery variable u.
        v_peak (float, optional):
            Spike cut-off value for the membrane potential.
        nonlin (float, optional):
            Nonlinearity of the input current. (1.0 = linear, 2.0 = square, etc.)
        I_app (float, optional):
            External applied input current.
        tau_ampa (float, optional):
            Time constant of the AMPA synapse.
        tau_gaba (float, optional):
            Time constant of the GABA synapse.
        E_ampa (float, optional):
            Reversal potential of the AMPA synapse.
        E_gaba (float, optional):
            Reversal potential of the GABA synapse.
        increase_noise (float, optional):
            Increase of AMPA conductance due to noise (equal to a Poisson distributed
            spike train as input).
        rates_noise (float, optional):
            Rate of the noise in the AMPA conductance.
        params_for_pop (bool, optional):
            If True, the parameters are population-wide and not neuron-specific.
        init (dict, optional):
            Initial values for the variables.

    Variables to record:
        - g_ampa
        - g_gaba
        - I_v
        - v
        - u
        - s
        - n
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        C: float = 20.0,
        k: float = 1.0,
        b_n: float = 0.1,
        a_s: float = 0.1,
        a_n: float = 0.1,
        v_r: float = -55.0,
        v_t: float = -40.0,
        a: float = 0.1,
        b: float = -2.0,
        c: float = -50.0,
        d: float = 2.0,
        v_peak: float = 25.0,
        nonlin: float = 0.1,
        I_app: float = 0.0,
        tau_ampa: float = 2.0,
        tau_gaba: float = 5.0,
        E_ampa: float = 0.0,
        E_gaba: float = -80.0,
        increase_noise: float = 0.0,
        rates_noise: float = 0.0,
        params_for_pop: bool = False,
        init: dict = {},
    ):
        # Create the arguments
        parameters = f"""
            C              = {C} {': population' if params_for_pop else ''}
            k              = {k} {': population' if params_for_pop else ''}
            b_n            = {b_n} {': population' if params_for_pop else ''}
            a_s            = {a_s} {': population' if params_for_pop else ''}
            a_n            = {a_n} {': population' if params_for_pop else ''}
            v_r            = {v_r} {': population' if params_for_pop else ''}
            v_t            = {v_t} {': population' if params_for_pop else ''}
            a              = {a} {': population' if params_for_pop else ''}
            b              = {b} {': population' if params_for_pop else ''}
            c              = {c} {': population' if params_for_pop else ''}
            d              = {d} {': population' if params_for_pop else ''}
            v_peak         = {v_peak} {': population' if params_for_pop else ''}
            nonlin         = {nonlin} {': population' if params_for_pop else ''}
            tau_ampa       = {tau_ampa} {': population' if params_for_pop else ''}
            tau_gaba       = {tau_gaba} {': population' if params_for_pop else ''}
            E_ampa         = {E_ampa} {': population' if params_for_pop else ''}
            E_gaba         = {E_gaba} {': population' if params_for_pop else ''}
            I_app          = {I_app} # pA
            increase_noise = {increase_noise} {': population' if params_for_pop else ''}
            rates_noise    = {rates_noise}
        """

        syn = _syn_noisy
        i_v = f"root_func(I_app {_I_syn}, nonlin) - n"
        affix = """
            ds/dt     = a_s*(pos(u)**0.1 - s)
            dn/dt     = a_n*(b_n*(pos(u)**0.1-s) - n)
        """

        # get equations
        equations = _get_equation_izhikevich_2007(syn=syn, i_v=i_v, affix=affix)

        # set initial values
        equations = _set_init(equations, init)

        super().__init__(
            parameters=parameters,
            equations=equations,
            functions="""
                root_func(x,y)=((abs(x))**(1/y))/((x+1e-20)/(abs(x)+ 1e-20))
            """,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="Izhikevich2007_Corbit_FSI_noisy_AMPA",
            description="""
                Standard neuron model from Izhikevich (2007) with additional
                conductance based synapses for AMPA and GABA currents with noise
                in AMPA conductance. Additional slow currents were added to fit
                the striatal FSI neuron model from Corbit et al. (2016).
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2007CorbitFsiNoisyBase(Neuron):
    """
    TEMPLATE

    [Izhikevich (2007)](https://isbnsearch.org/isbn/9780262090438)-like neuron model
    with conductance-based AMPA and GABA synapses with noise in the baseline current.
    Additional slow currents were added to fit the striatal FSI neuron model from
    [Corbit et al. (2016)](https://doi.org/10.1523/JNEUROSCI.0339-16.2016). The
    additional currents should allow the neuron to produce late spiking.

    Parameters:
        C (float, optional):
            Membrane capacitance.
        k (float, optional):
            Scaling factor for the quadratic term in the membrane potential.
        b_n (float, optional):
            Sensitivity of the slow current n to the difference between the slow current
            s and the recovery variable u.
        a_s (float, optional):
            Time scale of the slow current s.
        a_n (float, optional):
            Time scale of the slow current n.
        v_r (float, optional):
            Resting membrane potential.
        v_t (float, optional):
            Instantaneous activation threshold potential.
        a (float, optional):
            Time scale of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential.
        d (float, optional):
            After-spike change of the recovery variable u.
        v_peak (float, optional):
            Spike cut-off value for the membrane potential.
        nonlin (float, optional):
            Nonlinearity of the input current. (1.0 = linear, 2.0 = square, etc.)
        I_app (float, optional):
            External applied input current.
        tau_ampa (float, optional):
            Time constant of the AMPA synapse.
        tau_gaba (float, optional):
            Time constant of the GABA synapse.
        E_ampa (float, optional):
            Reversal potential of the AMPA synapse.
        E_gaba (float, optional):
            Reversal potential of the GABA synapse.
        base_mean (float, optional):
            Mean of the baseline current.
        base_noise (float, optional):
            Standard deviation of the baseline current noise.
        rate_base_noise (float, optional):
            Rate of the noise update (Poisson distributed) in the baseline current.
        params_for_pop (bool, optional):
            If True, the parameters are population-wide and not neuron-specific.
        init (dict, optional):
            Initial values for the variables.

    Variables to record:
        - offset_base
        - I_base
        - g_ampa
        - g_gaba
        - I_v
        - v
        - u
        - s
        - n
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        C: float = 20.0,
        k: float = 1.0,
        b_n: float = 0.1,
        a_s: float = 0.1,
        a_n: float = 0.1,
        v_r: float = -55.0,
        v_t: float = -40.0,
        a: float = 0.1,
        b: float = -2.0,
        c: float = -50.0,
        d: float = 2.0,
        v_peak: float = 25.0,
        nonlin: float = 0.1,
        I_app: float = 0.0,
        tau_ampa: float = 2.0,
        tau_gaba: float = 5.0,
        E_ampa: float = 0.0,
        E_gaba: float = -80.0,
        base_mean: float = 0.0,
        base_noise: float = 0.0,
        rate_base_noise: float = 0.0,
        params_for_pop: bool = False,
        init: dict = {},
    ):
        # Create the arguments
        parameters = f"""
            C              = {C} {': population' if params_for_pop else ''}
            k              = {k} {': population' if params_for_pop else ''}
            b_n            = {b_n} {': population' if params_for_pop else ''}
            a_s            = {a_s} {': population' if params_for_pop else ''}
            a_n            = {a_n} {': population' if params_for_pop else ''}
            v_r            = {v_r} {': population' if params_for_pop else ''}
            v_t            = {v_t} {': population' if params_for_pop else ''}
            a              = {a} {': population' if params_for_pop else ''}
            b              = {b} {': population' if params_for_pop else ''}
            c              = {c} {': population' if params_for_pop else ''}
            d              = {d} {': population' if params_for_pop else ''}
            v_peak         = {v_peak} {': population' if params_for_pop else ''}
            nonlin         = {nonlin} {': population' if params_for_pop else ''}
            tau_ampa       = {tau_ampa} {': population' if params_for_pop else ''}
            tau_gaba       = {tau_gaba} {': population' if params_for_pop else ''}
            E_ampa         = {E_ampa} {': population' if params_for_pop else ''}
            E_gaba         = {E_gaba} {': population' if params_for_pop else ''}
            I_app          = {I_app} # pA
            base_mean      = {base_mean}
            base_noise     = {base_noise}
            rate_base_noise = {rate_base_noise}
        """

        syn = _syn_default
        i_v = f"root_func(I_app {_I_syn}, nonlin) - n + I_base"
        prefix = _I_base_noise
        affix = """
            ds/dt     = a_s*(pos(u)**0.1 - s)
            dn/dt     = a_n*(b_n*(pos(u)**0.1-s) - n)
        """

        # get equations
        equations = _get_equation_izhikevich_2007(
            syn=syn, i_v=i_v, prefix=prefix, affix=affix
        )

        # set initial values
        equations = _set_init(equations, init)

        super().__init__(
            parameters=parameters,
            equations=equations,
            functions="""
                root_func(x,y)=((abs(x))**(1/y))/((x+1e-20)/(abs(x)+ 1e-20))
            """,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="Izhikevich2007_Corbit_FSI_noisy_base",
            description="""
                Standard neuron model from Izhikevich (2007) with additional
                conductance based synapses for AMPA and GABA currents with noise
                in the baseline current. Additional slow currents were added to fit
                the striatal FSI neuron model from Corbit et al. (2016).
            """,
        )

        # For reporting
        self._instantiated.append(True)


class Izhikevich2007NoisyAmpaOscillating(Neuron):
    """
    TEMPLATE

    [Izhikevich (2007)](https://isbnsearch.org/isbn/9780262090438)-like neuron model
    with conductance-based AMPA and GABA synapses with noise in the AMPA conductance.
    An additional oscillating current was added to the model.

    Parameters:
        C (float, optional):
            Membrane capacitance.
        k (float, optional):
            Scaling factor for the quadratic term in the membrane potential.
        v_r (float, optional):
            Resting membrane potential.
        v_t (float, optional):
            Instantaneous activation threshold potential.
        a (float, optional):
            Time scale of the recovery variable u.
        b (float, optional):
            Sensitivity of the recovery variable u to the the membrane potential v.
        c (float, optional):
            After-spike reset value of the membrane potential.
        d (float, optional):
            After-spike change of the recovery variable u.
        v_peak (float, optional):
            Spike cut-off value for the membrane potential.
        I_app (float, optional):
            External applied input current.
        tau_ampa (float, optional):
            Time constant of the AMPA synapse.
        tau_gaba (float, optional):
            Time constant of the GABA synapse.
        E_ampa (float, optional):
            Reversal potential of the AMPA synapse.
        E_gaba (float, optional):
            Reversal potential of the GABA synapse.
        increase_noise (float, optional):
            Increase of AMPA conductance due to noise (equal to a Poisson distributed
            spike train as input).
        rates_noise (float, optional):
            Rate of the noise in the AMPA conductance.
        freq (float, optional):
            Frequency of the oscillating current.
        amp (float, optional):
            Amplitude of the oscillating current.
        params_for_pop (bool, optional):
            If True, the parameters are population-wide and not neuron-specific.
        init (dict, optional):
            Initial values for the variables.

    Variables to record:
        - osc
        - g_ampa
        - g_gaba
        - I_v
        - v
        - u
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        C: float = 20.0,
        k: float = 1.0,
        v_r: float = -55.0,
        v_t: float = -40.0,
        a: float = 0.1,
        b: float = -2.0,
        c: float = -50.0,
        d: float = 2.0,
        v_peak: float = 25.0,
        I_app: float = 0.0,
        tau_ampa: float = 2.0,
        tau_gaba: float = 5.0,
        E_ampa: float = 0.0,
        E_gaba: float = -80.0,
        increase_noise: float = 0.0,
        rates_noise: float = 0.0,
        freq: float = 0.0,
        amp: float = 300.0,
        params_for_pop: bool = False,
        init: dict = {},
    ):
        # Create the arguments
        parameters = f"""
            C              = {C} {': population' if params_for_pop else ''}
            k              = {k} {': population' if params_for_pop else ''}
            v_r            = {v_r} {': population' if params_for_pop else ''}
            v_t            = {v_t} {': population' if params_for_pop else ''}
            a              = {a} {': population' if params_for_pop else ''}
            b              = {b} {': population' if params_for_pop else ''}
            c              = {c} {': population' if params_for_pop else ''}
            d              = {d} {': population' if params_for_pop else ''}
            v_peak         = {v_peak} {': population' if params_for_pop else ''}
            tau_ampa       = {tau_ampa} {': population' if params_for_pop else ''}
            tau_gaba       = {tau_gaba} {': population' if params_for_pop else ''}
            E_ampa         = {E_ampa} {': population' if params_for_pop else ''}
            E_gaba         = {E_gaba} {': population' if params_for_pop else ''}
            I_app          = {I_app} # pA
            increase_noise = {increase_noise} {': population' if params_for_pop else ''}
            rates_noise    = {rates_noise}
            freq           = {freq}
            amp            = {amp}
        """

        syn = _syn_noisy
        i_v = f"I_app {_I_syn} + osc"
        prefix = "osc = amp * sin(t * 2 * pi * (freq  /1000))"

        # get equations
        equations = _get_equation_izhikevich_2007(syn=syn, i_v=i_v, prefix=prefix)

        # set initial values
        equations = _set_init(equations, init)

        super().__init__(
            parameters=parameters,
            equations=equations,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="Izhikevich2007_noisy_AMPA_oscillating",
            description="""
                Standard neuron model from Izhikevich (2007) with additional
                conductance based synapses for AMPA and GABA currents with noise
                in AMPA conductance. An additional oscillating current was added
                to the model.
            """,
        )

        # For reporting
        self._instantiated.append(True)


### create objects for backwards compatibility
Izhikevich2007_record_currents = Izhikevich2007RecCur()
Izhikevich2007_voltage_clamp = Izhikevich2007VoltageClamp()
Izhikevich2007_syn = Izhikevich2007Syn()
Izhikevich2007_noisy_AMPA = Izhikevich2007NoisyAmpa()
Izhikevich2007_noisy_I = Izhikevich2007NoisyBase()
Izhikevich2007_fsi_noisy_AMPA = Izhikevich2007FsiNoisyAmpa()
Izhikevich2007_Corbit_FSI_noisy_AMPA = Izhikevich2007CorbitFsiNoisyAmpa()
Izhikevich2007_Corbit_FSI_noisy_I = Izhikevich2007CorbitFsiNoisyBase()
Izhikevich2007_noisy_AMPA_oscillating = Izhikevich2007NoisyAmpaOscillating()
