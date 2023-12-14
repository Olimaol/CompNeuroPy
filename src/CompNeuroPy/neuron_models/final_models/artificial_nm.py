from ANNarchy import Neuron


### artificial neuron models


class IntegratorNeuron(Neuron):
    """
    TEMPLATE

    Integrator Neuron for stop_condition in spiking models.

    The variable g_ampa increases for incoming spikes (target ampa) and decreases
    exponentially with time constant tau. If g_ampa reaches a threshold, the neuron's
    variable decision, which is by default -1, changes to the neuron_id. This can be
    used to cause the stop_condition of ANNarchy's simulate_until() function
    (stop_codnition="decision>=0 : any"). In case of multiple integrator neurons,
    the neuron_id can be used to identify the neuron that reached the threshold.

    !!! warning
        You have to define the variable neuron_id for each neuron in the Integrator
        population.

    Parameters:
        tau (float, optional):
            Time constant in ms of the neuron. Default: 1.
        threshold (float, optional):
            Threshold for the decision g_ampa has to reach. Default: 1.

    Examples:
        ```python
        from ANNarchy import Population, simulate_until
        from CompNeuroPy.neuron_models import Integrator

        # Create a population of 10 integrator neurons
        integrator_neurons = Population(
            geometry=10,
            neuron=IntegratorNeuron(tau=1, threshold=1),
            stop_condition="decision>=0 : any",
            name="integrator_neurons",)

        # set the neuron_id for each neuron
        integrator_neurons.neuron_id = range(10)

        # simulate until one neuron reaches the threshold
        simulate_until(max_duration=1000, population=integrator_neurons)

        # check if simulation stop due to stop_codnition and which neuron reached the
        # threshold
        if (integrator_neurons.decision >= 0).any():
            neurons_reached_thresh = integrator_neurons.neuron_id[
                integrator_neurons.decision >= 0
            ]
            print(f"Neuron(s) {neurons_reached_thresh} reached threshold.")
        else:
            print("No neuron reached threshold.")
        ```

    Variables to record:
        - g_ampa
        - decision
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(self, tau: float = 1, threshold: float = 1):
        # Create the arguments
        parameters = f"""
            tau = {tau} : population
            threshold = {threshold} : population
            neuron_id = 0
        """

        super().__init__(
            parameters=parameters,
            equations="""
                dg_ampa/dt = - g_ampa / tau
                ddecision/dt = 0 : init = -1
            """,
            spike="""
                g_ampa >= threshold
            """,
            reset="""
                decision = neuron_id
            """,
            name="integrator_neuron",
            description="""
                Integrator Neuron, which integrates incoming spikes with value g_ampa
                and emits a spike when reaching a threshold. After spike decision
                changes, which can be used as for stop condition""",
        )

        # For reporting
        self._instantiated.append(True)


class IntegratorNeuronSimple(Neuron):
    """
    TEMPLATE

    Integrator Neuron for stop_condition in spiking models.

    The variable g_ampa increases for incoming spikes (target ampa) and decreases
    exponentially with time constant tau. You can check g_ampa and use it for the
    stop_condition of ANNarchy's simulate_until() function
    (stop_codnition="g_ampa>=some_value : any"). In case of multiple integrator neurons,
    the neuron_id can be used to identify the neuron that reached the threshold.

    !!! warning
        You have to define the variable neuron_id for each neuron in the Integrator
        population.

    Parameters:
        tau (float, optional):
            Time constant in ms of the neuron. Default: 1.

    Examples:
        ```python
        from ANNarchy import Population, simulate_until
        from CompNeuroPy.neuron_models import Integrator

        # Create a population of 10 integrator neurons
        integrator_neurons = Population(
            geometry=10,
            neuron=IntegratorNeuronSimple(tau=1),
            stop_condition="g_ampa>=5 : any",
            name="integrator_neurons",)

        # set the neuron_id for each neuron
        integrator_neurons.neuron_id = range(10)

        # simulate until one neuron reaches the threshold
        simulate_until(max_duration=1000, population=integrator_neurons)

        # check if simulation stop due to stop_codnition and which neuron reached the
        # threshold
        if (integrator_neurons.g_ampa >= 5).any():
            neurons_reached_thresh = integrator_neurons.neuron_id[
                integrator_neurons.g_ampa >= 5
            ]
            print(f"Neuron(s) {neurons_reached_thresh} reached threshold.")
        else:
            print("No neuron reached threshold.")
        ```

    Variables to record:
        - g_ampa
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(self, tau: float = 1):
        # Create the arguments
        parameters = f"""
            tau = {tau} : population
            neuron_id = 0
        """

        super().__init__(
            parameters=parameters,
            equations="""
                dg_ampa/dt = - g_ampa / tau
                r = 0
            """,
            name="integrator_neuron_simple",
            description="""
                Integrator Neuron, which integrates incoming spikes with value g_ampa,
                which can be used as a stop condition
            """,
        )

        # For reporting
        self._instantiated.append(True)


class PoissonNeuron(Neuron):
    """
    TEMPLATE

    Poisson neuron whose rate can be specified and is reached instantaneous. The
    neuron emits spikes following a Poisson distribution, the average firing rate
    is given by the parameter rates.

    Parameters:
        rates (float, optional):
            The average firing rate of the neuron in Hz. Default: 0.

    Variables to record:
        - p
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(self, rates: float = 0):
        # Create the arguments
        parameters = f"""
            rates = {rates}
        """

        super().__init__(
            parameters=parameters,
            equations="""
                p = Uniform(0.0, 1.0) * 1000.0 / dt
            """,
            spike="""
                p <= rates
            """,
            reset="""
                p = 0.0
            """,
            name="poisson_neuron",
            description="""
                Poisson neuron whose rate can be specified and is reached instantaneous.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class PoissonNeuronUpDown(Neuron):
    """
    TEMPLATE

    The neuron emits spikes following a Poisson distribution, the average firing rate is
    given by the parameter rates and is reached with time constants tau_up and tau_down.

    Attributes:
        rates (float, optional):
            The average firing rate of the neuron in Hz. Default: 0.
        tau_up (float, optional):
            Time constant in ms for increasing the firing rate. Default: 1.
        tau_down (float, optional):
            Time constant in ms for decreasing the firing rate. Default: 1.
    """

    # For reporting
    _instantiated = []

    def __init__(self, rates: float = 0, tau_up: float = 1, tau_down: float = 1):
        # Create the arguments
        parameters = f"""
            rates = {rates}
            tau_up = {tau_up}
            tau_down = {tau_down}
        """

        super().__init__(
            parameters=parameters,
            equations="""
                p = Uniform(0.0, 1.0) * 1000.0 / dt
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
            description="""Poisson neuron whose rate can be specified and is reached
                with time constants tau_up and tau_down.
            """,
        )

        # For reporting
        self._instantiated.append(True)


class PoissonNeuronSin(Neuron):
    """
    TEMPLATE

    Neuron emitting spikes following a Poisson distribution, the average firing rate
    is given by a sinus function.

    Parameters:
        amplitude (float, optional):
            Amplitude of the sinus function. Default: 0.
        base (float, optional):
            Base (offset) of the sinus function. Default: 0.
        frequency (float, optional):
            Frequency of the sinus function. Default: 0.
        phase (float, optional):
            Phase of the sinus function. Default: 0.

    Variables to record:
        - rates
        - p
        - r
    """

    # For reporting
    _instantiated = []

    def __init__(
        self,
        amplitude: float = 0,
        base: float = 0,
        frequency: float = 0,
        phase: float = 0,
    ):
        # Create the arguments
        parameters = f"""
            amplitude = {amplitude}
            base = {base}
            frequency = {frequency}
            phase = {phase}
        """

        super().__init__(
            parameters=parameters,
            equations="""
                rates = amplitude * sin((2*pi*frequency)*(t/1000-phase)) + base
                p     = Uniform(0.0, 1.0) * 1000.0 / dt
            """,
            spike="""
                p <= rates
            """,
            reset="""
                p = 0.0
            """,
            name="poisson_neuron_sin",
            description="Poisson neuron whose rate varies with a sinus function.",
        )

        # For reporting
        self._instantiated.append(True)


### create neurons models with old names for backwards compatibility
integrator_neuron = IntegratorNeuron()
integrator_neuron_simple = IntegratorNeuronSimple()
poisson_neuron = PoissonNeuron()
poisson_neuron_up_down = PoissonNeuronUpDown()
poisson_neuron_sin = PoissonNeuronSin()
