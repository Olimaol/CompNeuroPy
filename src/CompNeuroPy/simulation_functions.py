from ANNarchy import simulate, get_population, dt


def current_step(pop, t1=500, t2=500, a1=0, a2=100):
    """
    Stimulates a given population in two periods with two input currents.

    Args:
        pop (str):
            population name of population, which should be stimulated with input current
            neuron model of population has to contain "I_app" as input current
        t1 (int):
            time in ms before current step
        t2 (int):
            time in ms after current step
        a1 (int):
            current amplitude before current step
        a2 (int):
            current amplitude after current step

    Returns:
        return_dict (dict):
            dictionary containing:

            - duration (int): duration of the simulation
    """

    ### save prev input current
    I_prev = get_population(pop).I_app

    ### first/pre current step simulation
    get_population(pop).I_app = a1
    simulate(t1)

    ### second/post current step simulation
    get_population(pop).I_app = a2
    simulate(t2)

    ### reset input current to previous value
    get_population(pop).I_app = I_prev

    ### return some additional information which could be usefull
    return {"duration": t1 + t2}


def current_stim(pop, t=500, a=100):
    """
    Stimulates a given population during specified period 't' with input current with
    amplitude 'a', after this stimulation the current is reset to initial value
    (before stimulation).

    Args:
        pop (str):
            population name of population, which should be stimulated with input current
            neuron model of population has to contain "I_app" as input current
        t (int):
            duration in ms
        a (int):
            current amplitude
    """

    return current_step(pop, t1=t, t2=0, a1=a, a2=0)


def current_ramp(pop, a0, a1, dur, n):
    """
    Conducts multiple current stimulations with constantly changing current inputs.
    After this current_ramp stimulation the current amplitude is reset to the initial
    value (before current ramp).


    Args:
        pop (str):
            population name of population, which should be stimulated with input current
            neuron model of population has to contain "I_app" as input current
        a0 (int):
            initial current amplitude (of first stimulation)
        a1 (int):
            final current amplitude (of last stimulation)
        dur (int):
            duration of the complete current ramp (all stimulations)
        n (int):
            number of stimulations

    !!! warning
        dur/n should be divisible by the simulation time step without remainder

    Returns:
        return_dict (dict):
            dictionary containing:

            - da (int): current step size
            - dur_stim (int): duration of one stimulation

    Raises:
        AssertionError: if resulting duration of one stimulation is not divisible by the
            simulation time step without remainder
    """

    assert (dur / n) / dt() % 1 == 0, (
        "ERROR current_ramp: dur/n should result in a duration (for a single stimulation) which is divisible by the simulation time step (without remainder)\ncurrent duration = "
        + str(dur / n)
        + ", timestep = "
        + str(dt())
        + "!\n"
    )

    da = (a1 - a0) / (n - 1)  # for n stimulations only n-1 steps occur
    dur_stim = dur / n
    amp = a0
    for _ in range(n):
        current_stim(pop, t=dur_stim, a=amp)
        amp = amp + da

    return {"da": da, "dur_stim": dur_stim}


def increasing_current(pop, a0, da, nr_steps, dur_step):
    """
    Conducts multiple current stimulations with constantly increasing current inputs.
    After this increasing_current stimulation the current amplitude is reset to the
    initial value (before increasing_current).

    Args:
        pop (str):
            population name of population, which should be stimulated with input current
            neuron model of population has to contain "I_app" as input current
        a0 (int):
            initial current amplitude (of first stimulation)
        da (int):
            current step size
        nr_steps (int):
            number of stimulations
        dur_step (int):
            duration of one stimulation

    Returns:
        return_dict (dict):
            dictionary containing:

            - current_list (list): list of current amplitudes for each stimulation
    """
    current_list = []
    a = a0
    for _ in range(nr_steps):
        current_list.append(a)
        current_stim(pop, t=dur_step, a=a)
        a += da

    return {"current_list": current_list}
