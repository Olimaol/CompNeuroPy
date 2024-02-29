from ANNarchy import simulate, get_population, dt


def attr_sim(pop: str, attr_dict, t=500):
    """
    Simulates a period 't' setting the attributes of a given population to the values
    specified in 'attr_list', after this simulation the attributes are reset to initial
    values (before simulation).

    Args:
        pop (str):
            population name of population whose attributes should be set
        attr_dict (dict):
            dictionary containing the attributes and their values
        t (int):
            duration in ms
    """

    ### save prev attr
    v_prev_dict = {
        attr: getattr(get_population(pop), attr) for attr in attr_dict.keys()
    }

    ### set attributes
    for attr, v in attr_dict.items():
        setattr(get_population(pop), attr, v)

    ### simulate
    simulate(t)

    ### reset attributes to previous values
    for attr, v in v_prev_dict.items():
        setattr(get_population(pop), attr, v)


def attribute_step(pop: str, attr, t1=500, t2=500, v1=0, v2=100):
    """
    Simulates an attribute step for a given population.

    Args:
        pop (str):
            population name of population whose attribute should be changed
        attr (str):
            name of attribute which should be changed
        t1 (int):
            time in ms before step
        t2 (int):
            time in ms after step
        v1 (int):
            value of attribute for t1
        v2 (int):
            value of attribute for t2

    Returns:
        return_dict (dict):
            dictionary containing:

            - duration (int): duration of the simulation
    """

    ### first/pre step simulation
    attr_sim(pop, {attr: v1}, t=t1)

    ### second/post step simulation
    attr_sim(pop, {attr: v2}, t=t2)

    ### return duration of the simulation
    return {"duration": t1 + t2}


def attr_ramp(pop: str, attr, v0, v1, dur, n):
    """
    Simulating while constantly changing the attribute of a given population.
    After this attr_ramp simulation the attribute value is reset to the initial
    value (before simulation).

    Args:
        pop (str):
            population name of population whose attribute should be changed
        attr (str):
            name of attribute which should be changed
        v0 (int):
            initial value of attribute (of first stimulation)
        v1 (int):
            final value of attribute (of last stimulation)
        dur (int):
            duration of the complete ramp simulation
        n (int):
            number of steps for changing the attribute

    !!! warning
        dur/n should be divisible by the simulation time step without remainder

    Returns:
        return_dict (dict):
            dictionary containing:

            - dv (int): step size of attribute
            - dur_stim (int): duration of single steps

    Raises:
        ValueError: if resulting duration of one stimulation is not divisible by the
            simulation time step without remainder
    """

    if (dur / n) / dt() % 1 != 0:
        raise ValueError(
            "ERROR current_ramp: dur/n should result in a duration (for a single stimulation) which is divisible by the simulation time step (without remainder)\ncurrent duration = "
            + str(dur / n)
            + ", timestep = "
            + str(dt())
            + "!\n"
        )

    dv = (v1 - v0) / (n - 1)  # for n stimulations only n-1 steps occur
    dur_stim = dur / n
    v = v0
    for _ in range(n):
        attr_sim(pop, attr_dict={attr: v}, t=dur_stim)
        v = v + dv

    return {"dv": dv, "dur_stim": dur_stim}


def increasing_attr(pop: str, attr, v0, dv, nr_steps, dur_step):
    """
    Conducts multiple simulations while constantly increasing the attribute of a given
    population. After this simulation the attribute value is reset to the initial value
    (before simulation).

    Args:
        pop (str):
            population name of population whose attribute should be changed
        v0 (int):
            initial attribute value (of first stimulation)
        dv (int):
            attribute step size
        nr_steps (int):
            number of simulations with different attribute values
        dur_step (int):
            duration of one step simulation

    Returns:
        return_dict (dict):
            dictionary containing:

            - attr_list (list): list of attribute values for each step simulation
    """
    attr_list = []
    v = v0
    for _ in range(nr_steps):
        attr_list.append(v)
        attr_sim(pop, {attr: v}, t=dur_step)
        v += dv

    return {"attr_list": attr_list}


def current_step(pop: str, t1=500, t2=500, a1=0, a2=100):
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
    return attribute_step(pop, "I_app", t1=t1, t2=t2, v1=a1, v2=a2)


def current_stim(pop: str, t=500, a=100):
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
    attr_sim(pop, {"I_app": a}, t=t)


def current_ramp(pop: str, a0, a1, dur, n):
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
        ValueError: if resulting duration of one stimulation is not divisible by the
            simulation time step without remainder
    """
    attr_ramp_return = attr_ramp(pop, "I_app", a0, a1, dur, n)
    return {"da": attr_ramp_return["dv"], "dur_stim": attr_ramp_return["dur_stim"]}


def increasing_current(pop: str, a0, da, nr_steps, dur_step):
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
    increasing_attr_return = increasing_attr(pop, "I_app", a0, da, nr_steps, dur_step)
    return {"current_list": increasing_attr_return["attr_list"]}
