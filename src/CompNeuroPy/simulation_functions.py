from CompNeuroPy import ann
from CompNeuroPy import analysis_functions as af
import numpy as np
from typing import Callable


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

    Returns:
        attr_list_dict (dict):
            dictionary containing the attribute values for each time step, keys are the
            attribute names
    """

    ### save prev attr
    v_prev_dict = {
        attr: getattr(ann.get_population(pop), attr) for attr in attr_dict.keys()
    }

    ### set attributes
    for attr, v in attr_dict.items():
        setattr(ann.get_population(pop), attr, v)

    ### simulate
    ann.simulate(t)

    ### reset attributes to previous values
    for attr, v in v_prev_dict.items():
        setattr(ann.get_population(pop), attr, v)

    ### return the values for the attribute for each time step
    attr_list_dict = {
        attr: [v] * int(round(t / ann.dt())) for attr, v in attr_dict.items()
    }
    return attr_list_dict


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
            - v_arr (np.array): array of attribute values for each time step
    """
    v_list = []
    ### first/pre step simulation
    attr_list_dict = attr_sim(pop, {attr: v1}, t=t1)
    v_list.extend(attr_list_dict[attr])
    ### second/post step simulation
    attr_list_dict = attr_sim(pop, {attr: v2}, t=t2)
    v_list.extend(attr_list_dict[attr])

    ### return duration of the simulation and the attribute values
    return {"duration": t1 + t2, "v_arr": np.array(v_list)}


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
            - v_arr (np.array): array of attribute values for each time step

    Raises:
        ValueError: if resulting duration of one stimulation is not divisible by the
            simulation time step without remainder
    """

    if (dur / n) / ann.dt() % 1 != 0:
        raise ValueError(
            "ERROR current_ramp: dur/n should result in a duration (for a single stimulation) which is divisible by the simulation time step (without remainder)\ncurrent duration = "
            + str(dur / n)
            + ", timestep = "
            + str(ann.dt())
            + "!\n"
        )

    dv = (v1 - v0) / (n - 1)  # for n stimulations only n-1 steps occur
    dur_stim = dur / n
    v = v0
    v_list = []
    for _ in range(n):
        attr_list_dict = attr_sim(pop, attr_dict={attr: v}, t=dur_stim)
        v_list.extend(attr_list_dict[attr])
        v = v + dv

    return {"dv": dv, "dur_stim": dur_stim, "v_arr": np.array(v_list)}


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
            - v_arr (np.array): array of attribute values for each time step
    """
    attr_list = []
    v = v0
    v_list = []
    for _ in range(nr_steps):
        attr_list.append(v)
        attr_list_dict = attr_sim(pop, {attr: v}, t=dur_step)
        v_list.extend(attr_list_dict[attr])
        v += dv

    return {"attr_list": attr_list, "v_arr": np.array(v_list)}


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
            - current_arr (np.array): array of current values for each time step
    """
    attribute_step_ret = attribute_step(pop, "I_app", t1=t1, t2=t2, v1=a1, v2=a2)
    return {
        "duration": attribute_step_ret["duration"],
        "current_arr": attribute_step_ret["v_arr"],
    }


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

    Returns:
        current_arr (np.array):
            array of current values for each time step
    """
    attr_list_dict = attr_sim(pop, {"I_app": a}, t=t)
    return np.array(attr_list_dict["I_app"])


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
            - current_arr (np.array): array of current values for each time step

    Raises:
        ValueError: if resulting duration of one stimulation is not divisible by the
            simulation time step without remainder
    """
    attr_ramp_return = attr_ramp(pop, "I_app", a0, a1, dur, n)
    return {
        "da": attr_ramp_return["dv"],
        "dur_stim": attr_ramp_return["dur_stim"],
        "current_arr": attr_ramp_return["v_arr"],
    }


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
            - current_arr (np.array): array of current values for each time step
    """
    increasing_attr_return = increasing_attr(pop, "I_app", a0, da, nr_steps, dur_step)
    return {
        "current_list": increasing_attr_return["attr_list"],
        "current_arr": increasing_attr_return["v_arr"],
    }


class SimulationEvents:
    """
    Class to create a Simulation consiting of multiple events. Add the effects
    (functions) of the events in a class which inherits from SimulationEvents. Within
    the effect functions you can use the attributes of the class which inherits from
    SimulationEvents. Do never simulate within the effect functions of the events. The
    simulation is done between the events.

    !!! warning
        The onset of events and trigger times should be given in simulation steps (not
        in ms). The 'end' event has to be triggered to end the simulation (otherwise
        it will be triggered right at the beginning).

    Example:
        ```python
        from CompNeuroPy import SimulationEvents

        ### define a class which inherits from SimulationEvents
        ### define the effects of the events in the class
        class MySim(SimulationEvents):

            def __init__(
                self,
                p=0.8,
                verbose=False,
            ):
                ### set attributes which should be used in the effect functions
                self.p = p
                super().__init__(verbose=verbose)

            def effect1(self):
                ### set the parameter of a population to the value of p
                pop.parameter = self.p

            def effect2(self):
                ### set the parameter of a population to 0
                pop.parameter = 0

        ### create the simulation object
        my_sim = MySim()

        ### add events to the simulation
        ### start event right at the beginning which triggers event1 after 100 ms
        my_sim.add_event(name="start", trigger={"event1": 100})
        ### event1 causes effect1 and triggers event2 after 200 ms
        my_sim.add_event(name="event1", effect=my_sim.effect1, trigger={"event2": 200})
        ### event2 causes effect2 and triggers end event after 300 ms
        my_sim.add_event(name="event2", effect=my_sim.effect2, trigger={"end": 300})

        ### run the simulation
        my_sim.run()
        ```
    """

    def __init__(self, verbose=False):
        """
        Args:
            verbose (bool):
                if True, additional information is printed during simulation
        """
        ### set verbose
        self.verbose = verbose
        ### initialize events
        self._initialize()
        ### list for storing added events, without changing them
        self.stored_event_list = []
        self.called_during_restore = False
        ### add the end event
        self.add_event(name="end", effect=self._end_sim)

    def _initialize(self):
        """
        initialize locals
        """
        if self.verbose:
            print("initialize locals")
        ### list of events
        self.event_list = []
        self.event_name_list = []
        ### as long as end == False simulation runs
        self.end = False
        ### if events occur depends on happened events
        self.happened_event_list = []
        ### initialize model triggers empty, before first simulation, there should not be model_trigger_events
        ### model_trigger_list = name of populations of which the decision should be checked
        self.model_trigger_list = []
        self.past_model_trigger_list = []

    def add_event(
        self,
        name: str,
        onset: int = None,
        model_trigger: str = None,
        requirement_string: str = None,
        effect: Callable = None,
        trigger: dict[str, int | Callable[[], int]] = None,
    ):
        """
        Adds an event to the simulation. You always have to trigger the end event to end
        the simulation.

        Args:
            name (str):
                name of the event
            onset (int):
                time in simulation steps when the event should occur
            model_trigger (str):
                name of population which can trigger the event (by setting variable
                decision to -1)
            requirement_string (str):
                string containing the requirements for the event to occur TODO: replace with function
            effect (function):
                Function which is executed during the event. Within the effect function
                you can use the attributes of the class which inherits from
                SimulationEvents.
            trigger (dict):
                dictionary containing the names of other events as keys and the
                relative time in simulation steps to the onset of the current event as
                values. The values can also be callable functions which return the
                time (without any aruments). They are called when this event is triggered.
        """
        self.event_list.append(
            self._Event(
                trial_procedure=self,
                name=name,
                onset=onset,
                model_trigger=model_trigger,
                requirement_string=requirement_string,
                effect=effect,
                trigger=trigger,
            )
        )
        self.event_name_list.append(name)

        if not self.called_during_restore:
            self.stored_event_list.append(
                {
                    "name": name,
                    "onset": onset,
                    "model_trigger": model_trigger,
                    "requirement_string": requirement_string,
                    "effect": effect,
                    "trigger": trigger,
                }
            )

    def _restore_event_list(self):
        """
        Restore the event list after simulation to the state before the first call of
        run. To be able to run the simulation multiple times.
        """
        self.called_during_restore = True
        for event in self.stored_event_list:
            self.add_event(**event)
        self.called_during_restore = False

    def run(self):
        """
        Run the simulation. The simulation runs until the end event is triggered. The
        simulation can be run multiple times by calling this function multiple times.
        """
        ### for all events with given onset, change onset to current step + onset
        ### (otherwise run would need to be called at time 0)
        for event in self.event_list:
            if event.onset is not None:
                event.onset = ann.get_current_step() + event.onset

        ### check if there are events which have no onset and are not triggered by other
        ### events and have no model_trigger --> they would never start
        ### --> set their onset to current step --> they are run directly after calling run
        triggered_events = []
        for event in self.event_list:
            if event.trigger is not None:
                triggered_events.extend(list(event.trigger.keys()))
        for event in self.event_list:
            if (
                event.onset is None
                and event.model_trigger is None
                and event.name not in triggered_events
            ):
                event.onset = ann.get_current_step()
                if self.verbose:
                    print(event.name, "set onset to start of run")

        ### run simulation
        while not (self.end):
            ### check if model triggers were activated --> if yes run the corresponding events, model_trigger events can trigger other events (with onset) --> run current_step events after model trigger events
            ### if that's the case --> model trigger event would run twice (because during first run it gets an onset) --> define here run_event_list which prevents events run twice
            self.run_event_list = []
            self._run_model_trigger_events()
            ### run the events of the current time, based on mode and happened events
            self._run_current_events()
            ### if event triggered end --> end simulation / skip rest
            if self.end:
                if self.verbose:
                    print("end event triggered --> end simulation")
                continue
            ### check then next events occur
            next_events_time = self._get_next_events_time()
            ### check if there are model triggers
            self.model_trigger_list = self._get_model_trigger_list()
            ### simulate until next event(s) or model triggers
            if self.verbose:
                print("check_triggers:", self.model_trigger_list)
            if len(self.model_trigger_list) > 1:
                ### multiple model triggers
                ann.simulate_until(
                    max_duration=next_events_time,
                    population=[
                        ann.get_population(pop_name)
                        for pop_name in self.model_trigger_list
                    ],
                    operator="or",
                )
            elif len(self.model_trigger_list) > 0:
                ### a single model trigger
                ann.simulate_until(
                    max_duration=next_events_time,
                    population=ann.get_population(self.model_trigger_list[0]),
                )
            else:
                ### no model_triggers
                ann.simulate(next_events_time)

        ### after run finishes initialize again
        self._initialize()

        ### restore event_list
        self._restore_event_list()

    def _run_current_events(self):
        """
        Run all events with start == current step
        """
        ### run all events of the current step
        ### repeat this until no event was run, because events can set the onset of other events to the current step
        ### due to repeat --> prevent that same event is run twice
        event_run = True
        while event_run:
            event_run = False
            for event in self.event_list:
                if (
                    event.onset == ann.get_current_step()
                    and not (event.name in self.run_event_list)
                    and event._check_requirements()
                ):
                    event.run()
                    event_run = True
                    self.run_event_list.append(event.name)

    def _run_model_trigger_events(self):
        """
        check the current model triggers stored in self.model_trigger_list
        if they are activated --> run corresponding events
        prevent that these model triggers are stored again in self.model_trigger_list
        """
        ### loop to check if model trigger got active
        for model_trigger in self.model_trigger_list:
            ### TODO this is not generalized yet, only works if the model_trigger populations have the variable decision which is set to -1 if the model trigger is active
            if int(ann.get_population(model_trigger).decision[0]) == -1:
                ### -1 means got active
                ### find the events triggerd by the model_trigger and run them
                for event in self.event_list:
                    if event.model_trigger == model_trigger:
                        event.run()
                        self.run_event_list.append(event.name)
                ### prevent that these model_triggers are used again
                self.past_model_trigger_list.append(model_trigger)

    def _get_next_events_time(self):
        """
        go through all events and get onsets
        get onset which are > current_step
        return smallest diff in ms (ms value = full timesteps!)

        Returns:
            time (float):
                time in ms until the next event, rounded to full timesteps
        """
        next_event_time = np.inf
        for event in self.event_list:
            ### skip events without onset
            if event.onset == None:
                continue
            ### check if onset in the future and nearest
            if (
                event.onset > ann.get_current_step()
                and (event.onset - ann.get_current_step()) < next_event_time
            ):
                next_event_time = event.onset - ann.get_current_step()
        ### return difference (simulation duration until nearest next event) in ms, round to full timesteps
        return round(next_event_time * ann.dt(), af.get_number_of_decimals(ann.dt()))

    def _get_model_trigger_list(self):
        """
        check if there are events with model_triggers
        check if these model triggers already happened
        check if the requirements of the events are met
        not happend + requirements met --> add model_trigger to model_trigger_list
        returns the (new) model_trigger_list

        Returns:
            model_trigger_list (list):
                list of model triggers which are not in past_model_trigger_list and
                have their requirements met
        """
        ret = []
        for event in self.event_list:
            if event.model_trigger != None:
                if (
                    not (event.model_trigger in self.past_model_trigger_list)
                    and event._check_requirements()
                ):
                    ret.append(event.model_trigger)
        return ret

    def _end_sim(self):
        """
        Event to end the simulation
        """
        self.end = True

    class _Event:
        """
        Class for events in the simulation
        """

        def __init__(
            self,
            trial_procedure,
            name,
            onset=None,
            model_trigger=None,
            requirement_string=None,
            effect=None,
            trigger=None,
        ):
            """
            Args:
                trial_procedure (SimulationEvents):
                    SimulationEvents object
                name (str):
                    name of the event
                onset (int):
                    time in simulation steps when the event should occur
                model_trigger (str):
                    name of population which can trigger the event (by setting variable
                    decision to -1)
                requirement_string (str):
                    string containing the requirements for the event to occur TODO: replace with function
                effect (function):
                    function which is executed during the event
                trigger (dict):
                    dictionary containing the names of other events as keys and the
                    relative time in simulation steps to the onset of the current event as
                    values. The values can also be callable functions which return the
                    time (without any aruments). They are called when this event is triggered.
            """
            self.trial_procedure = trial_procedure
            self.name = name
            self.onset = onset
            self.model_trigger = model_trigger
            self.requirement_string = requirement_string
            self.effect = effect
            self.trigger = trigger

        def run(self):
            """
            Run the event i.e. execute the effect of the event and trigger other events
            """
            ### check requirements
            if self._check_requirements():
                ### run the event
                if self.trial_procedure.verbose:
                    print("run event:", self.name, ann.get_time())
                ### for events which are triggered by model --> set onset
                if self.onset == None:
                    self.onset = ann.get_current_step()
                ### run the effect
                if self.effect is not None:
                    self.effect()
                ### trigger other events
                if self.trigger is not None:
                    ### loop over all triggered events
                    for name, delay in self.trigger.items():
                        ### get the other event
                        event_idx = self.trial_procedure.event_name_list.index(name)
                        ### set onset of other event
                        if callable(delay):
                            add = delay()
                            self.trial_procedure.event_list[event_idx].onset = (
                                self.onset + add
                            )
                        else:
                            self.trial_procedure.event_list[event_idx].onset = (
                                self.onset + delay
                            )
                ### store event in happened events
                self.trial_procedure.happened_event_list.append(self.name)

        ### TODO replace requirement_string with a function (which has access to the
        ### attributes) checking the requirements
        def _check_requirements(self):
            """
            Check if the requirements for the event are met

            Returns:
                met (bool):
                    True if requirements are met, False otherwise
            """
            if self.requirement_string != None:
                ### check requirement with requirement string
                return self._eval_requirement_string()
            else:
                ### no requirement
                return True

        def _eval_requirement_string(self):
            """
            evaluates a condition string in format like 'XXX==XXX and (XXX==XXX or
            XXX==XXX)'

            Returns:
                met (bool):
                    True if requirements are met, False otherwise
            """
            ### split condition string
            string = self.requirement_string
            string = string.split(" and ")
            string = [sub_string.split(" or ") for sub_string in string]

            ### loop over string splitted string parts
            final_string = []
            for sub_idx, sub_string in enumerate(string):
                ### combine outer list eelemts with and
                ### and combine inner list elements with or
                if len(sub_string) == 1:
                    if sub_idx < len(string) - 1:
                        final_string.append(
                            self._get_condition_part(sub_string[0]) + " and "
                        )
                    else:
                        final_string.append(self._get_condition_part(sub_string[0]))
                else:
                    for sub_sub_idx, sub_sub_string in enumerate(sub_string):
                        if sub_sub_idx < len(sub_string) - 1:
                            final_string.append(
                                self._get_condition_part(sub_sub_string) + " or "
                            )
                        elif sub_idx < len(string) - 1:
                            final_string.append(
                                self._get_condition_part(sub_sub_string) + " and "
                            )
                        else:
                            final_string.append(
                                self._get_condition_part(sub_sub_string)
                            )
            return eval("".join(final_string))

        def _get_condition_part(self, string):
            """
            converts a string in format like '((XXX==XXX)' into '((True)'
            """
            ### remove spaces from string
            string = string.strip()
            string = string.split()
            string = "".join(string)

            ### recursively remove brackets
            ### at the end evaluate term (without brackets) and then return the evaluated value with the former brackets
            if string[0] == "(":
                return "(" + self._get_condition_part(string[1:])
            elif string[-1] == ")":
                return self._get_condition_part(string[:-1]) + ")"
            else:
                return str(self._eval_condition_part(string))

        def _eval_condition_part(self, string):
            """
            gets string in format 'XXX==XXX'

            evaluates the term for mode and happened events

            returns True/False
            """

            var = string.split("==")[0]
            val = string.split("==")[1]
            if var == "mode":
                test = self.trial_procedure.mode == val
            elif var == "happened_event_list":
                ### remove brackets
                val = val.strip("[]")
                ### split entries
                val = val.split(",")
                ### remove spaces from entries
                happened_event_list_from_string = [val_val.strip() for val_val in val]
                ### check if all events are in happened_event_list, if not --> return False
                test = True
                for event in happened_event_list_from_string:
                    if not (event in self.trial_procedure.happened_event_list):
                        test = False
            return test
