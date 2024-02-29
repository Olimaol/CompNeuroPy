from ANNarchy import reset
from CompNeuroPy.monitors import RecordingTimes
from CompNeuroPy import CompNeuroMonitors
from CompNeuroPy import model_functions as mf
from copy import deepcopy


class CompNeuroExp:
    """
    Experiment combining simulations and recordings.

    Use this class as a parent class for your experiment. You have to additionally
    implement a run function which runs the simulations and controlls the recordings.
    The run function should return the results of the experiment by calling the results
    function of the CompNeuroExp class.

    Attributes:
        monitors (CompNeuroMonitors):
            CompNeuroMonitors object for recordings
        data (dict):
            dict for storing optional data

    Examples:
        ```python
        from CompNeuroPy import CompNeuroExp
        from ANNarchy import simulate

        class MyExperiment(CompNeuroExp):
            def run(self):
                # run simulations and control recordings
                self.monitors.start()
                simulate(1000)
                self.reset()
                simulate(1000)
                # store optional data
                self.data["duration"] = 2000
                # return results
                return self.results()
        ```
    """

    def __init__(
        self,
        monitors: CompNeuroMonitors | None = None,
    ):
        """
        Initialize the experiment.

        Args:
            monitors (CompNeuroMonitors):
                CompNeuroMonitors object for recordings
        """
        self.monitors = monitors
        self.data = {}  # dict for optional data
        self._model_state = None

    def store_model_state(self, compartment_list: list[str]):
        """
        Store the state of the model. If this is called, reset does not reset the model
        to compile state but to the state stored here.

        Args:
            compartment_list (list[str]):
                list of compartments to store the state of
        """
        self._model_state = mf._get_all_attributes(compartment_list)

    def reset_model_state(self):
        """
        Reset the stored model state.
        """
        self._model_state = None

    def reset(
        self,
        populations=True,
        projections=False,
        synapses=False,
        model=True,
        model_state=True,
        parameters=True,
    ):
        """
        Reset the ANNarchy model and monitors and the CompNeuroMonitors used for the
        experiment.

        !!! warning
            If you want the network to have the same state at the beginning of each
            experiment run, you should call this function at the beginning of the run
            function of the CompNeuroExp class (except using OptNeuron)! If you only
            want to have the same time for the network at the beginning of each
            experiment run, set populations, projections, and synapses to False and
            model to True. If you want to set parameters during the experiment and also
            reset the dynamic variables without resetting the parameters, set parameters
            to False.

        Args:
            populations (bool, optional):
                reset populations. Defaults to True.
            projections (bool, optional):
                reset projections. Defaults to False.
            synapses (bool, optional):
                reset synapses. Defaults to False.
            model (bool, optional):
                If False, do ignore all other arguments (the network state doesn't
                change) and only reset the CompNeuroMonitors (creating new chunk)
                Default: True.
            model_state (bool, optional):
                If True, reset the model to the stored model state instead of
                compilation state (all compartments not stored in the model state will
                still be resetted to compilation state). Default: True.
            parameters (bool, optional):
                If True, reset the parameters of the model (either to compile or stored
                state). Default: True.
        """
        reset_kwargs = {}
        reset_kwargs["populations"] = populations
        reset_kwargs["projections"] = projections
        reset_kwargs["synapses"] = synapses
        reset_kwargs["monitors"] = True

        ### reset CompNeuroMonitors and ANNarchy model
        if self.monitors is not None:
            ### there are monitors, therefore use theri reset function
            self.monitors.reset(model=model, **reset_kwargs, parameters=parameters)
            ### after reset, set the state of the model to the stored state
            if model_state and self._model_state is not None and model is True:
                ### if parameters=False, they are not set
                mf._set_all_attributes(self._model_state, parameters=parameters)
        elif model is True:
            if parameters is False:
                ### if parameters=False, get parameters before reset and set them after
                ### reset
                parameters_dict = mf._get_all_parameters()
            ### there are no monitors, but model should be resetted, therefore use
            ### ANNarchy's reset function
            reset(**reset_kwargs)
            if parameters is False:
                ### if parameters=False, set parameters after reset
                mf._set_all_parameters(parameters_dict)
            ### after reset, set the state of the model to the stored state
            if model_state and self._model_state is not None:
                ### if parameters=False, they are not set
                mf._set_all_attributes(self._model_state, parameters=parameters)

    def results(self):
        """
        !!! warning
            Call this function at the end of the run function of the CompNeuroExp class!

        !!! warning
            Calling this function resets the CompNeuroMonitors. For example, if you
            simulate two recording chunks in the run function and you run the experiment
            twice, you will get two recording chunks for each experiment run (not two
            for the first and four for the second run). But ANNarchy is not resetted
            automatically! So the network time and state (activity etc.) at the
            beginning of the second run is the same as at the end of the first run. To
            prevent this use the reset function of the CompNeuroExp class.

        Returns:
            results_obj (CompNeuroExp._ResultsCl):
                Object with attributes:
                    recordings (list):
                        list of recordings
                    recording_times (recording_times_cl):
                        recording times object
                    mon_dict (dict):
                        dict of recorded variables of the monitors
                    data (dict):
                        dict with optional data stored during the experiment
        """
        obj = self._ResultsCl()
        if self.monitors is not None:
            (
                obj.recordings,
                obj.recording_times,
            ) = self.monitors.get_recordings_and_clear()
            obj.mon_dict = self.monitors.mon_dict
        else:
            obj.recordings = []
            obj.recording_times = None
            obj.mon_dict = {}
        ### need deepcopy here because experiment can be run mutliple times and within
        ### experiment the entries of self.data can be changed, and without deepcopy
        ### the data of older results objects would also be changed
        obj.data = deepcopy(self.data)

        return obj

    class _ResultsCl:
        """
        Class for storing the results of the experiment.

        Attributes:
            recordings (list):
                list of recordings
            recording_times (recording_times_cl):
                recording times object
            mon_dict (dict):
                dict of recorded variables of the monitors
            data (dict):
                dict with optional data stored during the experiment
        """

        def __init__(self) -> None:
            self.recordings: list
            self.recording_times: RecordingTimes
            self.mon_dict: dict
            self.data: dict

    def run(self) -> _ResultsCl:
        """
        !!! warning
            This function has to be implemented by the user!
        """
        raise NotImplementedError(
            """
                You have to implement a run function which runs the simulations and
                controlls the recordings. The run function should return the results of
                the experiment by calling the results function of the CompNeuroExp class.
            """
        )


### old name for backward compatibility, TODO remove
Experiment = CompNeuroExp
