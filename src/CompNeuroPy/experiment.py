from ANNarchy import reset
from CompNeuroPy.monitors import RecordingTimes
from CompNeuroPy import CompNeuroMonitors
from CompNeuroPy import model_functions as mf


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
        self.recordings = {}  # save dict for monitor recordings
        self.monitors = monitors
        self.data = {}  # dict for optional data

    def reset(
        self,
        populations=True,
        projections=False,
        synapses=False,
        model=True,
        parameters=True,
    ):
        """
        Reset the ANNarchy model and monitors and the CompNeuroMonitors used for the
        experiment.

        !!! warning
            If you want the network to have the same state at the beginning of each
            experiment run, you should call this function at the beginning of the run
            function of the CompNeuroExp class! If you only want to have the same time
            for the network at the beginning of each experiment run, set populations,
            projections, and synapses to False.

        Args:
            populations (bool, optional):
                reset populations. Defaults to True.
            projections (bool, optional):
                reset projections. Defaults to False.
            synapses (bool, optional):
                reset synapses. Defaults to False.
            model (bool, optional):
                If False, do ignore the arguments populations, projections, and
                synapses (the network state doesn't change) and only reset the
                CompNeuroMonitors Default: True.
            parameters (bool, optional):
                If False, do not reset the parameters of the model. Default: True.
        """
        reset_kwargs = {}
        reset_kwargs["populations"] = populations
        reset_kwargs["projections"] = projections
        reset_kwargs["synapses"] = synapses
        reset_kwargs["monitors"] = True

        ### reset CompNeuroMonitors and ANNarchy model
        if self.monitors is not None:
            self.monitors.reset(model=model, parameters=parameters, **reset_kwargs)
        elif model is True:
            if parameters is False:
                ### if parameters=False, get parameters before reset and set them after
                ### reset
                parameters = mf._get_all_parameters()
            reset(**reset_kwargs)
            if parameters is False:
                ### if parameters=False, set parameters after reset
                mf._set_all_parameters(parameters)

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
        obj.data = self.data

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
