from ANNarchy import reset
from CompNeuroPy.Monitors import RecordingTimes
from CompNeuroPy import Monitors


class CompNeuroExp:
    """
    Experiment combining simulations and recordings.

    Use this class as a parent class for your experiment. You have to additionally
    implement a run function which runs the simulations and controlls the recordings.
    The run function should return the results of the experiment by calling the results
    function of the CompNeuroExp class.

    Attributes:
        mon (Monitors):
            Monitors object for recordings
        data (dict):
            dict for storing optional data

    Examples:
        >>> from CompNeuroPy import CompNeuroExp
        >>> from ANNarchy import simulate
        >>>
        >>> class MyExperiment(CompNeuroExp):
        >>>     def run(self):
        >>>         # run simulations and control recordings
        >>>         self.mon.start()
        >>>         simulate(1000)
        >>>         self.reset()
        >>>         simulate(1000)
        >>>         # store optional data
        >>>         self.data["duration"] = 2000
        >>>         # return results
        >>>         return self.results()
    """

    def __init__(self, monitors: Monitors, reset_function=None, reset_kwargs={}):
        """
        Initialize the experiment.

        Args:
            monitors (Monitors):
                Monitors object for recordings
            reset_function (function, optional):
                A function which resets the ANNarchy model.
                Default: None, i.e., ANNarchys' reset function
            reset_kwargs (dict, optional):
                Arguments of the reset_function besides the ones which are used by
                ANNarchys' reset function. Default: {}.
        """
        self.recordings = {}  # save dict for monitor recordings
        self.mon = monitors
        self.data = {}  # dict for optional data

        ### check function to reset network
        if reset_function is None:
            self.reset_function = reset
        else:
            self.reset_function = reset_function
        self.reset_kwargs = reset_kwargs

    def reset(self, populations=True, projections=False, synapses=False):
        """
        Reset the ANNarchy model and monitors and the CompNeuroPy Monitors used for the
        experiment. The reset function of the CompNeuroExp class is used which can be
        set during initialization and can have additional arguments besides the ones
        which are used by ANNarchys' reset function which are also set during
        initialization.

        Args:
            populations (bool, optional):
                reset populations. Defaults to True.
            projections (bool, optional):
                reset projections. Defaults to False.
            synapses (bool, optional):
                reset synapses. Defaults to False.
            monitors (bool, optional):
                reset monitors. Defaults to True.
        """
        self.reset_kwargs["populations"] = populations
        self.reset_kwargs["projections"] = projections
        self.reset_kwargs["synapses"] = synapses
        self.reset_kwargs["monitors"] = True
        ### reset monitors
        self.mon.reset()
        ### reset ANNarchy model
        self.reset_function(**self.reset_kwargs)

    def results(self):
        """
        !!! warning
            Call this function at the end of the run function of the CompNeuroExp class!

        Returns:
            results_obj (CompNeuroExp._ResultsCl):
                Object with with attributes:
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
        obj.recordings, obj.recording_times = self.mon.get_recordings_and_clear()
        obj.mon_dict = self.mon.mon_dict
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


### old name for backward compatibility, TODO remove
Experiment = CompNeuroExp
