from ANNarchy import reset
from CompNeuroPy.Monitors import recording_times_cl
from CompNeuroPy import Monitors


class Experiment:
    def __init__(self, monitors: Monitors, reset_function=None, reset_kwargs={}):
        """
        Class for experiments

        Parameters
        ----------
        monitors: object
            Monitors object for recordings
        reset_function: function
            a function which resets the ANNarchy model
        reset_kwargs: dict
            arguments of the reset_function
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

    def reset(self, populations=True, projections=False, synapses=False, monitors=True):
        """
        reset the ANNarchy model and the monitors

        Parameters
        ----------
        populations: bool, optional, default=True
            reset populations
        projections: bool, optional, default=False
            reset projections
        synapses: bool, optional, default=False
            reset synapses
        monitors: bool, optional, default=True
            reset monitors
        """
        self.reset_kwargs["populations"] = populations
        self.reset_kwargs["projections"] = projections
        self.reset_kwargs["synapses"] = synapses
        self.reset_kwargs["monitors"] = monitors
        ### reset monitors
        self.mon.reset()
        ### reset ANNarchy model
        self.reset_function(**self.reset_kwargs)

    def results(self):
        """
        call this function at the end of the experiment

        returns an object with variables "recordings", "monDict", and "data"
            "recordings" and "monDict" are obtained automatically
            "data" has to be defined by yourself during the experiment

        Returns
        -------
        obj: object
            object with variables "recordings", "monDict", and "data"
        """
        obj = self.return_cl()
        obj.recordings, obj.recording_times = self.mon.get_recordings_and_clear()
        obj.monDict = self.mon.monDict
        obj.data = self.data

        return obj

    class return_cl:
        def __init__(self) -> None:
            self.recordings: list
            self.recording_times: recording_times_cl
            self.monDict: dict
            self.data: dict
