class Experiment:

    import ANNarchy as ann
    import CompNeuroPy as cnp

    def __init__(self, reset_function=None, reset_kwargs={}):
        """
        Args:
            reset_function: function
                a function which resets the ANNarchy model

            reset_kwargs: dict
                arguments of the reset_function
        """
        self.recordings = {}  # save dict for monitor recordings
        self.mon = self.cnp.Monitors()  # dict for monitors
        self.data = {}  # dict for optional data

        ### check function to reset network
        if reset_function is None:
            self.reset_function = self.ann.reset
        else:
            self.reset_function = reset_function
        self.reset_kwargs = reset_kwargs

    def reset(self, populations=True, projections=False, synapses=False, monitors=True):
        self.reset_kwargs["populations"] = populations
        self.reset_kwargs["projections"] = projections
        self.reset_kwargs["synapses"] = synapses
        self.reset_kwargs["monitors"] = monitors
        self.mon.reset(model=False)
        self.reset_function(
            **self.reset_kwargs
        )  # TODO reset funciton in opt_neuron also has to use the arguments from above

    def results(self):
        """
        call this function at the end of the experiment

        returns an object with variables "recordings", "monDict", and "data"
            "recordings" and "monDict" are obtained automatically
            "data" has to be defined by yourself during the experiment
        """
        obj = self.return_cl()
        obj.recordings = self.mon.get_recordings()
        obj.monDict = self.mon.monDict
        obj.data = self.data

        return obj

    class return_cl:
        def __init__(self) -> None:
            pass
