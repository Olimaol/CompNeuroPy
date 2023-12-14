### TODO show that you can run experiment multiple times and always get the same recordings/recording_times structure
from CompNeuroPy import (
    CompNeuroExp,
    CompNeuroSim,
    CompNeuroMonitors,
    CompNeuroModel,
    current_step,
    current_ramp,
    plot_recordings,
)
from CompNeuroPy.full_models import HHmodelBischop
from ANNarchy import dt, setup


### combine both simulations and recordings in an experiment
class MyExp(CompNeuroExp):
    """
    parent class CompNeuroExp provides the variables:
        self.mon = []) --> define during init, a CompNeuroMonitors object to do recordings
        self.data = {} --> a dictionary with any optional data
    and the functions:
        self.reset()   --> resets the model and monitors
        self.results() --> returns a results object (with recordings and optional data from self.data)
    """

    def __init__(
        self,
        model: CompNeuroModel,
        monitors: CompNeuroMonitors,
        reset_function=None,
        reset_kwargs={},
    ):
        self.model = model
        super().__init__(monitors, reset_function, reset_kwargs)

    ### we have to define some function in which the simulations and recordings are done
    def run(self):
        """
        do the simulations and recordings
        """

        ### define some simulations
        sim_step = CompNeuroSim(
            simulation_function=current_step,
            simulation_kwargs={
                "pop": self.model.populations[0],
                "t1": 500,
                "t2": 500,
                "a1": 0,
                "a2": 50,
            },
        )
        sim_ramp = CompNeuroSim(
            simulation_function=current_ramp,
            simulation_kwargs={
                "pop": self.model.populations[0],
                "a0": 0,
                "a1": 100,
                "dur": 1000,
                "n": 50,
            },
        )

        ### run simulations/recordings
        self.mon.start()
        sim_step.run()
        ### if you want to reset the model, you should use the objects reset()
        ### it's the same as the ANNarchy reset + it resets the CompNeuroMonitors
        ### creating a new chunk
        self.reset()
        sim_ramp.run()
        ### SIMULATION END

        ### optional: store anything you want in the data dict, for example information
        ### about the simulations
        self.data["sim"] = [sim_step.simulation_info(), sim_ramp.simulation_info()]
        self.data["population_name"] = self.model.populations[0]
        self.data["time_step"] = dt()

        ### return results, use the object's self.results() function which automatically
        ### returns an object with "recordings", "recording_times", "mon_dict", and "data"
        return self.results()


if __name__ == "__main__":
    ### create and compile a model
    setup(dt=0.01)
    model = HHmodelBischop()

    ### define recordings before experiment
    monitors = CompNeuroMonitors({f"pop;{model.populations[0]}": ["v"]})

    ### run the experiment
    my_exp = MyExp(monitors=monitors, model=model)
    results = my_exp.run()

    ### print results
    print("recordings:\n", results.recordings, "\n\n")
    print("data:\n", results.data, "\n\n")
    print("mon_dict:\n", results.mon_dict, "\n\n")

    ### quick plot of the membrane potential from the first and second chunk
    plot_recordings(
        figname="example_experiment_sim_step.png",
        recordings=results.recordings,
        recording_times=results.recording_times,
        chunk=0,
        shape=(1, 1),
        plan=[f"1;{results.data['population_name']};v;line"],
    )
    plot_recordings(
        figname="example_experiment_sim_ramp.png",
        recordings=results.recordings,
        recording_times=results.recording_times,
        chunk=1,
        shape=(1, 1),
        plan=[f"1;{results.data['population_name']};v;line"],
    )
