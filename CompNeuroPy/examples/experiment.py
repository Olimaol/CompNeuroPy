from CompNeuroPy import (
    Experiment,
    plot_recordings,
    generate_simulation,
    current_step,
    current_ramp,
    Monitors,
)
from CompNeuroPy.models import H_and_H_model_Bischop
from ANNarchy import dt, setup


### combine both simulations and recordings in an experiment
class my_exp(Experiment):
    """
    parent class Experiment provides the variables:
        self.mon = []) --> define during init, a CompNeuroPy Monitors object to do recordings
        self.data = {} --> a dictionary with any optional data
    and the functions:
        self.reset()   --> resets the model and monitors
        self.results() --> returns a results object (with recordings and optional data from self.data)
    """

    ### we have to define some funciton in which the simulations and recordings are done
    def run(self):
        """
        do the simulations and recordings
        """

        ### define some simulations
        sim_step = generate_simulation(
            simulation_function=current_step,
            simulation_kwargs={
                "pop": model.populations[0],
                "t1": 500,
                "t2": 500,
                "a1": 0,
                "a2": 50,
            },
        )
        sim_ramp = generate_simulation(
            simulation_function=current_ramp,
            simulation_kwargs={
                "pop": model.populations[0],
                "a0": 0,
                "a1": 100,
                "dur": 1000,
                "n": 50,
            },
        )

        ### run simulations/recordings
        self.mon.start()
        sim_step.run()
        ### if you want to reset the model, you can use the objects reset()
        ### it's the same as the ANNarchy reset (only necessary for the use with opt_neuron)
        self.reset()
        sim_ramp.run()
        ### SIMULATION END

        ### optional: store anything you want in the data dict, for example infomration about
        ### the simulations
        self.data["sim"] = [sim_step.simulation_info(), sim_ramp.simulation_info()]
        self.data["population_name"] = model.populations[0]
        self.data["time_step"] = dt()

        ### return results, use the object's self.results() function which automatically
        ### returns an object with "recordings", "recording_times", "monDict", and "data"
        return self.results()


if __name__ == "__main__":

    ### create and compile a model
    setup(dt=0.01)
    model = H_and_H_model_Bischop()

    ### define recordings before experiment
    monitors = Monitors({f"pop;{model.populations[0]}": ["v"]})

    ### run the experiment
    experiment_obj = my_exp(monitors=monitors)
    results = experiment_obj.run()

    print("recordings:\n", results.recordings, "\n\n")
    print("data:\n", results.data, "\n\n")
    print("monDict:\n", results.monDict, "\n\n")

    ### quick plot of the membrane potential from the first chunk
    chunk = 0
    plot_recordings(
        figname="example_experiment.svg",
        recordings=results.recordings,
        recording_times=results.recording_times,
        chunk=chunk,
        shape=(1, 1),
        plan=[f"1;{results.data['population_name']};v;line"],
    )
