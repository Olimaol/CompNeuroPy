"""
This example demonstrates how to use the CompNeuroExp class to combine simulations,
model and recordings in an experiment. It is shown how to define an experiment, how to
run it and how to get the results.
"""
from CompNeuroPy import (
    CompNeuroExp,
    CompNeuroSim,
    CompNeuroMonitors,
    CompNeuroModel,
    current_step,
    current_ramp,
    PlotRecordings,
)
from CompNeuroPy.full_models import HHmodelBischop
from ANNarchy import dt, setup, get_population


### combine both simulations and recordings in an experiment
class MyExp(CompNeuroExp):
    """
    Define an experiment by inheriting from CompNeuroExp.

    CompNeuroExp provides the attributes:

        monitors (CompNeuroMonitors):
            a CompNeuroMonitors object to do recordings, define during init otherwise
            None
        data (dict):
            a dictionary for storing any optional data

    and the functions:
        reset():
            resets the model and monitors
        results():
            returns a results object
    """

    def __init__(
        self,
        model: CompNeuroModel,
        sim_step: CompNeuroSim,
        sim_ramp: CompNeuroSim,
        monitors: CompNeuroMonitors,
    ):
        """
        Initialize the experiment and additionally store the model and simulations.

        Args:
            model (CompNeuroModel):
                a CompNeuroModel object
            sim_step (CompNeuroSim):
                a CompNeuroSim object for the step simulation
            sim_ramp (CompNeuroSim):
                a CompNeuroSim object for the ramp simulation
            monitors (CompNeuroMonitors):
                a CompNeuroMonitors object
        """
        self.model = model
        self.sim_step = sim_step
        self.sim_ramp = sim_ramp
        super().__init__(monitors)

    def run(self, E_L: float = -68.0):
        """
        Do the simulations and recordings.

        To use the CompNeuroExp class, you need to define a run function which
        does the simulations and recordings. The run function should return the
        results object which can be obtained by calling self.results().

        Args:
            E_L (float, optional):
                leak reversal potential of the population, which is set at the beginning
                of the experiment run. Default: -68 mV

        Returns:
            results (CompNeuroExp._ResultsCl):
                results object with attributes:
                    recordings (list):
                        list of recordings
                    recording_times (recording_times_cl):
                        recording times object
                    mon_dict (dict):
                        dict of recorded variables of the monitors
                    data (dict):
                        dict with optional data stored during the experiment
        """
        ### call reset at the beginning of the experiment to ensure that the model
        ### is in the same state at the beginning of each experiment run
        self.reset()

        ### also always start the monitors, they are stopped automatically at the end
        self.monitors.start()

        ### set the leak reversal potential of the population, be aware that this
        ### will be undone by the reset function if you don't set the parameters
        ### argument to False
        get_population(self.model.populations[0]).E_L = E_L

        ### SIMULATION START
        sim_step.run()
        ### if you want to reset the model, you should use the objects reset()
        ### it's the same as the ANNarchy reset + it resets the CompNeuroMonitors
        ### creating a new chunk, optionally not changing the parameters
        self.reset(parameters=False)
        sim_ramp.run()
        ### SIMULATION END

        ### optional: store anything you want in the data dict, for example information
        ### about the simulations
        self.data["sim"] = [sim_step.simulation_info(), sim_ramp.simulation_info()]
        self.data["population_name"] = self.model.populations[0]
        self.data["time_step"] = dt()

        ### return results using self.results()
        return self.results()


if __name__ == "__main__":
    ### create and compile a model
    setup(dt=0.01)
    model = HHmodelBischop()

    ### define recordings before experiment
    monitors = CompNeuroMonitors({model.populations[0]: ["v"]})

    ### define some simulations e.g. using CompNeuroSim
    sim_step = CompNeuroSim(
        simulation_function=current_step,
        simulation_kwargs={
            "pop": model.populations[0],
            "t1": 500,
            "t2": 500,
            "a1": 0,
            "a2": 50,
        },
    )
    sim_ramp = CompNeuroSim(
        simulation_function=current_ramp,
        simulation_kwargs={
            "pop": model.populations[0],
            "a0": 0,
            "a1": 100,
            "dur": 1000,
            "n": 50,
        },
    )

    ### init and run the experiment
    my_exp = MyExp(monitors=monitors, model=model, sim_step=sim_step, sim_ramp=sim_ramp)

    ### one use case is to run an experiment multiple times e.g. with different
    ### parameters
    results_run1 = my_exp.run()
    results_run2 = my_exp.run(E_L=-90.0)

    ### plot of the membrane potential from the first and second chunk using results
    ### experiment run 1
    PlotRecordings(
        figname="example_experiment_sim_step.png",
        recordings=results_run1.recordings,
        recording_times=results_run1.recording_times,
        chunk=0,
        shape=(1, 1),
        plan={
            "position": [1],
            "compartment": [results_run1.data["population_name"]],
            "variable": ["v"],
            "format": ["line"],
        },
    )
    PlotRecordings(
        figname="example_experiment_sim_ramp.png",
        recordings=results_run1.recordings,
        recording_times=results_run1.recording_times,
        chunk=1,
        shape=(1, 1),
        plan={
            "position": [1],
            "compartment": [results_run1.data["population_name"]],
            "variable": ["v"],
            "format": ["line"],
        },
    )
    ### experiment run 2
    PlotRecordings(
        figname="example_experiment2_sim_step.png",
        recordings=results_run2.recordings,
        recording_times=results_run2.recording_times,
        chunk=0,
        shape=(1, 1),
        plan={
            "position": [1],
            "compartment": [results_run2.data["population_name"]],
            "variable": ["v"],
            "format": ["line"],
        },
    )
    PlotRecordings(
        figname="example_experiment2_sim_ramp.png",
        recordings=results_run2.recordings,
        recording_times=results_run2.recording_times,
        chunk=1,
        shape=(1, 1),
        plan={
            "position": [1],
            "compartment": [results_run2.data["population_name"]],
            "variable": ["v"],
            "format": ["line"],
        },
    )

    ### print data and mon_dict from results
    print("\nrun1:")
    print("    data:")
    for key, value in results_run1.data.items():
        print(f"        {key}:", value)
    print("    mon_dict:")
    for key, value in results_run1.mon_dict.items():
        print(f"        {key}:", value)
    print("\nrun2:")
    print("    data:")
    for key, value in results_run2.data.items():
        print(f"        {key}:", value)
    print("    mon_dict:")
    for key, value in results_run2.mon_dict.items():
        print(f"        {key}:", value)
