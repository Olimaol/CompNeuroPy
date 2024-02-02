"""
This example demonstrates how to use the OptNeuron class to fit an ANNarchy neuron
model to the dynamics of another ANNarchy neuron model in a specific experiment.

The experiment and variable_bounds used are imported from the other example
'run_opt_neuron_from_data.py'.
"""

from CompNeuroPy import (
    CompNeuroExp,
    rmse,
    current_step,
    evaluate_expression_with_dict,
    print_df,
    PlotRecordings,
)
from CompNeuroPy.opt_neuron import OptNeuron
from CompNeuroPy.neuron_models import Izhikevich2007
from ANNarchy import dt, get_population


class MyExp(CompNeuroExp):
    """
    Benchmark experiment
    """

    def run(self, pop_name):
        """
        Run Benchmark simulation.

        Args:
            pop_name (str):
                name of the population with neurons of the tuned neuron model, this will
                be automatically provided by OptNeuron

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
        for atr_name in get_population(pop_name).attributes:
            print(f"{atr_name}: {getattr(get_population(pop_name), atr_name)}")
        self.monitors.start()
        current_step(pop_name, 25, 25, 30, 80)
        current_step(pop_name, 25, 25, 0, -30)
        self.data["pop_name"] = pop_name
        self.data["time_step"] = dt()
        return self.results()


def get_loss(
    results_ist: CompNeuroExp._ResultsCl, results_soll: CompNeuroExp._ResultsCl
):
    """
    Function which has to have the arguments results_ist and results_soll and should
    calculate and return the loss.
    """
    v_ist = results_ist.recordings[0][f"{results_ist.data['pop_name']};v"][:, 0]
    v_soll = results_soll.recordings[0][f"{results_soll.data['pop_name']};v"][:, 0]
    return rmse(v_soll, v_ist)


variables_bounds = {
    "C": [0.1, 100],
    "k": [0.01, 10],
    "v_r": [-90, -40],
    "v_t": "v_r + dv_r__v_t",
    "dv_r__v_t": [0, 30],
    "a": [0.01, 10],
    "b": [-20, 20],
    "c": [-100, -40],
    "d": [0, 30],
    "v_peak": [-30, 30],
    "v": "v_r",
    "u": 0,
}


variables_pick = {
    "C": 66.57360412862414,
    "k": 2.3095462300680003,
    "v_r": -63.81566591506167,
    "v_t": -63.81566591506167 + 0.602508704796174,
    "a": 6.365717580329044,
    "b": 17.095454384114184,
    "c": -52.68028922214341,
    "d": 21.738698612715623,
    "v_peak": 7.746989077180672,
    "init": {"v": -63.81566591506167, "u": 0},
}


def main():
    ### define optimization
    opt = OptNeuron(
        experiment=MyExp,
        get_loss_function=get_loss,
        variables_bounds=variables_bounds,
        neuron_model=Izhikevich2007(),
        target_neuron_model=Izhikevich2007(**variables_pick),
        time_step=0.1,
        compile_folder_name="benchmark_izhikevich",
        method="hyperopt",
        record=["v"],
    )

    ### run the optimization, define how often the experiment should be repeated
    fit = opt.run(
        max_evals=5,
        results_file_name="benchmark_izhikevich/result",
        deap_plot_file="benchmark_izhikevich/logbook.png",
        verbose=True,
    )

    ### print optimized parameters
    for key in variables_bounds.keys():
        if isinstance(variables_bounds[key], list):
            variables_bounds[key] = fit[key]
    for key in variables_bounds.keys():
        if isinstance(variables_bounds[key], str):
            variables_bounds[key] = evaluate_expression_with_dict(
                variables_bounds[key], variables_bounds
            )
    for key in variables_pick.keys():
        if key == "init":
            continue
        variables_pick[key] = [variables_pick[key], variables_bounds[key]]
    variables_pick.pop("init")
    print_df(variables_pick)

    PlotRecordings(
        figname="benchmark_izhikevich/results_soll.png",
        recordings=fit["results_soll"].recordings,
        recording_times=fit["results_soll"].recording_times,
        shape=(1, 1),
        plan={
            "position": [1],
            "compartment": [
                fit["results_soll"].data["pop_name"],
            ],
            "variable": ["v"],
            "format": ["line"],
        },
    )
    PlotRecordings(
        figname="benchmark_izhikevich/results.png",
        recordings=fit["results"].recordings,
        recording_times=fit["results"].recording_times,
        shape=(1, 1),
        plan={
            "position": [1],
            "compartment": [
                fit["results"].data["pop_name"],
            ],
            "variable": ["v"],
            "format": ["line"],
        },
    )


if __name__ == "__main__":
    main()
