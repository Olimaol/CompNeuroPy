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
    save_variables,
)
from CompNeuroPy.opt_neuron import OptNeuron
from CompNeuroPy.neuron_models import Izhikevich2007
import numpy as np
from run_benchmark import METHOD, MAX_EVALS
import sys


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
        self.monitors.start()
        current_step(pop_name, 25, 25, 30, 80)
        current_step(pop_name, 25, 25, 0, -30)
        self.data["pop_name"] = pop_name
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
    SIM_ID = int(sys.argv[1])
    file_appendix = f"_{METHOD}_{SIM_ID}"

    ### define optimization
    opt = OptNeuron(
        experiment=MyExp,
        get_loss_function=get_loss,
        variables_bounds=variables_bounds,
        neuron_model=Izhikevich2007(),
        target_neuron_model=Izhikevich2007(**variables_pick),
        time_step=0.1,
        compile_folder_name=f"benchmark_izhikevich{file_appendix}",
        method=METHOD,
        record=["v"],
    )

    ### run the optimization, define how often the experiment should be repeated
    fit = opt.run(
        max_evals=MAX_EVALS,
        results_file_name=f"benchmark_izhikevich_results/result{file_appendix}",
        deap_plot_file=f"benchmark_izhikevich_plots/logbook{file_appendix}.png",
        verbose=False,
    )

    ### compare target and optimized parameters
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
    ###TODO remove this printing, this is just for testing
    print_df(variables_pick)

    ### calculate the difference between the optimized and the target parameters
    parameter_difference = np.linalg.norm(
        np.array([variables_pick[key][0] for key in variables_pick.keys()])
        - np.array([variables_pick[key][1] for key in variables_pick.keys()])
    )

    ### get best loss and when it was obtained
    best_loss_idx = np.argmin(opt.loss_history[:, 0])
    best_loss = opt.loss_history[best_loss_idx, 0]
    best_loss_time = opt.loss_history[best_loss_idx, 1]
    best_loss_evals = best_loss_idx + 1

    ### get loss history
    loss_history = opt.loss_history

    ### save variables TODO
    save_variables(
        variable_list=[
            {parameter_difference:parameter_difference,
            best_loss,
            best_loss_time,
            best_loss_evals,
            loss_history,}
        ],
        name_list=
    )

    ### plot recordings
    PlotRecordings(
        figname=f"benchmark_izhikevich_plots/recordings_soll{file_appendix}.png",
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
        figname=f"benchmark_izhikevich_plots/recordings_ist{file_appendix}.png",
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
