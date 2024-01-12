"""
This example demonstrates how to plot recordings (from CompNeuroMonitors) using the
PlotRecordings class. The different plotting formats for spiking and non-spiking data
(populations and projections) are demonstrated.

This example loads data generated with other example "run_and_monitor_simulations.py".
"""
from CompNeuroPy import load_variables, PlotRecordings


def main():
    ### load data generated with other example "run_and_monitor_simulations.py"
    loaded_dict = load_variables(
        name_list=["recordings", "recording_times", "increase_rates_pop_info"],
        path="run_and_monitor_simulations/",
    )

    ### define what should be plotted in which subplot, here 14 subplots are defined to
    ### demonstrate the different plotting formats for spiking and non-spiking variables
    plan_dict = {
        "position": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14],
        "compartment": [
            "first_poisson",
            "first_poisson",
            "first_poisson",
            "first_poisson",
            "first_poisson",
            "second_poisson",
            "second_poisson",
            "second_poisson",
            "second_poisson",
            "ampa_proj",
            "ampa_proj",
            "ampa_proj",
            "ampa_proj",
        ],
        "variable": [
            "spike",
            "spike",
            "spike",
            "spike",
            "spike",
            "p",
            "p",
            "p",
            "p",
            "w",
            "w",
            "w",
            "w",
        ],
        "format": [
            "raster",
            "mean",
            "hybrid",
            "interspike",
            "cv",
            "line",
            "line_mean",
            "matrix",
            "matrix_mean",
            "line",
            "line_mean",
            "matrix",
            "matrix_mean",
        ],
    }

    ### plot first chunk
    PlotRecordings(
        figname="run_and_monitor_simulations/my_two_poissons_chunk_0.png",
        recordings=loaded_dict["recordings"],
        recording_times=loaded_dict["recording_times"],
        shape=(3, 5),
        plan=plan_dict,
    )
    ### plot second chunk
    PlotRecordings(
        figname="run_and_monitor_simulations/my_two_poissons_chunk_1.png",
        recordings=loaded_dict["recordings"],
        recording_times=loaded_dict["recording_times"],
        shape=(3, 5),
        plan=plan_dict,
        chunk=1,
    )

    return 1


if __name__ == "__main__":
    main()
