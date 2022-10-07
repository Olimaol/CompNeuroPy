import numpy as np
from CompNeuroPy import plot_recordings


def make_plot(recordings, recording_timings, chunk, period="all"):
    ### with plot_recodings one can easily plot the recodings of one chunk
    ### plot_recordings needs the time limits (in ms) and the idx limits for the data to plot, which can be obtained for example with the recording_timings object (they can of course also be set manually to specific values)
    if period == "all":
        time_lims = recording_timings.time_lims(chunk=chunk)
    else:
        time_lims = recording_timings.time_lims(chunk=chunk, period=period)
    ### the last two arguments of plot_recordings define how subplots are arranged and which recordings are shown in which subplot (plot_list)
    structure = (2, 2)
    ### the plot_list entries consist of strings with following format:
    ### 'sub_plot_nr;model_compartment;recorded_data;plotting_specifications'
    plot_list = [
        "1;first_poisson;spike;hybrid",
        "2;second_poisson;spike;hybrid",
        "3;first_poisson;p;line",
        "4;second_poisson;p;line",
    ]

    plot_recordings(
        figname=f"results/my_two_poissons_chunk_{chunk}_period_{period}.png",
        recordings=recordings,
        recording_times=recording_timings,
        chunk=chunk,
        time_lim=time_lims,
        shape=structure,
        plan=plot_list,
        dpi=300,
    )


def main():
    ### load data generated with script "run_and_monitor_simulations.py"
    recordings = np.load(
        "dataRaw/run_and_monitor_simulations/recordings.npy", allow_pickle=True
    )
    recording_timings = np.load(
        "dataRaw/run_and_monitor_simulations/recording_times.npy", allow_pickle=True
    ).item()

    ### plot first chunk (only one period)
    make_plot(recordings, recording_timings, 0)
    ### plot second chunk
    make_plot(recordings, recording_timings, 1)
    ### plot first period of second chunk
    make_plot(recordings, recording_timings, 1, 0)
    ### plot second period of second chunk
    make_plot(recordings, recording_timings, 1, 1)

    return 1


if __name__ == "__main__":
    main()
