from ANNarchy import Population, Izhikevich, setup, simulate, get_population
from CompNeuroPy import generate_model, Monitors, create_dir
import pylab as plt
import numpy as np

setup(dt=0.1)
create_dir("results")


def make_chunk_plot(fig_name, times, data_arr, recording_times):
    """
    this function creates and saves a plot
    used below, at the end of the script
    """
    plt.figure()
    ### mark the border between chunks
    plt.plot(times, data_arr[:, 0], color="k")
    plt.xlabel("time [ms]")
    plt.ylabel("membrane potential [mV]")
    plt.axvline(
        recording_times.time_lims(chunk=0, compartment="my_pop2")[1],
        color="gray",
        ls="dotted",
    )
    plt.text(
        recording_times.time_lims(chunk=0, compartment="my_pop2")[1],
        np.nanmax(data_arr),
        "1st chunk ",
        ha="right",
        va="top",
    )
    plt.text(
        recording_times.time_lims(chunk=0, compartment="my_pop2")[1],
        np.nanmax(data_arr),
        " 2nd chunk",
        ha="left",
        va="top",
    )
    plt.tight_layout()
    plt.savefig(f"results/{fig_name}")


def main():
    ### first we create a simple model, consisting of two populaitons, each consist of 1 neuron
    def create_model():
        a = Population(1, neuron=Izhikevich(), name="my_pop1")
        a.i_offset = 10
        Population(1, neuron=Izhikevich(), name="my_pop2")

    ### create and compile the model
    generate_model(
        model_creation_function=create_model,
        name="monitor_recordings_model",
        description="my simple example model",
        compile_folder_name="annarchy_monitor_recordings_model",
    )

    ### after compilation we can define the monitors using the monitor_dictionary
    ### and the Monitors class
    monitor_dictionary = {"pop;my_pop1;2": ["v", "spike"], "pop;my_pop2": ["v"]}
    mon = Monitors(monitor_dictionary)

    ### now lets do a simulation with some resets and pauses, creating recording chunks and periods
    for _ in ["lets do the simulation"]:
        ### first chunk, one period
        simulate(100)  # 100 ms not recorded
        mon.start()  # start all monitors
        simulate(100)  # 100 ms recorded

        ### reset --> second chunk
        ### with pause --> two periods
        mon.reset()  # model reset, beginning of new chunk
        simulate(
            100
        )  # 100 ms recorded (monitors were active before reset --> still active)

        mon.pause(["pop;my_pop1"])  # pause recordings of my_pop1
        simulate(100)  # 100 ms not recorded

        mon.start()  # start all monitors
        simulate(50)  # 50 ms recorded
        get_population("my_pop1").i_offset = 50  # increase activity during last period
        get_population("my_pop2").i_offset = 50  # increase activity during last period
        simulate(50)  # 50 ms recorded

    ### get all recordings and recording_times from the monitor object
    recordings = mon.get_recordings()
    recording_times = mon.get_recording_times()

    ### in the following we only work with the recorded variable "v" of my_pop1
    y1 = recordings[0]["my_pop1;v"]  # variable v of my_pop1 from 1st chunk
    y2 = recordings[1]["my_pop1;v"]  # variable v of my_pop1 from 2nd chunk

    ### Let's print the recoding times of my_pop1 (which had a pause in the second chunk, in contrast to my_pop2)
    for _ in ["lets print something"]:
        print("\nrecording time limits of my_pop1")
        print(
            f"{'':<25} {'in ms':<18} {'as index (dt=0.1 ms but recording period=2 ms)':<18}"
        )
        ### start and end of 1st chunk in ms and as indizes
        ### for the 1st chunk no period has to be defined, because there is only a single period
        time_ms = str(recording_times.time_lims(chunk=0, compartment="my_pop1"))
        time_idx = str(recording_times.idx_lims(chunk=0, compartment="my_pop1"))
        print(f"{'1st chunk':<25} {time_ms:<18} {time_idx:<18}")
        ### start and end of 1st period of 2nd chunk in ms and as indizes
        time_ms = str(
            recording_times.time_lims(chunk=1, period=0, compartment="my_pop1")
        )
        time_idx = str(
            recording_times.idx_lims(chunk=1, period=0, compartment="my_pop1")
        )
        print(f"{'2nd chunk, 1st period':<25} {time_ms:<18} {time_idx:<18}")
        ### start and end of 2nd period of 2nd chunk in ms and as indizes
        time_ms = str(
            recording_times.time_lims(chunk=1, period=1, compartment="my_pop1")
        )
        time_idx = str(
            recording_times.idx_lims(chunk=1, period=1, compartment="my_pop1")
        )
        print(f"{'2nd chunk, 2nd period':<25} {time_ms:<18} {time_idx:<18}")

    ### The indizes from recording_times can be used for the recroding arrays obtained with get_recordings
    chunk, period = [0, None]
    start_time, end_time = recording_times.time_lims(chunk=chunk, compartment="my_pop1")
    start_idx, end_idx = recording_times.idx_lims(chunk=chunk, compartment="my_pop1")
    period_time = recordings[chunk]["my_pop1;period"]
    x1 = np.arange(start_time, end_time + period_time, period_time)
    v1 = y1[start_idx : end_idx + 1, 0]

    ### same thing again for the 2nd chunk 1st period of my_pop1
    chunk, period = [1, 0]
    start_time, end_time = recording_times.time_lims(
        chunk=chunk, period=period, compartment="my_pop1"
    )
    start_idx, end_idx = recording_times.idx_lims(
        chunk=chunk, period=period, compartment="my_pop1"
    )
    period_time = recordings[chunk]["my_pop1;period"]
    x2 = np.arange(start_time, end_time + period_time, period_time)
    v2 = y2[start_idx : end_idx + 1, 0]

    ### and finally for the 2nd chunk 2nd period of my_pop1
    chunk, period = [1, 1]
    start_time, end_time = recording_times.time_lims(
        chunk=chunk, period=period, compartment="my_pop1"
    )
    start_idx, end_idx = recording_times.idx_lims(
        chunk=chunk, period=period, compartment="my_pop1"
    )
    period_time = recordings[chunk]["my_pop1;period"]
    x3 = np.arange(start_time, end_time + period_time, period_time)
    v3 = y2[start_idx : end_idx + 1, 0]

    ### plot the data of the periods of my_pop1
    plt.figure()
    plt.subplot(211)
    plt.title("my_pop1\n1st chunk")
    plt.plot(x1, v1, color="k")
    plt.xlabel("time [ms]")
    plt.ylabel("membrane potential [mV]")
    plt.subplot(212)
    plt.title("2nd chunk")
    plt.plot(x2, v2, color="g", label="1st period")
    plt.plot(x3, v3, color="r", label="2nd period")
    plt.xlabel("time [ms]")
    plt.ylabel("membrane potential [mV]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/monitor_recordings_periods_my_pop1.svg")

    ### one can combine recordings of multiple chunks, here for example for my_pop1
    ### the combine_chunks function of the recording_times object directly returns
    ### an array with time values and an array with values of the recorded variable
    ### recording pauses are filled with nan values
    times, data_arr = recording_times.combine_chunks(
        recordings, "my_pop1;v", mode="consecutive"
    )

    ### use both arrays to plot the data of all chunks of my_pop1
    make_chunk_plot(
        fig_name="monitor_recordings_chunk_my_pop1",
        times=times,
        data_arr=data_arr,
        recording_times=recording_times,
    )

    ### do the same for my_pop2
    times, data_arr = recording_times.combine_chunks(
        recordings, "my_pop2;v", mode="consecutive"
    )
    make_chunk_plot(
        fig_name="monitor_recordings_chunk_my_pop2",
        times=times,
        data_arr=data_arr,
        recording_times=recording_times,
    )

    return 1


if __name__ == "__main__":
    main()


### console output of this file:
"""
recording time limits of my_pop1
                        in ms              as index (dt=0.1 ms but recording period=2 ms)
1st chunk                 [100.0, 198.0]     [0, 49]
2nd chunk, 1st period     [0.0, 98.0]        [0, 49]
2nd chunk, 2nd period     [200.0, 298.0]     [50, 99]
"""
