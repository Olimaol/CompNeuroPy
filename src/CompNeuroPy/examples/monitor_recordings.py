"""
demonstrate use cases for CompNeuroMonitors

init
all paused

### 1st chunk
### demonstrate starting/pausing all
>>> start all
all started
>>> pause all
all paused
>>> start all
all started/resumed

### demonstrate pausing single compartments
>>> pause part
part paused
>>> start all
all started/resumed

### demonstrate starting single compartments
>>> pause all
all paused
>>> start part
part started/resumed
>>> start all
all started/resumed

### demonstrate chunking recordings by reset
>>> reset without model
2nd chunk, model not resetted
>>>reset with model
3rd chunk, model resetted

### demonstrate getting recordings during simulation
>>> get_recordings_and_clear
all recordings from aboth should be obtained
>>> simulate again and reset model creating 2 chunks --> check if model or ANNarchy montirs or both need to be resetted
>>> get_recordings_and_clear
new recordings (2 chunks) should be obtianed
>>> simulate again a single chunk
>>> get_recordings and get_recording_times
new recordings (1 chunk) should be obtained


"""
from ANNarchy import Population, Izhikevich, setup, simulate, get_population
from CompNeuroPy import generate_model, CompNeuroMonitors, create_dir
import pylab as plt
import numpy as np


def make_chunk_plot(fig_name, times, data_arr, recording_times):
    """
    this function creates and saves a plot
    used below, at the end of the main() function
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


def get_time_and_data_arr(
    recordings,
    recording_times,
    compartment,
    variable,
    chunk,
    period=None,
):
    """
    returns a time array and data array for the given recorded variable of the given compartment for the given chunk and period

    Args:
        recordings: list
            obtained with CompNeuroMonitors.get_recordings()
        recording_times: obj
            obtained with CompNeuroMonitors.get_recording_times()
        compartment: str
            name of the model compartment
        variable: str
            name of the recorded variable of the model compartment
        chunk: int
            index of chunk (1st chunk --> chunk=0)
        period: int, optional, default=None
            index of period (1st period --> period=0)
    """
    ### with the time limits one can generate an array with all recording time steps
    start_time, end_time = recording_times.time_lims(
        chunk=chunk, period=period, compartment=compartment
    )
    period_time = recordings[chunk][f"{compartment};period"]
    time_arr = np.arange(start_time, end_time + period_time, period_time)
    ### with the index limits one can obtain the data from the recording array
    start_idx, end_idx = recording_times.idx_lims(
        chunk=chunk, period=period, compartment=compartment
    )
    data_arr = recordings[chunk][f"{compartment};{variable}"]
    data_arr = data_arr[start_idx : end_idx + 1, 0]

    return [time_arr, data_arr]


def main():
    ### setup ANNarchy timestep adn create results folder
    setup(dt=0.1)
    create_dir("results")

    ### first we create a simple model, consisting of two populaitons, each consist of 1 neuron
    def create_model():
        a = Population(1, neuron=Izhikevich(), name="my_pop1")
        a.i_offset = 10
        a.v = -70
        a.u = -14
        b = Population(1, neuron=Izhikevich(), name="my_pop2")
        b.v = -70
        b.u = -14

    ### create and compile the model
    generate_model(
        model_creation_function=create_model,
        name="monitor_recordings_model",
        description="my simple example model",
        compile_folder_name="annarchy_monitor_recordings_model",
    )

    ### after compilation we can define the monitors using the monitor_dictionary
    ### and the CompNeuroMonitors class
    ### for my_pop1 we use a recording period of 2 ms
    ### for my_pop2 we do not give a recording preiod, therefore record every timestep
    monitor_dictionary = {"pop;my_pop1;2": ["v", "spike"], "pop;my_pop2": ["v"]}
    mon = CompNeuroMonitors(monitor_dictionary)

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

        mon.pause(["my_pop1"])  # pause recordings of my_pop1
        simulate(100)  # 100 ms not recorded

        mon.start()  # start all monitors --> resume recordings of my_pop1
        simulate(50)  # 50 ms recorded
        get_population("my_pop1").i_offset = 50  # increase activity during last period
        get_population("my_pop2").i_offset = 50  # increase activity during last period
        simulate(50)  # 50 ms recorded

    ### get all recordings and recording_times from the monitor object
    recordings = mon.get_recordings()
    recording_times = mon.get_recording_times()

    ### Let's print the recoding times of my_pop1 (which had a pause in the second chunk, in contrast to my_pop2)
    for _ in ["lets print recoding times of my_pop1"]:
        print("\nrecording time limits of my_pop1")
        ### print the number of periods for both chunks
        print(
            f"1st chunk, nr periods = {recording_times.nr_periods(chunk=0, compartment='my_pop1')}"
        )
        print(
            f"2nd chunk, nr periods = {recording_times.nr_periods(chunk=1, compartment='my_pop1')}"
        )
        ### then print the time limits
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

    ### Let's print also the recoding times of my_pop2
    for _ in ["lets print recoding times of my_pop2"]:
        print("\nrecording time limits of my_pop2")
        ### print the number of periods for both chunks
        print(
            f"1st chunk, nr periods = {recording_times.nr_periods(chunk=0, compartment='my_pop2')}"
        )
        print(
            f"2nd chunk, nr periods = {recording_times.nr_periods(chunk=1, compartment='my_pop2')}"
        )
        ### then print the time limits
        print(
            f"{'':<25} {'in ms':<18} {'as index (dt=0.1 ms and recording period=dt)':<18}"
        )
        ### start and end of 1st chunk in ms and as indizes
        ### for the 1st chunk no period has to be defined, because there is only a single period
        time_ms = str(recording_times.time_lims(chunk=0, compartment="my_pop2"))
        time_idx = str(recording_times.idx_lims(chunk=0, compartment="my_pop2"))
        print(f"{'1st chunk':<25} {time_ms:<18} {time_idx:<18}")
        ### also for the 2nd chunk no period has to be defined, because there is only a single period (pause was only in my_pop1)
        time_ms = str(recording_times.time_lims(chunk=1, compartment="my_pop2"))
        time_idx = str(recording_times.idx_lims(chunk=1, compartment="my_pop2"))
        print(f"{'2nd chunk':<25} {time_ms:<18} {time_idx:<18}")

    ### With recording_times and recordings one can combine the times and the recorded data
    for _ in ["obtain and plot data of my_pop1"]:
        ### let's get the start and end times and indizes for the data of my_pop1 from the first chunk
        x1, y1 = get_time_and_data_arr(
            recordings,
            recording_times,
            compartment="my_pop1",
            variable="v",
            chunk=0,
        )

        ### same for the 2nd chunk 1st period of my_pop1
        x2, y2 = get_time_and_data_arr(
            recordings,
            recording_times,
            compartment="my_pop1",
            variable="v",
            chunk=1,
            period=0,
        )

        ### and finally for the 2nd chunk 2nd period of my_pop1
        x3, y3 = get_time_and_data_arr(
            recordings,
            recording_times,
            compartment="my_pop1",
            variable="v",
            chunk=1,
            period=1,
        )

        ### plot the data of my_pop1
        plt.figure()
        plt.subplot(211)
        plt.title("my_pop1\n1st chunk")
        plt.plot(x1, y1, color="k")
        plt.xlabel("time [ms]")
        plt.ylabel("membrane potential [mV]")
        plt.subplot(212)
        plt.title("2nd chunk")
        plt.plot(x2, y2, color="g", label="1st period")
        plt.plot(x3, y3, color="r", label="2nd period")
        plt.xlabel("time [ms]")
        plt.ylabel("membrane potential [mV]")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/monitor_recordings_my_pop1.png")

    ### also obtain and plot the data of my_pop2
    for _ in ["obtain and plot data of my_pop2"]:
        ### for my_pop2 we have only a single period for each chunk
        ### 1st chunk :
        x1, y1 = get_time_and_data_arr(
            recordings,
            recording_times,
            compartment="my_pop2",
            variable="v",
            chunk=0,
        )
        ### 2nd chunk :
        x2, y2 = get_time_and_data_arr(
            recordings,
            recording_times,
            compartment="my_pop2",
            variable="v",
            chunk=1,
        )
        ### plot
        plt.figure()
        plt.subplot(211)
        plt.title("my_pop2\n1st chunk")
        plt.plot(x1, y1, color="k")
        plt.xlabel("time [ms]")
        plt.ylabel("membrane potential [mV]")
        plt.subplot(212)
        plt.title("2nd chunk")
        plt.plot(x2, y2, color="k")
        plt.xlabel("time [ms]")
        plt.ylabel("membrane potential [mV]")
        plt.tight_layout()
        plt.savefig("results/monitor_recordings_my_pop2.png")

    ### one can combine recordings of multiple chunks
    ### the combine_chunks function of the recording_times object directly returns
    ### an array with time values and an array with values of the recorded variable
    ### recording pauses are filled with nan values
    times, data_arr = recording_times.combine_chunks(
        recordings, "my_pop1;v", mode="consecutive"
    )
    ### use both arrays to plot the data of all chunks of my_pop1
    make_chunk_plot(
        fig_name="monitor_recordings_all_chunks_my_pop1",
        times=times,
        data_arr=data_arr,
        recording_times=recording_times,
    )

    ### do the same for my_pop2
    times, data_arr = recording_times.combine_chunks(
        recordings, "my_pop2;v", mode="consecutive"
    )
    make_chunk_plot(
        fig_name="monitor_recordings_all_chunks_my_pop2",
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
1st chunk, nr periods = 1
2nd chunk, nr periods = 2
                          in ms              as index (dt=0.1 ms but recording period=2 ms)
1st chunk                 [100.0, 198.0]     [0, 49]
2nd chunk, 1st period     [0.0, 98.0]        [0, 49]
2nd chunk, 2nd period     [200.0, 298.0]     [50, 99]

recording time limits of my_pop2
1st chunk, nr periods = 1
2nd chunk, nr periods = 1
                          in ms              as index (dt=0.1 ms and recording period=dt)
1st chunk                 [100.0, 199.9]     [0, 999]
2nd chunk                 [0.0, 299.9]       [0, 2999]
"""
