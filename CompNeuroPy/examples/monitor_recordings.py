from ANNarchy import Population, Izhikevich, setup, simulate, get_population
from CompNeuroPy import generate_model, Monitors, create_dir
import pylab as plt
import numpy as np

setup(dt=0.1)
create_dir("results")

### first we create a simple model, consisting of two populaitons, each consist of 1 neuron
def create_model():
    a = Population(1, neuron=Izhikevich(), name="my_pop1")
    a.i_offset = 10
    Population(1, neuron=Izhikevich(), name="my_pop2")


my_model = generate_model(
    model_creation_function=create_model,
    name="my_model",
    description="my simple example model",
    compile_folder_name="annarchy_my_model",
)


### after compilation we can define the monitors using the monitor_dicitonary
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
    time_ms = str(recording_times.time_lims(chunk=1, period=0, compartment="my_pop1"))
    time_idx = str(recording_times.idx_lims(chunk=1, period=0, compartment="my_pop1"))
    print(f"{'2nd chunk, 1st period':<25} {time_ms:<18} {time_idx:<18}")
    ### start and end of 2nd period of 2nd chunk in ms and as indizes
    time_ms = str(recording_times.time_lims(chunk=1, period=1, compartment="my_pop1"))
    time_idx = str(recording_times.idx_lims(chunk=1, period=1, compartment="my_pop1"))
    print(f"{'2nd chunk, 2nd period':<25} {time_ms:<18} {time_idx:<18}")


### The indizes from recording_times can be used for the arrays from get_recordings
start_time = recording_times.time_lims(chunk=0, compartment="my_pop1")[0]
start_idx = recording_times.idx_lims(chunk=0, compartment="my_pop1")[0]
end_time = recording_times.time_lims(chunk=0, compartment="my_pop1")[1]
end_idx = recording_times.idx_lims(chunk=0, compartment="my_pop1")[1]
x1 = np.arange(start_time, end_time, recordings[0]["my_pop1;period"])
v1 = y1[start_idx:end_idx, 0]

### same thing again for the 2nd chunk 1st period of my_pop1
start_time = recording_times.time_lims(chunk=1, period=0, compartment="my_pop1")[0]
start_idx = recording_times.idx_lims(chunk=1, period=0, compartment="my_pop1")[0]
end_time = recording_times.time_lims(chunk=1, period=0, compartment="my_pop1")[1]
end_idx = recording_times.idx_lims(chunk=1, period=0, compartment="my_pop1")[1]
x2 = np.arange(start_time, end_time, recordings[0]["my_pop1;period"])
v2 = y2[start_idx:end_idx, 0]

### and finally for the 2nd chunk 2nd period of my_pop1
start_time = recording_times.time_lims(chunk=1, period=1, compartment="my_pop1")[0]
start_idx = recording_times.idx_lims(chunk=1, period=1, compartment="my_pop1")[0]
end_time = recording_times.time_lims(chunk=1, period=1, compartment="my_pop1")[1]
end_idx = recording_times.idx_lims(chunk=1, period=1, compartment="my_pop1")[1]
x3 = np.arange(start_time, end_time, recordings[0]["my_pop1;period"])
v3 = y2[start_idx:end_idx, 0]


### plot the data of my_pop1
plt.figure()
plt.subplot(211)
plt.title("my_pop1\n1st chunk")
plt.plot(x1, v1, color="k")
plt.subplot(212)
plt.title("2nd chunk 2nd period")
plt.plot(x2, v2, color="g", label="1st period")
plt.plot(x3, v3, color="r", label="2nd period")
plt.legend()
plt.tight_layout()
plt.savefig("results/monitor_recordings.svg")


### if there are no pauses in between recordings one can combine recordings of multiple chunks, here for example for my_pop2
### the combine_chunks function of the recording_times object directly returns an array with time values and an array with
### values of the recorded variable
times, data_arr = recording_times.combine_chunks(
    recordings, "my_pop2;v", mode="consecutive"
)


### use both arrays to plot the data
plt.figure()
plt.title("my_pop2")
### mark the border between chunks
plt.axvline(
    recording_times.time_lims(chunk=0, compartment="my_pop2")[1],
    color="gray",
    ls="dotted",
)
plt.text(
    recording_times.time_lims(chunk=0, compartment="my_pop2")[1],
    np.max(data_arr),
    "1st chunk ",
    ha="right",
    va="top",
)
plt.text(
    recording_times.time_lims(chunk=0, compartment="my_pop2")[1],
    np.max(data_arr),
    " 2nd chunk",
    ha="left",
    va="top",
)
plt.plot(times, data_arr[:, 0], color="k")
plt.tight_layout()
plt.savefig("results/monitor_recordings2.svg")


### console output of this file:
"""
recording time limits of my_pop1
                          in ms              as index (dt=0.1)
1st chunk                 [100.0, 200.0]     [0, 1000]
2nd chunk, 1st period     [0, 100.0]         [0, 1000]
2nd chunk, 2nd period     [200.0, 300.0]     [1000, 2000]
"""
