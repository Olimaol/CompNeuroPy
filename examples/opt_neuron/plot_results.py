import numpy as np
import matplotlib.pyplot as plt
from CompNeuroPy import create_dir


class data:
    def __init__(self) -> None:
        """
        some class for storing data
        """
        pass


opt = data()
target = data()


def get_experiment_info(obj, results):
    """
    function to get the information from the results, store them into obj (object)
    """
    obj.recordings = results.recordings
    obj.data = results.data
    obj.sim = obj.data["sim"]
    obj.recording_times = obj.data["recording_times"]
    obj.pop_name = obj.data["population_name"]
    obj.time_step = obj.data["time_step"]


def combine_chunks(obj):
    """
    function which combines the chunks of obj.recordings
    with the combine_chunks function of recording_times
    """
    obj.times, obj.values = obj.recording_times.combine_chunks(
        recordings=obj.recordings,
        recording_data_str=f"{obj.pop_name};r",
        mode="consecutive",
    )


### define loading/saving directory
load_path = "dataRaw/parameter_fit/"
save_path = "results/parameter_fit/"
create_dir(save_path)


### define of which file the results should be plotted
file = ["", "_with_exp"][0]  # either emtpy string or "_with_exp"


### load the saved optimization file
best = np.load(f"{load_path}best{file}.npy", allow_pickle=True).item()


### get the information such as recordings and further data of the experiment with the optimized neuron
get_experiment_info(opt, best["results"])


### get the target data
if file == "_with_exp":
    ### the results_soll were generated in the same experiment with the target neuron model
    ### --> get the information like for the results of the optimized neuron
    get_experiment_info(target, best["results_soll"])
else:
    ### the results_soll were directly provided to opt_neuron / not generated in experiment
    ### see in the script run_opt_neuron.py
    ### with [:,None] we add the 2nd dimension, which represents the neuron number
    target.values = np.concatenate([best["results_soll"][0], best["results_soll"][1]])[
        :, None
    ]


### in the experiment, we ran two chunks of simulation without any pauses in between
### --> we can use combine_chunks function of recording_times to directly get a time and a values array
combine_chunks(opt)


### if target data was also generated with same experiment --> do the same analyses
if file == "_with_exp":
    combine_chunks(target)


### the simulations were simple current steps, thus we can get the input currents from the simulations with the fucntion get_current_arr
### the simulation was run two times consecutively, thus we can use flat=True
### we don't need target.current, should be the same as opt
opt.current = opt.sim.get_current_arr(opt.time_step, flat=True)


### generate the plot
plt.figure()
### plot the data(optimized and target) in first subplot
plt.subplot(211)
plt.plot(opt.times, target.values[:, 0], label="target", color="r")
plt.plot(opt.times, opt.values[:, 0], label="optimized", color="k", ls="dashed")
plt.xlabel("time in ms")
plt.ylabel("r")
plt.legend()
### plot the input in second subplot
plt.subplot(212)
plt.plot(opt.times, opt.current, color="k")
plt.xlabel("time in ms")
plt.ylabel("I_app")
plt.tight_layout()
plt.savefig(f"{save_path}fitted_r{file}.svg")
