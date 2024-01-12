import matplotlib.pyplot as plt
from CompNeuroPy import load_variables
from CompNeuroPy.experiment import CompNeuroExp


def get_recordings_and_combine_chunks(results_obj: CompNeuroExp._ResultsCl):
    """
    Function to get the recordings from the results object and combine the chunks into
    a single time and value array.

    Args:
        results_obj (CompNeuroExp._ResultsCl):
            results object returned by the run function of CompNeuroExp
    """
    ### get data
    rec = results_obj.recordings
    rec_times = results_obj.recording_times
    neuron_name = results_obj.data["population_name"]
    ### combine chunks into single time and value array
    time_arr, r_arr = rec_times.combine_chunks(
        recordings=rec,
        recording_data_str=f"{neuron_name};r",
        mode="consecutive",
    )

    return time_arr, r_arr


### load data
loaded_dict = load_variables(
    name_list=["best_from_data", "best_from_neuron"], path="parameter_fit"
)
### get results objects (returned by the CompNeuroExp)
results_from_data = loaded_dict["best_from_data"]["results"]
results_from_target_neuron = loaded_dict["best_from_neuron"]["results_soll"]
results_from_optimized_neuron = loaded_dict["best_from_neuron"]["results"]


### fitted results from data
plt.figure()
### get recordings and combine chunks into single time and value array
time_arr, r_arr = get_recordings_and_combine_chunks(results_from_data)
### plot recordings
plt.plot(time_arr, r_arr[:, 0], color="k")
### plotting settings
plt.title(
    "Target Data:\n 1st 1000 ms = step from 2 to 6\n 2nd 1000 ms = step from 2 to 10"
)
plt.xlabel("time [ms]")
plt.tight_layout()
plt.savefig("parameter_fit/from_data.png")
plt.close("all")


### fitted results from neuron
plt.figure()
### optimized neuron
time_arr, r_arr = get_recordings_and_combine_chunks(results_from_optimized_neuron)
plt.plot(time_arr, r_arr[:, 0], label="optimized neuron")
### target neuron
time_arr, r_arr = get_recordings_and_combine_chunks(results_from_target_neuron)
plt.plot(time_arr, r_arr[:, 0], label="target neuron")
### plotting settings
plt.xlabel("time [ms]")
plt.title("Target Data created with target neuron")
plt.legend()
plt.tight_layout()
plt.savefig("parameter_fit/from_neuron.png")
plt.close("all")
