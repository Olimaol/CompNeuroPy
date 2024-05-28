from ANNarchy import (
    Neuron,
    Population,
    compile,
    simulate,
    get_time,
    setup,
    dt,
    Projection,
)
from CompNeuroPy import (
    CompNeuroMonitors,
    PlotRecordings,
    interactive_plot,
    timing_decorator,
)
from CompNeuroPy.neuron_models import PoissonNeuron
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.stats import binom
from functools import wraps
import time
from collections.abc import Iterable
from tqdm import tqdm

setup(dt=0.1)

neuron_izh = Neuron(
    parameters="""
        C = 100.0 : population
        k = 0.7 : population
        v_r = -60.0 : population
        v_t = -40.0 : population
        a = 0.03 : population
        b = -2.0 : population
        c = -50.0 : population
        d = 100.0 : population
        v_peak = 0.0 : population
        I_app = 0.0
        E_ampa = 0.0 : population
        tau_ampa = 10.0 : population
    """,
    equations="""
        ### synaptic current
        tau_ampa * dg_ampa/dt = -g_ampa
        I_ampa = -neg(g_ampa*(v - E_ampa))
        ### Izhikevich spiking
        I_v        = I_app + I_ampa
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_v
        du/dt      = a*(b*(v - v_r) - u)
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
    """,
)

pop_pre = Population(100, neuron=PoissonNeuron(rates=10.0), name="pre")
pop_post = Population(1000, neuron=neuron_izh, name="post")

CONNECTION_PROB = 0.1
WEIGHTS = 1.0
proj = Projection(pre=pop_pre, post=pop_post, target="ampa", name="proj")
proj.connect_fixed_probability(weights=WEIGHTS, probability=CONNECTION_PROB)


monitors = CompNeuroMonitors(
    mon_dict={"pre": ["spike"], "post": ["v", "spike", "I_ampa", "g_ampa"]}
)

compile()

monitors.start()

start = time.time()
simulate(100.0)
pop_pre.rates = 1000.0
simulate(100.0)
print("simulate time:", time.time() - start)

recordings = monitors.get_recordings()
recording_times = monitors.get_recording_times()

PlotRecordings(
    figname="test.png",
    recordings=recordings,
    recording_times=recording_times,
    shape=(2, 2),
    plan={
        "position": [1, 3, 4],
        "compartment": ["pre", "post", "post"],
        "variable": ["spike", "spike", "v"],
        "format": ["hybrid", "hybrid", "line_mean"],
    },
)


def get_binned_spikes(
    spikes_dict: dict, time_lims_steps: tuple[int, int], BIN_SIZE_STEPS: int
):
    """
    Bin the given spike dictionary into time bins of the given size.

    Args:
        spikes_dict (dict):
            A dictionary of spike times for each neuron.
        time_lims_steps (tuple[int, int]):
            The time limits of the spike_dict in steps.
        BIN_SIZE_STEPS (int):
            The size of the bins in steps.

    Returns:
        counts_matrix (np.ndarray):
            The binned spike counts for each neuron with shape (n_bins, n_neurons, 1).
    """
    ### get the spike distribution of each time bin
    bins_array = np.arange(
        time_lims_steps[0],
        (time_lims_steps[1] - time_lims_steps[0]) + BIN_SIZE_STEPS // 2,
        BIN_SIZE_STEPS,
    ).astype(int)

    counts_array_list = [
        np.histogram(spikes_list, bins=bins_array)[0]
        for spikes_list in spikes_dict.values()
    ]
    counts_matrix = np.stack(counts_array_list, axis=1)
    counts_matrix = counts_matrix.reshape(counts_matrix.shape + (1,))

    return counts_matrix


def get_binned_variable(var_array: np.ndarray, BIN_SIZE_STEPS: int):
    """
    Bin the given variable array into time bins of the given size. Bins are created on
    the first axis of the array. The values of the bins are the average of the original
    values in the bin.

    Args:
        var_array (np.ndarray):
            The variable array to bin. First axis is the time axis and should be
            divisible by the bin size. Second axis is the number of time serieses.
        BIN_SIZE_STEPS (int):
            The size of the bins in steps.

    Returns:
        np.ndarray:
            The binned variable array with shape (n_bins, n_time_serieses, 1).
    """
    ### reshape the array to bin the first axis
    reshaped = var_array.reshape(
        var_array.shape[0] // BIN_SIZE_STEPS, BIN_SIZE_STEPS, var_array.shape[1]
    )
    ### get the average of the values in each bin
    averages: np.ndarray = np.mean(reshaped, axis=1)
    averages = averages.reshape(averages.shape + (1,))

    return averages


def get_top_n_percentile(x, pdf, percentile):
    """
    Get the top n percentile of the given data.

    Args:
        x (np.ndarray):
            The data values.
        pdf (np.ndarray):
            The PDF values for the corresponding data values.
        percentile (float):
            The percentile to get.

    Returns:
        x (np.ndarray):
            The data values of the top n percentile.
        pdf (np.ndarray):
            The PDF values for the corresponding data values of the top n percentile.
    """
    ### sort the data based on the pdf values
    sort_indices = np.argsort(pdf)
    x = x[sort_indices]
    pdf = pdf[sort_indices]
    ### get the top n percentile
    pdf_norm = pdf / np.sum(pdf)
    pdf_cumsum = np.cumsum(pdf_norm)
    top_n_percentile = np.where(pdf_cumsum > 1 - percentile / 100)[0][0]
    x = x[top_n_percentile:]
    pdf = pdf[top_n_percentile:]
    ### sort the data based on the data values
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    pdf = pdf[sort_indices]
    return x, pdf


BIN_SIZE_MS = 5
BIN_SIZE_STEPS = int(round(BIN_SIZE_MS / dt()))


class DistPreSpikes:

    def __init__(self, spikes_dict, time_lims_steps):
        """
        Create a distribution object for the given spike dictionary.

        Args:
            spikes_dict (dict):
                A dictionary of spike times for each neuron.
            time_lims_steps (tuple[int, int]):
                The time limits of the spike_dict in steps.
        """

        self.spikes_binned = get_binned_spikes(
            spikes_dict=spikes_dict,
            time_lims_steps=time_lims_steps,
            BIN_SIZE_STEPS=BIN_SIZE_STEPS,
        )
        self._pdf_dict = {}

    def pdf(self, time_bin=0):
        """
        Get the PDF of the spike counts over the population for the given time bin.

        Args:
            time_bin (int):
                The time bin to get the PDF for.

        Returns:
            x (np.ndarray):
                The spike count values of the PDF.
            pdf (np.ndarray):
                The PDF values for the corresponding spike count values.
        """
        ret = self._pdf_dict.get(time_bin)
        if ret is not None:
            return ret

        # scale data to have standard deviation of 1
        # if all values are the same return pdf with 1
        if np.std(self.spikes_binned[time_bin]) == 0:
            return (np.array([self.spikes_binned[time_bin][0]]), np.array([1]))
        else:
            # create histogram from 0 to pdf_max with stepsize 1
            pdf_max = (
                int(
                    round(
                        np.max(self.spikes_binned[time_bin])
                        + np.std(self.spikes_binned[time_bin])
                    )
                )
                + 1
            )
            ret = np.histogram(
                self.spikes_binned[time_bin],
                range=(0, pdf_max),
                bins=np.arange(pdf_max + 1),
                density=True,
            )
            return (ret[1][:-1], ret[0])

    def show_dist(self, time_bin=0):
        """
        Show the distribution of the spike counts over the population for the given time
        bin.

        Args:
            time_bin (int):
                The time bin to show the distribution for.
        """
        x, pdf = self.pdf(time_bin=time_bin)
        plt.bar(x, pdf, alpha=0.5, width=1)
        plt.xlabel("Spikes")
        plt.ylabel("Density")
        plt.title("Spikes Distribution")
        plt.show()


class DistIncomingSpikesFromModel:

    def __init__(self, pre_spikes_dict, proj, time_lims_steps):
        """
        Create a distribution object for the incoming spikes.

        Args:
            pre_spikes_dict (dict):
                A dictionary of spike times for each neuron of the pre population.
            proj (Projection):
                The projection object from the pre to the post population.
            time_lims_steps (tuple[int, int]):
                The time limits of the pre_spikes_dict in steps.
        """
        ### TODO check if calculated incoming spikes distribution (calculated using pre spikes distribution + synapse distribution) is correct
        ### obtain this distribution from the model
        ### use the spike dictionary of the pre pop and then transform it into incoming spikee dict for the post pop and then create a distribution
        ### to transform the spike dict into an incoming spikes dict, wee need to know for each post neuron, to which pre neurons it is connected

        incoming_spikes_dict = {}
        ### loop over each post neuron
        for post_neuron_idx, post_neuron in enumerate(proj):
            incoming_spikes_dict[post_neuron_idx] = []
            if post_neuron is not None:
                ### loop over each pre neuron connected to the post neuron
                for pre_neuron in post_neuron.pre_ranks:
                    ### store the spikes emitted by the pre neuron as incoming spikes for the post neuron
                    incoming_spikes_dict[post_neuron_idx].extend(
                        pre_spikes_dict[pre_neuron]
                    )
                ### sort the incoming spikes
                incoming_spikes_dict[post_neuron_idx].sort()

        ### bin the incoming spikes
        self.incoming_spikes_binned = get_binned_spikes(
            spikes_dict=incoming_spikes_dict,
            time_lims_steps=time_lims_steps,
            BIN_SIZE_STEPS=BIN_SIZE_STEPS,
        )
        self._pdf_dict = {}

    def pdf(self, time_bin=0):
        """
        Get the PDF of the incoming spike counts over the (post) population for the
        given time bin.

        Args:
            time_bin (int):
                The time bin to get the PDF for.

        Returns:
            x (np.ndarray):
                The incoming spike count values of the PDF.
            pdf (np.ndarray):
                The PDF values for the corresponding incoming spike count values.
        """
        ret = self._pdf_dict.get(time_bin)
        if ret is not None:
            return ret

        # if all values are the same return pdf with 1
        if np.std(self.incoming_spikes_binned[time_bin]) == 0:
            return (np.array([self.incoming_spikes_binned[time_bin][0]]), np.array([1]))
        else:
            # create histogram from 0 to pdf_max with stepsize 1
            pdf_max = (
                int(
                    round(
                        np.max(self.incoming_spikes_binned[time_bin])
                        + np.std(self.incoming_spikes_binned[time_bin])
                    )
                )
                + 1
            )
            ret = np.histogram(
                self.incoming_spikes_binned[time_bin],
                range=(0, pdf_max),
                bins=np.arange(pdf_max + 1),
                density=True,
            )
            pdf = ret[0]
            spike_values = ret[1][:-1]
            return (spike_values[pdf > 0], pdf[pdf > 0])

    def show_dist(self, time_bin=0):
        """
        Show the distribution of the spike counts over the population for the given time
        bin.

        Args:
            time_bin (int):
                The time bin to show the distribution for.
        """
        x, pdf = self.pdf(time_bin=time_bin)
        plt.bar(x, pdf, alpha=0.5, width=1)
        plt.xlabel("Spikes")
        plt.ylabel("Density")
        plt.title("Spikes Distribution")
        plt.show()


class DistPostSpikesAndVoltage:

    def __init__(self, spikes_dict, time_lims_steps, voltage_array):
        """
        Create a distribution object for the given spike dictionary and voltage array.

        Args:
            spikes_dict (dict):
                A dictionary of spike times for each neuron of the post population.
            time_lims_steps (tuple[int, int]):
                The time limits of the spike_dict in steps.
            voltage_array (np.ndarray):
                The voltage array of the post population.
        """
        ### bin spikes and voltage over time
        self.spikes_post_binned = get_binned_spikes(
            spikes_dict=spikes_dict,
            time_lims_steps=time_lims_steps,
            BIN_SIZE_STEPS=BIN_SIZE_STEPS,
        )
        self.voltage_binned = get_binned_variable(
            voltage_array, BIN_SIZE_STEPS=BIN_SIZE_STEPS
        )
        ### initial pdf dict
        self._pdf_dict = {}

    def pdf(self, time_bin=0):
        """
        Get the PDF of the spike counts and voltage values over the population for the
        given time bin.

        Args:
            time_bin (int):
                The time bin to get the PDF for.

        Returns:
            x (np.ndarray):
                The spike and voltage values of the PDF. Array of shape (100**2, 2).
                (:, 0) are the spike values and (:, 1) are the voltage values.
            pdf (np.ndarray):
                The PDF values for the corresponding spike and voltage value pairs.
        """
        ret = self._pdf_dict.get(time_bin)
        if ret is not None:
            return ret

        # Create the KDE object
        kde = KernelDensity(kernel="linear")
        # scale data to have standard deviation of 1
        # if all values are the same, scale is 1
        if np.std(self.spikes_post_binned[time_bin]) == 0:
            scale_spikes = 1
        else:
            scale_spikes = 1 / np.std(self.spikes_post_binned[time_bin])
        if np.std(self.voltage_binned[time_bin]) == 0:
            scale_voltage = 1
        else:
            scale_voltage = 1 / np.std(self.voltage_binned[time_bin])
        # combine the spike and voltage data
        train_data = np.concatenate(
            [
                scale_spikes * self.spikes_post_binned[time_bin],
                scale_voltage * self.voltage_binned[time_bin],
            ],
            axis=1,
        )
        # Fit the data to the KDE
        kde.fit(train_data)
        # Create points for which to estimate the PDF, here, 2D for spike and voltage
        # spike between 0 and pdf_spikes_max (depends on data), voltage between -100 and 0
        pdf_spikes_min = 0
        pdf_spikes_max = int(
            round(
                np.max(self.spikes_post_binned[time_bin])
                + np.std(self.spikes_post_binned[time_bin])
            )
        )
        pdf_spikes_max = max(pdf_spikes_max, 1)
        x = np.mgrid[pdf_spikes_min:pdf_spikes_max:100j, -100:0:100j].reshape(2, -1).T
        # Estimate the PDF for these points
        x_estimate = np.copy(x)
        x_estimate[:, 0] = scale_spikes * x_estimate[:, 0]
        x_estimate[:, 1] = scale_voltage * x_estimate[:, 1]
        log_density = kde.score_samples(x_estimate)
        pdf = np.exp(log_density)
        # spikes are discrete, sum values between 0 and 1, and between 1 and 2, etc.
        pdf = pdf.reshape(100, 100)
        x = x.reshape(100, 100, 2)
        pdf_discrete_size = int(round(pdf_spikes_max - pdf_spikes_min) + 1)
        pdf_discrete = np.zeros((pdf_discrete_size, 100))
        for i in range(pdf_discrete_size):
            pdf_discrete[i] = np.sum(
                pdf[(x[:, 0, 0] >= i) & (x[:, 0, 0] < i + 1)], axis=0
            )
        x_discrete_spikes = np.arange(pdf_spikes_min, pdf_spikes_max + 0.5).astype(int)
        x_discrete_voltage = np.linspace(-100, 0, 100)
        # x_discrete are all combinations of x_discrete_spikes and x_discrete_voltage
        x_discrete = np.zeros((pdf_discrete_size * 100, 2))
        for i in range(pdf_discrete_size):
            x_discrete[i * 100 : (i + 1) * 100, 0] = x_discrete_spikes[i]
            x_discrete[i * 100 : (i + 1) * 100, 1] = x_discrete_voltage
        x_discrete = x_discrete.reshape(pdf_discrete_size, 100, 2)
        ### normalize so that sum*stepsize = 1 (stepsize of spikes is 1)
        stepsize = 1 * (x_discrete[0, 1, 1] - x_discrete[0, 0, 1])
        pdf_discrete = pdf_discrete / np.sum(pdf_discrete) / stepsize
        # get indices of rows and columns where sum of pdf is not 0
        row_indices = np.where(np.sum(pdf_discrete, axis=1) > 0)[0]
        col_indices = np.where(np.sum(pdf_discrete, axis=0) > 0)[0]
        # get the pdf values and corresponding x values for the non-zero values
        pdf_discrete = pdf_discrete[row_indices][:, col_indices]
        x_discrete = x_discrete[row_indices][:, col_indices]
        # store the pdf
        ret = (x_discrete[pdf_discrete > 0], pdf_discrete[pdf_discrete > 0])
        self._pdf_dict[time_bin] = ret
        # return the pdf
        return ret

    def show_dist(self, time_bin=0):
        """
        Show the distribution of the spike counts and voltage values over the population
        for the given time bin.

        Args:
            time_bin (int):
                The time bin to show the distribution for.
        """
        x, pdf = self.pdf(time_bin=time_bin)
        if x.shape[0] == 1:
            plt.bar(x[0, :, 1], pdf[0, :])
            plt.xlabel("Voltage")
            plt.ylabel("Density")
            plt.title(f"Voltage Distribution, Spikes={x[0, 0, 0]}")
            plt.show()
        else:
            plt.pcolormesh(x[:, :, 0], x[:, :, 1], pdf)
            plt.xlabel("Spikes")
            plt.ylabel("Voltage")
            plt.title("Voltage-Spikes Distribution")
        plt.show()


class DistPostConductance:

    def __init__(self, conductance_array):
        """
        Create a distribution object for the given conductance array.

        Args:
            conductance_array (np.ndarray):
                The voltage array of the post population.
        """
        ### bin conductance over time
        self.g_binned = get_binned_variable(
            conductance_array, BIN_SIZE_STEPS=BIN_SIZE_STEPS
        )
        ### initial pdf dict
        self._pdf_dict = {}

    def pdf(self, time_bin=0):
        """
        Get the PDF of the conductance values over the population for the given time bin.

        Args:
            time_bin (int):
                The time bin to get the PDF for.

        Returns:
            x (np.ndarray):
                The conductance values of the PDF. Array of shape (100,).
            pdf (np.ndarray):
                The PDF values for the corresponding conductance values.
        """
        ret = self._pdf_dict.get(time_bin)
        if ret is not None:
            return ret

        # Create the KDE object
        kde = KernelDensity(kernel="linear")
        # scale data to have standard deviation of 1
        # if all values are the same, return this value with pdf 1
        if np.std(self.g_binned[time_bin]) == 0:
            return (np.array([self.g_binned[time_bin][0]]), np.array([1]))
        else:
            scale = 1 / np.std(self.g_binned[time_bin])
        # Fit the data to the KDE
        kde.fit(scale * self.g_binned[time_bin].reshape(-1, 1))
        # Create points for which to estimate the PDF, values can only be between 0 and max
        pdf_max = self.g_binned[time_bin].max() + np.std(self.g_binned[time_bin])
        x = np.linspace(0, pdf_max, 100).reshape(-1, 1)
        # Estimate the PDF for these points
        log_density = kde.score_samples(x * scale)
        pdf = np.exp(log_density)
        ### normalize so that sum*stepsize = 1
        stepsize = x[1, 0] - x[0, 0]
        pdf = pdf / np.sum(pdf) / stepsize
        conductance_values = x[:, 0]
        ret = (conductance_values[pdf > 0], pdf[pdf > 0])

        return ret

    def show_dist(self, time_bin=0):
        """
        Show the distribution of the conductances over the population for the given time
        bin.

        Args:
            time_bin (int):
                The time bin to show the distribution for.
        """
        x, pdf = self.pdf(time_bin=time_bin)
        plt.bar(x, pdf, alpha=0.5, width=x[1] - x[0])
        plt.xlabel("Conductance")
        plt.ylabel("Density")
        plt.title("Conductance Distribution")
        plt.show()


class DistSynapses:
    def __init__(self, pre_pop_size, connection_probability):
        """
        Create a distribution object for the number of synapses of the post population
        neurons.

        Args:
            pre_pop_size (int):
                The size of the pre population.
            connection_probability (float):
                The probability of connection between the pre and post populations.
        """
        number_synapses = binom(pre_pop_size, connection_probability)
        self._x = np.arange(
            number_synapses.ppf(0.05), number_synapses.ppf(0.95) + 1
        ).astype(int)
        self._pdf = number_synapses.pmf(self._x)
        ### normalize the pdf to sum to 1 (since stepsize is 1)
        self._pdf = self._pdf / np.sum(self._pdf)

    def pdf(self):
        """
        Get the PDF of the number of synapses of the post population neurons.

        Returns:
            x (np.ndarray):
                The synapse count values of the PDF.
            pdf (np.ndarray):
                The PDF values for the corresponding synapse count values.
        """
        return self._x[self._pdf > 0], self._pdf[self._pdf > 0]

    def show_dist(self):
        """
        Show the distribution of the number of synapses of the post population neurons.
        """
        x, pdf = self.pdf()
        plt.bar(x, pdf, alpha=0.5, width=1)
        plt.xlabel("Synapses")
        plt.ylabel("Density")
        plt.title("Synapses Distribution")
        plt.show()


class DistIncomingSpikes:

    def __init__(self, dist_pre_spikes: DistPreSpikes, dist_synapses: DistSynapses):
        """
        Create a distribution object for the incoming spike counts over the post
        population for the given pre spike and synapse distributions.

        Args:
            dist_pre_spikes (DistPreSpikes):
                The distribution of the pre spike counts.
            dist_synapses (DistSynapses):
                The distribution of the number of synapses of the post population neurons.
        """
        self.dist_pre_spikes = dist_pre_spikes
        self.dist_synapses = dist_synapses
        self._pdf_dict = {}

    def pdf(self, time_bin=0):
        """
        Get the PDF of the incoming spike counts over the post population for the given
        time bin.

        Args:
            time_bin (int):
                The time bin to get the PDF for.

        Returns:
            x (np.ndarray):
                The incoming spike count values of the PDF.
            pdf (np.ndarray):
                The PDF values for the corresponding incoming spike count values.
        """
        ret = self._pdf_dict.get(time_bin)
        if ret is not None:
            return ret

        ### get pdfs of pre spikes and synapses
        spike_count_arr, pdf_spike_count_arr = self.dist_pre_spikes.pdf(
            time_bin=time_bin
        )
        synapse_count_arr, pdf_synapse_count_arr = self.dist_synapses.pdf()
        ### if using pdfs to combine them and calculate new pdfs --> only use i.e. the top 95%
        spike_count_arr, pdf_spike_count_arr = get_top_n_percentile(
            spike_count_arr, pdf_spike_count_arr, 95
        )
        synapse_count_arr, pdf_synapse_count_arr = get_top_n_percentile(
            synapse_count_arr, pdf_synapse_count_arr, 95
        )
        ### for each possible number of synapses, calculate the pdf of the sum of
        ### incoming spikes and weight it with the probability of the corresponding
        ### number of synapses
        incoming_spike_count_list = []
        pdf_incoming_spike_count_list = []
        for synapse_count, pdf_synapse_count in zip(
            synapse_count_arr, pdf_synapse_count_arr
        ):
            incoming_spike_count, pdf_incoming_spike_count = self.pdf_of_sum(
                spike_count_arr, pdf_spike_count_arr, n=synapse_count
            )
            incoming_spike_count_list.extend(incoming_spike_count)
            pdf_incoming_spike_count_list.extend(
                pdf_incoming_spike_count * pdf_synapse_count
            )
        ### combine the pdf of unique values in incoming_spike_count_list
        incoming_spike_count_arr = np.array(incoming_spike_count_list)
        pdf_incoming_spike_count_arr = np.array(pdf_incoming_spike_count_list)
        sort_indices = np.argsort(incoming_spike_count_arr)
        incoming_spike_count_arr = incoming_spike_count_arr[sort_indices]
        pdf_incoming_spike_count_arr = pdf_incoming_spike_count_arr[sort_indices]
        ### combine unique values of incoming spikes and sum the corresponding pdf values (slice
        ### the pdf array at the unique indices and sum the values between the indices)
        incoming_spike_count_arr, unique_indices = np.unique(
            incoming_spike_count_arr, return_index=True
        )
        pdf_incoming_spike_count_arr = np.add.reduceat(
            pdf_incoming_spike_count_arr, unique_indices
        )
        ### sort pdf and values based on pdf
        sort_indices = np.argsort(pdf_incoming_spike_count_arr)
        incoming_spike_count_arr = incoming_spike_count_arr[sort_indices]
        pdf_incoming_spike_count_arr = pdf_incoming_spike_count_arr[sort_indices]
        ### normalize the pdf to sum to 1 (since stepsize is 1) (it is already
        ### normalized but to not accumulate errors, normalize it again)
        pdf_incoming_spike_count_arr = pdf_incoming_spike_count_arr / np.sum(
            pdf_incoming_spike_count_arr
        )
        ### store the pdf
        ret = (
            incoming_spike_count_arr[pdf_incoming_spike_count_arr > 0],
            pdf_incoming_spike_count_arr[pdf_incoming_spike_count_arr > 0],
        )
        self._pdf_dict[time_bin] = ret
        ### return the pdf
        return ret

    def show_dist(self, time_bin=0):
        """
        Show the distribution of the incoming spike counts over the post population for
        the given time bin.
        """
        x, pdf = self.pdf(time_bin=time_bin)
        plt.bar(x, pdf, alpha=0.5, width=1)
        plt.xlabel("Incoming Spikes")
        plt.ylabel("Density")
        plt.title("Incoming Spikes Distribution")
        plt.show()

    def pdf_of_sum(self, x_values, pdf_values, n):
        """
        Calculate the PDF of the n-th sum of a random variable X, where X has the PDF given by
        x_values and pdf_values.

        Args:
            x_values(np.ndarray):
                The values of the random variable X.
            pdf_values(np.ndarray):
                The PDF values for the corresponding values of the random variable X.
        """
        result_x = x_values.copy()
        result_pdf = pdf_values.copy()
        for _ in range(n - 1):
            result_x_new = []
            result_pdf_new = []
            # multiply combinations for pdf_values and add combinations for x_values
            result_x_new = np.add.outer(result_x, x_values).flatten()
            result_pdf_new = np.outer(result_pdf, pdf_values).flatten()
            # sort based on x values
            sort_indices = np.argsort(result_x_new)
            result_x_new = result_x_new[sort_indices]
            result_pdf_new = result_pdf_new[sort_indices]
            # sum the pdf_values for each unique value in x_sum
            result_x, unique_indices = np.unique(result_x_new, return_index=True)
            result_pdf = np.add.reduceat(result_pdf_new, unique_indices)

        return result_x, result_pdf


class ConductanceCalc:

    def __init__(self, tau, w):
        """
        Create a conductance calculator object with the given synaptic decay time
        constant and weight.

        Args:
            tau (float):
                The synaptic decay time constant.
            w (float):
                The synaptic weight.
        """
        self.tau = tau
        self.w = w

    def g_mean(self, nbr_spikes: int | np.ndarray, g_init: np.ndarray):
        """
        Calculate the mean conductance of the post population neurons for the given number
        of incoming spikes and initial (prev) conductances.

        Args:
            nbr_spikes (int | np.ndarray):
                The number of incoming spikes.
            g_init (np.ndarray):
                The initial (prev) conductances of the post population neurons.

        Returns:
            g_mean_arr (np.ndarray):
                The mean conductance values for the given number of spikes and initial
                (prev) conductances. First axis is the number of spikes and second axis
                is the initial conductance values.
            g_end_arr (np.ndarray):
                The end conductance values for the given number of spikes and initial
                (prev) conductances. First axis is the number of spikes and second axis
                is the initial conductance values.
        """

        # initial (prev) conductance
        self.g_init = g_init
        # single number of spikes (check if nbr_spikes is iterable)
        if not isinstance(nbr_spikes, Iterable):
            # isi duration in ms if spikes are evenly distributed
            self.d = BIN_SIZE_MS / (int(nbr_spikes) + 1)
            # mean exp for calculating the mean conductance
            self.mean_exp = np.mean(np.exp(-np.linspace(0, self.d, 100) / self.tau))
            # calculate the mean conductance
            g_mean_arr = np.zeros((1, g_init.size))
            g_end_arr = np.zeros((1, g_init.size))
            g_mean_arr[0] = self.mean_exp * np.mean(
                np.stack(self._g_mean_recursive(lvl=int(nbr_spikes))), axis=0
            )
            g_end_arr[0] = self._foo(lvl=int(nbr_spikes)) - self.w
            return g_mean_arr, g_end_arr
        # multiple number of spikes
        else:
            g_mean_arr = np.zeros((nbr_spikes.size, g_init.size))
            g_end_arr = np.zeros((nbr_spikes.size, g_init.size))
            for lvl_idx, lvl in enumerate(nbr_spikes):
                # isi duration in ms if spikes are evenly distributed
                self.d = BIN_SIZE_MS / (int(lvl) + 1)
                # mean exp for calculating the mean conductance
                self.mean_exp = np.mean(np.exp(-np.linspace(0, self.d, 100) / self.tau))
                # calculate the mean conductance
                g_mean_arr[lvl_idx] = self.mean_exp * np.mean(
                    np.stack(self._g_mean_recursive(lvl=int(lvl))), axis=0
                )
                g_end_arr[lvl_idx] = self._foo(lvl=int(lvl)) - self.w
            return g_mean_arr, g_end_arr

    def _foo(self, lvl):
        if lvl == 0:
            ret = self.g_init * np.exp(-self.d / self.tau) + self.w
            return ret
        return self._foo(lvl - 1) * np.exp(-self.d / self.tau) + self.w

    def _g_mean_recursive(self, lvl):
        if lvl == 0:
            return [self.g_init]
        ret_rec = self._g_mean_recursive(lvl - 1)
        ret_rec.append(self._foo(lvl - 1))
        return ret_rec

    def show_conductance(self, nbr_spikes: int, g_init: np.ndarray):
        """
        Show the conductance of the post population neurons for the given number of spikes
        and initial (prev) conductances.

        Args:
            nbr_spikes (int):
                The number of incoming spikes.
            g_init (np.ndarray):
                The initial (prev) conductances of the post population neurons.
        """
        # calculate g_mean and g_end and print values
        g_mean, g_end = self.g_mean(nbr_spikes=nbr_spikes, g_init=g_init)
        print(f"Mean Conductance: {g_mean}\nEnd Conductance: {g_end}")
        # generate conductance over time
        timestep = 0.0001
        # time over bin
        t_arr = np.arange(0, BIN_SIZE_MS, timestep)
        # when spikes occur
        spike_idx = np.arange(
            t_arr.size // (nbr_spikes + 1),
            t_arr.size - (t_arr.size // (nbr_spikes + 1)) / 2,
            t_arr.size // (nbr_spikes + 1),
        )
        # loop over time and calculate conductance
        g = g_init
        g_list = []
        for t_idx, t in enumerate(t_arr):
            g = g - (g / self.tau) * timestep
            if t_idx in spike_idx:
                g = g + self.w
            g_list.append(g)
        # plot the conductance
        g_mean = np.mean(np.stack(g_list), axis=0)
        g_end = g_list[-1]
        for g_value in g_end:
            plt.axhline(y=g_value, color="r", linestyle="--")
        plt.title(f"Mean Conductance: {g_mean}\nEnd Conductance: {g_end}")
        plt.plot(t_arr, g_list)
        plt.xlabel("Time (ms)")
        plt.ylabel("Conductance")
        plt.tight_layout()
        plt.show()


class DistCurrentConductance:

    def __init__(self, tau, w):
        """
        Create a current conductance object with the given synaptic decay time constant
        and weight.

        Args:
            tau (float):
                The synaptic decay time constant.
            w (float):
                The synaptic weight.
        """
        self.w = w
        self.conductance_calc = ConductanceCalc(tau=tau, w=w)

    def pdf(
        self,
        incoming_spikes_count_arr,
        pdf_incoming_spikes_count_arr,
        prev_g_arr=np.array([0, 10]),
        pdf_prev_g_arr=np.array([0.8, 0.2]),
    ):
        """
        Get the PDF of the current conductances of the post population for the given
        incoming spikes distribution and previous conductances distribution.

        Args:
            incoming_spikes_count_arr (np.ndarray):
                The incoming spike count values.
            pdf_incoming_spikes_count_arr (np.ndarray):
                The PDF values for the corresponding incoming spike count values.
            prev_g_arr (np.ndarray):
                The previous conductance values.
            pdf_prev_g_arr (np.ndarray):
                The PDF values for the corresponding previous conductance values.

        Returns:
            x (np.ndarray):
                The current conductance values of the PDF.
            pdf (np.ndarray):
                The PDF values for the corresponding current conductance values.
        """
        ### if using pdfs to combine them and calculate new pdfs --> only use i.e. the top 95%
        incoming_spikes_count_arr, pdf_incoming_spikes_count_arr = get_top_n_percentile(
            incoming_spikes_count_arr, pdf_incoming_spikes_count_arr, 95
        )
        prev_g_arr, pdf_prev_g_arr = get_top_n_percentile(
            prev_g_arr, pdf_prev_g_arr, 95
        )
        ### get the pdf by combining the pdfs of the incoming spikes and previous conductances
        pdf_current_g_arr = np.outer(pdf_incoming_spikes_count_arr, pdf_prev_g_arr)

        current_g_arr = np.empty((incoming_spikes_count_arr.size, prev_g_arr.size))
        current_g_end_arr = np.empty((incoming_spikes_count_arr.size, prev_g_arr.size))
        for incoming_spikes_count_idx, incoming_spikes_count in enumerate(
            incoming_spikes_count_arr
        ):
            (
                current_g_arr[incoming_spikes_count_idx],
                current_g_end_arr[incoming_spikes_count_idx],
            ) = conductance_calc.g_mean(
                nbr_spikes=incoming_spikes_count, g_init=prev_g_arr
            )

        ### get distribution for g_mean and g_end
        ret_g_mean = self._get_pdf_of_g_arr(
            pdf_current_g_arr.flatten(), current_g_arr.flatten()
        )
        ret_g_end = self._get_pdf_of_g_arr(
            pdf_current_g_arr.flatten(), current_g_end_arr.flatten()
        )

        return ret_g_mean, ret_g_end

    def _get_pdf_of_g_arr(self, pdf_current_g_arr, current_g_arr):

        ### use the conductance and corresponding pdf samples to estimate the density
        # scale data to have standard deviation of 1
        # if all values are the same return pdf with 1
        if np.std(current_g_arr) == 0:
            return (np.array([current_g_arr[0]]), np.array([1]))
        else:
            # create histogram from 0 to pdf_max with stepsize self.w/100
            pdf_max = np.max(current_g_arr)
            ret = np.histogram(
                current_g_arr,
                weights=pdf_current_g_arr,
                range=(0, pdf_max),
                bins=np.arange(0, pdf_max + (self.w / 100) / 2, self.w / 100),
                density=True,
            )
            pdf = ret[0]
            g_values = ret[1][:-1]
            return (g_values[pdf > 0], pdf[pdf > 0])

    def show_dist(
        self,
        incoming_spikes_count_arr,
        pdf_incoming_spikes_count_arr,
        prev_g_arr,
        pdf_prev_g_arr,
    ):
        """
        Show the distribution of the current conductances of the post population for the
        given incoming spikes distribution and previous conductances distribution.
        """

        dist_mean, dist_end = self.pdf(
            incoming_spikes_count_arr=incoming_spikes_count_arr,
            pdf_incoming_spikes_count_arr=pdf_incoming_spikes_count_arr,
            prev_g_arr=prev_g_arr,
            pdf_prev_g_arr=pdf_prev_g_arr,
        )
        mask = dist_mean[1] > 0
        x_mean = (dist_mean[0][mask], dist_mean[1][mask])
        mask = dist_end[1] > 0
        x_end = (dist_end[0][mask], dist_end[1][mask])
        plt.subplot(411)
        plt.bar(
            incoming_spikes_count_arr, pdf_incoming_spikes_count_arr, alpha=0.5, width=1
        )
        plt.xlabel("Incoming Spikes")
        plt.ylabel("Density")
        plt.title("Incoming Spikes Distribution")

        plt.subplot(412)
        if len(prev_g_arr) > 1:
            width = prev_g_arr[1] - prev_g_arr[0]
        else:
            width = 1
        plt.bar(prev_g_arr, pdf_prev_g_arr, alpha=0.5, width=width)
        plt.xlabel("Conductance")
        plt.ylabel("Density")
        plt.title("Previous Conductance Distribution")

        plt.subplot(413)
        plt.bar(
            x_mean[0],
            x_mean[1],
            alpha=0.5,
            width=np.min(np.diff(x_mean[0])),
        )
        plt.xlim(
            np.min(np.concatenate([x_mean[0], x_end[0]]))
            - np.min(np.diff(x_mean[0])) / 2,
            np.max(np.concatenate([x_mean[0], x_end[0]]))
            + np.min(np.diff(x_mean[0])) / 2,
        )
        plt.xlabel("Conductance Mean")
        plt.ylabel("Density")
        plt.title("Conductance Mean Distribution")

        plt.subplot(414)
        plt.bar(
            x_end[0],
            x_end[1],
            alpha=0.5,
            width=np.min(np.diff(x_end[0])),
        )
        plt.xlim(
            np.min(
                np.concatenate([x_mean[0], x_end[0]]) - np.min(np.diff(x_end[0])) / 2
            ),
            np.max(
                np.concatenate([x_mean[0], x_end[0]]) + np.min(np.diff(x_end[0])) / 2
            ),
        )
        plt.xlabel("Conductance End")
        plt.ylabel("Density")
        plt.title("Conductance End Distribution")
        plt.tight_layout()
        plt.show()


dist_pre_spikes = DistPreSpikes(
    spikes_dict=recordings[0]["pre;spike"], time_lims_steps=(0, 2000)
)

dist_post = DistPostSpikesAndVoltage(
    spikes_dict=recordings[0]["post;spike"],
    time_lims_steps=(0, 2000),
    voltage_array=recordings[0]["post;v"],
)

dist_synapses = DistSynapses(
    pre_pop_size=pop_pre.size, connection_probability=CONNECTION_PROB
)

dist_incoming_spikes = DistIncomingSpikes(
    dist_pre_spikes=dist_pre_spikes, dist_synapses=dist_synapses
)

conductance_calc = ConductanceCalc(tau=pop_post.tau_ampa, w=WEIGHTS)

dist_current_conductance = DistCurrentConductance(tau=pop_post.tau_ampa, w=WEIGHTS)

dist_post_conductance = DistPostConductance(
    conductance_array=recordings[0]["post;g_ampa"]
)

dist_incoming_spikes_from_model = DistIncomingSpikesFromModel(
    pre_spikes_dict=recordings[0]["pre;spike"],
    proj=proj,
    time_lims_steps=(0, 2000),
)

# Plot the results
PLOT_EXAMPLES = False
if PLOT_EXAMPLES:
    ### dist pre spikes (directly from recordings)
    # dist_pre_spikes.show_dist(time_bin=10)
    # dist_pre_spikes.show_dist(time_bin=-1)
    ### dist post spikes and voltage (directly from recordings)
    # dist_post.show_dist(time_bin=10)
    # dist_post.show_dist(time_bin=-1)
    ### dist synapses (calculated from connection probability)
    # dist_synapses.show_dist()
    ### dist incoming spikes (calculated from dist pre spikes and dist synapses)
    dist_incoming_start = dist_incoming_spikes.pdf(time_bin=5)
    dist_incoming_end = dist_incoming_spikes.pdf(time_bin=25)
    dist_incoming_from_model_start = dist_incoming_spikes_from_model.pdf(time_bin=5)
    dist_incoming_from_model_end = dist_incoming_spikes_from_model.pdf(time_bin=25)
    plt.subplot(211)
    plt.bar(
        dist_incoming_start[0],
        dist_incoming_start[1],
        alpha=0.5,
        width=1,
        color="b",
        label="calculated",
    )
    plt.bar(
        dist_incoming_from_model_start[0],
        dist_incoming_from_model_start[1],
        alpha=0.5,
        width=1,
        color="r",
        label="model",
    )
    plt.legend()
    plt.xlabel("Incoming Spikes")
    plt.ylabel("Density")
    plt.title("Incoming Spikes Distribution Start")
    plt.subplot(212)
    plt.bar(
        dist_incoming_end[0],
        dist_incoming_end[1],
        alpha=0.5,
        width=1,
        color="b",
        label="calculated",
    )
    plt.bar(
        dist_incoming_from_model_end[0],
        dist_incoming_from_model_end[1],
        alpha=0.5,
        width=1,
        color="r",
        label="model",
    )
    plt.legend()
    plt.xlabel("Incoming Spikes")
    plt.ylabel("Density")
    plt.title("Incoming Spikes Distribution End")
    plt.tight_layout()
    plt.show()
    ### conductance calc (update conductance values based on incoming spikes)
    # conductance_calc.show_conductance(nbr_spikes=5, g_init=np.array([0, 0.5, 1.0, 8.0]))
    ### dist current conductance (calculated from dist incoming spikes and some example previous conductance)
    # incoming_spikes_count_arr, pdf_incoming_spikes_count_arr = dist_incoming_spikes.pdf(
    #     time_bin=10
    # )
    # dist_current_conductance.show_dist(
    #     incoming_spikes_count_arr=incoming_spikes_count_arr,
    #     pdf_incoming_spikes_count_arr=pdf_incoming_spikes_count_arr,
    #     prev_g_arr=np.array([0, 180.0]),
    #     pdf_prev_g_arr=np.array([0.5, 0.5]),
    # )
    # incoming_spikes_count_arr, pdf_incoming_spikes_count_arr = dist_incoming_spikes.pdf(
    #     time_bin=-1
    # )
    # dist_current_conductance.show_dist(
    #     incoming_spikes_count_arr=incoming_spikes_count_arr,
    #     pdf_incoming_spikes_count_arr=pdf_incoming_spikes_count_arr,
    #     prev_g_arr=np.array([0, 180.0]),
    #     pdf_prev_g_arr=np.array([0.5, 0.5]),
    # )
    ### dist post conductance (directly from recordings)
    # dist_post_conductance.show_dist(time_bin=10)
    # dist_post_conductance.show_dist(time_bin=-1)
    pass

CONDUCTANCE_LOOP = True
if CONDUCTANCE_LOOP:
    nbr_bins = dist_pre_spikes.spikes_binned.shape[0]
    g_end_dist = (np.array([0.0]), np.array([1.0]))
    g_arr_list = []
    pdf_g_arr_list = []
    g_model_arr_list = []
    pdf_g_model_arr_list = []
    incoming_spikes_count_arr_list, pdf_incoming_spikes_count_arr_list = [], []
    incoming_spikes_count_arr_model_list, pdf_incoming_spikes_count_arr_model_list = (
        [],
        [],
    )
    start = time.time()
    for bin in tqdm(range(nbr_bins)):
        ### calculate incoming spikes distribution from pre spikes and synapse distribution
        incoming_spikes_count_arr, pdf_incoming_spikes_count_arr = (
            dist_incoming_spikes.pdf(time_bin=bin)
        )
        ### get incoming spikes distribution directly from model
        incoming_spikes_count_arr_model, pdf_incoming_spikes_count_arr_model = (
            dist_incoming_spikes_from_model.pdf(time_bin=bin)
        )
        ### calculate current conductance distribution
        g_mean_dist, g_end_dist = dist_current_conductance.pdf(
            incoming_spikes_count_arr=incoming_spikes_count_arr,
            pdf_incoming_spikes_count_arr=pdf_incoming_spikes_count_arr,
            prev_g_arr=g_end_dist[0],
            pdf_prev_g_arr=g_end_dist[1],
        )
        ### get current conductance distribution from model
        g_model_arr, pdf_g_model_arr = dist_post_conductance.pdf(time_bin=bin)
        ### store the conductance distribution
        g_arr_list.append(g_mean_dist[0])
        pdf_g_arr_list.append(g_mean_dist[1])
        g_model_arr_list.append(g_model_arr)
        pdf_g_model_arr_list.append(pdf_g_model_arr)
        ### store the incoming spikes distribution
        incoming_spikes_count_arr_list.append(incoming_spikes_count_arr)
        pdf_incoming_spikes_count_arr_list.append(pdf_incoming_spikes_count_arr)
        incoming_spikes_count_arr_model_list.append(incoming_spikes_count_arr_model)
        pdf_incoming_spikes_count_arr_model_list.append(
            pdf_incoming_spikes_count_arr_model
        )
    print("loop time:", time.time() - start)

    start = time.time()
    # Plotting
    plt.subplot(411)
    for x, y, z in zip(range(len(g_arr_list)), g_arr_list, pdf_g_arr_list):
        norm = plt.Normalize(z.min(), z.max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(z))
        plt.hlines(y=y, xmin=x - norm(z) * 0.5, xmax=x + norm(z) * 0.5, color=colors)
    plt.ylim(0, 200)
    plt.xlabel("Time Bins")
    plt.ylabel("Conductance")
    plt.title("Conductance Distribution over Time Bins")
    plt.subplot(412)
    for x, y, z in zip(
        range(len(g_model_arr_list)), g_model_arr_list, pdf_g_model_arr_list
    ):
        norm = plt.Normalize(z.min(), z.max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(z))
        plt.hlines(y=y, xmin=x - norm(z) * 0.5, xmax=x + norm(z) * 0.5, color=colors)
    plt.ylim(0, 200)
    plt.xlabel("Time Bins")
    plt.ylabel("Conductance")
    plt.title("Model Conductance Distribution over Time Bins")
    plt.subplot(413)
    for x, y, z in zip(
        range(len(incoming_spikes_count_arr_list)),
        incoming_spikes_count_arr_list,
        pdf_incoming_spikes_count_arr_list,
    ):
        norm = plt.Normalize(z.min(), z.max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(z))
        plt.hlines(y=y, xmin=x - norm(z) * 0.5, xmax=x + norm(z) * 0.5, color=colors)
    plt.xlabel("Time Bins")
    plt.ylabel("Incoming Spikes")
    plt.title("Incoming Spikes Distribution over Time Bins")
    plt.subplot(414)
    for x, y, z in zip(
        range(len(incoming_spikes_count_arr_model_list)),
        incoming_spikes_count_arr_model_list,
        pdf_incoming_spikes_count_arr_model_list,
    ):
        norm = plt.Normalize(z.min(), z.max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(z))
        plt.hlines(y=y, xmin=x - norm(z) * 0.5, xmax=x + norm(z) * 0.5, color=colors)
    plt.xlabel("Time Bins")
    plt.ylabel("Incoming Spikes")
    plt.title("Model Incoming Spikes Distribution over Time Bins")
    plt.tight_layout()
    print("plot time:", time.time() - start)
    plt.show()
