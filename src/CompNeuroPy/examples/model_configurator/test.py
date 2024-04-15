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

setup(dt=0.1)

neuron_izh = Neuron(
    parameters="""
        C = 100.0
        k = 0.7
        v_r = -60.0
        v_t = -40.0
        a = 0.03
        b = -2.0
        c = -50.0
        d = 100.0
        v_peak = 35.0
        I_app = 0.0
        E_ampa = 0.0
    """,
    equations="""
        ### synaptic current
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
pop_post = Population(100, neuron=neuron_izh, name="post")

proj = Projection(pre=pop_pre, post=pop_post, target="ampa", name="proj")
proj.connect_fixed_probability(weights=10, probability=0.1)


monitors = CompNeuroMonitors(
    mon_dict={"pre": ["spike"], "post": ["v", "spike", "I_ampa"]}
)

compile()

monitors.start()

simulate(100.0)
pop_pre.rates = 200.0
simulate(100.0)

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
        "format": ["hybrid", "hybrid", "line"],
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


BIN_SIZE_STEPS = int(round(5 / dt()))


class DistPreSpikes:

    def __init__(self, spikes_dict, time_lims_steps):

        self.spikes_binned = get_binned_spikes(
            spikes_dict=spikes_dict,
            time_lims_steps=time_lims_steps,
            BIN_SIZE_STEPS=BIN_SIZE_STEPS,
        )
        self._pdf_dict = {}

    @property
    def pdf(self, time_bin=0):
        return self._pdf_dict.get(time_bin, self._get_pdf(time_bin))

    def _get_pdf(self, time_bin=0):
        # Create the KDE object
        kde = KernelDensity()

        # Fit the data to the KDE
        kde.fit(self.spikes_binned[time_bin])

        # Create points for which to estimate the PDF
        x = np.linspace(0, 10, 100).reshape(-1, 1)

        # Estimate the PDF for these points
        log_density = kde.score_samples(x)
        pdf = np.exp(log_density)

        # spikes can only be positive
        pdf[x[:, 0] >= 0] = pdf[x[:, 0] >= 0] * (
            1 + np.sum(pdf[x[:, 0] < 0]) / np.sum(pdf[x[:, 0] >= 0])
        )
        pdf[x[:, 0] < 0] = 0

        # store the pdf
        self._pdf_dict[time_bin] = (x, pdf)

        return x, pdf


class DistPostSpikesAndVoltage:

    def __init__(self, spikes_dict, time_lims_steps, voltage_array):

        self.spikes_post_binned = get_binned_spikes(
            spikes_dict=spikes_dict,
            time_lims_steps=time_lims_steps,
            BIN_SIZE_STEPS=BIN_SIZE_STEPS,
        )
        self.voltage_binned = get_binned_variable(
            voltage_array, BIN_SIZE_STEPS=BIN_SIZE_STEPS
        )
        self._pdf_dict = {}

    @property
    def pdf(self, time_bin=0):
        return self._pdf_dict.get(time_bin, self._get_pdf(time_bin))

    def _get_pdf(self, time_bin=0):
        # Create the KDE object
        kde = KernelDensity()

        # Fit the data to the KDE
        kde.fit(self.spikes_binned[time_bin])

        # Create points for which to estimate the PDF
        x = np.linspace(0, 10, 100).reshape(-1, 1)

        # Estimate the PDF for these points
        log_density = kde.score_samples(x)
        pdf = np.exp(log_density)

        # spikes can only be positive
        pdf[x[:, 0] >= 0] = pdf[x[:, 0] >= 0] * (
            1 + np.sum(pdf[x[:, 0] < 0]) / np.sum(pdf[x[:, 0] >= 0])
        )
        pdf[x[:, 0] < 0] = 0

        # store the pdf
        self._pdf_dict[time_bin] = (x, pdf)

        return x, pdf


dist_pre_spikes = DistPreSpikes(
    spikes_dict=recordings[0]["pre;spike"], time_lims_steps=(0, 2000)
)

DistPostSpikesAndVoltage(
    spikes_dict=recordings[0]["post;spike"],
    time_lims_steps=(0, 2000),
    voltage_array=recordings[0]["post;v"],
)

# Plot the results
x, pdf_spikes_pre = dist_pre_spikes.pdf(time_bin=0)
plt.fill_between(x[:, 0], pdf_spikes_pre, alpha=0.5)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Kernel Density Estimation")
plt.show()


def get_pdf_post():
    # Create the KDE object
    kde = KernelDensity()

    time_bin = -1
    # combine the spike and voltage data
    data = np.concatenate(
        [spikes_post_binned[time_bin], voltage_binned[time_bin]], axis=1
    )

    print(data)
    # Fit the data to the KDE
    kde.fit(data)
    # Create points for which to estimate the PDF, here, 2D for spike and voltage
    # spike between 0 and 10, voltage between -100 and 100
    x = np.mgrid[0:10:100j, -100:100:100j].reshape(2, -1).T

    print(x)
    print(x.shape)

    # Estimate the PDF for these points
    log_density = kde.score_samples(x)
    pdf = np.exp(log_density)

    ### spikes can only be positive
    pdf[x[:, 0] >= 0] = pdf[x[:, 0] >= 0] * (
        1 + np.sum(pdf[x[:, 0] < 0]) / np.sum(pdf[x[:, 0] >= 0])
    )
    pdf[x[:, 0] < 0] = 0
    # print(x[18:22])
    # print(pdf[18:22])
    # print(x[118:122])
    # print(pdf[118:122])
    pdf = pdf.reshape(100, 100)

    print(np.sum(pdf) * (20 / 100) * (200 / 100))

    return x, pdf


# plot the results
x, pdf_post = get_pdf_post()
plt.imshow(pdf_post, aspect="auto", extent=(-100, 100, 0, 10), origin="lower")
plt.xlabel("voltages")
plt.ylabel("spikes")
plt.title("Kernel Density Estimation")
plt.show()


# ### discrete because spikes are integers
# x = np.arange(-0.5, 20.5, 1.0)
# kde_discrete = np.histogram(counts_matrix[0], x, density=True)[0]
# kde_discrete2 = np.histogram(counts_matrix[-1], x, density=True)[0]
# plt.subplot(211)
# plt.bar(np.arange(kde_discrete.size).astype(int), kde_discrete, alpha=0.5)
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.title("Kernel Density Estimation")
# plt.subplot(212)
# plt.bar(np.arange(kde_discrete.size).astype(int), kde_discrete2, alpha=0.5)
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.show()
