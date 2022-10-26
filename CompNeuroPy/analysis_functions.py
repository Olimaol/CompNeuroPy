import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
from ANNarchy import raster_plot, dt
import warnings
from CompNeuroPy import system_functions as sf
from CompNeuroPy import extra_functions as ef
from scipy.interpolate import interp1d
from multiprocessing import Process


def my_raster_plot(spikes):
    """
    Returns two vectors representing for each recorded spike 1) the spike times and 2) the ranks of the neurons.

    The spike times are always in simulation steps (in contrast to default ANNarchy raster_plot)
    """
    t, n = raster_plot(spikes)
    t = t / dt()
    return t, n


def get_nanmean(a, axis=None, dtype=None):
    """
    np.nanmean without warnings
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ret = np.nanmean(a, axis=axis, dtype=dtype)
    return ret


def get_nanstd(a, axis=None, dtype=None):
    """
    np.nanstd without warnings
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ret = np.nanstd(a, axis=axis, dtype=dtype)
    return ret


def hanning_split_overlap(seq, size, overlap):
    """
    splits a sequence (array) in as many hanning-windowed subsequences as possible

    seq: full original sequence
    size: size of subsequences to be generated from the full sequence, subsequences are hanning-windowed
    overlap: overlap of subsequences

    iterates over complete size of subsequences
    --> = start positions
    from startpositions iterate to reach end of fullsequence with stepsize subsequencesize-overlap
    --> for each start position n sequence positions are obtained = indizes of seq for building the subsequences
        first row:  [0, 0+(subsequencesize-overlap), 0+2*(subsequencesize-overlap), ...]
        second row: [1, 1+(subsequencesize-overlap), 1+2*(subsequencesize-overlap), ...]
    --> similar to matrix, columns = indizes for building the subsequences
    """
    # seq[i::stepsize] == seq[range(i,seq.size,stepsize)]
    return np.array(
        [
            x * np.hanning(size)
            for x in zip(*[seq[i :: size - overlap] for i in range(size)])
        ]
    )


def get_population_power_spectrum(
    spikes,
    time_step,
    t_start=None,
    t_end=None,
    fft_size=None,
):
    """
    generates power spectrum of population spikes, returns list with [frequency_arr, power_spectrum]
    using the Welch methode from: Welch, P. (1967). The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. IEEE Transactions on audio and electroacoustics, 15(2), 70-73.

    The spike arrays are splitted into multiple arrays and then multiple FFTs are performed and the results are averaged.

    Size of splitted signals and the time step of the simulation determine the frequency resolution and the maximum frequency:
        maximum frequency [Hz] = 500 / time_step
        frequency resolution [Hz] = 1000 / (time_step * fftSize)

    Args:
        spikes: dicitonary
            ANNarchy spike dict of one population

        time_step: float
            time step of the simulation in ms

        t_start: float or int, optional, default = time of first spike
            start time of analyzed data in ms

        t_end: float or int, optional, default = time of last spike
            end time of analyzed data in ms

        fft_size: int, optional, default = maximum
            signal size for the FFT (size of splitted arrays)
            has to be a power of 2
    """
    populations_size = len(list(spikes.keys()))

    def ms_to_s(x):
        return x / 1000

    simulation_time = t_end - t_start  # in ms
    sampling_frequency = 1 / ms_to_s(time_step)  # in Hz

    ### get fft_size
    ### if None --> as large as possible
    if fft_size is None:
        pow = 1
        while (2 ** (pow + 1)) / sampling_frequency < ms_to_s(simulation_time):
            pow = pow + 1
        fft_size = 2**pow

    if ms_to_s(simulation_time) < (fft_size / sampling_frequency):
        ### catch a too large fft_size
        print(
            f"Too large fft_size {fft_size} for duration {simulation_time} ms. FFT_size has to be smaller than {int(ms_to_s(simulation_time)*sampling_frequency)}!"
        )
        return [np.zeros(int(fft_size / 2 - 2)), np.zeros(int(fft_size / 2 - 2))]
    elif (np.log2(fft_size) - int(np.log2(fft_size))) != 0:
        ### catch fft_size if its not power of 2
        print("FFT_size hast to be power of 2!")
        return [np.zeros(int(fft_size / 2 - 2)), np.zeros(int(fft_size / 2 - 2))]
    else:
        print(
            f"power sepctrum, min = {1000 / (time_step * fft_size)}, max = {500 / time_step}"
        )
        ### calculate frequency powers
        spectrum = np.zeros((populations_size, fft_size))
        for neuron in range(populations_size):
            ### sampling steps array
            spiketrain = np.zeros(
                int(np.round(ms_to_s(simulation_time) * sampling_frequency))
            )
            ### spike times as sampling steps
            idx = (
                np.round(
                    ms_to_s((np.array(spikes[neuron]) * time_step)) * sampling_frequency
                )
            ).astype(np.int32)
            ### cut the spikes before t_start and after t_end
            idx_start = ms_to_s(t_start) * sampling_frequency
            idx_end = ms_to_s(t_end) * sampling_frequency
            mask = ((idx > idx_start).astype(int) * (idx < idx_end).astype(int)).astype(
                bool
            )
            idx = (idx[mask] - idx_start).astype(np.int32)

            ### set spiketrain array to one if there was a spike at sampling step
            spiketrain[idx] = 1

            ### generate multiple overlapping sequences out of the spike trains
            spiketrain_sequences = hanning_split_overlap(
                spiketrain, fft_size, int(fft_size / 2)
            )

            ### generate power spectrum
            spectrum[neuron] = get_nanmean(
                np.abs(np.fft.fft(spiketrain_sequences)) ** 2, 0
            )

        ### mean spectrum over all neurons
        spectrum = get_nanmean(spectrum, 0)

        frequency_arr = np.fft.fftfreq(fft_size, 1.0 / sampling_frequency)

        return [frequency_arr[2 : int(fft_size / 2)], spectrum[2 : int(fft_size / 2)]]


def get_power_spektrum_from_time_array(
    arr,
    presimulationTime,
    simulationTime,
    simulation_dt,
    samplingfrequency=250,
    fftSize=1024,
):
    """
    generates power spectrum of time signal (returns [frequencies,power])
    using the Welch methode (Welch,1967)

    arr: time array, value for each timestep
    presimulationTime: simulation time which will not be analyzed
    simulationTime: analyzed simulation time

    samplingfrequency: to sample the arr, in Hz --> max frequency = samplingfrequency / 2
    fftSize: signal size for FFT, duration (in s) = fftSize / samplingfrequency --> frequency resolution = samplingfrequency / fftSize
    """

    if (simulationTime / 1000) < (fftSize / samplingfrequency):
        print("Simulation time has to be >=", fftSize / samplingfrequency, "s for FFT!")
        return [np.zeros(int(fftSize / 2 - 2)), np.zeros(int(fftSize / 2 - 2))]
    else:
        ### sampling steps array
        sampling_arr = arr[0 :: int((1 / samplingfrequency) * 1000 / simulation_dt)]

        ### generate multiple overlapping sequences
        sampling_arr_sequences = hanning_split_overlap(
            sampling_arr, fftSize, int(fftSize / 2)
        )

        ### generate power spectrum
        spektrum = get_nanmean(np.abs(np.fft.fft(sampling_arr_sequences)) ** 2, 0)

        frequenzen = np.fft.fftfreq(fftSize, 1.0 / samplingfrequency)

        return [frequenzen[2 : int(fftSize / 2)], spektrum[2 : int(fftSize / 2)]]


def get_pop_rate_old(
    spikes, duration, dt=1, t_start=0, t_smooth_ms=-1
):  # TODO: maybe makes errors with few/no spikes... check this TODO: automatically detect t_smooth does not work for strongly varying activity... implement something different
    """
    spikes: spikes dictionary from ANNarchy
    duration: duration of period after (optional) initial period (t_start) from which rate is calculated in ms
    dt: timestep of simulation
    t_start: starting simulation time, from which rates should be calculated
    t_smooth_ms: time window size for rate calculation in ms, optional, standard = -1 which means automatic window size

    returns smoothed population rate from period after rampUp period until duration
    """
    duration_init = duration
    temp_duration = duration + t_start
    t, n = raster_plot(spikes)
    if len(t) > 1:  # check if there are spikes in population at all
        if t_smooth_ms == -1:
            ISIs = []
            minTime = np.inf
            duration = 0
            for idx, key in enumerate(spikes.keys()):
                times = np.array(spikes[key]).astype(int)
                if len(times) > 1:  # check if there are spikes in neuron
                    ISIs += (np.diff(times) * dt).tolist()  # ms
                    minTime = np.min([minTime, times.min()])
                    duration = np.max([duration, times.max()])
                else:  # if there is only 1 spike set ISI to 10ms
                    ISIs += [10]
            t_smooth_ms = np.min(
                [(duration - minTime) / 2.0 * dt, np.mean(np.array(ISIs)) * 10 + 10]
            )

        rate = np.zeros((len(list(spikes.keys())), int(temp_duration / dt)))
        rate[:] = np.NaN
        binSize = int(t_smooth_ms / dt)
        bins = np.arange(0, int(temp_duration / dt) + binSize, binSize)
        binsCenters = bins[:-1] + binSize // 2
        timeshift_start = -binSize // 2
        timeshift_end = binSize // 2
        timeshift_step = np.max([binSize // 10, 1])
        for idx, key in enumerate(spikes.keys()):
            times = np.array(spikes[key]).astype(int)
            for timeshift in np.arange(
                timeshift_start, timeshift_end, timeshift_step
            ).astype(int):
                hist, edges = np.histogram(times, bins + timeshift)
                rate[
                    idx, np.clip(binsCenters + timeshift, 0, rate.shape[1] - 1)
                ] = hist / (t_smooth_ms / 1000.0)

        poprate = get_nanmean(rate, 0)
        timesteps = np.arange(0, int(temp_duration / dt), 1).astype(int)
        time = timesteps[np.logical_not(np.isnan(poprate))]
        poprate = poprate[np.logical_not(np.isnan(poprate))]
        poprate = np.interp(timesteps, time, poprate)

        ret = poprate[int(t_start / dt) :]
    else:
        ret = np.zeros(int(duration / dt))

    return [np.arange(t_start, t_start + duration_init, dt), ret]


def recursive_rate(
    spike_arr,
    t0,
    t1,
    duration_init,
    nr_neurons,
    nr_spikes,
    c_pre=np.inf,
    bin_nr_best_pre=1,
):
    """
    spike_arr: spike times in s
    t0, t1: start/end of period in s
    duration_init: full duration in s
    nr_neurons: number of neurons
    time_step: simulation_timestep in s
    """
    duration = t1 - t0

    threshold_nr_spikes = 5  # np.max([0.1 * nr_spikes, 10])
    threshold_duration = 5 / 1000.0

    nr_sub_divide = 10

    ### go up the nr_bins until threshold duration is reached with bin_size
    ### for each bin_size collect c
    ### collect min c --> c_min

    bin_nr_temp = 2
    bin_size = duration / bin_nr_temp
    c = c_pre + 1
    c_min = c_pre + 1
    bin_nr_best = 0
    while bin_size > threshold_duration:
        hist, edges = np.histogram(spike_arr, bin_nr_temp, range=(t0, t1))
        bin_size = duration / bin_nr_temp
        c = (2 * np.mean(hist) - np.var(hist)) / (duration) ** 2
        if c < c_min:
            c_min = c
            bin_nr_best = bin_nr_temp
        bin_nr_temp = bin_nr_temp + 1

    ### if c_min better than c_pre --> continue subsplitting /2
    ### if c_min worse than c_pre --> stop subsplitting and split the current spike array with the previous best bin size
    ### split the spike_array with bin_nr/nr_sub_divide (because the bin_nr is from the previous run, where the duration was nr_sub_divide times larger)
    ### if spike_arr.size < threshold_nr_spikes or duration < threshold_duration:
    if c_pre < c_min or (
        spike_arr.size < threshold_nr_spikes or duration < threshold_duration
    ):
        bin_nr_best_final = bin_nr_best_pre // nr_sub_divide
        if bin_nr_best_final < 2:
            rate = [spike_arr.size / (duration * nr_neurons)]
            time = [t0 + (t1 - t0) * (1 / 2.0)]
        else:
            hist, edges = np.histogram(spike_arr, bin_nr_best_final, range=(t0, t1))
            bin_size = duration / (bin_nr_best_final)
            rate = [hist[idx] / (bin_size * nr_neurons) for idx in range(hist.size)]
            time = [t0 + bin_size * idx + bin_size / 2 for idx in range(hist.size)]
        return [np.array([time[idx], rate[idx]]) for idx in range(len(rate))]
    else:
        ### continue sub dividing
        ###  get spike_sub_arr from histogramm
        hist, edges = np.histogram(spike_arr, nr_sub_divide, range=(t0, t1))
        spike_sub_arr_list = []
        t0_t1_list = []
        for idx in range(hist.size):
            if idx < hist.size - 1:
                mask = (spike_arr >= edges[idx]).astype(int) * (
                    spike_arr < edges[idx + 1]
                ).astype(int)
            else:
                mask = (spike_arr >= edges[idx]).astype(int) * (
                    spike_arr <= edges[idx + 1]
                ).astype(int)
            spike_sub_arr_list.append(spike_arr[mask.astype(bool)])
            t0_t1_list.append([edges[idx], edges[idx + 1]])

        return [
            recursive_rate(
                spike_sub_arr_list[idx],
                t0=t0_t1_list[idx][0],
                t1=t0_t1_list[idx][1],
                duration_init=duration_init,
                nr_neurons=nr_neurons,
                nr_spikes=nr_spikes,
                c_pre=c_min,
                bin_nr_best_pre=bin_nr_best,
            )
            for idx in range(hist.size)
        ]


def get_pop_rate(
    spikes, duration, dt=1, t_start=0, t_smooth_ms=-1
):  # TODO: maybe makes errors with few/no spikes... check this TODO: automatically detect t_smooth does not work for strongly varying activity... implement something different
    """
    spikes: spikes dictionary from ANNarchy
    duration: duration of period after (optional) initial period (t_start) from which rate is calculated in ms
    dt: timestep of simulation
    t_start: starting simulation time, from which rates should be calculated
    t_smooth_ms: time window size for rate calculation in ms, optional, standard = -1 which means automatic window size

    returns smoothed population rate from period after rampUp period until duration
    """

    if t_smooth_ms > 0:
        return get_pop_rate_old(
            spikes, duration, dt=dt, t_start=t_start, t_smooth_ms=t_smooth_ms
        )

    t, _ = raster_plot(spikes)
    if len(t) > 1:  # check if there are spikes in population at all
        ### concatenate all spike times and sort them
        spike_arr = dt * np.sort(
            np.concatenate(
                [np.array(spikes[neuron]).astype(int) for neuron in spikes.keys()]
            )
        )
        nr_neurons = len(list(spikes.keys()))
        nr_spikes = spike_arr.size

        ### use recursive_rate to get firing rate
        ### spike array is splitted in time bins
        ### time bins widths are automatically found
        rate = recursive_rate(
            spike_arr / 1000.0,
            t0=t_start / 1000.0,
            t1=(t_start + duration) / 1000.0,
            duration_init=duration / 1000.0,
            nr_neurons=nr_neurons,
            nr_spikes=nr_spikes,
        )
        ### returns lists of lists --> needs to be flattened
        rate = np.array(ef.flatten_list(rate))
        ### returns time values and rate values
        time_population_rate, population_rate = [rate[:, 0], rate[:, 1]]
        ### time_population_rate was returned in s --> transform it into ms
        time_population_rate = time_population_rate * 1000

        ### interpolate all 5 ms linear
        time_arr0 = np.arange(t_start, t_start + duration, 5)
        interpolate_func = interp1d(
            time_population_rate,
            population_rate,
            kind="linear",
            bounds_error=False,
            fill_value=(population_rate[0], population_rate[-1]),
        )
        population_rate = interpolate_func(time_arr0)

        ### interpolate all dt quadratic
        time_arr = np.arange(t_start, t_start + duration, dt)
        interpolate_func = interp1d(
            time_arr0,
            population_rate,
            kind="quadratic",
            bounds_error=False,
            fill_value=(population_rate[0], population_rate[-1]),
        )
        population_rate = interpolate_func(time_arr)

        ret = population_rate
    else:
        ret = np.zeros(int(duration / dt))

    return [np.arange(t_start, t_start + duration, dt), ret]


def plot_recordings(
    figname, recordings, recording_times, chunk, shape, plan, time_lim=[], dpi=300
):
    """
    Plots the recordings for the given recording_times specified in plan.

    Args:
        figname: str
            path + name of figure (e.g. "figures/my_figure.png")

        recordings: list
            a recordings list from CompNeuroPy obtained with the function
            get_recordings() from a Monitors object.

        recording_times: object
            recording_times object from CompNeuroPy obtained with the
            function get_recording_times() from a Monitors object.

        chunk: int
            which chunk of recordings should be used (the index of chunk)

        shape: tuple
            Defines the subplot arrangement e.g. (3,2) = 3 rows, 2 columns

        plan: list of strings
            Defines which recordings are plotted in which subplot and how.
            Entries of the list have the structure: "subplot_nr;model_component_name;variable_to_plot;format",
            e.g. "1,my_pop1;v;line".
            mode: defines how the data is plotted, available modes:
                - for spike data: raster, mean, hybrid
                - for other data: line, mean, matrix
                - only for projection data: matrix_mean

        time_lim: list, optional, default = time lims of chunk
            Defines the x-axis for all subplots. The list contains two
            numbers: start and end time in ms. The times have to be
            within the chunk.

        dpi: int, optional, default = 300
            The dpi of the saved figure

    """
    proc = Process(
        target=__plot_recordings,
        args=(figname, recordings, recording_times, chunk, shape, plan, time_lim, dpi),
    )
    proc.start()
    proc.join()


def __plot_recordings(
    figname, recordings, recording_times, chunk, shape, plan, time_lim=[], dpi=300
):
    """
    Plots the recordings for the given recording_times specified in plan.

    Args:
        figname: str
            path + name of figure (e.g. "figures/my_figure.png")

        recordings: list
            a recordings list from CompNeuroPy obtained with the function
            get_recordings() from a Monitors object.

        recording_times: object
            recording_times object from CompNeuroPy obtained with the
            function get_recording_times() from a Monitors object.

        chunk: int
            which chunk of recordings should be used (the index of chunk)

        shape: tuple
            Defines the subplot arrangement e.g. (3,2) = 3 rows, 2 columns

        plan: list of strings
            Defines which recordings are plotted in which subplot and how.
            Entries of the list have the structure: "subplot_nr;model_component_name;variable_to_plot;format",
            e.g. "1,my_pop1;v;line".
            mode: defines how the data is plotted, available modes:
                - for spike data: raster, mean, hybrid
                - for other data: line, mean, matrix
                - only for projection data: matrix_mean

        time_lim: list, optional, default = time lims of chunk
            Defines the x-axis for all subplots. The list contains two
            numbers: start and end time in ms. The times have to be
            within the chunk.

        dpi: int, optional, default = 300
            The dpi of the saved figure

    """
    print(f"generate fig {figname}", end="... ", flush=True)
    recordings = recordings[chunk]
    if len(time_lim) == 0:
        time_lim = recording_times.time_lims(chunk=chunk)
    start_time = time_lim[0]
    end_time = time_lim[1]
    compartment_list = []
    for plan_str in plan:
        compartment = plan_str.split(";")[1]
        if not (compartment in compartment_list):
            compartment_list.append(compartment)

    ### get idx_lim for all compartments, in parallel check that there are no pauses
    time_arr_dict = {}
    time_step = recordings["dt"]
    for compartment in compartment_list:
        actual_period = recordings[f"{compartment};period"]

        time_arr_part = []

        ### loop over periods
        nr_periods = recording_times.__get_nr_periods__(
            chunk=chunk, compartment=compartment
        )
        for period in range(nr_periods):
            ### get the time_lim and idx_lim of the period
            time_lims = recording_times.time_lims(
                chunk=chunk, compartment=compartment, period=period
            )
            time_arr_part.append(
                np.arange(time_lims[0], time_lims[1] + actual_period, actual_period)
            )

        time_arr_dict[compartment] = np.concatenate(time_arr_part)

    plt.figure(figsize=([6.4 * shape[1], 4.8 * shape[0]]))
    for subplot in plan:
        try:
            nr, part, variable, mode = subplot.split(";")
            nr = int(nr)
            style = ""
        except:
            try:
                nr, part, variable, mode, style = subplot.split(";")
                nr = int(nr)
            except:
                print(
                    '\nERROR plot_recordings: for each subplot give plan-string as: "nr;part;variable;mode" or "nr;part;variable;mode;style" if style is available!\nWrong string: '
                    + subplot
                    + "\n"
                )
                quit()
        try:
            data = recordings[f"{part};{variable}"]
        except:
            print(
                "\nWARNING plot_recordings: data",
                ";".join([part, variable]),
                "not in recordings\n",
            )
            plt.subplot(shape[0], shape[1], nr)
            plt.text(
                0.5,
                0.5,
                " ".join([part, variable]) + " not available",
                va="center",
                ha="center",
            )
            continue

        plt.subplot(shape[0], shape[1], nr)
        if variable == "spike" and (
            mode == "raster" or mode == "single"
        ):  # "single" only for down compatibility
            t, n = my_raster_plot(data)
            t = t * time_step  # convert time steps into ms
            mask = ((t >= start_time).astype(int) * (t <= end_time).astype(int)).astype(
                bool
            )
            if mask.size == 0:
                plt.title("Spikes " + part)
                print(
                    "\nWARNING plot_recordings: data",
                    ";".join([part, variable]),
                    "does not contain any spikes in the given time interval.\n",
                )
                plt.text(
                    0.5,
                    0.5,
                    " ".join([part, variable]) + " does not contain any spikes.",
                    va="center",
                    ha="center",
                )
            else:
                if np.unique(n).size == 1:
                    marker, size = ["|", 3000]
                else:
                    marker, size = [".", 3]
                if style != "":
                    color = style
                else:
                    color = "k"

                plt.scatter(
                    t[mask], n[mask], color=color, marker=marker, s=size, linewidth=0.1
                )
                plt.xlim(start_time, end_time)
                plt.xlabel("time [ms]")
                plt.ylabel("# neurons")
                plt.title("Spikes " + part)
        elif variable == "spike" and mode == "mean":
            time_arr, firing_rate = get_pop_rate(
                data,
                end_time - start_time,
                dt=time_step,
                t_start=start_time,
            )
            plt.plot(time_arr, firing_rate, color="k")
            plt.xlim(start_time, end_time)
            plt.xlabel("time [ms]")
            plt.ylabel("Mean firing rate [Hz]")
            plt.title("Mean firing rate " + part)
        elif variable == "spike" and mode == "hybrid":
            t, n = my_raster_plot(data)
            t = t * time_step  # convert time steps into ms
            mask = ((t >= start_time).astype(int) * (t <= end_time).astype(int)).astype(
                bool
            )
            if mask.size == 0:
                plt.title("Spikes " + part)
                print(
                    "\nWARNING plot_recordings: data",
                    ";".join([part, variable]),
                    "does not contain any spikes in the given time interval.\n",
                )
                plt.text(
                    0.5,
                    0.5,
                    " ".join([part, variable]) + " does not contain any spikes.",
                    va="center",
                    ha="center",
                )
            else:

                plt.plot(
                    t[mask], n[mask], "k.", markersize=np.sqrt(3), markeredgewidth=0.1
                )
                plt.ylabel("# neurons")
                ax = plt.gca().twinx()
                time_arr, firing_rate = get_pop_rate(
                    data,
                    end_time - start_time,
                    dt=time_step,
                    t_start=start_time,
                    t_smooth_ms=-1,
                )
                ax.plot(
                    time_arr,
                    firing_rate,
                    color="r",
                )
                plt.ylabel("Mean firing rate [Hz]", color="r")
                ax.tick_params(axis="y", colors="r")
                plt.xlim(start_time, end_time)
                plt.xlabel("time [ms]")
                plt.title("Activity " + part)
        elif variable != "spike" and mode == "line":
            if len(data.shape) == 1:
                plt.plot(time_arr_dict[part], data, color="k")
                plt.title(f"Variable {variable} of {part} (1)")
            elif len(data.shape) == 2 and isinstance(data[0, 0], list) is not True:
                ### population: data[time,neurons]
                for neuron in range(data.shape[1]):
                    plt.plot(
                        time_arr_dict[part],
                        data[:, neuron],
                        color="k",
                    )
                plt.title(f"Variable {variable} of {part} ({data.shape[1]})")
            elif len(data.shape) == 3 or (
                len(data.shape) == 2 and isinstance(data[0, 0], list) is True
            ):
                if len(data.shape) == 3:
                    ### projection data: data[time, postneurons, preneurons]
                    for post_neuron in range(data.shape[1]):
                        for pre_neuron in range(data.shape[2]):
                            plt.plot(
                                time_arr_dict[part],
                                data[:, post_neuron, pre_neuron],
                                color="k",
                            )
                else:
                    ### data[time, postneurons][preneurons] (with different number of preneurons)
                    for post_neuron in range(data.shape[1]):
                        for pre_neuron in range(len(data[0, post_neuron])):
                            plt.plot(
                                time_arr_dict[part],
                                np.array(
                                    [
                                        data[t, post_neuron][pre_neuron]
                                        for t in range(data.shape[0])
                                    ]
                                ),
                                color="k",
                            )

                plt.title(f"Variable {variable} of {part} ({data.shape[1]})")
            else:
                print(
                    "\nERROR plot_recordings: shape not accepted,",
                    ";".join([part, variable]),
                    "\n",
                )
            plt.xlim(start_time, end_time)
            plt.xlabel("time [ms]")
        elif variable != "spike" and mode == "mean":
            if len(data.shape) == 1:
                plt.plot(time_arr_dict[part], data, color="k")
                plt.title(f"Variable {variable} of {part} (1)")
            elif len(data.shape) == 2 and isinstance(data[0, 0], list) is not True:
                ### population: data[time,neurons]
                nr_neurons = data.shape[1]
                data = np.mean(data, 1)
                plt.plot(
                    time_arr_dict[part],
                    data[:],
                    color="k",
                )
                plt.title(f"Variable {variable} of {part} ({nr_neurons}, mean)")
            elif len(data.shape) == 3 or (
                len(data.shape) == 2 and isinstance(data[0, 0], list) is True
            ):
                if len(data.shape) == 3:
                    ### projection data: data[time, postneurons, preneurons]
                    for post_neuron in range(data.shape[1]):
                        plt.plot(
                            time_arr_dict[part],
                            np.mean(data[:, post_neuron, :], 1),
                            color="k",
                        )
                else:
                    ### data[time, postneurons][preneurons] (with different number of preneurons)
                    for post_neuron in range(data.shape[1]):
                        avg_data = []
                        for pre_neuron in range(len(data[0, post_neuron])):
                            avg_data.append(
                                np.array(
                                    [
                                        data[t, post_neuron][pre_neuron]
                                        for t in range(data.shape[0])
                                    ]
                                )
                            )
                        plt.plot(
                            time_arr_dict[part],
                            np.mean(avg_data, 0),
                            color="k",
                        )

                plt.title(
                    f"Variable {variable} of {part}, mean for {data.shape[1]} post neurons"
                )
            else:
                print(
                    "\nERROR plot_recordings: shape not accepted,",
                    ";".join([part, variable]),
                    "\n",
                )
            plt.xlim(start_time, end_time)
            plt.xlabel("time [ms]")

        elif variable != "spike" and mode == "matrix_mean":

            if len(data.shape) == 3 or (
                len(data.shape) == 2 and isinstance(data[0, 0], list) is True
            ):
                if len(data.shape) == 3:
                    ### average over pre neurons --> get 2D array [time, postneuron]
                    data_avg = np.mean(data, 2)

                    ### after cerating 2D array --> same procedure as for populations
                    ### get the times and data between time_lims
                    mask = (
                        (time_arr_dict[part] >= start_time).astype(int)
                        * (time_arr_dict[part] <= end_time).astype(int)
                    ).astype(bool)
                    time_arr = time_arr_dict[part][mask]
                    data_arr = data_avg[mask, :]

                    ### check with the actual_period and the times array if there is data missing
                    ###     from time_lims and actual period opne should get all times at which data points should be
                    actual_period = recordings[f"{part};period"]
                    actual_start_time = (
                        np.ceil(start_time / actual_period) * actual_period
                    )
                    actual_end_time = (
                        np.ceil(end_time / actual_period - 1) * actual_period
                    )
                    soll_times = np.arange(
                        actual_start_time,
                        actual_end_time + actual_period,
                        actual_period,
                    )

                    ### check if there are time points, where data is missing
                    plot_data_arr = np.empty((soll_times.size, data_arr.shape[1]))
                    plot_data_arr[:] = None
                    for time_point_idx, time_point in enumerate(soll_times):
                        if time_point in time_arr:
                            ### data at time point is available --> use data
                            idx_available_data = time_arr == time_point
                            plot_data_arr[time_point_idx, :] = data_arr[
                                idx_available_data, :
                            ]
                        ### if data is not available it stays none

                    vmin = np.nanmin(plot_data_arr)
                    vmax = np.nanmax(plot_data_arr)

                    masked_array = np.ma.array(
                        plot_data_arr.T, mask=np.isnan(plot_data_arr.T)
                    )
                    cmap = matplotlib.cm.viridis
                    cmap.set_bad("red", 1.0)

                    plt.title(
                        f"Variable {variable} of {part} ({data.shape[1]}) [{ef.sci(vmin)}, {ef.sci(vmax)}]"
                    )

                    plt.gca().imshow(
                        masked_array,
                        aspect="auto",
                        vmin=vmin,
                        vmax=vmax,
                        extent=[
                            np.min(soll_times) - 0.5,
                            np.max(soll_times) - 0.5,
                            data.shape[1] - 0.5,
                            -0.5,
                        ],
                        cmap=cmap,
                        interpolation="none",
                    )
                    if data.shape[1] == 1:
                        plt.yticks([0])
                    else:
                        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xlabel("time [ms]")

                else:
                    ### data[time, postneurons][preneurons] (with different number of preneurons)
                    ### average over pre neurons --> get 2D array [time, postneuron]
                    data_avg = np.empty((data.shape[0], data.shape[1]))
                    for post_neuron in range(data.shape[1]):
                        avg_post = []
                        for pre_neuron in range(len(data[0, post_neuron])):
                            avg_post.append(
                                np.array(
                                    [
                                        data[t, post_neuron][pre_neuron]
                                        for t in range(data.shape[0])
                                    ]
                                )
                            )
                        data_avg[:, post_neuron] = np.mean(avg_post, 0)

                    ### after cerating 2D array --> same procedure as for populations
                    ### get the times and data between time_lims
                    mask = (
                        (time_arr_dict[part] >= start_time).astype(int)
                        * (time_arr_dict[part] <= end_time).astype(int)
                    ).astype(bool)
                    time_arr = time_arr_dict[part][mask]
                    data_arr = data_avg[mask, :]

                    ### check with the actual_period and the times array if there is data missing
                    ###     from time_lims and actual period opne should get all times at which data points should be
                    actual_period = recordings[f"{part};period"]
                    actual_start_time = (
                        np.ceil(start_time / actual_period) * actual_period
                    )
                    actual_end_time = (
                        np.ceil(end_time / actual_period - 1) * actual_period
                    )
                    soll_times = np.arange(
                        actual_start_time,
                        actual_end_time + actual_period,
                        actual_period,
                    )

                    ### check if there are time points, where data is missing
                    plot_data_arr = np.empty((soll_times.size, data_arr.shape[1]))
                    plot_data_arr[:] = None
                    for time_point_idx, time_point in enumerate(soll_times):
                        if time_point in time_arr:
                            ### data at time point is available --> use data
                            idx_available_data = time_arr == time_point
                            plot_data_arr[time_point_idx, :] = data_arr[
                                idx_available_data, :
                            ]
                        ### if data is not available it stays none

                    vmin = np.nanmin(plot_data_arr)
                    vmax = np.nanmax(plot_data_arr)

                    masked_array = np.ma.array(
                        plot_data_arr.T, mask=np.isnan(plot_data_arr.T)
                    )
                    cmap = matplotlib.cm.viridis
                    cmap.set_bad("red", 1.0)

                    plt.title(
                        f"Variable {variable} of {part} ({data.shape[1]}) [{ef.sci(vmin)}, {ef.sci(vmax)}]"
                    )

                    plt.gca().imshow(
                        masked_array,
                        aspect="auto",
                        vmin=vmin,
                        vmax=vmax,
                        extent=[
                            np.min(soll_times) - 0.5,
                            np.max(soll_times) - 0.5,
                            data.shape[1] - 0.5,
                            -0.5,
                        ],
                        cmap=cmap,
                        interpolation="none",
                    )
                    if data.shape[1] == 1:
                        plt.yticks([0])
                    else:
                        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xlabel("time [ms]")

                plt.title(
                    f"Variable {variable} of {part}, mean for {data.shape[1]} post neurons [{ef.sci(vmin)}, {ef.sci(vmax)}]"
                )
            else:
                print(
                    "\nERROR plot_recordings: shape not accepted,",
                    ";".join([part, variable]),
                    "\n",
                )
            plt.xlim(start_time, end_time)
            plt.xlabel("time [ms]")

        elif variable != "spike" and mode == "matrix":
            # data[start_step:end_step,neuron]
            if len(data.shape) == 2 and isinstance(data[0, 0], list) is not True:
                ### data from population [times,neurons]
                ### get the times and data between time_lims
                mask = (
                    (time_arr_dict[part] >= start_time).astype(int)
                    * (time_arr_dict[part] <= end_time).astype(int)
                ).astype(bool)
                time_arr = time_arr_dict[part][mask]
                data_arr = data[mask, :]

                ### check with the actual_period and the times array if there is data missing
                ###     from time_lims and actual period opne should get all times at which data points should be
                actual_period = recordings[f"{part};period"]
                actual_start_time = np.ceil(start_time / actual_period) * actual_period
                actual_end_time = np.ceil(end_time / actual_period - 1) * actual_period
                soll_times = np.arange(
                    actual_start_time, actual_end_time + actual_period, actual_period
                )

                ### check if there are time points, where data is missing
                plot_data_arr = np.empty((soll_times.size, data_arr.shape[1]))
                plot_data_arr[:] = None
                for time_point_idx, time_point in enumerate(soll_times):
                    if time_point in time_arr:
                        ### data at time point is available --> use data
                        idx_available_data = time_arr == time_point
                        plot_data_arr[time_point_idx, :] = data_arr[
                            idx_available_data, :
                        ]
                    ### if data is not available it stays none

                vmin = np.nanmin(plot_data_arr)
                vmax = np.nanmax(plot_data_arr)

                masked_array = np.ma.array(
                    plot_data_arr.T, mask=np.isnan(plot_data_arr.T)
                )
                cmap = matplotlib.cm.viridis
                cmap.set_bad("red", 1.0)

                plt.title(
                    f"Variable {part} {variable} ({data.shape[1]}) [{ef.sci(vmin)}, {ef.sci(vmax)}]"
                )

                plt.gca().imshow(
                    masked_array,
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                    extent=[
                        np.min(soll_times) - 0.5,
                        np.max(soll_times) - 0.5,
                        data.shape[1] - 0.5,
                        -0.5,
                    ],
                    cmap=cmap,
                    interpolation="none",
                )
                if data.shape[1] == 1:
                    plt.yticks([0])
                else:
                    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
                plt.xlabel("time [ms]")
            elif len(data.shape) == 3 or (
                len(data.shape) == 2 and isinstance(data[0, 0], list) is True
            ):
                ### data from projection
                if len(data.shape) == 3:
                    ### projection data: data[time, postneurons, preneurons]
                    ### create a 2D array from the 3D array
                    data_resh = data.reshape(
                        (data.shape[0], int(data.shape[1] * data.shape[2]))
                    )
                    data_split = np.split(data_resh, data.shape[1], axis=1)
                    ### separate the post_neurons by nan vectors
                    data_with_none = np.concatenate(
                        [
                            np.concatenate(
                                [
                                    data_split[idx],
                                    np.zeros((data.shape[0], 1)) * np.nan,
                                ],
                                axis=1,
                            )
                            for idx in range(len(data_split))
                        ],
                        axis=1,
                    )[:, :-1]

                    ### after cerating 2D array --> same procedure as for populations
                    ### get the times and data between time_lims
                    mask = (
                        (time_arr_dict[part] >= start_time).astype(int)
                        * (time_arr_dict[part] <= end_time).astype(int)
                    ).astype(bool)
                    time_arr = time_arr_dict[part][mask]
                    data_arr = data_with_none[mask, :]

                    ### check with the actual_period and the times array if there is data missing
                    ###     from time_lims and actual period opne should get all times at which data points should be
                    actual_period = recordings[f"{part};period"]
                    actual_start_time = (
                        np.ceil(start_time / actual_period) * actual_period
                    )
                    actual_end_time = (
                        np.ceil(end_time / actual_period - 1) * actual_period
                    )
                    soll_times = np.arange(
                        actual_start_time,
                        actual_end_time + actual_period,
                        actual_period,
                    )

                    ### check if there are time points, where data is missing
                    plot_data_arr = np.empty((soll_times.size, data_arr.shape[1]))
                    plot_data_arr[:] = None
                    for time_point_idx, time_point in enumerate(soll_times):
                        if time_point in time_arr:
                            ### data at time point is available --> use data
                            idx_available_data = time_arr == time_point
                            plot_data_arr[time_point_idx, :] = data_arr[
                                idx_available_data, :
                            ]
                        ### if data is not available it stays none

                    vmin = np.nanmin(plot_data_arr)
                    vmax = np.nanmax(plot_data_arr)

                    masked_array = np.ma.array(
                        plot_data_arr.T, mask=np.isnan(plot_data_arr.T)
                    )
                    cmap = matplotlib.cm.viridis
                    cmap.set_bad("red", 1.0)

                    plt.title(
                        f"Variable {variable} of {part} ({data.shape[1]}) [{ef.sci(vmin)}, {ef.sci(vmax)}]"
                    )

                    plt.gca().imshow(
                        masked_array,
                        aspect="auto",
                        vmin=vmin,
                        vmax=vmax,
                        extent=[
                            np.min(soll_times) - 0.5,
                            np.max(soll_times) - 0.5,
                            data.shape[1] - 0.5,
                            -0.5,
                        ],
                        cmap=cmap,
                        interpolation="none",
                    )
                    if data.shape[1] == 1:
                        plt.yticks([0])
                    else:
                        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xlabel("time [ms]")
                else:
                    ### data[time, postneurons][preneurons] (with different number of preneurons)
                    pass

            else:
                print(
                    "\nERROR plot_recordings: shape not accepted,",
                    ";".join([part, variable]),
                    "\n",
                )
                quit()
        else:
            print(
                "\nERROR plot_recordings: mode",
                mode,
                "not available for variable",
                variable,
                "\n",
            )
            quit()
    plt.tight_layout()

    ### save plot
    figname_parts = figname.split("/")
    if len(figname_parts) > 1:
        save_dir = "/".join(figname_parts[:-1])
        sf.create_dir(save_dir)
    plt.savefig(figname, dpi=dpi)
    plt.close()
    print("Done")


def get_number_of_zero_decimals(nr):
    decimals = 0
    if nr != 0:
        while abs(nr) < 1:
            nr = nr * 10
            decimals = decimals + 1

    return decimals


def get_number_of_decimals(nr):
    nr = nr - int(nr)
    decimals = 0
    if nr != 0:
        while abs(nr) < 1:
            nr = nr * 10
            decimals = decimals + 1

    return decimals


def sample_data_with_timestep(time_arr, data_arr, timestep):
    """
    time_arr: times of data_arr in ms
    data_arr: array with values
    timestep: timestep in ms for sampling
    """
    interpolate_func = interp1d(
        time_arr, data_arr, bounds_error=False, fill_value="extrapolate"
    )
    min_time = round(
        round(time_arr[0] / timestep, 0) * timestep,
        get_number_of_zero_decimals(timestep),
    )
    max_time = round(
        round(time_arr[-1] / timestep, 0) * timestep,
        get_number_of_zero_decimals(timestep),
    )
    new_time_arr = np.arange(min_time, max_time + timestep, timestep)
    new_data_arr = interpolate_func(new_time_arr)

    return [new_time_arr, new_data_arr]


def time_data_add_nan(time_arr, data_arr):
    """
    if there are gaps in time_arr --> fill them with respecitve time values
    fill the corresponding data_arr values with nan
    time_arr = 1D array
    data_arr = nD array
    both have same first dimension size!
    """

    time_diff_arr = np.diff(time_arr)
    time_diff_unique_arr = np.unique(time_diff_arr, return_counts=True)
    period_time = time_diff_unique_arr[0][np.argmax(time_diff_unique_arr[1])]

    time = time_arr[0]
    idx = 0
    while time < time_arr[-1]:
        if time != time_arr[idx]:
            ### time_arr of idx did not increase by period_time
            time_arr = np.insert(time_arr, idx, time_arr[idx - 1] + period_time)
            data_arr = np.insert(
                data_arr, idx, np.nan * np.zeros(data_arr.shape[1:]), axis=0
            )
        idx = idx + 1
        time = time + period_time

    return [time_arr, data_arr]
