import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
from ANNarchy import raster_plot, dt, inter_spike_interval, coefficient_of_variation
import warnings
from CompNeuroPy import system_functions as sf
from CompNeuroPy import extra_functions as ef
from CompNeuroPy.monitors import RecordingTimes
from scipy.interpolate import interp1d
from multiprocessing import Process
from typingchecker import check_types


def my_raster_plot(spikes: dict):
    """
    Returns two vectors representing for each recorded spike 1) the spike times and 2)
    the ranks of the neurons. The spike times are always in simulation steps (in
    contrast to default ANNarchy raster_plot).

    Args:
        spikes (dict):
            ANNarchy spike dict of one population

    Returns:
        t (array):
            spike times in simulation steps
        n (array):
            ranks of the neurons
    """
    t, n = raster_plot(spikes)
    np.zeros(10)
    t = np.round(t / dt(), 0).astype(int)
    return t, n


def get_nanmean(a, axis=None, dtype=None):
    """
    Same as np.nanmean but without printing warnings.

    Args:
        a (array_like):
            Array containing numbers whose mean is desired. If `a` is not an
            array, a conversion is attempted.
        axis (None or int or tuple of ints, optional):
            Axis or axes along which the means are computed. The default is to
            compute the mean of the flattened array.

            .. numpy versionadded:: 1.7.0

            If this is a tuple of ints, a mean is performed over multiple axes,
            instead of a single axis or all the axes as before.
        dtype (data-type, optional):
            Type to use in computing the mean.  For integer inputs, the default
            is `float64`; for floating point inputs, it is the same as the
            input dtype.

    Returns:
        m (ndarray, see dtype parameter above):
            If `out=None`, returns a new array containing the mean values,
            otherwise a reference to the output array is returned.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ret = np.nanmean(a, axis=axis, dtype=dtype)
    return ret


def get_nanstd(a, axis=None, dtype=None):
    """
    Same as np.nanstd but without printing warnings.

    Args:
        a (array_like):
            Calculate the standard deviation of these values.
        axis (None or int or tuple of ints, optional):
            Axis or axes along which the standard deviation is computed. The
            default is to compute the standard deviation of the flattened array.

            .. numpy versionadded:: 1.7.0

            If this is a tuple of ints, a standard deviation is performed over
            multiple axes, instead of a single axis or all the axes as before.
        dtype (dtype, optional):
            Type to use in computing the standard deviation. For arrays of
            integer type the default is float64, for arrays of float types it is
            the same as the array type.

    Returns:
        standard_deviation (ndarray, see dtype parameter above):
            If `out` is None, return a new array containing the standard deviation,
            otherwise return a reference to the output array.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ret = np.nanstd(a, axis=axis, dtype=dtype)
    return ret


def _hanning_split_overlap(seq, size, overlap):
    """
    Splits a sequence (array) in as many hanning-windowed subsequences as possible.

    Iterates over complete size of subsequences.
    --> = start positions
    From startpositions iterate to reach end of fullsequence with stepsize
    subsequencesize-overlap.
    --> for each start position n sequence positions are obtained = indizes of seq
    for building the subsequences
        first row:  [0, 0+(subsequencesize-overlap), 0+2*(subsequencesize-overlap), ...]
        second row: [1, 1+(subsequencesize-overlap), 1+2*(subsequencesize-overlap), ...]
    --> similar to matrix, columns = indizes for building the subsequences

    Args:
        seq (array):
            full original sequence
        size (int):
            size of subsequences to be generated from the full sequence, subsequences
            are hanning-windowed
        overlap (int):
            overlap of subsequences

    Returns:
        array (array):
            array of subsequences

    """
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
    Generates power spectrum of population spikes, returns frequency_arr and
    power_spectrum_arr. Using the Welch methode from: Welch, P. (1967). The use of fast
    Fourier transform for the estimation of power spectra: a method based on time
    averaging over short, modified periodograms. IEEE Transactions on audio and
    electroacoustics, 15(2), 70-73.

    The spike arrays are splitted into multiple arrays and then multiple FFTs are
    performed and the results are averaged.

    Size of splitted signals and the time step of the simulation determine the frequency
    resolution and the maximum frequency:
        maximum frequency [Hz] = 500 / time_step
        frequency resolution [Hz] = 1000 / (time_step * fftSize)

    Args:
        spikes (dicitonary):
            ANNarchy spike dict of one population
        time_step (float):
            time step of the simulation in ms
        t_start (float or int, optional):
            start time of analyzed data in ms. Default: time of first spike
        t_end (float or int, optional):
            end time of analyzed data in ms. Default: time of last spike
        fft_size (int, optional):
            signal size for the FFT (size of splitted arrays)
            has to be a power of 2. Default: maximum

    Returns:
        frequency_arr (array):
            array with frequencies
        spectrum (array):
            array with power spectrum
    """

    def ms_to_s(x):
        return x / 1000

    ### get population_size / sampling_frequency
    populations_size = len(list(spikes.keys()))
    sampling_frequency = 1 / ms_to_s(time_step)  # in Hz

    ### check if there are spikes in data
    t, _ = my_raster_plot(spikes)
    if len(t) < 2:
        ### there are no 2 spikes
        print("WARNING: get_population_power_spectrum: <2 spikes!")
        ### --> return None or zeros
        if fft_size == None:
            print(
                "ERROR: get_population_power_spectrum: <2 spikes and no fft_size given!"
            )
            quit()
        else:
            frequency_arr = np.fft.fftfreq(fft_size, 1.0 / sampling_frequency)
            frequency_arr_ret = frequency_arr[2 : int(fft_size / 2)]
            spectrum_ret = np.zeros(frequency_arr_ret.shape)
            return [frequency_arr_ret, spectrum_ret]

    ### check if t_start / t_end are None
    if t_start == None:
        t_start = round(t.min() * time_step, get_number_of_decimals(time_step))
    if t_end == None:
        t_end = round(t.max() * time_step, get_number_of_decimals(time_step))

    ### calculate time
    simulation_time = round(t_end - t_start, get_number_of_decimals(time_step))  # in ms

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
            spiketrain_sequences = _hanning_split_overlap(
                spiketrain, fft_size, int(fft_size / 2)
            )

            ### generate power spectrum
            spectrum[neuron] = get_nanmean(
                np.abs(np.fft.fft(spiketrain_sequences)) ** 2, 0
            )

        ### mean spectrum over all neurons
        spectrum = get_nanmean(spectrum, 0)

        frequency_arr = np.fft.fftfreq(fft_size, 1.0 / sampling_frequency)

        return (frequency_arr[2 : int(fft_size / 2)], spectrum[2 : int(fft_size / 2)])


def get_power_spektrum_from_time_array(
    arr,
    presimulationTime,
    simulationTime,
    simulation_dt,
    samplingfrequency=250,
    fftSize=1024,
):
    """
    Generates power spectrum of time signal (returns frequencies_arr and power_arr).
    Using the Welch methode (Welch,1967).

    amplingfrequency: to sample the arr, in Hz --> max frequency = samplingfrequency / 2
    fftSize: signal size for FFT, duration (in s) = fftSize / samplingfrequency
    --> frequency resolution = samplingfrequency / fftSize

    Args:
        arr (array):
            time array, value for each timestep
        presimulationTime (float or int):
            simulation time which will not be analyzed
        simulationTime (float or int):
            analyzed simulation time
        simulation_dt (float or int):
            simulation timestep
        samplingfrequency (float or int, optional):
            sampling frequency for sampling the time array. Default: 250
        fftSize (int, optional):
            signal size for the FFT (size of splitted arrays)
            has to be a power of 2. Default: 1024

    Returns:
        frequency_arr (array):
            array with frequencies
        spectrum (array):
            array with power spectrum
    """

    if (simulationTime / 1000) < (fftSize / samplingfrequency):
        print("Simulation time has to be >=", fftSize / samplingfrequency, "s for FFT!")
        return [np.zeros(int(fftSize / 2 - 2)), np.zeros(int(fftSize / 2 - 2))]
    else:
        ### sampling steps array
        sampling_arr = arr[0 :: int((1 / samplingfrequency) * 1000 / simulation_dt)]

        ### generate multiple overlapping sequences
        sampling_arr_sequences = _hanning_split_overlap(
            sampling_arr, fftSize, int(fftSize / 2)
        )

        ### generate power spectrum
        spektrum = get_nanmean(np.abs(np.fft.fft(sampling_arr_sequences)) ** 2, 0)

        frequenzen = np.fft.fftfreq(fftSize, 1.0 / samplingfrequency)

        return (frequenzen[2 : int(fftSize / 2)], spektrum[2 : int(fftSize / 2)])


# TODO: maybe makes errors with few/no spikes... check this TODO: automatically detect t_smooth does not work for strongly varying activity... implement something different
def _get_pop_rate_old(spikes, duration, dt=1, t_start=0, t_smooth_ms=-1):
    """
    Returns smoothed population rate from period after ramp up period until duration.

    Args:
        spikes (dict):
            ANNarchy spike dict of one population
        duration (float or int):
            duration of period after (optional) initial period (t_start) from which rate
            is calculated in ms
        dt (float or int, optional):
            timestep of simulation. Default: 1
        t_start (float or int, optional):
            starting simulation time, from which rates should be calculated. Default: 0
        t_smooth_ms (float or int, optional):
            time window size for rate calculation in ms, optional, standard = -1 which
            means automatic window size. Default: -1

    Returns:
        time_arr (array):
            array with time steps in ms
        rate (array):
            array with population rate
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

        rate = np.zeros((len(list(spikes.keys())), int(round(temp_duration / dt))))
        rate[:] = np.NaN
        binSize = int(round(t_smooth_ms / dt))
        bins = np.arange(0, int(round(temp_duration / dt)) + binSize, binSize)
        binsCenters = bins[:-1] + binSize // 2
        timeshift_start = -binSize // 2
        timeshift_end = binSize // 2
        timeshift_step = int(np.max([binSize // 10, 1]))
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
        timesteps = np.arange(0, int(round(temp_duration / dt)), 1).astype(int)
        time = timesteps[np.logical_not(np.isnan(poprate))]
        poprate = poprate[np.logical_not(np.isnan(poprate))]
        poprate = np.interp(timesteps, time, poprate)

        ret = poprate[int(round(t_start / dt)) :]
    else:
        ret = np.zeros(int(round(duration / dt)))

    return (
        np.arange(
            round(t_start, get_number_of_decimals(dt)),
            round(t_start + duration_init, get_number_of_decimals(dt)),
            round(dt, get_number_of_decimals(dt)),
        ),
        ret,
    )


def _recursive_get_bin_times_list(
    spike_arr,
    t0,
    t1,
    duration_init,
    nr_neurons,
    nr_spikes,
    c_pre=np.inf,
):
    """
    Recursive function to get the bin times for the _recursive_rate function.

    Args:
        spike_arr (array):
            spike times in s
        t0 (float or int):
            start of period in s
        t1 (float or int):
            end of period in s
        duration_init (float or int):
            full duration in s
        nr_neurons (int):
            number of neurons
        nr_spikes (int):
            number of spikes
        c_pre (float or int, optional):
            c value of previous bin. Default: np.inf

    Returns:
        bin_times_list (list):
            list with bin times
    """

    duration = t1 - t0

    threshold_nr_spikes = 5  # np.max([0.1 * nr_spikes, 10])
    threshold_duration = 5 / 1000.0
    nr_sub_divide = 5  # the higher the more fast chagnes are detected TODO: make this an argument for the user

    ### go up the nr_bins until threshold duration is reached with bin_size
    ### for each bin_size collect c
    ### collect min c --> c_min
    ### thus check if further sub dividing will improve the histogram
    bin_nr_temp = 2
    bin_size = duration / bin_nr_temp
    c = c_pre + 1
    c_min = c_pre + 1
    while bin_size > threshold_duration:
        hist, edges = np.histogram(spike_arr, bin_nr_temp, range=(t0, t1))
        bin_size = duration / bin_nr_temp
        c = (2 * np.mean(hist) - np.var(hist)) / (duration) ** 2
        if c < c_min:
            c_min = c
        bin_nr_temp = bin_nr_temp + 1

    ### if c_min better than c_pre --> continue subsplitting
    ### if c_min worse than c_pre --> stop subsplitting
    if c_pre < c_min:
        ### stop splitting because previous binsize was better than further subplitting
        ### --> current array is the complete bin
        return np.array([t0, t1])
    elif spike_arr.size < threshold_nr_spikes or duration < threshold_duration:
        ### stop splitting because thresholds are reached
        return np.array([t0, t1])
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
            _recursive_get_bin_times_list(
                spike_sub_arr_list[idx],
                t0=t0_t1_list[idx][0],
                t1=t0_t1_list[idx][1],
                duration_init=duration_init,
                nr_neurons=nr_neurons,
                nr_spikes=nr_spikes,
                c_pre=c_min,
            )
            for idx in range(hist.size)
        ]


def _recursive_rate(
    spike_arr,
    t0,
    t1,
    duration_init,
    nr_neurons,
    nr_spikes,
    c_pre=np.inf,
):
    """
    Recursive function to get the firing rate for the given spike array.

    Args:
        spike_arr (array):
            spike times in s
        t0 (float or int):
            start of period in s
        t1 (float or int):
            end of period in s
        duration_init (float or int):
            full duration in s
        nr_neurons (int):
            number of neurons
        nr_spikes (int):
            number of spikes
        c_pre (float or int, optional):
            c value of previous bin. Default: np.inf

    Returns:
        time_arr (array):
            array with time steps in s
        rate (array):
            array with firing rate
    """
    bin_times_list = _recursive_get_bin_times_list(
        spike_arr,
        t0,
        t1,
        duration_init,
        nr_neurons,
        nr_spikes,
        c_pre=c_pre,
    )

    bin_times_arr = np.array(ef.flatten_list(bin_times_list))
    if len(bin_times_arr.shape) == 1:
        ### only a single bin --> need to reshape array
        bin_times_arr = bin_times_arr[None, :]

    ### get borders for spikes
    time_first_spike = spike_arr.min()
    time_last_spike = spike_arr.max()
    t_min = np.max([time_first_spike, t0])
    t_max = np.min([time_last_spike, t1])

    ### get rates for all bins
    rate_list = []
    time_list = []
    ### for loop over bins
    for rec_bin in bin_times_arr:
        t0_bin = rec_bin[0]
        t1_bin = rec_bin[1]
        dur_bin = np.diff(rec_bin)[0]

        samples_per_bin = 10
        time_sample_arr = np.linspace(t0_bin, t1_bin, samples_per_bin, endpoint=True)
        ### for loop over bin samples
        for time_sample in time_sample_arr:
            t0_sample = np.clip(time_sample - dur_bin / 2, t0, t1 - dur_bin)
            t1_sample = np.clip(time_sample + dur_bin / 2, t0 + dur_bin, t1)
            nr_spikes_sample = np.sum(
                (spike_arr >= t0_sample).astype(int)
                * (spike_arr < t1_sample).astype(int)
            )
            time_list.append(time_sample)
            rate_list.append(nr_spikes_sample / (dur_bin * nr_neurons))

    ### move all values which are outside the spike borders to the borders
    time_arr = np.array(time_list)
    rate_arr = np.array(rate_list)
    mask_min = time_arr < t_min
    mask_max = time_arr > t_max
    time_arr[mask_min] = t_min
    time_arr[mask_max] = t_max

    ### now apped and prepend rate= 0
    ### time is first interspikeinterval before t_min and last interspikeinterval after t_max
    isi_arr = np.diff(np.sort(time_arr))
    time_arr = np.append(time_arr, [t_min - isi_arr[0], t_max + isi_arr[-1]])
    rate_arr = np.append(rate_arr, [0, 0])
    ### in case this added values outside of the time range --> remove them
    mask = ((time_arr >= t0).astype(int) * (time_arr <= t1).astype(int)).astype(bool)
    time_arr = time_arr[mask]
    rate_arr = rate_arr[mask]

    ### for edges of bins two values were calculated
    ### --> get these double values of the edges and
    ### use the average rate for these times
    time_unique_arr, time_unique_idx_arr, time_unique_counts_arr = np.unique(
        time_arr, return_counts=True, return_index=True
    )
    ### get idx of double times
    multi_times_idx_arr = np.where(time_unique_counts_arr > 1)[0]
    ### average the rates at these indixes
    for multi_times_idx in multi_times_idx_arr:
        rate_arr[time_arr == time_unique_arr[multi_times_idx]] = np.mean(
            rate_arr[time_arr == time_unique_arr[multi_times_idx]]
        )
    ### combine times and rates
    time_rate_arr = np.array([time_arr, rate_arr])

    ### return only the unique times (thus no two values for bin edges)
    return time_rate_arr[:, time_unique_idx_arr]


def get_pop_rate(
    spikes: dict,
    t_start: float | int | None = None,
    t_end: float | int | None = None,
    time_step: float | int = 1,
    t_smooth_ms: float | int = -1,
):
    """
    Generates a smoothed population firing rate. Returns a time array and a firing rate
    array.

    Args:
        spikes (dictionary):
            ANNarchy spike dict of one population
        t_start (float or int, optional):
            start time of analyzed data in ms. Default: time of first spike
        t_end (float or int, optional):
            end time of analyzed data in ms. Default: time of last spike
        time_step (float or int, optional):
            time step of the simulation in ms. Default: 1
        t_smooth_ms (float or int, optional):
            time window for firing rate calculation in ms, if -1 --> time window sizes
            are automatically detected. Default: -1

    Returns:
        time_arr (array):
            array with time steps in ms
        rate (array):
            array with population rate in Hz for each time step
    """
    dt = time_step

    t, _ = my_raster_plot(spikes)

    ### check if there are spikes in population at all
    if len(t) > 1:
        if t_start == None:
            t_start = round(t.min() * time_step, get_number_of_decimals(time_step))
        if t_end == None:
            t_end = round(t.max() * time_step, get_number_of_decimals(time_step))

        duration = round(t_end - t_start, get_number_of_decimals(time_step))

        ### if t_smooth is given --> use classic time_window method
        if t_smooth_ms > 0:
            return _get_pop_rate_old(
                spikes, duration, dt=dt, t_start=t_start, t_smooth_ms=t_smooth_ms
            )
        else:
            ### concatenate all spike times and sort them
            spike_arr = dt * np.sort(
                np.concatenate(
                    [np.array(spikes[neuron]).astype(int) for neuron in spikes.keys()]
                )
            )
            nr_neurons = len(list(spikes.keys()))
            nr_spikes = spike_arr.size

            ### use _recursive_rate to get firing rate
            ### spike array is splitted in time bins
            ### time bins widths are automatically found
            time_population_rate, population_rate = _recursive_rate(
                spike_arr / 1000.0,
                t0=t_start / 1000.0,
                t1=(t_start + duration) / 1000.0,
                duration_init=duration / 1000.0,
                nr_neurons=nr_neurons,
                nr_spikes=nr_spikes,
            )
            ### time_population_rate was returned in s --> transform it into ms
            time_population_rate = time_population_rate * 1000
            time_arr0 = np.arange(t_start, t_start + duration, dt)
            if len(time_population_rate) > 1:
                ### interpolate
                interpolate_func = interp1d(
                    time_population_rate,
                    population_rate,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(population_rate[0], population_rate[-1]),
                )
                population_rate_arr = interpolate_func(time_arr0)
            else:
                population_rate_arr = np.zeros(len(time_arr0))
                mask = time_arr0 == time_population_rate[0]
                population_rate_arr[mask] = population_rate[0]

            ret = population_rate_arr
    else:
        if t_start == None or t_end == None:
            return None
        else:
            duration = t_end - t_start
            ret = np.zeros(int(duration / dt))

    return (np.arange(t_start, t_start + duration, dt), ret)


@check_types()
def plot_recordings(
    figname: str,
    recordings: list,
    recording_times: RecordingTimes,
    chunk: int,
    shape: tuple,
    plan: list[str],
    time_lim: None | tuple = None,
    dpi: int = 300,
):
    """
    Plots the recordings of a single chunk from recordings. Plotted variables are
    specified in plan.

    Args:
        figname (str):
            path + name of figure (e.g. "figures/my_figure.png")
        recordings (list):
            a recordings list from CompNeuroPy obtained with the function
            get_recordings() from a CompNeuroMonitors object.
        recording_times (object):
            recording_times object from CompNeuroPy obtained with the
            function get_recording_times() from a CompNeuroMonitors object.
        chunk (int):
            which chunk of recordings should be used (the index of chunk)
        shape (tuple):
            Defines the subplot arrangement e.g. (3,2) = 3 rows, 2 columns
        plan (list of strings):
            Defines which recordings are plotted in which subplot and how.
            Entries of the list have the structure:
                "subplot_nr;model_component_name;variable_to_plot;format",
                e.g. "1,my_pop1;v;line".
                mode: defines how the data is plotted, available modes:
                    - for spike data: raster, mean, hybrid
                    - for other data: line, mean, matrix
                    - only for projection data: matrix_mean
        time_lim (tuple, optional):
            Defines the x-axis for all subplots. The list contains two
            numbers: start and end time in ms. The times have to be
            within the chunk. Default: None, i.e., time lims of chunk
        dpi (int, optional):
            The dpi of the saved figure. Default: 300
    """
    proc = Process(
        target=_plot_recordings,
        args=(figname, recordings, recording_times, chunk, shape, plan, time_lim, dpi),
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        quit()


def _plot_recordings(
    figname: str,
    recordings: list,
    recording_times: RecordingTimes,
    chunk: int,
    shape: tuple,
    plan: list[str],
    time_lim: None | tuple,
    dpi: int,
):
    """
    Plots the recordings for the given recording_times specified in plan.

    Args:
        figname (str):
            path + name of figure (e.g. "figures/my_figure.png")
        recordings (list):
            a recordings list from CompNeuroPy obtained with the function
            get_recordings() from a CompNeuroMonitors object.
        recording_times (object):
            recording_times object from CompNeuroPy obtained with the
            function get_recording_times() from a CompNeuroMonitors object.
        chunk (int):
            which chunk of recordings should be used (the index of chunk)
        shape (tuple):
            Defines the subplot arrangement e.g. (3,2) = 3 rows, 2 columns
        plan (list of strings):
            Defines which recordings are plotted in which subplot and how.
                Entries of the list have the structure:
                "subplot_nr;model_component_name;variable_to_plot;format",
                e.g. "1,my_pop1;v;line".
                mode: defines how the data is plotted, available modes:
                    - for spike data: raster, mean, hybrid
                    - for other data: line, mean, matrix
                    - only for projection data: matrix_mean
        time_lim (tuple):
            Defines the x-axis for all subplots. The list contains two
            numbers: start and end time in ms. The times have to be
            within the chunk.
        dpi (int):
            The dpi of the saved figure.
    """
    print(f"generate fig {figname}", end="... ", flush=True)
    recordings = recordings[chunk]
    if isinstance(time_lim, type(None)):
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
        nr_periods = recording_times._get_nr_periods(
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
            ### check if variable is equation
            variable_is_equation = (
                "+" in variable or "-" in variable or "*" in variable or "/" in variable
            )
            if variable_is_equation:
                ### evalueate the equation
                value_dict = {}
                for rec_key, rec_val in recordings.items():
                    if rec_key is f"{part};parameter_dict":
                        continue
                    if ";" in rec_key:
                        rec_var_name = rec_key.split(";")[1]
                    else:
                        rec_var_name = rec_key
                    value_dict[rec_var_name] = rec_val
                for param_key, param_val in recordings[
                    f"{part};parameter_dict"
                ].items():
                    value_dict[param_key] = param_val
                ### evaluate
                evaluated_variable = ef.evaluate_expression_with_dict(
                    expression=variable, value_dict=value_dict
                )
                ### add the evaluated variable to the recordings
                recordings[f"{part};{variable}"] = evaluated_variable

            ### set data
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
        if (variable == "spike" or variable == "axon_spike") and (
            mode == "raster" or mode == "single"
        ):  # "single" only for down compatibility
            nr_neurons = len(list(data.keys()))
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
                plt.ylim(0 - 0.5, nr_neurons - 0.5)
                plt.xlabel("time [ms]")
                plt.ylabel("# neurons")
                plt.title("Spikes " + part)
        elif (variable == "spike" or variable == "axon_spike") and mode == "mean":
            time_arr, firing_rate = get_pop_rate(
                spikes=data,
                t_start=start_time,
                t_end=end_time,
                time_step=time_step,
            )
            plt.plot(time_arr, firing_rate, color="k")
            plt.xlim(start_time, end_time)
            plt.xlabel("time [ms]")
            plt.ylabel("Mean firing rate [Hz]")
            plt.title("Mean firing rate " + part)
        elif (variable == "spike" or variable == "axon_spike") and mode == "hybrid":
            nr_neurons = len(list(data.keys()))
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
                    marker, size = ["|", np.sqrt(3000)]
                else:
                    marker, size = [".", np.sqrt(3)]

                plt.plot(
                    t[mask], n[mask], f"k{marker}", markersize=size, markeredgewidth=0.1
                )
                plt.ylabel("# neurons")
                plt.ylim(0 - 0.5, nr_neurons - 0.5)
                ax = plt.gca().twinx()
                time_arr, firing_rate = get_pop_rate(
                    spikes=data,
                    t_start=start_time,
                    t_end=end_time,
                    time_step=time_step,
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
        elif (variable != "spike" and variable != "axon_spike") and mode == "line":
            if len(data.shape) == 1:
                plt.plot(time_arr_dict[part], data, color="k")
                plt.title(f"Variable {variable} of {part} (1)")
            elif len(data.shape) == 2 and isinstance(data[0, 0], list) is not True:
                ### population: data[time,neurons]
                for neuron in range(data.shape[1]):
                    # in case of gaps file time gaps and add nan to data TODO also for other plots
                    plot_x, plot_y = time_data_add_nan(
                        time_arr_dict[part], data[:, neuron]
                    )

                    plt.plot(
                        plot_x,
                        plot_y,
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
        elif (variable != "spike" and variable != "axon_spike") and mode == "mean":
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

        elif (
            variable != "spike" and variable != "axon_spike"
        ) and mode == "matrix_mean":
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

        elif (variable != "spike" and variable != "axon_spike") and mode == "matrix":
            # data[start_step:end_step,neuron]
            if len(data.shape) == 2 and isinstance(data[0, 0], list) is not True:
                ### data from population [times,neurons]
                ### get the times and data between time_lims
                mask = (
                    (time_arr_dict[part] >= start_time).astype(int)
                    * (time_arr_dict[part] <= end_time).astype(int)
                ).astype(bool)

                time_decimals = get_number_of_decimals(time_step)

                time_arr = np.round(time_arr_dict[part][mask], time_decimals)
                data_arr = data[mask, :]

                ### check with the actual_period and the times array if there is data missing
                ###     from time_lims and actual period opne should get all times at which data points should be
                actual_period = recordings[f"{part};period"]
                actual_start_time = np.ceil(start_time / actual_period) * actual_period
                actual_end_time = np.ceil(end_time / actual_period - 1) * actual_period
                soll_times = np.round(
                    np.arange(
                        actual_start_time,
                        actual_end_time + actual_period,
                        actual_period,
                    ),
                    time_decimals,
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
    """
    For numbers which are smaller than zero get the number of digits after the decimal
    point which are zero (plus 1). For the number 0 or numbers >=1 return zero, e.g.:

    Args:
        nr (float or int):
            the number from which the number of digits are obtained

    Returns:
        decimals (int):
            number of digits after the decimal point which are zero (plus 1)

    Examples:
        >>> get_number_of_zero_decimals(0.12)
        1
        >>> get_number_of_zero_decimals(0.012)
        2
        >>> get_number_of_zero_decimals(1.012)
        0
    """
    decimals = 0
    if nr != 0:
        while abs(nr) < 1:
            nr = nr * 10
            decimals = decimals + 1

    return decimals


def get_number_of_decimals(nr):
    """
    Get number of digits after the decimal point.

    Args:
        nr (float or int):
            the number from which the number of digits are obtained

    Returns:
        decimals (int):
            number of digits after the decimal point

    Examples:
        >>> get_number_of_decimals(5)
        0
        >>> get_number_of_decimals(5.1)
        1
        >>> get_number_of_decimals(0.0101)
        4
    """

    if nr != int(nr):
        decimals = len(str(nr).split(".")[1])
    else:
        decimals = 0

    return decimals


def sample_data_with_timestep(time_arr, data_arr, timestep):
    """
    Samples a data array each timestep using interpolation

    Args:
        time_arr (array):
            times of data_arr in ms
        data_arr (array):
            array with data values from which will be sampled
        timestep (float or int):
            timestep in ms for sampling

    Returns:
        time_arr (array):
            sampled time array
        data_arr (array):
            sampled data array
    """
    interpolate_func = interp1d(
        time_arr, data_arr, bounds_error=False, fill_value="extrapolate"
    )
    min_time = round(
        round(time_arr[0] / timestep, 0) * timestep,
        get_number_of_decimals(timestep),
    )
    max_time = round(
        round(time_arr[-1] / timestep, 0) * timestep,
        get_number_of_decimals(timestep),
    )
    new_time_arr = np.arange(min_time, max_time + timestep, timestep)
    new_data_arr = interpolate_func(new_time_arr)

    return (new_time_arr, new_data_arr)


def time_data_add_nan(time_arr, data_arr, fill_time_step=None, axis=0):
    """
    If there are gaps in time_arr --> fill them with respective time values.
    Fill the corresponding data_arr values with nan.

    By default it is tried to fill the time array with continuously increasing times
    based on the smallest time difference found there can still be discontinuities after
    filling the arrays (because existing time values are not changed).

    But one can also give a fixed fill time step.

    Args:
        time_arr (1D array):
            times of data_arr in ms
        data_arr (nD array):
            the size of the specified dimension of data array must have the same length
            as time_arr
        fill_time_step (number, optional, default=None):
            if there are gaps they are filled with this time step
        axis (int):
            which dimension of the data_arr belongs to the time_arr

    Returns:
        time_arr (1D array):
            time array with gaps filled
        data_arr (nD array):
            data array with gaps filled
    """
    time_arr = time_arr.astype(float)
    data_arr = data_arr.astype(float)
    data_arr_shape = data_arr.shape

    if data_arr_shape[axis] != time_arr.size:
        print(
            "ERROR time_data_add_nan: time_arr must have same length as specified axis (default=0) of data_arr!"
        )
        quit()

    ### find gaps
    time_diff_arr = np.round(np.diff(time_arr), 6)
    if isinstance(fill_time_step, type(None)):
        time_diff_min = time_diff_arr.min()
    else:
        time_diff_min = fill_time_step
    gaps_arr = time_diff_arr > time_diff_min

    ### split arrays at gaps
    time_arr_split = np.split(
        time_arr, indices_or_sections=np.where(gaps_arr)[0] + 1, axis=0
    )
    data_arr_split = np.split(
        data_arr, indices_or_sections=np.where(gaps_arr)[0] + 1, axis=axis
    )

    ### fill gaps between splits
    data_arr_append_shape = list(data_arr_shape)
    for split_arr_idx in range(len(time_arr_split) - 1):
        ### get gaps boundaries
        current_end = time_arr_split[split_arr_idx][-1]
        next_start = time_arr_split[split_arr_idx + 1][0]
        ### create gap filling arrays
        time_arr_append = np.arange(
            current_end + time_diff_min, next_start, time_diff_min
        )
        data_arr_append_shape[axis] = time_arr_append.size
        data_arr_append = np.zeros(tuple(data_arr_append_shape)) * np.nan
        ### append gap filling arrays to splitted arrays
        time_arr_split[split_arr_idx] = np.append(
            arr=time_arr_split[split_arr_idx],
            values=time_arr_append,
            axis=0,
        )
        data_arr_split[split_arr_idx] = np.append(
            arr=data_arr_split[split_arr_idx],
            values=data_arr_append,
            axis=axis,
        )

    ### combine splitted arrays again
    time_arr = np.concatenate(time_arr_split, axis=0)
    data_arr = np.concatenate(data_arr_split, axis=axis)

    return (time_arr, data_arr)


def rmse(a, b):
    """
    Calculates the root-mean-square error between two arrays.

    Args:
        a (array):
            first array
        b (array):
            second array

    Returns:
        rmse (float):
            root-mean-square error
    """

    return np.sqrt(np.mean((a - b) ** 2))


def rsse(a, b):
    """
    Calculates the root-sum-square error between two arrays.

    Args:
        a (array):
            first array
        b (array):
            second array

    Returns:
        rsse (float):
            root-sum-square error
    """

    return np.sqrt(np.sum((a - b) ** 2))


def get_minimum(input_data: list | np.ndarray | tuple | float):
    """
    Returns the minimum of the input data.

    Args:
        input_data (list, np.ndarray, tuple, or float):
            The input data from which the minimum is to be obtained.

    Returns:
        minimum (float):
            The minimum of the input data.
    """
    if isinstance(input_data, (list, np.ndarray, tuple)):
        # If the input is a list, numpy array, or tuple, we handle them as follows
        flattened_list = [
            item
            for sublist in input_data
            for item in (
                sublist if isinstance(sublist, (list, np.ndarray, tuple)) else [sublist]
            )
        ]
        return float(min(flattened_list))
    else:
        # If the input is a single value, return it as the minimum
        return float(input_data)


def get_maximum(input_data: list | np.ndarray | tuple | float):
    """
    Returns the maximum of the input data.

    Args:
        input_data (list, np.ndarray, tuple, or float):
            The input data from which the maximum is to be obtained.

    Returns:
        maximum (float):
            The maximum of the input data.
    """

    if isinstance(input_data, (list, np.ndarray, tuple)):
        # If the input is a list, numpy array, or tuple, we handle them as follows
        flattened_list = [
            item
            for sublist in input_data
            for item in (
                sublist if isinstance(sublist, (list, np.ndarray, tuple)) else [sublist]
            )
        ]
        return float(max(flattened_list))
    else:
        # If the input is a single value, return it as the maximum
        return float(input_data)


# plan = {
#     "position": [1, 2, 3],
#     "compartment": ["pop1", "pop2", "pop3"],
#     "variable": ["v", "v", "v"],
#     "format": ["line", "line", "line"],
#     "color": ["k", "k", "k"],
# }


class PlotRecordings:
    """
    Plot recordings from CompNeuroMonitors.

    TODO: CHeck if there are memory issues with large recordings or many subplots.
    """

    @check_types()
    def __init__(
        self,
        figname: str,
        recordings: list[dict],
        recording_times: RecordingTimes,
        shape: tuple[int, int],
        plan: dict,
        chunk: int = 0,
        time_lim: None | tuple[float, float] = None,
        dpi: int = 300,
    ) -> None:
        """
        Create and save the plot.

        Args:
            figname (str):
                The name of the figure to be saved.
            recordings (list):
                A recordings list obtained from CompNeuroMonitors.
            recording_times (RecordingTimes):
                The RecordingTimes object containing the recording times obtained from
                CompNeuroMonitors.
            shape (tuple):
                The shape of the figure. (number of rows, number of columns)
            plan (dict):
                Defines which recordings are plotted in which subplot and how. The plan
                has to contain the following keys: "position", "compartment",
                "variable", "format". The values of the keys have to be lists of the
                same length. The values of the key "position" have to be integers
                between 1 and the number of subplots (defined by shape). The values of
                the key "compartment" have to be the names of the model compartments as
                strings. The values of the key "variable" have to be strings containing
                the names of the recorded variables or equations using the recorded
                variables. The values of the key "format" have to be strings defining
                how the recordings are plotted. The following formats are available for
                spike recordings: "raster", "mean", "hybrid", "interspike". The
                following formats are available for other recordings: "line",
                "line_mean", "matrix", "matrix_mean".
            chunk (int, optional):
                The chunk of the recordings to be plotted. Default: 0.
            time_lim (tuple, optional):
                Defines the x-axis for all subplots. The tuple contains two
                numbers: start and end time in ms. The times have to be
                within the chunk. Default: None, i.e., the whole chunk is plotted.
            dpi (int, optional):
                The dpi of the saved figure. Default: 300.
        """
        ### print start message
        print(f"Generate fig {figname}", end="... ", flush=True)

        ### set attributes
        self.figname = figname
        self.recordings = recordings
        self.recording_times = recording_times
        self.shape = shape
        self.plan = plan
        self.chunk = chunk
        self.time_lim = time_lim
        self.dpi = dpi

        ### get available compartments (from recordings) and recorded variables for each
        ### compartment
        (
            self._compartment_list,
            self._compartment_recordings_dict,
        ) = self._get_compartment_recordings()

        ### check plan keys and values
        self._check_plan()

        ### get start and end time for plotting and timestep
        self._start_time, self._end_time, self._time_step = self._get_start_end_time()

        ### get compbined time array for recordings of each compartment
        self._time_arr_list = self._get_time_arr_list()

        ### get data from recordings for each subplot
        self._raw_data_list = self._get_raw_data_list()

        ### create plot
        self._plot()

        ### print end message
        print("Done\n")

    def _get_compartment_recordings(self):
        """
        Get available compartment names from recordings.
        Get recorded variables (names) for each compartment.

        Returns:
            compartment_list (list):
                List of compartment names.
            compartment_recordings_dict (dict):
                Dictionary with compartment names as keys and list of recorded variables
                as values.
        """
        ### check if chunk is valid
        if self.chunk >= len(self.recordings) or self.chunk < 0:
            print(
                f"\nERROR PlotRecordings: chunk {self.chunk} is not valid.\n"
                f"Number of chunks: {len(self.recordings)}\n"
            )
            quit()

        ### get compartment names and recorded variables for each compartment
        compartment_list = []
        compartment_recordings_dict = {}
        for recordings_key in self.recordings[self.chunk].keys():
            if ";" not in recordings_key:
                continue

            ### get compartment
            compartment, recorded_variable = recordings_key.split(";")
            if compartment not in compartment_list:
                compartment_list.append(compartment)
                compartment_recordings_dict[compartment] = []

            ### get recordings for compartment
            if recorded_variable != "period" and recorded_variable != "parameter_dict":
                compartment_recordings_dict[compartment].append(recorded_variable)

        return compartment_list, compartment_recordings_dict

    def _check_plan(self):
        """
        Check if plan is valid.
        """

        ### check if plan keys are valid
        valid_keys = ["position", "compartment", "variable", "format"]
        for key in self.plan.keys():
            if key not in valid_keys:
                print(
                    f"\nERROR PlotRecordings: plan key {key} is not valid.\n"
                    f"Valid keys are {valid_keys}.\n"
                )
                quit()

        ### check if plan values are valid (have same length)
        for key in self.plan.keys():
            if len(self.plan[key]) != len(self.plan["position"]):
                print(
                    f"\nERROR PlotRecordings: plan value of key '{key}' has not the same length as plan value of key 'position'.\n"
                )
                quit()

        ### check if plan positions are valid
        ### check if min and max are valid
        if get_minimum(self.plan["position"]) < 1:
            print(
                f"\nERROR PlotRecordings: plan position has to be >= 1.\n"
                f"plan position: {self.plan['position']}\n"
            )
            quit()
        if get_maximum(self.plan["position"]) > self.shape[0] * self.shape[1]:
            print(
                f"\nERROR PlotRecordings: plan position has to be <= shape[0] * shape[1].\n"
                f"plan position: {self.plan['position']}\n"
                f"shape: {self.shape}\n"
            )
            quit()
        ### check if plan positions are unique
        if len(np.unique(self.plan["position"])) != len(self.plan["position"]):
            print(
                f"\nERROR PlotRecordings: plan position has to be unique.\n"
                f"plan position: {self.plan['position']}\n"
            )
            quit()

        ### check if plan compartments are valid
        for compartment in self.plan["compartment"]:
            if compartment not in self._compartment_list:
                print(
                    f"\nERROR PlotRecordings: plan compartment {compartment} is not valid.\n"
                    f"Valid compartments are {self._compartment_list}.\n"
                )
                quit()

        ### check if plan variables are valid
        for plot_idx in range(len(self.plan["variable"])):
            compartment = self.plan["compartment"][plot_idx]
            variable: str = self.plan["variable"][plot_idx]
            ### check if variable contains a mathematical expression
            if "+" in variable or "-" in variable or "*" in variable or "/" in variable:
                ### separate variables
                variable = variable.replace(" ", "")
                variable = variable.replace("+", " ")
                variable = variable.replace("-", " ")
                variable = variable.replace("*", " ")
                variable = variable.replace("/", " ")
                variables_list = variable.split(" ")
                ### remove numbers
                variables_list = [var for var in variables_list if not var.isdigit()]
                ### spike and axon_spike are not allowed in equations
                if "spike" in variables_list or "axon_spike" in variables_list:
                    print(
                        f"\nERROR PlotRecordings: plan variable {variable} is not valid.\n"
                        f"Variables 'spike' and 'axon_spike' are not allowed in equations.\n"
                    )
                    quit()
            else:
                variables_list = [variable]
            ### check if variables are valid
            for var in variables_list:
                if var not in self._compartment_recordings_dict[compartment]:
                    print(
                        f"\nERROR PlotRecordings: plan variable {var} is not valid for compartment {compartment}.\n"
                        f"Valid variables are {self._compartment_recordings_dict[compartment]}.\n"
                    )
                    quit()

        ### check if plan formats are valid
        valid_formats_spike = ["raster", "mean", "hybrid", "interspike", "cv"]
        valid_formats_other = ["line", "line_mean", "matrix", "matrix_mean"]
        for plot_idx in range(len(self.plan["format"])):
            variable = self.plan["variable"][plot_idx]
            format = self.plan["format"][plot_idx]
            ### check if format is valid
            if variable == "spike" or variable == "axon_spike":
                if format not in valid_formats_spike:
                    print(
                        f"\nERROR PlotRecordings: plan format {format} is not valid for variable {variable}.\n"
                        f"Valid formats are {valid_formats_spike}.\n"
                    )
                    quit()
            else:
                if format not in valid_formats_other:
                    print(
                        f"\nERROR PlotRecordings: plan format {format} is not valid for variable {variable}.\n"
                        f"Valid formats are {valid_formats_other}.\n"
                    )
                    quit()

    def _get_start_end_time(self):
        """
        Check if time_lim is given and valid. If it's not given get it from recordings.
        Get timestep from recordings.

        Returns:
            start_time (float):
                The start time of the recordings.
            end_time (float):
                The end time of the recordings.
            time_step (float):
                The timestep of the recordings.

        Raises:
            ValueError: If given time_lim is not within the chunk.
        """

        chunk_time_lims = self.recording_times.time_lims(chunk=self.chunk)
        ### check if time_lim is given
        if isinstance(self.time_lim, type(None)):
            ### get start and end time from recording_times
            start_time, end_time = chunk_time_lims
        else:
            ### check if time_lim is within chunk
            if (
                self.time_lim[0] < chunk_time_lims[0]
                or self.time_lim[1] > chunk_time_lims[1]
            ):
                raise ValueError(
                    f"\nERROR PlotRecordings: time_lim {self.time_lim} is not within chunk.\n"
                    f"chunk time lims: {chunk_time_lims[0]} - {chunk_time_lims[1]}\n"
                )
            start_time, end_time = self.time_lim

        ### get timestep
        time_step = self.recordings[self.chunk]["dt"]

        return start_time, end_time, time_step

    def _get_time_arr_list(self):
        """
        Get combined time array for each subplot of plan.

        Returns:
            time_arr_list (list):
                List with time arrays for each subplot of plan.
        """
        ### loop over compartments of plan
        time_arr_dict = {}
        for compartment in np.unique(self.plan["compartment"]):
            actual_period = self.recordings[self.chunk][f"{compartment};period"]

            ### get time array for each recording period of the chunk
            time_arr_period_list = []
            nr_periods = self.recording_times._get_nr_periods(
                chunk=self.chunk, compartment=compartment
            )
            for period in range(nr_periods):
                time_lims = self.recording_times.time_lims(
                    chunk=self.chunk, compartment=compartment, period=period
                )
                start_time_preiod = time_lims[0]
                end_time_period = round(
                    time_lims[1] + actual_period, get_number_of_decimals(actual_period)
                )
                time_arr_period_list.append(
                    np.arange(start_time_preiod, end_time_period, actual_period)
                )

            ### combine time arrays of periods
            time_arr_dict[compartment] = np.concatenate(time_arr_period_list)

        ### get time array for each subplot of plan
        time_arr_list = []
        for plot_idx in range(len(self.plan["position"])):
            compartment = self.plan["compartment"][plot_idx]
            time_arr_list.append(time_arr_dict[compartment])

        return time_arr_list

    def _get_raw_data_list(self):
        """
        Get raw data for each subplot of plan.

        Returns:
            data_list (dict):
                List with data for each subplot of plan.
        """
        data_list = []
        ### loop over subplots of plan
        for plot_idx in range(len(self.plan["position"])):
            compartment = self.plan["compartment"][plot_idx]
            variable: str = self.plan["variable"][plot_idx]
            ### check if variable is equation
            if "+" in variable or "-" in variable or "*" in variable or "/" in variable:
                ### get the values of the recorded variables of the compartment, store
                ### them in dict
                value_dict = {
                    rec_var_name: self.recordings[self.chunk][
                        f"{compartment};{rec_var_name}"
                    ]
                    for rec_var_name in self._compartment_recordings_dict[compartment]
                }
                ### evaluate equation with these values
                variable_data = ef.evaluate_expression_with_dict(
                    expression=variable, value_dict=value_dict
                )
            else:
                ### get data from recordings
                variable_data = self.recordings[self.chunk][f"{compartment};{variable}"]
            ### append data to data_list
            data_list.append(variable_data)

        return data_list

    def _plot(self):
        """
        Create plot.
        """
        ### create figure
        plt.figure(figsize=([6.4 * self.shape[1], 4.8 * self.shape[0]]))

        ### loop over subplots of plan
        for plot_idx in range(len(self.plan["position"])):
            ### create subplot
            plt.subplot(self.shape[0], self.shape[1], self.plan["position"][plot_idx])

            ### fill subplot
            self._fill_subplot(plot_idx)

        ### save figure
        plt.tight_layout()
        figname_parts = self.figname.split("/")
        if len(figname_parts) > 1:
            save_dir = "/".join(figname_parts[:-1])
            sf.create_dir(save_dir)
        plt.savefig(self.figname, dpi=self.dpi)
        plt.close()

    def _fill_subplot(self, plot_idx):
        """
        Fill subplot with data.

        Args:
            plot_idx (int):
                The index of the subplot in the plan.
        """
        variable: str = self.plan["variable"][plot_idx]

        ### general subplot settings
        plt.xlabel("time [ms]")
        plt.xlim(self._start_time, self._end_time)

        if variable == "spike" or variable == "axon_spike":
            ### spike recordings
            self._fill_subplot_spike(plot_idx)
        else:
            ### other (array) recordings
            self._fill_subplot_other(plot_idx)

    def _fill_subplot_spike(self, plot_idx):
        """
        Fill subplot with spike data.

        Args:
            plot_idx (int):
                The index of the subplot in the plan.
        """
        ### get data
        compartment = self.plan["compartment"][plot_idx]
        format: str = self.plan["format"][plot_idx]
        data = self._raw_data_list[plot_idx]

        ### get spike times and ranks
        spike_times, spike_ranks = my_raster_plot(data)
        spike_times = spike_times * self._time_step

        ### get spikes within time_lims
        mask: np.ndarray = (
            (spike_times >= self._start_time).astype(int)
            * (spike_times <= self._end_time).astype(int)
        ).astype(bool)

        ### check if there are no spikes
        if mask.size == 0:
            ### set title
            plt.title(f"Spikes {compartment}")
            ### print warning
            print(
                f"\n  WARNING PlotRecordings: {compartment} does not contain any spikes in the given time interval."
            )
            ### plot text
            plt.text(
                0.5,
                0.5,
                f"{compartment} does not contain any spikes.",
                va="center",
                ha="center",
            )
            plt.xticks([])
            plt.yticks([])
            plt.xlim(0, 1)
            plt.xlabel("")
            return

        ### plot raster plot
        if format == "raster" or format == "hybrid":
            self._raster_plot(compartment, spike_ranks, spike_times, mask)

        ### plot mean firing rate
        if format == "mean" or format == "hybrid":
            self._mean_firing_rate_plot(compartment, data, format)

        ### plot interspike interval histogram
        if format == "interspike":
            self._interspike_interval_plot(compartment, data)

        ### plot coefficient of variation histogram
        if format == "cv":
            self._coefficient_of_variation_plot(compartment, data)

    def _raster_plot(self, compartment, spike_ranks, spike_times, mask):
        """
        Plot raster plot.

        Args:
            compartment (str):
                The name of the compartment.
            spike_ranks (array):
                The spike ranks.
            spike_times (array):
                The spike times.
            mask (array):
                The mask for the spike times.
        """
        ### set title
        plt.title(f"Spikes {compartment} ({spike_ranks.max() + 1})")
        ### check if there is only one neuron
        if spike_ranks.max() == 0:
            marker, size = ["|", 3000]
        else:
            marker, size = [".", 3]
        ### plot spikes
        plt.scatter(
            spike_times[mask],
            spike_ranks[mask],
            color="k",
            marker=marker,
            s=size,
            linewidth=0.1,
        )
        ### set limits
        plt.ylim(-0.5, spike_ranks.max() + 0.5)
        ### set ylabel
        plt.ylabel("# neurons")
        ### set yticks
        if spike_ranks.max() == 0:
            plt.yticks([0])
        else:
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    def _mean_firing_rate_plot(self, compartment, data, format):
        """
        Plot mean firing rate.

        Args:
            compartment (str):
                The name of the compartment.
            data (array):
                The spike data.
            format (str):
                The format of the plot.
        """
        ### set title
        plt.title(f"Activity {compartment} ({len(data)})")
        ### set axis
        ax = plt.gca()
        color = "k"
        ### for hybrid format plot mean firing rate in second y-axis
        if format == "hybrid":
            ax = plt.gca().twinx()
            color = "r"
        ### get mean firing rate
        time_arr, firing_rate = get_pop_rate(
            spikes=data,
            t_start=self._start_time,
            t_end=self._end_time,
            time_step=self._time_step,
        )
        ### plot mean firing rate
        ax.plot(time_arr, firing_rate, color=color)
        ### set limits
        ax.set_xlim(self._start_time, self._end_time)
        ### set ylabel
        ax.set_ylabel("Mean firing rate [Hz]", color=color)
        ax.tick_params(axis="y", colors=color)

    def _interspike_interval_plot(self, compartment, data):
        """
        Plot interspike interval histogram.

        Args:
            compartment (str):
                The name of the compartment.
            data (dict):
                The spike data.
        """
        ### set title
        plt.title(f"Interspike interval histogram {compartment} ({len(data)})")
        ### get interspike intervals
        interspike_intervals_list = inter_spike_interval(spikes=data)
        ### plot histogram
        plt.hist(
            interspike_intervals_list,
            bins=100,
            range=(0, 200),
            density=True,
            color="k",
        )
        ### set limits
        plt.xlim(0, 200)
        ### set ylabel
        plt.ylabel("Probability")
        plt.xlabel("Interspike interval [ms]")

    def _coefficient_of_variation_plot(self, compartment, data):
        """
        Plot coefficient of variation histogram.

        Args:
            compartment (str):
                The name of the compartment.
            data (dict):
                The spike data.
        """
        ### set title
        plt.title(f"Coefficient of variation histogram {compartment} ({len(data)})")
        ### get coefficient of variation
        coefficient_of_variation_dict = coefficient_of_variation(
            spikes=data,
            per_neuron=True,
        )
        coefficient_of_variation_list = list(coefficient_of_variation_dict.values())
        ### plot histogram
        plt.hist(
            coefficient_of_variation_list,
            bins=100,
            range=(0, 2),
            density=True,
            color="k",
        )
        ### set limits
        plt.xlim(0, 2)
        ### set ylabel
        plt.ylabel("Probability")
        plt.xlabel("Coefficient of variation")

    def _fill_subplot_other(self, plot_idx):
        """
        Fill subplot with array data.

        Args:
            plot_idx (int):
                The index of the subplot in the plan.
        """
        ### get data
        compartment = self.plan["compartment"][plot_idx]
        variable: str = self.plan["variable"][plot_idx]
        format: str = self.plan["format"][plot_idx]
        data_arr = self._raw_data_list[plot_idx]
        time_arr = self._time_arr_list[plot_idx]

        ### get data within time_lims
        mask: np.ndarray = (
            (time_arr >= self._start_time).astype(int)
            * (time_arr <= self._end_time).astype(int)
        ).astype(bool)

        ### fill gaps in time_arr and data_arr with nan
        time_arr, data_arr = time_data_add_nan(
            time_arr=time_arr[mask], data_arr=data_arr[mask], axis=0
        )

        ### plot line plot
        if "line" in format:
            self._line_plot(
                compartment,
                variable,
                time_arr,
                data_arr,
                plot_idx,
                mean="mean" in format,
            )

        ### plot matrix plot
        if "matrix" in format:
            self._matrix_plot(
                compartment,
                variable,
                time_arr,
                data_arr,
                plot_idx,
                mean="mean" in format,
            )

    def _line_plot(self, compartment, variable, time_arr, data_arr, plot_idx, mean):
        """
        Plot line plot.

        Args:
            compartment (str):
                The name of the compartment.
            variable (str):
                The name of the variable.
            time_arr (array):
                The time array.
            data_arr (array):
                The data array.
            plot_idx (int):
                The index of the subplot in the plan.
            mean (bool):
                If True, plot the mean of the data. Population: average over neurons.
                Projection: average over preneurons (results in one line for each
                postneuron).
        """

        ### set title
        plt.title(f"Variable {variable} of {compartment} ({data_arr.shape[1]})")

        ### Shape of data defines how to plot
        ### 2D array where elements are no lists
        ### = population data [time, neurons]
        ### --> plot line for each neuron
        if len(data_arr.shape) == 2 and isinstance(data_arr[0, 0], list) is not True:
            ### mean -> average over neurons
            if mean:
                data_arr = np.mean(data_arr, 1, keepdims=True)
            ### plot line for each neuron
            for neuron in range(data_arr.shape[1]):
                plt.plot(
                    time_arr,
                    data_arr[:, neuron],
                    color="k",
                )

        ### 2D array where elements are lists
        ### = projection data [time, postneurons][preneurons]
        ### 3D array
        ### = projection data [time, postneurons, preneurons]
        ### --> plot line for each preneuron postneuron pair
        elif len(data_arr.shape) == 3 or (
            len(data_arr.shape) == 2 and isinstance(data_arr[0, 0], list) is True
        ):
            ### plot line for each preneuron postneuron pair
            for post_neuron in range(data_arr.shape[1]):
                ### the post_neuron has a constant number of preneurons
                ### --> create array with preneuron indices [time, preneurons]
                post_neuron_data = np.array(data_arr[:, post_neuron])
                ### mean -> average over preneurons
                if mean:
                    post_neuron_data = np.mean(post_neuron_data, 1, keepdims=True)
                for pre_neuron in range(post_neuron_data.shape[1]):
                    plt.plot(
                        time_arr,
                        post_neuron_data[:, pre_neuron],
                        color="k",
                    )
        else:
            print(
                f"\nERROR PlotRecordings: shape of data not supported, {compartment}, {variable} in plot {plot_idx}.\n"
            )

    def _matrix_plot(self, compartment, variable, time_arr, data_arr, plot_idx, mean):
        """
        Plot matrix plot.

        Args:
            compartment (str):
                The name of the compartment.
            variable (str):
                The name of the variable.
            time_arr (array):
                The time array.
            data_arr (array):
                The data array.
            plot_idx (int):
                The index of the subplot in the plan.
            mean (bool):
                If True, plot the mean of the data. Population: average over neurons.
                Projection: average over preneurons (results in one line for each
                postneuron).
        """
        ### number of neurons i.e. postneurons
        nr_neurons = data_arr.shape[1]

        ### Shape of data defines how to plot
        ### 2D array where elements are no lists
        ### = population data [time, neurons]
        ### --> plot matrix row for each neuron
        ### mean -> average over neurons
        if len(data_arr.shape) == 2 and isinstance(data_arr[0, 0], list) is not True:
            ### mean -> average over neurons
            if mean:
                data_arr = np.mean(data_arr, 1, keepdims=True)

        ### 2D array where elements are lists
        ### = projection data [time, postneurons][preneurons]
        ### 3D array
        ### = projection data [time, postneurons, preneurons]
        ### --> plot matrix row for each preneuron postneuron pair (has to reshape to 2D array [time, neuron pair])
        ### mean -> average over preneurons
        elif len(data_arr.shape) == 3 or (
            len(data_arr.shape) == 2 and isinstance(data_arr[0, 0], list) is True
        ):
            array_2D_list = []
            ### loop over postneurons
            for post_neuron in range(data_arr.shape[1]):
                ### the post_neuron has a constant number of preneurons
                ### --> create array with preneuron indices [time, preneurons]
                post_neuron_data = np.array(data_arr[:, post_neuron])
                ### mean --> average over preneurons
                if mean:
                    post_neuron_data = np.mean(post_neuron_data, 1, keepdims=True)
                ### append all preneurons arrays to array_2D_list
                for pre_neuron in range(post_neuron_data.shape[1]):
                    array_2D_list.append(post_neuron_data[:, pre_neuron])
                ### append a None array to array_2D_list to separate postneurons
                array_2D_list.append(np.empty(post_neuron_data.shape[0]) * np.nan)

            ### convert array_2D_list to 2D array, not use last None array
            data_arr = np.array(array_2D_list[:-1]).T

        ### some other shape not supported
        else:
            print(
                f"\nERROR PlotRecordings: shape of data not supported, {compartment}, {variable} in plot {plot_idx}.\n"
            )

        ### plot matrix row for each neuron or preneuron postneuron pair
        plt.imshow(
            data_arr.T,
            aspect="auto",
            vmin=np.nanmin(data_arr),
            vmax=np.nanmax(data_arr),
            extent=[
                time_arr.min()
                - self.recordings[self.chunk][f"{compartment};period"] / 2,
                time_arr.max()
                + self.recordings[self.chunk][f"{compartment};period"] / 2,
                data_arr.shape[1] - 0.5,
                -0.5,
            ],
            cmap="viridis",
            interpolation="none",
        )
        if data_arr.shape[1] == 1:
            plt.yticks([0])
        else:
            ### all y ticks
            y_tick_positions_all_arr = np.arange(data_arr.shape[1])
            ### boolean array of valid y ticks
            valid_y_ticks = np.logical_not(np.isnan(data_arr).any(axis=0))
            ### get y tick labels
            if False in valid_y_ticks:
                ### there are nan entries
                ### split at nan entries
                y_tick_positions_split_list = np.array_split(
                    y_tick_positions_all_arr, np.where(np.logical_not(valid_y_ticks))[0]
                )
                ### decrease by 1 after each nan entry
                y_tick_positions_split_list = [
                    y_tick_positions_split - idx_split
                    for idx_split, y_tick_positions_split in enumerate(
                        y_tick_positions_split_list
                    )
                ]
                ### join split arrays
                y_tick_labels_all_arr = np.concatenate(y_tick_positions_split_list)
            else:
                y_tick_labels_all_arr = y_tick_positions_all_arr

            valid_y_ticks_selected_idx_arr = np.linspace(
                0,
                np.sum(valid_y_ticks),
                num=min([10, np.sum(valid_y_ticks)]),
                dtype=int,
                endpoint=False,
            )
            valid_y_ticks_selected_arr = y_tick_positions_all_arr[valid_y_ticks][
                valid_y_ticks_selected_idx_arr
            ]
            valid_y_ticks_labels_selected_arr = y_tick_labels_all_arr[valid_y_ticks][
                valid_y_ticks_selected_idx_arr
            ]

            plt.yticks(valid_y_ticks_selected_arr, valid_y_ticks_labels_selected_arr)

        ### set title
        plt.title(
            f"Variable {variable} of {compartment} ({nr_neurons}) [{ef.sci(np.nanmin(data_arr))}, {ef.sci(np.nanmax(data_arr))}]"
        )
