import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ANNarchy import raster_plot, dt
import warnings
from CompNeuroPy import system_functions as sf
from CompNeuroPy import extra_functions as ef
from scipy.interpolate import interp1d


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
    presimulationTime,
    simulationTime,
    simulation_dt,
    spikesSamplingfrequency=250,
    fftSize=1024,
):
    """
    generates power spectrum of population spikes
    using the Welch methode (Welch,1967)

    spikes: ANNarchy spike dict of one population
    presimulationTime: simulation time which will not be analyzed
    simulationTime: analyzed simulation time

    spikesSamplingfrequency: to sample the spike times, in Hz --> max frequency = samplingfrequency / 2
    fftSize: signal size for FFT, duration (in s) = fftSize / samplingfrequency --> frequency resolution = samplingfrequency / fftSize
    """
    populationSize = len(list(spikes.keys()))

    if (simulationTime / 1000) < (fftSize / spikesSamplingfrequency):
        print(
            "Simulation time has to be >=",
            fftSize / spikesSamplingfrequency,
            "s for FFT!",
        )
        return [np.zeros(int(fftSize / 2 - 2)), np.zeros(int(fftSize / 2 - 2))]
    else:
        spektrum = np.zeros((populationSize, fftSize))
        for neuron in range(populationSize):
            ### sampling steps array
            spiketrain = np.zeros(
                int((simulationTime / 1000.0) * spikesSamplingfrequency)
            )
            ### spike times as sampling steps
            idx = (
                ((np.array(spikes[neuron]) * simulation_dt) / 1000.0)
                * spikesSamplingfrequency
            ).astype(np.int32)
            ### cut the spikes from presimulation
            idx = idx - (presimulationTime / 1000.0) * spikesSamplingfrequency
            idx = idx[idx >= 0].astype(np.int32)
            ### sampling steps array, one if there was a spike at sampling step = spike train
            spiketrain[idx] = 1

            ###generate multiple overlapping sequences out of the spike trains
            spiketrainSequences = hanning_split_overlap(
                spiketrain, fftSize, int(fftSize / 2)
            )

            ###generate power spectrum
            spektrum[neuron] = get_nanmean(
                np.abs(np.fft.fft(spiketrainSequences)) ** 2, 0
            )

        ###mean spectrum over all neurons
        spektrum = get_nanmean(spektrum, 0)

        frequenzen = np.fft.fftfreq(fftSize, 1.0 / spikesSamplingfrequency)

        return [frequenzen[2 : int(fftSize / 2)], spektrum[2 : int(fftSize / 2)]]


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

        ###generate multiple overlapping sequences
        sampling_arr_sequences = hanning_split_overlap(
            sampling_arr, fftSize, int(fftSize / 2)
        )

        ###generate power spectrum
        spektrum = get_nanmean(np.abs(np.fft.fft(sampling_arr_sequences)) ** 2, 0)

        frequenzen = np.fft.fftfreq(fftSize, 1.0 / samplingfrequency)

        return [frequenzen[2 : int(fftSize / 2)], spektrum[2 : int(fftSize / 2)]]


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
        for idx, key in enumerate(spikes.keys()):
            times = np.array(spikes[key]).astype(int)
            for timeshift in np.arange(
                -binSize // 2, binSize // 2 + binSize // 10, binSize // 10
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

    return ret


def plot_recordings(
    figname, recordings, recording_times, chunk, time_lim, shape, plan, dpi=300
):
    """
    recordings: list with recordings
    shape: tuple, shape of subplots
    plan: list of strings, strings defin where to plot which data and how
    """
    recordings = recordings[chunk]
    compartment_list = []
    for plan_str in plan:
        compartment = plan_str.split(";")[1]
        if not (compartment in compartment_list):
            compartment_list.append(compartment)

    ### get idx_lim for all compartments, in parallel check that there are no pauses
    idx_lim = {}
    for compartment in compartment_list:
        idx_lim[compartment] = []
        nr_periods = recording_times.__get_nr_periods__(
            chunk=chunk, compartment=compartment
        )
        time_lim_list = []
        idx_lim_list = []
        for period in range(nr_periods):
            ### get the time_lim and idx_lim of the period
            time_lim_list.append(
                recording_times.time_lims(
                    chunk=chunk, compartment=compartment, period=period
                )
            )
            idx_lim_list.append(
                recording_times.idx_lims(
                    chunk=chunk, compartment=compartment, period=period
                )
            )
        ### check for both time_lim values, in which period they fall
        for idx_time_lim_val, time_lim_val in enumerate(time_lim):
            for period in range(nr_periods):
                ### if period is found --> calculate idx_lim of compartment
                if (
                    time_lim_val >= time_lim_list[period][0]
                    and time_lim_val <= time_lim_list[period][1]
                ):
                    calc_i0 = idx_lim_list[period][0]
                    calc_i1 = idx_lim_list[period][1]
                    calc_t0 = time_lim_list[period][0]
                    calc_t1 = time_lim_list[period][1]

                    idx_lim[compartment].append(
                        (calc_i1 - calc_i0) / (calc_t1 - calc_t0) * time_lim_val
                        + calc_i0
                        - calc_t0 * (calc_i1 - calc_i0) / (calc_t1 - calc_t0)
                    )
                    if idx_time_lim_val == 0:
                        ### first index within time limits --> ceil
                        idx_lim[compartment][-1] = np.ceil(idx_lim[compartment][-1])
                    else:
                        ### last index within time limits --> floor
                        idx_lim[compartment][-1] = np.floor(idx_lim[compartment][-1])

                ### if period is found --> check were first recording step is for compartment --> actual start time

    ### check if there are no pauses between periods
    # if period < nr_periods - 1:
    #     this_period_lims = recording_times.time_lims(
    #         chunk=chunk, compartment=compartment, period=period
    #     )
    #     next_period_lims = recording_times.time_lims(
    #         chunk=chunk, compartment=compartment, period=period + 1
    #     )
    #     if this_period_lims[1] != next_period_lims[0]:
    #         print(
    #             "ERROR plot_recordings, time_lim and idx_lim do not fit! Maybe multiple periods separated by pauses in recordings?"
    #         )
    #         quit()

    ### define times and indizes for plots
    print(figname, end=":\t")
    print(time_lim)

    start_time = time_lim[0]
    end_time = time_lim[1]
    times = {}
    start_step = {}
    end_step = {}
    for compartment in compartment_list:
        times[compartment] = np.arange(
            start_time, end_time, recordings[f"{compartment};period"]
        )
        start_step[compartment] = int(idx_lim[compartment][0])
        end_step[compartment] = int(idx_lim[compartment][1])

        ### check if there are pauses between periods:
        if not (
            (end_time - start_time) / recordings[f"{compartment};period"]
            == end_step[compartment] - start_step[compartment]
        ):
            print(
                "ERROR plot_recordings, time_lim and idx_lim do not fit! Maybe multiple periods separated by pauses in recordings?"
            )
            # quit()
        print(f"{compartment} with period {recordings[f'{compartment};period']}")
        print(f"recordings shape: {recordings[f'{compartment};p'].shape}")
        print(f"time range: {start_time} - {end_time}")
        print(f"start step: {start_step[compartment]}")
        print(f"end step: {end_step[compartment]}")
        print(f"times shape: {times[compartment].shape}\n")

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
            data = recordings[";".join([part, variable])]
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
        if variable == "spike" and mode == "raster":
            t, n = my_raster_plot(data)
            if np.unique(n).size == 0:
                plt.title("Spikes " + part)
                print(
                    "\nWARNING plot_recordings: data",
                    ";".join([part, variable]),
                    "does not contain any spikes.\n",
                )
                plt.text(
                    0.5,
                    0.5,
                    " ".join([part, variable]) + " does not contain any spikes.",
                    va="center",
                    ha="center",
                )
            else:
                t = t * recordings["dt"]  # convert time steps into ms
                mask = (
                    (t >= start_time).astype(int) * (t <= end_time).astype(int)
                ).astype(bool)
                if style != "":
                    plt.scatter(
                        t[mask], n[mask], color=style, marker=".", s=3, linewidth=0.1
                    )
                else:
                    plt.scatter(
                        t[mask], n[mask], color="k", marker=".", s=3, linewidth=0.1
                    )
                plt.xlim(start_time, end_time)
                plt.xlabel("time [ms]")
                plt.ylabel("# neurons")
                plt.title("Spikes " + part)
        elif variable == "spike" and mode == "mean":
            firing_rate = get_pop_rate(
                data,
                end_time - start_time,
                dt=recordings["dt"],
                t_start=start_time,
            )
            plt.plot(times[part], firing_rate, color="k")
            plt.xlim(start_time, end_time)
            plt.xlabel("time [ms]")
            plt.ylabel("Mean firing rate [Hz]")
            plt.title("Mean firing rate " + part)
        elif variable == "spike" and mode == "hybrid":
            t, n = my_raster_plot(data)
            if np.unique(n).size == 0:
                plt.title("Spikes " + part)
                print(
                    "\nWARNING plot_recordings: data",
                    ";".join([part, variable]),
                    "does not contain any spikes.\n",
                )
                plt.text(
                    0.5,
                    0.5,
                    " ".join([part, variable]) + " does not contain any spikes.",
                    va="center",
                    ha="center",
                )
            else:
                t = t * recordings["dt"]  # convert time steps into ms
                mask = (
                    (t >= start_time).astype(int) * (t <= end_time).astype(int)
                ).astype(bool)
                plt.plot(
                    t[mask], n[mask], "k.", markersize=np.sqrt(3), markeredgewidth=0.1
                )
                plt.ylabel("# neurons")
                ax = plt.gca().twinx()
                firing_rate = get_pop_rate(
                    data,
                    end_time - start_time,
                    dt=recordings["dt"],
                    t_start=start_time,
                )
                ax.plot(
                    times[part],
                    firing_rate[
                        np.arange(
                            0, firing_rate.shape[0], recordings[f"{part};period"]
                        ).astype(int)
                    ],
                    color="r",
                )
                plt.ylabel("Mean firing rate [Hz]", color="r")
                ax.tick_params(axis="y", colors="r")
                plt.xlim(start_time, end_time)
                plt.xlabel("time [ms]")
                plt.title("Activity " + part)
        elif variable == "spike" and mode == "single":
            t, n = my_raster_plot(data)
            if np.unique(n).size == 0:
                plt.title("Spikes " + part)
                print(
                    "\nWARNING plot_recordings: data",
                    ";".join([part, variable]),
                    "does not contain any spikes.\n",
                )
                plt.text(
                    0.5,
                    0.5,
                    " ".join([part, variable]) + " does not contain any spikes.",
                    va="center",
                    ha="center",
                )
            elif np.unique(n).size == 1:
                t = t * recordings["dt"]  # convert time steps into ms
                mask = (
                    (t >= start_time).astype(int) * (t <= end_time).astype(int)
                ).astype(bool)
                if style != "":
                    plt.scatter(
                        t[mask], n[mask], color=style, marker="|", s=3000, linewidth=0.1
                    )
                else:
                    plt.scatter(
                        t[mask], n[mask], color="k", marker="|", s=3000, linewidth=0.1
                    )
                plt.xlim(start_time, end_time)
                plt.xlabel("time [ms]")
                plt.ylabel("# neurons")
                plt.title("Spikes " + part)
            else:
                plt.title("Spikes " + part)
                print(
                    "\nWARNING plot_recordings: data",
                    ";".join([part, variable]),
                    'multiple neurons. Mode "single" not available.\n',
                )
                plt.text(
                    0.5,
                    0.5,
                    " ".join([part, variable])
                    + ' multiple neurons. Mode "single" not available.',
                    va="center",
                    ha="center",
                )
        elif variable != "spike" and mode == "line":
            if len(data.shape) == 1:
                plt.plot(
                    times[part], data[start_step[part] : end_step[part]], color="k"
                )
                plt.title("Variable " + part + " " + variable + "(1)")
            elif len(data.shape) == 2:
                for neuron in range(data.shape[1]):
                    plt.plot(
                        times[part],
                        data[start_step[part] : end_step[part], neuron],
                        color="k",
                    )
                    plt.title(
                        "Variable "
                        + part
                        + " "
                        + variable
                        + "("
                        + str(data.shape[1])
                        + ")"
                    )
            else:
                print(
                    "\nERROR plot_recordings: only data of 1D neuron populations accepted,",
                    ";".join([part, variable]),
                    "\n",
                )  ### TODO this seems to not be neccessary, because data of 2D populations have also shape (time, nr_neurons)
            plt.xlim(start_time, end_time)
            plt.xlabel("time [ms]")
        elif variable != "spike" and mode == "mean":
            pass
        elif variable != "spike" and mode == "matrix":
            # data[start_step:end_step,neuron]
            if len(data.shape) == 2:

                vmin = np.min(data[start_step[part] : end_step[part], :])
                vmax = np.max(data[start_step[part] : end_step[part], :])

                plt.title(
                    f"Variable {part} {variable} ({data.shape[1]}) [{ef.sci(vmin)}, {ef.sci(vmax)}]"
                )

                plt.gca().imshow(
                    data[start_step[part] : end_step[part], :].T,
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                    extent=[
                        np.min(times[part]) - 0.5,
                        np.max(times[part]) - 0.5,
                        data.shape[1] - 0.5,
                        -0.5,
                    ],
                    interpolation="none",
                )
                if data.shape[1] == 1:
                    plt.yticks([0])
                else:
                    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            else:
                print(
                    "\nERROR plot_recordings: only 2D data accepted,",
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


def get_number_of_zero_decimals(nr):
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
