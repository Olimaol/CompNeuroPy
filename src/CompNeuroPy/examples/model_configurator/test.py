from ANNarchy import Neuron, Population, compile, simulate, get_time, setup, dt
from CompNeuroPy import (
    CompNeuroMonitors,
    PlotRecordings,
    interactive_plot,
    timing_decorator,
)
from CompNeuroPy.neuron_models import Izhikevich2007
import numpy as np

setup(dt=0.1)

neuron = Neuron(
    parameters="""
        C = 100.0,
        k = 0.7,
        v_r = -60.0,
        v_t = -40.0,
        a = 0.03,
        b = -2.0,
        c = -50.0,
        d = 100.0,
        v_peak = 35.0,
        I_app = 0.0,
        tau = 300
    """,
    equations="""
        ### Izhikevich spiking
        I_v        = I_app
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I_v
        du/dt      = a*(b*(v - v_r) - u)
        ### Spike tracking
        tau * dspike_track/dt = - spike_track
        ### I tracking
        tau * dI_track/dt = - I_track + I_v
    """,
    spike="v >= v_peak",
    reset="""
        v = c
        u = u + d
        spike_track = 1.0
    """,
)

pop = Population(100, neuron=neuron, name="pop")


monitors = CompNeuroMonitors(
    mon_dict={"pop;1": ["I_v", "spike_track", "I_track", "spike"]}
)

compile()

monitors.start()

### create an array with amplitudes between -200 and 200
I_app_arr = np.arange(-200, 200, 5)

### create an array with durations between 10 ms and 200 ms
duration_arr = np.arange(10, 200 + 10, 10)

### TODO alwys draw random duration from duration_arr and set I_app of the whole population to the shuffeled I_app_arr
### --> I_app_arr is the size of the population
### --> or just draw from I_app_arr for each neuron... and the population is as large as we want... maybe better
total_duration = 1000
duration_list = []
while sum(duration_list) < total_duration:
    duration_list.append(np.random.choice(duration_arr))

for duration in duration_list:
    pop.I_app = np.random.choice(I_app_arr, size=pop.size)
    simulate(duration)

recordings = monitors.get_recordings()

### concatenate the recorded arrays of all neurons
I_v = np.concatenate(recordings[0]["pop;I_v"].T)
spike_track = np.concatenate(recordings[0]["pop;spike_track"].T)
I_track = np.concatenate(recordings[0]["pop;I_track"].T)
### spikesw vom ersten neuron dann spiekes vom zweiten + simulierte gesamtzeit dann spikes vom dritten + simulierte gesamtzeit *2
spike_times = np.concatenate(
    [
        dt() * np.array(recordings[0]["pop;spike"][i]) + i * sum(duration_list)
        for i in range(pop.size)
    ]
)
### round spike times to full ms to be compatible with the other recording arrays
spike_times = np.round(spike_times, 0)
spikes_onehot = np.zeros(I_track.size)
spikes_onehot[spike_times.astype(int)] = 1


def create_plot(axs, sliders):

    end_time = int(sliders[0]["slider"].val)
    print(end_time)

    ### plot the variables
    ### I tracking
    axs[0].plot(I_v[end_time - 1000 : end_time], label="I_v")
    axs[0].plot(I_track[end_time - 1000 : end_time], label="I_track")
    axs[0].set_ylim(-200, 200)
    axs[0].set_xlim(0, 1000)
    axs[0].legend(loc="upper left")
    ### spike tracking
    axs[1].plot(spike_track[end_time - 1000 : end_time], label="spike_track")
    axs[1].plot(spikes_onehot[end_time - 1000 : end_time], label="spikes")
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(0, 1000)
    axs[1].legend(loc="upper left")


interactive_plot(
    nrows=2,
    ncols=1,
    sliders=[
        {"label": "end time", "valmin": 1000, "valmax": I_track.size, "valinit": 1000},
    ],
    create_plot=create_plot,
)

### TODO: data looks good now train categorization model to predict spikes


# max_len = 1000
# t_list = list(range(-max_len, 0, 1))
# spike_track_list = [0] * max_len
# I_track_list = [0] * max_len
# I_v_list = [0] * max_len


# def track_var(var_list, var_name):
#     """
#     Track a variable of the population pop. The current variable value is stored in the
#     last element of the var_list. The first element is removed.
#     """
#     var_list.append(getattr(pop, var_name)[0])
#     var_list.pop(0)


# def create_plot(axs, sliders, **kwargs):

#     ### update the rates variable
#     pop.I_app = sliders[0]["slider"].val

#     ### plot the variables
#     ### spike tracking
#     axs[0].plot(kwargs["t_list"], kwargs["spike_track_list"], label="f")
#     axs[0].set_ylim(0, 1)
#     axs[0].set_xlim(kwargs["t_list"][0], kwargs["t_list"][-1])
#     ### I tracking
#     axs[1].plot(kwargs["t_list"], kwargs["I_track_list"], label="f_0")
#     axs[1].plot(kwargs["t_list"], kwargs["I_v_list"], label="f_1")
#     axs[1].set_ylim(0, sliders[0]["slider"].val + 20)
#     axs[1].set_xlim(kwargs["t_list"][0], kwargs["t_list"][-1])
#     ### legend
#     axs[1].legend(loc="upper left")


# def update_loop(**kwargs):
#     simulate(1.0)
#     ### update the variable lists
#     track_var(kwargs["spike_track_list"], "spike_track")
#     track_var(kwargs["I_track_list"], "I_track")
#     track_var(kwargs["I_v_list"], "I_v")
#     ### update the time list
#     kwargs["t_list"].append(get_time())
#     kwargs["t_list"].pop(0)


# interactive_plot(
#     nrows=2,
#     ncols=1,
#     sliders=[
#         {"label": "I_app", "valmin": 0.0, "valmax": 200.0, "valinit": 0.0},
#     ],
#     create_plot=lambda axs, sliders: create_plot(
#         axs,
#         sliders,
#         spike_track_list=spike_track_list,
#         I_track_list=I_track_list,
#         I_v_list=I_v_list,
#         t_list=t_list,
#     ),
#     update_loop=lambda: update_loop(
#         spike_track_list=spike_track_list,
#         I_track_list=I_track_list,
#         I_v_list=I_v_list,
#         t_list=t_list,
#     ),
#     figure_frequency=20,
#     update_frequency=100,
# )
