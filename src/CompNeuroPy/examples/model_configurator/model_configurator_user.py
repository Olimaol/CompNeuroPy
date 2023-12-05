from ANNarchy import (
    Population,
    Projection,
    setup,
    simulate,
    get_population,
    get_projection,
    dt,
)
from CompNeuroPy.neuron_models import (
    poisson_neuron_up_down,
    Izhikevich2003_flexible_noisy_I,
)
from CompNeuroPy import generate_model, Monitors, plot_recordings, my_raster_plot
from model_configurator_cnp import model_configurator
import matplotlib.pyplot as plt
import numpy as np


def BGM_part_function(params):
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor_exc = Population(
        params["cor_exc.size"],
        poisson_neuron_up_down,
        name="cor_exc",
    )
    cor_exc.tau_up = params["cor_exc.tau_up"]
    cor_exc.tau_down = params["cor_exc.tau_down"]
    cor_inh = Population(
        params["cor_inh.size"],
        poisson_neuron_up_down,
        name="cor_inh",
    )
    cor_inh.tau_up = params["cor_inh.tau_up"]
    cor_inh.tau_down = params["cor_inh.tau_down"]
    ### BG Populations
    stn = Population(
        params["stn.size"],
        Izhikevich2003_flexible_noisy_I,
        name="stn",
    )
    stn.a = params["stn.a"]
    stn.b = params["stn.b"]
    stn.c = params["stn.c"]
    stn.d = params["stn.d"]
    stn.n2 = params["stn.n2"]
    stn.n1 = params["stn.n1"]
    stn.n0 = params["stn.n0"]
    stn.tau_ampa = params["stn.tau_ampa"]
    stn.tau_gaba = params["stn.tau_gaba"]
    stn.E_ampa = params["stn.E_ampa"]
    stn.E_gaba = params["stn.E_gaba"]
    snr = Population(
        params["snr.size"],
        Izhikevich2003_flexible_noisy_I,
        name="snr",
    )
    snr.a = params["snr.a"]
    snr.b = params["snr.b"]
    snr.c = params["snr.c"]
    snr.d = params["snr.d"]
    snr.n2 = params["snr.n2"]
    snr.n1 = params["snr.n1"]
    snr.n0 = params["snr.n0"]
    snr.tau_ampa = params["snr.tau_ampa"]
    snr.tau_gaba = params["snr.tau_gaba"]
    snr.E_ampa = params["snr.E_ampa"]
    snr.E_gaba = params["snr.E_gaba"]
    gpe = Population(
        params["gpe.size"],
        Izhikevich2003_flexible_noisy_I,
        name="gpe",
    )
    gpe.a = params["gpe.a"]
    gpe.b = params["gpe.b"]
    gpe.c = params["gpe.c"]
    gpe.d = params["gpe.d"]
    gpe.n2 = params["gpe.n2"]
    gpe.n1 = params["gpe.n1"]
    gpe.n0 = params["gpe.n0"]
    gpe.tau_ampa = params["gpe.tau_ampa"]
    gpe.tau_gaba = params["gpe.tau_gaba"]
    gpe.E_ampa = params["gpe.E_ampa"]
    gpe.E_gaba = params["gpe.E_gaba"]
    thal = Population(
        params["thal.size"],
        Izhikevich2003_flexible_noisy_I,
        name="thal",
    )
    thal.a = params["thal.a"]
    thal.b = params["thal.b"]
    thal.c = params["thal.c"]
    thal.d = params["thal.d"]
    thal.n2 = params["thal.n2"]
    thal.n1 = params["thal.n1"]
    thal.n0 = params["thal.n0"]
    thal.tau_ampa = params["thal.tau_ampa"]
    thal.tau_gaba = params["thal.tau_gaba"]
    thal.E_ampa = params["thal.E_ampa"]
    thal.E_gaba = params["thal.E_gaba"]

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_exc__stn = Projection(
        pre=cor_exc,
        post=stn,
        target="ampa",
        name="cor_exc__stn",
    )
    cor_exc__stn.connect_fixed_probability(
        probability=params["cor_exc__stn.probability"], weights=1
    )

    cor_inh__stn = Projection(
        pre=cor_inh,
        post=stn,
        target="gaba",
        name="cor_inh__stn",
    )
    cor_inh__stn.connect_fixed_probability(
        probability=params["cor_inh__stn.probability"], weights=1
    )

    ### stn output
    stn__snr = Projection(
        pre=stn,
        post=snr,
        target="ampa",
        name="stn__snr",
    )
    stn__snr.connect_fixed_probability(
        probability=params["stn__snr.probability"], weights=1
    )
    stn__gpe = Projection(
        pre=stn,
        post=gpe,
        target="ampa",
        name="stn__gpe",
    )
    stn__gpe.connect_fixed_probability(
        probability=params["stn__gpe.probability"], weights=1
    )
    ### gpe proto output
    if params["general.more_complex"]:
        # gpe__stn = Projection(
        #     pre=gpe,
        #     post=stn,
        #     target="gaba",
        #     name="gpe__stn",
        # )
        # gpe__stn.connect_fixed_probability(
        #     probability=params["gpe__stn.probability"], weights=1
        # )
        gpe__snr = Projection(
            pre=gpe,
            post=snr,
            target="gaba",
            name="gpe__snr",
        )
        gpe__snr.connect_fixed_probability(
            probability=params["gpe__snr.probability"], weights=1
        )
    ### snr output
    snr__thal = Projection(
        pre=snr,
        post=thal,
        target="gaba",
        name="snr__thal",
    )
    snr__thal.connect_fixed_probability(
        probability=params["snr__thal.probability"], weights=1
    )
    if params["general.more_complex"]:
        snr__snr = Projection(
            pre=snr,
            post=snr,
            target="gaba",
            name="snr__snr",
        )
        snr__snr.connect_fixed_probability(
            probability=params["snr__snr.probability"], weights=1
        )


if __name__ == "__main__":
    #######   PARAMETERS   ######
    params = {}
    #######   POPULATIONS PARAMETERS   ######
    ### cortex / input populations
    params["cor_exc.size"] = 100
    params["cor_exc.tau_up"] = 10
    params["cor_exc.tau_down"] = 30
    params["cor_inh.size"] = 100
    params["cor_inh.tau_up"] = 10
    params["cor_inh.tau_down"] = 30
    ### BG Populations
    params["snr.size"] = 100
    params["snr.a"] = 0.005
    params["snr.b"] = 0.585
    params["snr.c"] = -65
    params["snr.d"] = 4
    params["snr.n2"] = 0.04
    params["snr.n1"] = 5
    params["snr.n0"] = 140
    params["snr.tau_ampa"] = 10
    params["snr.tau_gaba"] = 10
    params["snr.E_ampa"] = 0
    params["snr.E_gaba"] = -90
    params["stn.size"] = 50
    params["stn.a"] = 0.005
    params["stn.b"] = 0.265
    params["stn.c"] = -65
    params["stn.d"] = 2
    params["stn.n2"] = 0.04
    params["stn.n1"] = 5
    params["stn.n0"] = 140
    params["stn.tau_ampa"] = 10
    params["stn.tau_gaba"] = 10
    params["stn.E_ampa"] = 0
    params["stn.E_gaba"] = -90
    params["gpe.size"] = 100
    params["gpe.a"] = params["snr.a"]  # 0.039191890241715294
    params["gpe.b"] = params["snr.b"]  # 0.000548238111291427
    params["gpe.c"] = params["snr.c"]  # -49.88014418530518
    params["gpe.d"] = params["snr.d"]  # 108.0208225074675
    params["gpe.n2"] = params["snr.n2"]  # 0.08899515481507077
    params["gpe.n1"] = params["snr.n1"]  # 1.1929776239208976
    params["gpe.n0"] = params["snr.n0"]  # 24.2219699019072
    params["gpe.tau_ampa"] = 10
    params["gpe.tau_gaba"] = 10
    params["gpe.E_ampa"] = 0
    params["gpe.E_gaba"] = -90
    params["thal.size"] = 100
    params["thal.a"] = 0.02
    params["thal.b"] = 0.2
    params["thal.c"] = -65
    params["thal.d"] = 6
    params["thal.n2"] = 0.04
    params["thal.n1"] = 5
    params["thal.n0"] = 140
    params["thal.tau_ampa"] = 10
    params["thal.tau_gaba"] = 10
    params["thal.E_ampa"] = 0
    params["thal.E_gaba"] = -90
    #######   PROJECTIONS PARAMETERS   ######
    params["general.more_complex"] = True
    params["cor_exc__stn.probability"] = 0.2
    params["cor_inh__stn.probability"] = 0.2
    params["stn__snr.probability"] = 0.3
    params["stn__gpe.probability"] = 0.35
    params["gpe__stn.probability"] = 0.35
    params["gpe__snr.probability"] = 0.4
    params["snr__thal.probability"] = 0.6
    params["snr__snr.probability"] = 0.6

    ### create model which should be configurated
    ### if you create or compile has no effect
    setup(dt=0.1)
    model = generate_model(
        model_creation_function=BGM_part_function,
        model_kwargs={"params": params},
        name="BGM_part_model",
        description="Part of a BGM circuit",
        do_create=False,
    )

    ### model configurator should get target resting-state firing rates for the
    ### model populations one wants to configure and their afferents as input
    target_firing_rate_dict = {
        "cor_exc": 15,
        "cor_inh": 30,
        "stn": 30,
        "gpe": 50,
        "snr": 60,
        "thal": 5,
    }
    do_not_config_list = ["cor_exc", "cor_inh"]

    ### initialize model_configurator
    model_conf = model_configurator(
        model,
        target_firing_rate_dict,
        do_not_config_list=do_not_config_list,
        print_guide=True,
        I_app_variable="I_app",
        interpolation_grid_points=36,
    )

    ### obtain the maximum synaptic loads for the populations and the
    ### maximum weights of their afferent projections
    model_conf.get_max_syn(clear=False)

    ### now either set weights directly
    ### or define synaptic load of populations
    synaptic_load_dict = {
        "stn": [0.3, 0.3],
        "gpe": [0.4],
        "snr": [0.5, 0.3],
        "thal": [0.1],
    }
    ### and define the contributions of their afferent projections
    synaptic_contribution_dict = {"snr": {"gaba": {"gpe__snr": 0.7, "snr__snr": 0.3}}}
    synaptic_contribution_dict = model_conf.set_syn_load(
        synaptic_load_dict,
        synaptic_contribution_dict,
    )

    ### after setting the weights i.e. the synaptic load/contributions
    ### get the baseline currents
    I_base_dict = model_conf.set_base(I_base_variable="base_mean")
    print("user I_base:")
    print(I_base_dict)
    print("model cor_stn_weight:")
    print(get_projection("cor_exc__stn").w)

    for pop_name in model.populations:
        if "cor" in pop_name:
            continue
        get_population(pop_name).rate_base_noise = 10
        get_population(pop_name).base_noise = max(
            [
                model_conf.I_app_max_dict[pop_name] * 0.02,
                abs(get_population(pop_name).base_mean[0]) * 0.02,
            ]
        )
        print(f"pop_name: {pop_name}")
        print(f"base_mean: {get_population(pop_name).base_mean[0]}")
        print(f"base_noise: {get_population(pop_name).base_noise[0]}")
        print(f"rate_base_noise: {get_population(pop_name).rate_base_noise[0]}\n")

    ### do a test simulation
    mon = Monitors(
        {
            "pop;cor_exc": ["spike"],
            "pop;cor_inh": ["spike"],
            "pop;stn": ["spike", "g_ampa", "g_gaba"],
            "pop;gpe": ["spike", "g_ampa", "g_gaba"],
            "pop;snr": ["spike", "g_ampa", "g_gaba"],
            "pop;thal": ["spike", "g_ampa", "g_gaba"],
        }
    )
    get_population("cor_exc").rates = target_firing_rate_dict["cor_exc"]
    get_population("cor_inh").rates = target_firing_rate_dict["cor_inh"]
    simulate(1000)
    mon.start()
    simulate(4000)
    get_population("cor_exc").rates = 0
    get_population("cor_inh").rates = 0
    simulate(2000)

    ### get recordings
    recordings = mon.get_recordings()
    recording_times = mon.get_recording_times()

    stn_g_ampa = recordings[0]["stn;g_ampa"]
    stn_g_gaba = recordings[0]["stn;g_gaba"]
    cor_spike = recordings[0]["cor_exc;spike"]
    cor_spike_arr = np.zeros(stn_g_ampa.shape[0])
    t, n = my_raster_plot(cor_spike)
    values, counts = np.unique(t - 10000, return_counts=True)
    t = values.astype(int)
    cor_spike_arr[t] = counts
    plt.figure(figsize=(6.4, 4.8 * 2))
    plt.subplot(211)
    plt.ylabel("g_ampa")
    plt.plot(stn_g_ampa[:, 0], "k.")
    plt.subplot(212)
    plt.ylabel("g_gaba")
    plt.plot(stn_g_gaba[:, 0], "k.")
    plt.tight_layout()
    plt.savefig("stn_input_model.png", dpi=300)
    plt.close("all")

    ### print rates
    for pop_name in model.populations:
        spike_dict = recordings[0][f"{pop_name};spike"]
        t, n = my_raster_plot(spike_dict)
        nr_spikes_1st = np.sum(
            (t > int(round(1000 / dt(), 0))) * (t < int(round(5000 / dt(), 0)))
        )
        nr_spikes_2nd = np.sum((t > int(round(5000 / dt(), 0))))
        rate_1st = nr_spikes_1st / (4 * params[f"{pop_name}.size"])
        rate_2nd = nr_spikes_2nd / (2 * params[f"{pop_name}.size"])
        print(f"pop_name: {pop_name}, rate_1st: {rate_1st}, rate_2nd: {rate_2nd}")

    ### plot recordings
    plot_recordings(
        figname="model_recordings.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=0,
        shape=(5, 3),
        plan=[
            "1;cor_exc;spike;hybrid",
            "2;cor_inh;spike;hybrid",
            "4;stn;spike;hybrid",
            "5;stn;g_ampa;line",
            "6;stn;g_gaba;line",
            "7;gpe;spike;hybrid",
            "8;gpe;g_ampa;line",
            "9;gpe;g_gaba;line",
            "10;snr;spike;hybrid",
            "11;snr;g_ampa;line",
            "12;snr;g_gaba;line",
            "13;thal;spike;hybrid",
            "14;thal;g_ampa;line",
            "15;thal;g_gaba;line",
        ],
    )

    # TODO
    # it seems that there are problems with snr
    # it even gets wotse if I deactivate the lateral inhib
    # maybe check which g_ampa, g_gaba are expected based on weights and which actually are present
