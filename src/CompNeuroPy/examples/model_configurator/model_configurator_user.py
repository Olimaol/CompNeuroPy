from ANNarchy import (
    Population,
    Projection,
    setup,
    simulate,
    get_population,
    dt,
)
from CompNeuroPy.neuron_models import (
    poisson_neuron_up_down,
    Izhikevich2003NoisyBaseSNR,
)
from CompNeuroPy import (
    CompNeuroModel,
    CompNeuroMonitors,
    PlotRecordings,
    my_raster_plot,
)
from CompNeuroPy.examples.model_configurator.model_configurator_cnp import (
    ModelConfigurator,
)
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
    cor_exc.rates = params["cor_exc.rates"]
    cor_inh = Population(
        params["cor_inh.size"],
        poisson_neuron_up_down,
        name="cor_inh",
    )
    cor_inh.tau_up = params["cor_inh.tau_up"]
    cor_inh.tau_down = params["cor_inh.tau_down"]
    cor_inh.rates = params["cor_inh.rates"]
    ### BG Populations
    stn = Population(
        params["stn.size"],
        Izhikevich2003NoisyBaseSNR(
            a=params["stn.a"],
            b=params["stn.b"],
            c=params["stn.c"],
            d=params["stn.d"],
            n2=params["stn.n2"],
            n1=params["stn.n1"],
            n0=params["stn.n0"],
            tau_ampa=params["stn.tau_ampa"],
            tau_gaba=params["stn.tau_gaba"],
            E_ampa=params["stn.E_ampa"],
            E_gaba=params["stn.E_gaba"],
            noise=0,
            tau_power=10,
            snr_target=4,
            rate_noise=100,
        ),
        name="stn",
    )
    snr = Population(
        params["snr.size"],
        Izhikevich2003NoisyBaseSNR(
            a=params["snr.a"],
            b=params["snr.b"],
            c=params["snr.c"],
            d=params["snr.d"],
            n2=params["snr.n2"],
            n1=params["snr.n1"],
            n0=params["snr.n0"],
            tau_ampa=params["snr.tau_ampa"],
            tau_gaba=params["snr.tau_gaba"],
            E_ampa=params["snr.E_ampa"],
            E_gaba=params["snr.E_gaba"],
            noise=0,
            tau_power=10,
            snr_target=4,
            rate_noise=100,
        ),
        name="snr",
    )
    gpe = Population(
        params["gpe.size"],
        Izhikevich2003NoisyBaseSNR(
            a=params["gpe.a"],
            b=params["gpe.b"],
            c=params["gpe.c"],
            d=params["gpe.d"],
            n2=params["gpe.n2"],
            n1=params["gpe.n1"],
            n0=params["gpe.n0"],
            tau_ampa=params["gpe.tau_ampa"],
            tau_gaba=params["gpe.tau_gaba"],
            E_ampa=params["gpe.E_ampa"],
            E_gaba=params["gpe.E_gaba"],
            noise=0,
            tau_power=10,
            snr_target=4,
            rate_noise=100,
        ),
        name="gpe",
    )
    thal = Population(
        params["thal.size"],
        Izhikevich2003NoisyBaseSNR(
            a=params["thal.a"],
            b=params["thal.b"],
            c=params["thal.c"],
            d=params["thal.d"],
            n2=params["thal.n2"],
            n1=params["thal.n1"],
            n0=params["thal.n0"],
            tau_ampa=params["thal.tau_ampa"],
            tau_gaba=params["thal.tau_gaba"],
            E_ampa=params["thal.E_ampa"],
            E_gaba=params["thal.E_gaba"],
            noise=0,
            tau_power=10,
            snr_target=4,
            rate_noise=100,
        ),
        name="thal",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_exc__stn = Projection(
        pre=cor_exc,
        post=stn,
        target="ampa",
        name="cor_exc__stn",
    )
    cor_exc__stn.connect_fixed_probability(
        probability=params["cor_exc__stn.probability"], weights=0
    )

    cor_inh__stn = Projection(
        pre=cor_inh,
        post=stn,
        target="gaba",
        name="cor_inh__stn",
    )
    cor_inh__stn.connect_fixed_probability(
        probability=params["cor_inh__stn.probability"], weights=0
    )

    ### stn output
    stn__snr = Projection(
        pre=stn,
        post=snr,
        target="ampa",
        name="stn__snr",
    )
    stn__snr.connect_fixed_probability(
        probability=params["stn__snr.probability"], weights=0
    )
    stn__gpe = Projection(
        pre=stn,
        post=gpe,
        target="ampa",
        name="stn__gpe",
    )
    stn__gpe.connect_fixed_probability(
        probability=params["stn__gpe.probability"], weights=0
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
        #     probability=params["gpe__stn.probability"], weights=0
        # )
        gpe__snr = Projection(
            pre=gpe,
            post=snr,
            target="gaba",
            name="gpe__snr",
        )
        gpe__snr.connect_fixed_probability(
            probability=params["gpe__snr.probability"], weights=0
        )
    ### snr output
    snr__thal = Projection(
        pre=snr,
        post=thal,
        target="gaba",
        name="snr__thal",
    )
    snr__thal.connect_fixed_probability(
        probability=params["snr__thal.probability"], weights=0
    )
    if params["general.more_complex"]:
        snr__snr = Projection(
            pre=snr,
            post=snr,
            target="gaba",
            name="snr__snr",
        )
        snr__snr.connect_fixed_probability(
            probability=params["snr__snr.probability"], weights=0
        )


if __name__ == "__main__":
    #######   PARAMETERS   ######
    params = {}
    #######   POPULATIONS PARAMETERS   ######
    ### cortex / input populations
    params["cor_exc.size"] = 100
    params["cor_exc.tau_up"] = 10
    params["cor_exc.tau_down"] = 30
    params["cor_exc.rates"] = 15
    params["cor_inh.size"] = 100
    params["cor_inh.tau_up"] = 10
    params["cor_inh.tau_down"] = 30
    params["cor_inh.rates"] = 30
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
    ### create or compile have no effect
    setup(dt=0.1)
    model = CompNeuroModel(
        model_creation_function=BGM_part_function,
        model_kwargs={"params": params},
        name="BGM_part_model",
        description="Part of a BGM circuit",
        do_create=False,
    )

    # model.create()
    # mon = CompNeuroMonitors(
    #     {
    #         pop_name: [
    #             "I_noise",
    #             "I_signal",
    #             "I",
    #             "power_I_signal",
    #             "spike",
    #         ]
    #         for pop_name in ["stn"]
    #     }
    # )
    # mon.start()

    # simulate(500)
    # get_population("stn").I_app = 10
    # simulate(500)

    # recordings = mon.get_recordings()
    # recording_times = mon.get_recording_times()

    # PlotRecordings(
    #     recordings=recordings,
    #     recording_times=recording_times,
    #     chunk=0,
    #     shape=(5, 1),
    #     plan={
    #         "position": list(range(1, 5 + 1)),
    #         "compartment": ["stn"] * 5,
    #         "variable": [
    #             "I_noise",
    #             "I_signal",
    #             "I",
    #             "power_I_signal",
    #             "spike",
    #         ],
    #         "format": [
    #             "line",
    #             "line",
    #             "line",
    #             "line",
    #             "hybrid",
    #         ],
    #     },
    #     figname="model_recordings_noise.png",
    #     # time_lim=(495, 505),
    # )
    # quit()

    ### model configurator should get target resting-state firing rates for the
    ### model populations one wants to configure and their afferents as input
    ### TODO allow for target range
    target_firing_rate_dict = {
        "cor_exc": 15,
        "cor_inh": 30,
        "stn": 30,
        "gpe": 50,
        "snr": 60,
        "thal": 20,
    }
    do_not_config_list = ["cor_exc", "cor_inh"]

    ### initialize model_configurator
    model_conf = ModelConfigurator(
        model=model,
        target_firing_rate_dict=target_firing_rate_dict,
        max_psp=0.7,
        do_not_config_list=do_not_config_list,
        print_guide=True,
        I_app_variable="I_app",
        cache=True,
        clear_cache=True,
        log_file="model_configurator.log",
    )

    ### set syn load
    model_conf.set_syn_load(
        syn_load_dict={
            "stn": {"ampa": 1.0, "gaba": 1.0},
            "snr": {"ampa": 1.0, "gaba": 1.0},
            "gpe": {"ampa": 1.0},
            "thal": {"gaba": 1.0},
        },
        syn_contribution_dict={
            "stn": {"ampa": {"cor_exc__stn": 1.0}, "gaba": {"cor_inh__stn": 1.0}},
            "snr": {
                "ampa": {"stn__snr": 1.0},
                "gaba": {"gpe__snr": 1.0, "snr__snr": 1.0},
            },
            "gpe": {"ampa": {"stn__gpe": 1.0}},
            "thal": {"gaba": {"snr__thal": 1.0}},
        },
    )
    print(model_conf._weight_dicts.weight_dict)
    # ### or set weights
    # model_conf.set_weights(
    #     weight_dict={
    #         "cor_exc__stn": 0.14017251511767667,
    #         "cor_inh__stn": 0.3185158233680059,
    #         "stn__snr": 0.1411802181516728,
    #         "gpe__snr": 0.3210042713120005,
    #         "snr__snr": 0.3210042713120005,
    #         "stn__gpe": 0.1411802181516728,
    #         "snr__thal": 1.169558816450918,
    #     }
    # )

    I_base_dict = model_conf.get_base()
    print("I_base:")
    print(I_base_dict)

    model_conf.set_base()

    ### do a test simulation
    mon = CompNeuroMonitors(
        {
            "cor_exc": ["spike"],
            "cor_inh": ["spike"],
            "stn": ["spike"],
            "gpe": ["spike"],
            "snr": ["spike"],
            "thal": ["spike"],
        }
    )
    ### initial simulation
    simulate(1000)
    mon.start()
    ### first simulation with default inputs
    simulate(4000)
    get_population("cor_exc").rates = 0
    ### second simulation with changed inputs
    simulate(2000)

    ### get recordings
    recordings = mon.get_recordings()
    recording_times = mon.get_recording_times()

    ### print rates
    for pop_name in model.populations:
        spike_dict = recordings[0][f"{pop_name};spike"]
        t, n = my_raster_plot(spike_dict)
        nr_spikes_1st = np.sum(
            (t > int(round(1000 / dt()))) * (t < int(round(5000 / dt())))
        )
        nr_spikes_2nd = np.sum((t > int(round(5000 / dt()))))
        rate_1st = nr_spikes_1st / (4 * params[f"{pop_name}.size"])
        rate_2nd = nr_spikes_2nd / (2 * params[f"{pop_name}.size"])
        print(f"pop_name: {pop_name}, rate_1st: {rate_1st}, rate_2nd: {rate_2nd}")

    ### plot recordings
    PlotRecordings(
        figname="model_recordings.png",
        recordings=recordings,
        recording_times=recording_times,
        chunk=0,
        shape=(len(model.populations), 1),
        plan={
            "position": list(range(1, len(model.populations) + 1)),
            "compartment": model.populations,
            "variable": ["spike"] * len(model.populations),
            "format": ["hybrid"] * len(model.populations),
        },
    )
