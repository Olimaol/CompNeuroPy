from ANNarchy import Population, Projection
from CompNeuroPy.neuron_models import (
    poisson_neuron_up_down,
    Izhikevich2007_noisy_AMPA,
    Izhikevich2007_noisy_I,
    Izhikevich2007_fsi_noisy_AMPA,
    Izhikevich2003_noisy_AMPA,
    Izhikevich2003_flexible_noisy_AMPA,
    integrator_neuron,
    Izhikevich2007_Corbit_FSI_noisy_AMPA,
    Izhikevich2007_Corbit_FSI_noisy_I,
    poisson_neuron_sin,
    Izhikevich2007_noisy_AMPA_oscillating,
    Izhikevich2003_flexible_noisy_AMPA_oscillating,
    Izhikevich2003_flexible_noisy_I_nonlin,
)
from CompNeuroPy.synapse_models import factor_synapse, factor_synapse_without_max


def BGM_v01(self):
    """
    original model structure from Goenner et al. (2021)
    Goenner, L., Maith, O., Koulouri, I., Baladron, J., & Hamker, F. H. (2021). A spiking model of basal ganglia dynamics in stopping behavior supported by arkypallidal neurons. European Journal of Neuroscience, 53(7), 2296-2321.
    """
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor_go = Population(
        self.params["cor_go.size"], poisson_neuron_up_down, name="cor_go"
    )
    cor_pause = Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = Population(
        self.params["str_d1.size"], Izhikevich2007_noisy_AMPA, name="str_d1"
    )
    str_d2 = Population(
        self.params["str_d2.size"], Izhikevich2007_noisy_AMPA, name="str_d2"
    )
    str_fsi = Population(
        self.params["str_fsi.size"], Izhikevich2007_fsi_noisy_AMPA, name="str_fsi"
    )
    ### BG Populations
    stn = Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_proto",
    )
    gpe_arky = Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = Population(self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal")
    ### integrator Neurons
    integrator_go = Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )
    ### cortex pause output
    cor_pause__stn = Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )
    gpe_proto__str_fsi = Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = Projection(
        pre=thal,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_fsi",
    )


def BGM_v02(self):
    """
    difference to Goenner et al. (2021):
    str_fsi neuron model:
        new neuron model = fit to Hodgkin and Huxley neuron model from Corbit et al. (2016)
        Corbit, V. L., Whalen, T. C., Zitelli, K. T., Crilly, S. Y., Rubin, J. E., & Gittis, A. H. (2016). Pallidostriatal projections promote β oscillations in a dopamine-depleted biophysical network model. Journal of Neuroscience, 36(20), 5556-5571.
    """
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor_go = Population(
        self.params["cor_go.size"], poisson_neuron_up_down, name="cor_go"
    )
    cor_pause = Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = Population(
        self.params["str_d1.size"], Izhikevich2007_noisy_AMPA, name="str_d1"
    )
    str_d2 = Population(
        self.params["str_d2.size"], Izhikevich2007_noisy_AMPA, name="str_d2"
    )
    str_fsi = Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    stn = Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_proto",
    )
    gpe_arky = Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = Population(self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal")
    ### integrator Neurons
    integrator_go = Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )
    ### cortex pause output
    cor_pause__stn = Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )
    gpe_proto__str_fsi = Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = Projection(
        pre=thal,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_fsi",
    )


def BGM_vTEST(self):
    """
    difference to Goenner et al. (2021):
    cor_go neuron model:
        instead of poisson_neuron_up down it's poisson_neuron_sin
        --> can specify sinus oscillation as cor_go activity
    """
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor_go = Population(self.params["cor_go.size"], poisson_neuron_sin, name="cor_go")
    cor_pause = Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = Population(
        self.params["str_d1.size"], Izhikevich2007_noisy_AMPA, name="str_d1"
    )
    str_d2 = Population(
        self.params["str_d2.size"], Izhikevich2007_noisy_AMPA, name="str_d2"
    )
    str_fsi = Population(
        self.params["str_fsi.size"], Izhikevich2007_fsi_noisy_AMPA, name="str_fsi"
    )
    ### BG Populations
    stn = Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_proto",
    )
    gpe_arky = Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = Population(self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal")
    ### integrator Neurons
    integrator_go = Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )
    ### cortex pause output
    cor_pause__stn = Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )
    gpe_proto__str_fsi = Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = Projection(
        pre=thal,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_fsi",
    )


def BGM_v03(self):
    """
    difference to Goenner et al. (2021):
    str_fsi neuron model:
        new neuron model = fit to Hodgkin and Huxley neuron model from Corbit et al. (2016)
        Corbit, V. L., Whalen, T. C., Zitelli, K. T., Crilly, S. Y., Rubin, J. E., & Gittis, A. H. (2016). Pallidostriatal projections promote β oscillations in a dopamine-depleted biophysical network model. Journal of Neuroscience, 36(20), 5556-5571.

    difference to BGM_02 : added oscillation-term in Izhikevich2007_noisy_AMPA_oscillating-> replaced in str_d1 and str_d2 based on
    Corbit, V. L., Whalen, T. C., Zitelli, K. T., Crilly, S. Y., Rubin, J. E., & Gittis, A. H. (2016). Pallidostriatal projections promote β oscillations in a dopamine-depleted biophysical network model. Journal of Neuroscience, 36(20), 5556-5571.
    """
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor_go = Population(
        self.params["cor_go.size"], poisson_neuron_up_down, name="cor_go"
    )
    cor_pause = Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = Population(
        self.params["str_d1.size"],
        Izhikevich2007_noisy_AMPA_oscillating,
        name="str_d1",  # NEW NEURON MODEL
    )
    str_d2 = Population(
        self.params["str_d2.size"],
        Izhikevich2007_noisy_AMPA_oscillating,
        name="str_d2",  # NEW NEURON MODEL
    )
    str_fsi = Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    stn = Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_proto",
    )
    gpe_arky = Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = Population(self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal")
    ### integrator Neurons
    integrator_go = Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )
    ### cortex pause output
    cor_pause__stn = Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )
    gpe_proto__str_fsi = Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = Projection(
        pre=thal,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_fsi",
    )


def BGM_v04(self):
    """
    replication of small pallido-striatal network by Corbit et al.(2016) with noise -> switched off other connections
    """
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor_go = Population(
        self.params["cor_go.size"], poisson_neuron_up_down, name="cor_go"
    )
    cor_pause = Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = Population(
        self.params["str_d1.size"], Izhikevich2007_noisy_AMPA, name="str_d1"
    )
    str_d2 = Population(
        self.params["str_d2.size"], Izhikevich2007_noisy_AMPA, name="str_d2"
    )
    str_fsi = Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    stn = Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_proto",
    )
    gpe_arky = Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = Population(self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal")
    ### integrator Neurons
    integrator_go = Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )

    cor_stop__gpe_proto = Projection(  # NEW !
        pre=cor_stop,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_proto",
    )
    ### cortex pause output
    cor_pause__stn = Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )

    gpe_proto__gpe_proto = Projection(
        pre=gpe_proto,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_proto",
    )

    gpe_arky__gpe_arky = Projection(  # NEW, not in original BGM
        pre=gpe_arky,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_arky",
    )

    gpe_proto__str_fsi = Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = Projection(
        pre=thal,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_fsi",
    )


def BGM_v04oliver(self):
    """
    replication of small pallido-striatal network by Corbit et al.(2016) with noise -> switched off other connections
    """
    #######   POPULATIONS   ######
    ### Str Populations
    str_d2 = Population(
        self.params["str_d2.size"],
        Izhikevich2007_noisy_AMPA,
        name="str_d2",
    )
    str_fsi = Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    gpe_arky = Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )

    ######   PROJECTIONS   ######
    ### str d2 output
    str_d2__gpe_arky = Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_d2__gpe_arky",
    )
    str_d2__str_d2 = Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d2 = Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_fsi__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_fsi = Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_arky = Projection(  # NEW, not in original BGM
        pre=gpe_arky,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="gpe_arky__gpe_arky",
    )


def BGM_v04newgpe(self):
    """
    replication of small pallido-striatal network by Corbit et al.(2016)

    new gpe neuron model without refractory! (refitted data as in Goenner et al. 2021)
    also use now gpe_proto and not arky (based on more recent literatur about connectivity, see Lindi et al. 2023)
    """
    #######   POPULATIONS   ######
    ### Str Populations
    str_d2 = Population(
        self.params["str_d2.size"],
        Izhikevich2007_noisy_I,
        name="str_d2",
    )
    str_fsi = Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_I,
        name="str_fsi",
    )
    ### BG Populations
    gpe_proto = Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_I_nonlin,
        name="gpe_proto",
    )

    ######   PROJECTIONS   ######
    ### str d2 output
    str_d2__gpe_proto = Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_d2__gpe_proto",
    )
    str_d2__str_d2 = Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d2 = Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_fsi__str_fsi",
    )
    ### gpe proto output
    gpe_proto__str_fsi = Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="gpe_proto__str_fsi",
    )
    gpe_proto__gpe_proto = Projection(  # NEW, not in original BGM
        pre=gpe_proto,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="gpe_proto__gpe_proto",
    )


def BGM_v05(self):
    """
    replication of small pallido-striatal network by Corbit et al.(2016) with noise -> switched off other connections
    NEW : oscillation term in STR_D2, GPe Proto
    """
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor_go = Population(
        self.params["cor_go.size"], poisson_neuron_up_down, name="cor_go"
    )
    cor_pause = Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = Population(
        self.params["str_d1.size"],
        Izhikevich2007_noisy_AMPA_oscillating,
        name="str_d1",  # NEW NEURON MODEL
    )
    str_d2 = Population(
        self.params["str_d2.size"],
        Izhikevich2007_noisy_AMPA_oscillating,
        name="str_d2",  # NEW NEURON MODEL
    )
    str_fsi = Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    stn = Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,  # NEW NEURON MODEL
        name="gpe_proto",
    )
    gpe_arky = Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA_oscillating,
        name="gpe_arky",
    )
    gpe_cp = Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = Population(self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal")
    ### integrator Neurons
    integrator_go = Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )

    cor_stop__gpe_proto = Projection(  # NEW !
        pre=cor_stop,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_proto",
    )
    ### cortex pause output
    cor_pause__stn = Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )

    gpe_proto__gpe_proto = Projection(  # NEW, not in original BGM
        pre=gpe_proto,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_proto",
    )

    gpe_arky__gpe_arky = Projection(  # NEW, not in original BGM
        pre=gpe_arky,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_arky",
    )

    gpe_proto__str_fsi = Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = Projection(
        pre=thal,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_fsi",
    )


def BGM_v06(self):
    """
    replication of small pallido-striatal network by Corbit et al.(2016) with noise -> switched off other connections
    NEW : instead of oscillation term in STR_D2, GPe Proto, oscillatory poisson input for striatum and gpe -> strd2, gpe_proto original neuron models, cor_go, cor_stop now poisson_neuron_sin
    """
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor_go = Population(self.params["cor_go.size"], poisson_neuron_sin, name="cor_go")
    cor_pause = Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = Population(
        self.params["cor_stop.size"], poisson_neuron_sin, name="cor_stop"
    )
    ### Str Populations
    str_d1 = Population(
        self.params["str_d1.size"],
        Izhikevich2007_noisy_AMPA,
        name="str_d1",  # NEW NEURON MODEL
    )
    str_d2 = Population(
        self.params["str_d2.size"],
        Izhikevich2007_noisy_AMPA,
        name="str_d2",  # NEW NEURON MODEL
    )
    str_fsi = Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    stn = Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,  # NEW NEURON MODEL
        name="gpe_proto",
    )
    gpe_arky = Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = Population(self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal")
    ### integrator Neurons
    integrator_go = Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )

    cor_stop__gpe_proto = Projection(  # NEW !
        pre=cor_stop,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_proto",
    )
    ### cortex pause output
    cor_pause__stn = Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )

    gpe_proto__gpe_proto = Projection(  # NEW, not in original BGM
        pre=gpe_proto,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_proto",
    )

    gpe_arky__gpe_arky = Projection(  # NEW, not in original BGM
        pre=gpe_arky,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_arky",
    )

    gpe_proto__str_fsi = Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = Projection(
        pre=thal,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_fsi",
    )
