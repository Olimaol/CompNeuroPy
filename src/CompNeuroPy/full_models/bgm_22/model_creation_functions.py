from CompNeuroPy import ann
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
    Izhikevich2003FixedNoisyAmpa,
    Izhikevich2003NoisyBaseNonlin,
    Izhikevich2007Humphries2009SPND1,
    Izhikevich2007Humphries2009SPND2,
    Izhikevich2007Humphries2009FSI,
)
from CompNeuroPy.synapse_models import factor_synapse, factor_synapse_without_max
from CompNeuroPy.striatal_microcircuit.microcircuit import Microcircuit
from CompNeuroPy.striatal_microcircuit.cortical_inputs import CorticalInputs
import numpy as np


def BGM_v01(self):
    """
    original model structure from Goenner et al. (2021)
    Goenner, L., Maith, O., Koulouri, I., Baladron, J., & Hamker, F. H. (2021). A spiking model of basal ganglia dynamics in stopping behavior supported by arkypallidal neurons. European Journal of Neuroscience, 53(7), 2296-2321.
    """
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor_go = ann.Population(
        self.params["cor_go.size"], poisson_neuron_up_down, name="cor_go"
    )
    cor_pause = ann.Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = ann.Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = ann.Population(
        self.params["str_d1.size"], Izhikevich2007_noisy_AMPA, name="str_d1"
    )
    str_d2 = ann.Population(
        self.params["str_d2.size"], Izhikevich2007_noisy_AMPA, name="str_d2"
    )
    str_fsi = ann.Population(
        self.params["str_fsi.size"], Izhikevich2007_fsi_noisy_AMPA, name="str_fsi"
    )
    ### BG Populations
    stn = ann.Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = ann.Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = ann.Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_proto",
    )
    gpe_arky = ann.Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = ann.Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = ann.Population(
        self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal"
    )
    ### integrator Neurons
    integrator_go = ann.Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = ann.Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = ann.Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = ann.Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = ann.Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = ann.Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = ann.Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = ann.Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )
    ### cortex pause output
    cor_pause__stn = ann.Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = ann.Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = ann.Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = ann.Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = ann.Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = ann.Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = ann.Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = ann.Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = ann.Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = ann.Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = ann.Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = ann.Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = ann.Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = ann.Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = ann.Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = ann.Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = ann.Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = ann.Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = ann.Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = ann.Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = ann.Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )
    gpe_proto__str_fsi = ann.Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = ann.Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = ann.Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = ann.Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = ann.Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = ann.Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = ann.Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = ann.Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = ann.Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = ann.Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = ann.Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = ann.Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = ann.Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = ann.Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = ann.Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = ann.Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = ann.Projection(
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
    cor_go = ann.Population(
        self.params["cor_go.size"], poisson_neuron_up_down, name="cor_go"
    )
    cor_pause = ann.Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = ann.Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = ann.Population(
        self.params["str_d1.size"], Izhikevich2007_noisy_AMPA, name="str_d1"
    )
    str_d2 = ann.Population(
        self.params["str_d2.size"], Izhikevich2007_noisy_AMPA, name="str_d2"
    )
    str_fsi = ann.Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    stn = ann.Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = ann.Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = ann.Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_proto",
    )
    gpe_arky = ann.Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = ann.Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = ann.Population(
        self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal"
    )
    ### integrator Neurons
    integrator_go = ann.Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = ann.Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = ann.Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = ann.Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = ann.Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = ann.Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = ann.Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = ann.Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )
    ### cortex pause output
    cor_pause__stn = ann.Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = ann.Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = ann.Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = ann.Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = ann.Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = ann.Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = ann.Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = ann.Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = ann.Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = ann.Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = ann.Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = ann.Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = ann.Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = ann.Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = ann.Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = ann.Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = ann.Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = ann.Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = ann.Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = ann.Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = ann.Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )
    gpe_proto__str_fsi = ann.Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = ann.Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = ann.Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = ann.Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = ann.Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = ann.Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = ann.Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = ann.Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = ann.Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = ann.Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = ann.Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = ann.Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = ann.Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = ann.Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = ann.Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = ann.Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = ann.Projection(
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
    cor_go = ann.Population(
        self.params["cor_go.size"], poisson_neuron_sin, name="cor_go"
    )
    cor_pause = ann.Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = ann.Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = ann.Population(
        self.params["str_d1.size"], Izhikevich2007_noisy_AMPA, name="str_d1"
    )
    str_d2 = ann.Population(
        self.params["str_d2.size"], Izhikevich2007_noisy_AMPA, name="str_d2"
    )
    str_fsi = ann.Population(
        self.params["str_fsi.size"], Izhikevich2007_fsi_noisy_AMPA, name="str_fsi"
    )
    ### BG Populations
    stn = ann.Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = ann.Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = ann.Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_proto",
    )
    gpe_arky = ann.Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = ann.Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = ann.Population(
        self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal"
    )
    ### integrator Neurons
    integrator_go = ann.Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = ann.Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = ann.Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = ann.Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = ann.Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = ann.Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = ann.Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = ann.Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )
    ### cortex pause output
    cor_pause__stn = ann.Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = ann.Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = ann.Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = ann.Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = ann.Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = ann.Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = ann.Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = ann.Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = ann.Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = ann.Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = ann.Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = ann.Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = ann.Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = ann.Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = ann.Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = ann.Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = ann.Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = ann.Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = ann.Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = ann.Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = ann.Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )
    gpe_proto__str_fsi = ann.Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = ann.Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = ann.Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = ann.Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = ann.Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = ann.Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = ann.Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = ann.Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = ann.Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = ann.Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = ann.Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = ann.Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = ann.Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = ann.Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = ann.Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = ann.Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = ann.Projection(
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
    cor_go = ann.Population(
        self.params["cor_go.size"], poisson_neuron_up_down, name="cor_go"
    )
    cor_pause = ann.Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = ann.Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = ann.Population(
        self.params["str_d1.size"],
        Izhikevich2007_noisy_AMPA_oscillating,
        name="str_d1",  # NEW NEURON MODEL
    )
    str_d2 = ann.Population(
        self.params["str_d2.size"],
        Izhikevich2007_noisy_AMPA_oscillating,
        name="str_d2",  # NEW NEURON MODEL
    )
    str_fsi = ann.Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    stn = ann.Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = ann.Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = ann.Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_proto",
    )
    gpe_arky = ann.Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = ann.Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = ann.Population(
        self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal"
    )
    ### integrator Neurons
    integrator_go = ann.Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = ann.Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = ann.Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = ann.Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = ann.Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = ann.Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = ann.Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = ann.Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )
    ### cortex pause output
    cor_pause__stn = ann.Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = ann.Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = ann.Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = ann.Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = ann.Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = ann.Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = ann.Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = ann.Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = ann.Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = ann.Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = ann.Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = ann.Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = ann.Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = ann.Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = ann.Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = ann.Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = ann.Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = ann.Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = ann.Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = ann.Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = ann.Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )
    gpe_proto__str_fsi = ann.Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = ann.Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = ann.Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = ann.Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = ann.Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = ann.Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = ann.Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = ann.Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = ann.Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = ann.Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = ann.Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = ann.Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = ann.Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = ann.Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = ann.Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = ann.Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = ann.Projection(
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
    cor_go = ann.Population(
        self.params["cor_go.size"], poisson_neuron_up_down, name="cor_go"
    )
    cor_pause = ann.Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = ann.Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = ann.Population(
        self.params["str_d1.size"], Izhikevich2007_noisy_AMPA, name="str_d1"
    )
    str_d2 = ann.Population(
        self.params["str_d2.size"], Izhikevich2007_noisy_AMPA, name="str_d2"
    )
    str_fsi = ann.Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    stn = ann.Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = ann.Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = ann.Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_proto",
    )
    gpe_arky = ann.Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = ann.Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = ann.Population(
        self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal"
    )
    ### integrator Neurons
    integrator_go = ann.Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = ann.Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = ann.Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = ann.Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = ann.Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = ann.Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = ann.Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = ann.Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )

    cor_stop__gpe_proto = ann.Projection(  # NEW !
        pre=cor_stop,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_proto",
    )
    ### cortex pause output
    cor_pause__stn = ann.Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = ann.Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = ann.Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = ann.Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = ann.Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = ann.Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = ann.Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = ann.Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = ann.Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = ann.Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = ann.Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = ann.Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = ann.Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = ann.Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = ann.Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = ann.Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = ann.Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = ann.Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = ann.Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = ann.Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = ann.Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )

    gpe_proto__gpe_proto = ann.Projection(
        pre=gpe_proto,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_proto",
    )

    gpe_arky__gpe_arky = ann.Projection(  # NEW, not in original BGM
        pre=gpe_arky,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_arky",
    )

    gpe_proto__str_fsi = ann.Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = ann.Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = ann.Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = ann.Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = ann.Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = ann.Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = ann.Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = ann.Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = ann.Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = ann.Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = ann.Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = ann.Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = ann.Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = ann.Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = ann.Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = ann.Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = ann.Projection(
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
    str_d2 = ann.Population(
        self.params["str_d2.size"],
        Izhikevich2007_noisy_AMPA,
        name="str_d2",
    )
    str_fsi = ann.Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    gpe_arky = ann.Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )

    ######   PROJECTIONS   ######
    ### str d2 output
    str_d2__gpe_arky = ann.Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_d2__gpe_arky",
    )
    str_d2__str_d2 = ann.Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d2 = ann.Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = ann.Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_fsi__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_fsi = ann.Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_arky = ann.Projection(  # NEW, not in original BGM
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
    str_d2 = ann.Population(
        self.params["str_d2.size"],
        Izhikevich2007_noisy_I,
        name="str_d2",
    )
    str_fsi = ann.Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_I,
        name="str_fsi",
    )
    ### BG Populations
    gpe_proto = ann.Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_I_nonlin,
        name="gpe_proto",
    )

    ######   PROJECTIONS   ######
    ### str d2 output
    str_d2__gpe_proto = ann.Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_d2__gpe_proto",
    )
    str_d2__str_d2 = ann.Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d2 = ann.Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = ann.Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="str_fsi__str_fsi",
    )
    ### gpe proto output
    gpe_proto__str_fsi = ann.Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse_without_max,
        name="gpe_proto__str_fsi",
    )
    gpe_proto__gpe_proto = ann.Projection(  # NEW, not in original BGM
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
    cor_go = ann.Population(
        self.params["cor_go.size"], poisson_neuron_up_down, name="cor_go"
    )
    cor_pause = ann.Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = ann.Population(
        self.params["cor_stop.size"], poisson_neuron_up_down, name="cor_stop"
    )
    ### Str Populations
    str_d1 = ann.Population(
        self.params["str_d1.size"],
        Izhikevich2007_noisy_AMPA_oscillating,
        name="str_d1",  # NEW NEURON MODEL
    )
    str_d2 = ann.Population(
        self.params["str_d2.size"],
        Izhikevich2007_noisy_AMPA_oscillating,
        name="str_d2",  # NEW NEURON MODEL
    )
    str_fsi = ann.Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    stn = ann.Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = ann.Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = ann.Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,  # NEW NEURON MODEL
        name="gpe_proto",
    )
    gpe_arky = ann.Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA_oscillating,
        name="gpe_arky",
    )
    gpe_cp = ann.Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = ann.Population(
        self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal"
    )
    ### integrator Neurons
    integrator_go = ann.Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = ann.Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = ann.Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = ann.Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = ann.Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = ann.Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = ann.Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = ann.Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )

    cor_stop__gpe_proto = ann.Projection(  # NEW !
        pre=cor_stop,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_proto",
    )
    ### cortex pause output
    cor_pause__stn = ann.Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = ann.Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = ann.Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = ann.Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = ann.Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = ann.Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = ann.Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = ann.Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = ann.Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = ann.Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = ann.Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = ann.Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = ann.Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = ann.Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = ann.Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = ann.Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = ann.Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = ann.Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = ann.Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = ann.Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = ann.Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )

    gpe_proto__gpe_proto = ann.Projection(  # NEW, not in original BGM
        pre=gpe_proto,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_proto",
    )

    gpe_arky__gpe_arky = ann.Projection(  # NEW, not in original BGM
        pre=gpe_arky,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_arky",
    )

    gpe_proto__str_fsi = ann.Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = ann.Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = ann.Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = ann.Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = ann.Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = ann.Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = ann.Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = ann.Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = ann.Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = ann.Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = ann.Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = ann.Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = ann.Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = ann.Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = ann.Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = ann.Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = ann.Projection(
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
    cor_go = ann.Population(
        self.params["cor_go.size"], poisson_neuron_sin, name="cor_go"
    )
    cor_pause = ann.Population(
        self.params["cor_pause.size"], poisson_neuron_up_down, name="cor_pause"
    )
    cor_stop = ann.Population(
        self.params["cor_stop.size"], poisson_neuron_sin, name="cor_stop"
    )
    ### Str Populations
    str_d1 = ann.Population(
        self.params["str_d1.size"],
        Izhikevich2007_noisy_AMPA,
        name="str_d1",  # NEW NEURON MODEL
    )
    str_d2 = ann.Population(
        self.params["str_d2.size"],
        Izhikevich2007_noisy_AMPA,
        name="str_d2",  # NEW NEURON MODEL
    )
    str_fsi = ann.Population(
        self.params["str_fsi.size"],
        Izhikevich2007_Corbit_FSI_noisy_AMPA,
        name="str_fsi",
    )
    ### BG Populations
    stn = ann.Population(self.params["stn.size"], Izhikevich2003_noisy_AMPA, name="stn")
    snr = ann.Population(self.params["snr.size"], Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = ann.Population(
        self.params["gpe_proto.size"],
        Izhikevich2003_flexible_noisy_AMPA,  # NEW NEURON MODEL
        name="gpe_proto",
    )
    gpe_arky = ann.Population(
        self.params["gpe_arky.size"],
        Izhikevich2003_flexible_noisy_AMPA,
        name="gpe_arky",
    )
    gpe_cp = ann.Population(
        self.params["gpe_cp.size"], Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp"
    )
    thal = ann.Population(
        self.params["thal.size"], Izhikevich2003_noisy_AMPA, name="thal"
    )
    ### integrator Neurons
    integrator_go = ann.Population(
        self.params["integrator_go.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_go",
    )
    integrator_stop = ann.Population(
        self.params["integrator_stop.size"],
        integrator_neuron,
        stop_condition="decision>=0 : any",
        name="integrator_stop",
    )

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1 = ann.Projection(
        pre=cor_go,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d1",
    )
    cor_go__str_d2 = ann.Projection(
        pre=cor_go,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_d2",
    )
    cor_go__str_fsi = ann.Projection(
        pre=cor_go,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__str_fsi",
    )
    cor_go__thal = ann.Projection(
        pre=cor_go,
        post=thal,
        target="ampa",
        synapse=factor_synapse,
        name="cor_go__thal",
    )
    ### cortex stop output
    cor_stop__gpe_arky = ann.Projection(
        pre=cor_stop,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_arky",
    )
    cor_stop__gpe_cp = ann.Projection(
        pre=cor_stop,
        post=gpe_cp,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_cp",
    )

    cor_stop__gpe_proto = ann.Projection(  # NEW !
        pre=cor_stop,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="cor_stop__gpe_proto",
    )
    ### cortex pause output
    cor_pause__stn = ann.Projection(
        pre=cor_pause,
        post=stn,
        target="ampa",
        synapse=factor_synapse,
        name="cor_pause__stn",
    )
    ### str d1 output
    str_d1__snr = ann.Projection(
        pre=str_d1, post=snr, target="gaba", synapse=factor_synapse, name="str_d1__snr"
    )
    str_d1__gpe_cp = ann.Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__gpe_cp",
    )
    str_d1__str_d1 = ann.Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d1",
    )
    str_d1__str_d2 = ann.Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d1__str_d2",
    )
    ### str d2 output
    str_d2__gpe_proto = ann.Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_proto",
    )
    str_d2__gpe_arky = ann.Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_arky",
    )
    str_d2__gpe_cp = ann.Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__gpe_cp",
    )
    str_d2__str_d1 = ann.Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d1",
    )
    str_d2__str_d2 = ann.Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_d2__str_d2",
    )
    ### str fsi output
    str_fsi__str_d1 = ann.Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d1",
    )
    str_fsi__str_d2 = ann.Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_d2",
    )
    str_fsi__str_fsi = ann.Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="str_fsi__str_fsi",
    )
    ### stn output
    stn__snr = ann.Projection(
        pre=stn, post=snr, target="ampa", synapse=factor_synapse, name="stn__snr"
    )
    stn__gpe_proto = ann.Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_proto",
    )
    stn__gpe_arky = ann.Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        synapse=factor_synapse,
        name="stn__gpe_arky",
    )
    stn__gpe_cp = ann.Projection(
        pre=stn, post=gpe_cp, target="ampa", synapse=factor_synapse, name="stn__gpe_cp"
    )
    ### gpe proto output
    gpe_proto__stn = ann.Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__stn",
    )
    gpe_proto__snr = ann.Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__snr",
    )
    gpe_proto__gpe_arky = ann.Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_arky",
    )
    gpe_proto__gpe_cp = ann.Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_cp",
    )

    gpe_proto__gpe_proto = ann.Projection(  # NEW, not in original BGM
        pre=gpe_proto,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__gpe_proto",
    )

    gpe_arky__gpe_arky = ann.Projection(  # NEW, not in original BGM
        pre=gpe_arky,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_arky",
    )

    gpe_proto__str_fsi = ann.Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_proto__str_fsi",
    )
    ### gpe arky output
    gpe_arky__str_d1 = ann.Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d1",
    )
    gpe_arky__str_d2 = ann.Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_d2",
    )
    gpe_arky__str_fsi = ann.Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__str_fsi",
    )
    gpe_arky__gpe_proto = ann.Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_proto",
    )
    gpe_arky__gpe_cp = ann.Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_arky__gpe_cp",
    )
    ### gpe cp output
    gpe_cp__str_d1 = ann.Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d1",
    )
    gpe_cp__str_d2 = ann.Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_d2",
    )
    gpe_cp__str_fsi = ann.Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__str_fsi",
    )
    gpe_cp__gpe_proto = ann.Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_proto",
    )
    gpe_cp__gpe_arky = ann.Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        synapse=factor_synapse,
        name="gpe_cp__gpe_arky",
    )
    gpe_cp__integrator_stop = ann.Projection(
        pre=gpe_cp,
        post=integrator_stop,
        target="ampa",
        synapse=factor_synapse,
        name="gpe_cp__integrator_stop",
    )
    ### snr output
    snr__thal = ann.Projection(
        pre=snr, post=thal, target="gaba", synapse=factor_synapse, name="snr__thal"
    )
    ### thal output
    thal__integrator_go = ann.Projection(
        pre=thal,
        post=integrator_go,
        target="ampa",
        synapse=factor_synapse,
        name="thal__integrator_go",
    )
    thal__str_d1 = ann.Projection(
        pre=thal,
        post=str_d1,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d1",
    )
    thal__str_d2 = ann.Projection(
        pre=thal,
        post=str_d2,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_d2",
    )
    thal__str_fsi = ann.Projection(
        pre=thal,
        post=str_fsi,
        target="ampa",
        synapse=factor_synapse,
        name="thal__str_fsi",
    )


def BGM_v07(self):
    """
    difference to Goenner et al. (2021):
    - no cortical populations: they are replaced by spike count TimedArray inputs created in the Microcircuit and Cortical Inputs classes
        - the striatal inputs: are generated in the Microcircuit class
        - the cortex_go to thalamus, cortex_stop to GPe_arky and GPe_cp, and cortex_pause to STN inputs: are generated using the Cortical Inputs class after generating the corresponding BG pops
    - striatum popultions are generated within the Microcircuit class here at the begining
        - the populations str_d1, str_d2 and str_fsi are obtained from the Microcircuit class with the method create_model()
        - the local striatal connections are generated in the Microcircuit class, not here anymore
        - the projections between the striatal populations and other BG populations are still generated here
    - Integrators not used anymore
    - new neuron models
        - GPe parameters were refitted and GPe neurons use nonlinear function for external current (not baseline current)
        - implemented a new noise method (not noisy ampa anymore but noisy baseline current)
        - implemented numerical stabilization for conductances (factor and constant driving force for ampa)
    - using this model_creation_function requires the BGM class to have a model_creation_kwargs dict with the following entries:
        - "build_mc": bool, whether to build the microcircuit or load existing data
        - "build_ci": bool, whether to build the cortical inputs or load existing data
        - "mc.nx": size parameter for the microcircuit (number of neurons per dimension)
        - "mc.b": size parameter for the microcircuit (number of neurons per dimension) . just make them equal
        - "dbs": bool, whether to simulate under DBS condition or not
        - "timestep": simulation timestep in ms
        - "t.duration": max total simulation duration in ms
        - "update_time": time interval for updating the striatal inputs in ms
        - "mc.storage_dir": directory where to store/load the caudate microcircuit data
        - "mc.seed": random seed for the microcircuit generation
        - "mc.fitted_params_path": path to the fitted striatal connection probabilities
        - "mc.cortical_rate_path": path to the cortical rate time series (it's not used here but it needs to be the same as for creating the inputs for the Microcircuits)
        - "ci.storage_dir": directory where to store/load the putamen cortical inputs data
        - "ci.seed": random seed for the cortical input generation
        - "ci.n_thal": number of cortical input neurons for a thalamic neuron
        - "ci.n_gpe_arky": number of cortical input neurons for a GPe_arky neuron
        - "ci.n_gpe_cp": number of cortical input neurons for a GPe_cp neuron
        - "ci.n_stn": number of cortical input neurons for a STN neuron
    - do not use factor_synapse anymore -> use standard synapse, now change proj.w instead of proj.mod_factor
    """

    ### CREATE THE STRIATAL MICRO CIRCUIT FOR THE CURRENT LOOP
    mc = Microcircuit(
        name=self.model_creation_kwargs["mc.name"],
        nx=self.model_creation_kwargs["mc.nx"],
        b=self.model_creation_kwargs["mc.b"],
        dbs_condition=self.model_creation_kwargs["dbs"],
        build_connectivity=self.model_creation_kwargs["build_mc"],
        build_missing_gaba_input=self.model_creation_kwargs["build_mc"],
        build_cortical_input=self.model_creation_kwargs["build_ci"],
        dt=self.model_creation_kwargs["timestep"],
        T=self.model_creation_kwargs["t.duration"],
        update_time=self.model_creation_kwargs["update_time"],
        storage_dir=self.model_creation_kwargs[f"mc.storage_dir"],
        seed=self.model_creation_kwargs["mc.seed"],
        fitted_params_path=self.model_creation_kwargs["mc.fitted_params_path"],
        cortical_rate_path=self.model_creation_kwargs["mc.cortical_rate_path"],
        verbose=False,
    )
    str_pop = mc.create_model()

    #######   POPULATIONS   ######
    ### Str Populations now obtained from Microcircuit class
    str_d1 = str_pop["dSPN"]
    str_d2 = str_pop["iSPN"]
    str_fsi = str_pop["FS"]
    ### BG Populations
    stn = ann.Population(
        self.params["stn.size"],
        Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=False),
        name="stn",
    )
    snr = ann.Population(
        self.params["snr.size"],
        Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=False),
        name="snr",
    )
    gpe_proto = ann.Population(
        self.params["gpe_proto.size"],
        Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=True),
        name="gpe_proto",
    )
    gpe_arky = ann.Population(
        self.params["gpe_arky.size"],
        Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=True),
        name="gpe_arky",
    )
    gpe_cp = ann.Population(
        self.params["gpe_cp.size"],
        Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=True),
        name="gpe_cp",
    )
    thal = ann.Population(
        self.params["thal.size"],
        Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=False),
        name="thal",
    )

    ######   PROJECTIONS   ######

    ### str d1 output
    ann.Projection(
        pre=str_d1,
        post=snr,
        target="gaba",
        name=f"str_d1__{snr.name}",
    )
    ann.Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        name=f"str_d1__{gpe_cp.name}",
    )
    ### str d2 output
    ann.Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        name=f"str_d2__{gpe_proto.name}",
    )
    ann.Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        name=f"str_d2__{gpe_arky.name}",
    )
    ann.Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        name=f"str_d2__{gpe_cp.name}",
    )
    ### stn output
    ann.Projection(
        pre=stn,
        post=snr,
        target="ampa",
        name=f"{stn.name}__{snr.name}",
    )
    ann.Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        name=f"{stn.name}__{gpe_proto.name}",
    )
    ann.Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        name=f"{stn.name}__{gpe_arky.name}",
    )
    ann.Projection(
        pre=stn,
        post=gpe_cp,
        target="ampa",
        name=f"{stn.name}__{gpe_cp.name}",
    )
    ### gpe proto output
    ann.Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        name=f"{gpe_proto.name}__{stn.name}",
    )
    ann.Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        name=f"{gpe_proto.name}__{snr.name}",
    )
    ann.Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        name=f"{gpe_proto.name}__{gpe_arky.name}",
    )
    ann.Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        name=f"{gpe_proto.name}__{gpe_cp.name}",
    )
    ann.Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        name=f"{gpe_proto.name}__str_fsi",
    )
    ### gpe arky output
    ann.Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        name=f"{gpe_arky.name}__str_d1",
    )
    ann.Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        name=f"{gpe_arky.name}__str_d2",
    )
    ann.Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        name=f"{gpe_arky.name}__str_fsi",
    )
    ann.Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        name=f"{gpe_arky.name}__{gpe_proto.name}",
    )
    ann.Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        name=f"{gpe_arky.name}__{gpe_cp.name}",
    )
    ### gpe cp output
    ann.Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        name=f"{gpe_cp.name}__str_d1",
    )
    ann.Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        name=f"{gpe_cp.name}__str_d2",
    )
    ann.Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        name=f"{gpe_cp.name}__str_fsi",
    )
    ann.Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        name=f"{gpe_cp.name}__{gpe_proto.name}",
    )
    ann.Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        name=f"{gpe_cp.name}__{gpe_arky.name}",
    )
    ### snr output
    ann.Projection(
        pre=snr,
        post=thal,
        target="gaba",
        name=f"{snr.name}__{thal.name}",
    )
    ann.Projection(
        pre=thal,
        post=str_d1,
        target="glut",
        name=f"{thal.name}__str_d1",
    )
    ann.Projection(
        pre=thal,
        post=str_d2,
        target="glut",
        name=f"{thal.name}__str_d2",
    )
    ann.Projection(
        pre=thal,
        post=str_fsi,
        target="ampa",
        name=f"{thal.name}__str_fsi",
    )

    #######   CREATE THE CORTICAL INPUTS OF THE BG POPs   ######
    ci = CorticalInputs(
        populations=[
            thal,
            gpe_arky,
            gpe_cp,
            stn,
        ],
        N_cortical_inputs_dict={
            thal.name: self.model_creation_kwargs["ci.n_thal"],
            gpe_arky.name: self.model_creation_kwargs["ci.n_gpe_arky"],
            gpe_cp.name: self.model_creation_kwargs["ci.n_gpe_cp"],
            stn.name: self.model_creation_kwargs["ci.n_stn"],
        },
        dt=self.model_creation_kwargs["timestep"],
        update_time=self.model_creation_kwargs["update_time"],
        T=self.model_creation_kwargs["t.duration"],
        name=self.model_creation_kwargs["mc.name"],
        dbs_condition=self.model_creation_kwargs["dbs"],
        storage_dir=self.model_creation_kwargs["ci.storage_dir"],
        cortical_rate_path=self.model_creation_kwargs["mc.cortical_rate_path"],
        build_cortical_input=self.model_creation_kwargs["build_ci"],
        seed=self.model_creation_kwargs["ci.seed"],
        verbose=False,
    )
    ci.create_model()

    return mc, ci


def BGM_v08(self):
    """
    difference to Goenner et al. (2021):
    - no cortical populations: they are replaced by spike count TimedArray inputs
        - the following pops receive input from the TimedArray via CurrentProjection:
        - str_d1
        - str_d2
        - str_fsi
        - thal
        - gpe_arky
        - gpe_cp
        - stn
    - Integrators not used anymore
    - new neuron models
        - striatal neurons fully based on Humphries 2009
        - GPe parameters were refitted and GPe neurons use nonlinear function for external current (not baseline current)
        - implemented a new noise method (not noisy ampa anymore but noisy baseline current)
        - implemented numerical stabilization for conductances (factor and constant driving force for ampa)
        - implemented a exp_input: additional incoming exc spikes from exponential distribution with lambda "exp_input" (mean=1/lambda), weighted by the input "g_cor"
            - for labda values see: https://docs.google.com/spreadsheets/d/1yPWpbQnrIrALBvEErs7cz59YovRqgoXFFPaCmKbDYaw/edit?usp=sharing
    - using this model_creation_function requires the BGM class to have a model_creation_kwargs dict with the following entries:
        - "input.rates": array with (steps,) shape containing the input rates for each time step
        - "input.schedule": a single scalar with the schedule time in ms for updating the input rates
        - "timestep": simulation timestep in ms
    - do not use factor_synapse anymore -> use standard synapse, now change proj.w instead of proj.mod_factor
    """

    #######   POPULATIONS   ######
    ### Str Populations now obtained from Microcircuit class
    str_d1 = ann.Population(
        self.params["str_d1.size"],
        Izhikevich2007Humphries2009SPND1(
            current_based_excitation=True, exp_input=1 / 0.7, params_for_pop=True
        ),
        name="str_d1",
    )
    str_d2 = ann.Population(
        self.params["str_d2.size"],
        Izhikevich2007Humphries2009SPND2(
            current_based_excitation=True, exp_input=1 / 0.7, params_for_pop=True
        ),
        name="str_d2",
    )
    str_fsi = ann.Population(
        self.params["str_fsi.size"],
        Izhikevich2007Humphries2009FSI(
            current_based_excitation=True, exp_input=1 / 0.28, params_for_pop=True
        ),
        name="str_fsi",
    )
    ### BG Populations
    stn = ann.Population(
        self.params["stn.size"],
        Izhikevich2003NoisyBaseNonlin(
            stabilize=True, use_nonlin=False, exp_input=1 / 0.075
        ),
        name="stn",
    )
    snr = ann.Population(
        self.params["snr.size"],
        Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=False),
        name="snr",
    )
    gpe_proto = ann.Population(
        self.params["gpe_proto.size"],
        Izhikevich2003NoisyBaseNonlin(stabilize=True, use_nonlin=True),
        name="gpe_proto",
    )
    gpe_arky = ann.Population(
        self.params["gpe_arky.size"],
        Izhikevich2003NoisyBaseNonlin(
            stabilize=True, use_nonlin=True, exp_input=1 / 0.03
        ),
        name="gpe_arky",
    )
    gpe_cp = ann.Population(
        self.params["gpe_cp.size"],
        Izhikevich2003NoisyBaseNonlin(
            stabilize=True, use_nonlin=True, exp_input=1 / 0.03
        ),
        name="gpe_cp",
    )
    thal = ann.Population(
        self.params["thal.size"],
        Izhikevich2003NoisyBaseNonlin(
            stabilize=True, use_nonlin=False, exp_input=1 / 0.12
        ),
        name="thal",
    )

    ######   PROJECTIONS   ######

    ### str d1 output
    ann.Projection(
        pre=str_d1,
        post=snr,
        target="gaba",
        name=f"str_d1__{snr.name}",
    )
    ann.Projection(
        pre=str_d1,
        post=gpe_cp,
        target="gaba",
        name=f"str_d1__{gpe_cp.name}",
    )
    ann.Projection(
        pre=str_d1,
        post=str_d1,
        target="gaba",
        name="str_d1__str_d1",
    )
    ann.Projection(
        pre=str_d1,
        post=str_d2,
        target="gaba",
        name="str_d1__str_d2",
    )
    ### str d2 output
    ann.Projection(
        pre=str_d2,
        post=gpe_proto,
        target="gaba",
        name=f"str_d2__{gpe_proto.name}",
    )
    ann.Projection(
        pre=str_d2,
        post=gpe_arky,
        target="gaba",
        name=f"str_d2__{gpe_arky.name}",
    )
    ann.Projection(
        pre=str_d2,
        post=gpe_cp,
        target="gaba",
        name=f"str_d2__{gpe_cp.name}",
    )
    ann.Projection(
        pre=str_d2,
        post=str_d1,
        target="gaba",
        name="str_d2__str_d1",
    )
    ann.Projection(
        pre=str_d2,
        post=str_d2,
        target="gaba",
        name="str_d2__str_d2",
    )
    ### str fsi output
    ann.Projection(
        pre=str_fsi,
        post=str_d1,
        target="gaba",
        name="str_fsi__str_d1",
    )
    ann.Projection(
        pre=str_fsi,
        post=str_d2,
        target="gaba",
        name="str_fsi__str_d2",
    )
    ann.Projection(
        pre=str_fsi,
        post=str_fsi,
        target="gaba",
        name="str_fsi__str_fsi",
    )
    ### stn output
    ann.Projection(
        pre=stn,
        post=snr,
        target="ampa",
        name=f"{stn.name}__{snr.name}",
    )
    ann.Projection(
        pre=stn,
        post=gpe_proto,
        target="ampa",
        name=f"{stn.name}__{gpe_proto.name}",
    )
    ann.Projection(
        pre=stn,
        post=gpe_arky,
        target="ampa",
        name=f"{stn.name}__{gpe_arky.name}",
    )
    ann.Projection(
        pre=stn,
        post=gpe_cp,
        target="ampa",
        name=f"{stn.name}__{gpe_cp.name}",
    )
    ### gpe proto output
    ann.Projection(
        pre=gpe_proto,
        post=stn,
        target="gaba",
        name=f"{gpe_proto.name}__{stn.name}",
    )
    ann.Projection(
        pre=gpe_proto,
        post=snr,
        target="gaba",
        name=f"{gpe_proto.name}__{snr.name}",
    )
    ann.Projection(
        pre=gpe_proto,
        post=gpe_arky,
        target="gaba",
        name=f"{gpe_proto.name}__{gpe_arky.name}",
    )
    ann.Projection(
        pre=gpe_proto,
        post=gpe_cp,
        target="gaba",
        name=f"{gpe_proto.name}__{gpe_cp.name}",
    )
    ann.Projection(
        pre=gpe_proto,
        post=str_fsi,
        target="gaba",
        name=f"{gpe_proto.name}__str_fsi",
    )
    ### gpe arky output
    ann.Projection(
        pre=gpe_arky,
        post=str_d1,
        target="gaba",
        name=f"{gpe_arky.name}__str_d1",
    )
    ann.Projection(
        pre=gpe_arky,
        post=str_d2,
        target="gaba",
        name=f"{gpe_arky.name}__str_d2",
    )
    ann.Projection(
        pre=gpe_arky,
        post=str_fsi,
        target="gaba",
        name=f"{gpe_arky.name}__str_fsi",
    )
    ann.Projection(
        pre=gpe_arky,
        post=gpe_proto,
        target="gaba",
        name=f"{gpe_arky.name}__{gpe_proto.name}",
    )
    ann.Projection(
        pre=gpe_arky,
        post=gpe_cp,
        target="gaba",
        name=f"{gpe_arky.name}__{gpe_cp.name}",
    )
    ### gpe cp output
    ann.Projection(
        pre=gpe_cp,
        post=str_d1,
        target="gaba",
        name=f"{gpe_cp.name}__str_d1",
    )
    ann.Projection(
        pre=gpe_cp,
        post=str_d2,
        target="gaba",
        name=f"{gpe_cp.name}__str_d2",
    )
    ann.Projection(
        pre=gpe_cp,
        post=str_fsi,
        target="gaba",
        name=f"{gpe_cp.name}__str_fsi",
    )
    ann.Projection(
        pre=gpe_cp,
        post=gpe_proto,
        target="gaba",
        name=f"{gpe_cp.name}__{gpe_proto.name}",
    )
    ann.Projection(
        pre=gpe_cp,
        post=gpe_arky,
        target="gaba",
        name=f"{gpe_cp.name}__{gpe_arky.name}",
    )
    ### snr output
    ann.Projection(
        pre=snr,
        post=thal,
        target="gaba",
        name=f"{snr.name}__{thal.name}",
    )
    ann.Projection(
        pre=thal,
        post=str_d1,
        target="glut",
        name=f"{thal.name}__str_d1",
    )
    ann.Projection(
        pre=thal,
        post=str_d2,
        target="glut",
        name=f"{thal.name}__str_d2",
    )
    ann.Projection(
        pre=thal,
        post=str_fsi,
        target="ampa",
        name=f"{thal.name}__str_fsi",
    )
