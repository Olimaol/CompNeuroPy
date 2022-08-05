from ANNarchy import Population, Projection
from CompNeuroPy.neuron_models import poisson_neuron_up_down, poisson_neuron, Izhikevich2007_noisy_AMPA, Izhikevich2007_fsi_noisy_AMPA, Izhikevich2003_noisy_AMPA, Izhikevich2003_flexible_noisy_AMPA, integrator_neuron, integrator_neuron_simple
from CompNeuroPy.synapse_models import factor_synapse

def BGM_v01(self):
    #######   POPULATIONS   ######
    ### cortex / input populations
    cor_go    = Population(self.params['cor_go.size'],    poisson_neuron_up_down, name="cor_go")
    cor_pause = Population(self.params['cor_pause.size'], poisson_neuron_up_down, name="cor_pause")
    cor_stop  = Population(self.params['cor_stop.size'],  poisson_neuron_up_down, name="cor_stop")
    ### Str Populations
    str_d1  = Population(self.params['str_d1.size'],  Izhikevich2007_noisy_AMPA, name="str_d1")
    str_d2  = Population(self.params['str_d2.size'],  Izhikevich2007_noisy_AMPA, name="str_d2")
    str_fsi = Population(self.params['str_fsi.size'], Izhikevich2007_fsi_noisy_AMPA, name="str_fsi")
    ### BG Populations
    stn       = Population(self.params['stn.size'],       Izhikevich2003_noisy_AMPA, name="stn")
    snr       = Population(self.params['snr.size'],       Izhikevich2003_noisy_AMPA, name="snr")
    gpe_proto = Population(self.params['gpe_proto.size'], Izhikevich2003_flexible_noisy_AMPA, name="gpe_proto")
    gpe_arky  = Population(self.params['gpe_arky.size'],  Izhikevich2003_flexible_noisy_AMPA, name="gpe_arky")
    gpe_cp    = Population(self.params['gpe_cp.size'],    Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp")
    thal      = Population(self.params['thal.size'],      Izhikevich2003_noisy_AMPA, name="thal")
    ### integrator Neurons
    integrator_go   = Population(self.params['integrator_go.size'],   integrator_neuron, stop_condition="decision < 0 : any", name="integrator_go")
    integrator_stop = Population(self.params['integrator_stop.size'], integrator_neuron, stop_condition="decision < 0 : any", name="integrator_stop")

    ######   PROJECTIONS   ######
    ### cortex go output
    cor_go__str_d1  = Projection(pre=cor_go, post=str_d1,  target='ampa', synapse=factor_synapse, name='cor_go__str_d1')
    cor_go__str_d2  = Projection(pre=cor_go, post=str_d2,  target='ampa', synapse=factor_synapse, name='cor_go__str_d2')
    cor_go__str_fsi = Projection(pre=cor_go, post=str_fsi, target='ampa', synapse=factor_synapse, name='cor_go__str_fsi')
    cor_go__thal    = Projection(pre=cor_go, post=thal,    target='ampa', synapse=factor_synapse, name='cor_go__thal')
    ### cortex stop output
    cor_stop__gpe_arky = Projection(pre=cor_stop, post=gpe_arky, target='ampa', synapse=factor_synapse, name='cor_stop__gpe_arky')
    cor_stop__gpe_cp   = Projection(pre=cor_stop, post=gpe_cp,   target='ampa', synapse=factor_synapse, name='cor_stop__gpe_cp')
    ### cortex pause output
    cor_pause__stn = Projection(pre=cor_pause, post=stn, target='ampa', synapse=factor_synapse, name='cor_pause__stn')
    ### str d1 output
    str_d1__snr    = Projection(pre=str_d1, post=snr,    target='gaba', synapse=factor_synapse, name='str_d1__snr')
    str_d1__gpe_cp = Projection(pre=str_d1, post=gpe_cp, target='gaba', synapse=factor_synapse, name='str_d1__gpe_cp')
    str_d1__str_d1 = Projection(pre=str_d1, post=str_d1, target='gaba', synapse=factor_synapse, name='str_d1__str_d1')
    str_d1__str_d2 = Projection(pre=str_d1, post=str_d2, target='gaba', synapse=factor_synapse, name='str_d1__str_d2')
    ### str d2 output
    str_d2__gpe_proto = Projection(pre=str_d2, post=gpe_proto, target='gaba', synapse=factor_synapse, name='str_d2__gpe_proto')
    str_d2__gpe_arky  = Projection(pre=str_d2, post=gpe_arky,  target='gaba', synapse=factor_synapse, name='str_d2__gpe_arky')
    str_d2__gpe_cp    = Projection(pre=str_d2, post=gpe_cp,    target='gaba', synapse=factor_synapse, name='str_d2__gpe_cp')
    str_d2__str_d1    = Projection(pre=str_d2, post=str_d1,    target='gaba', synapse=factor_synapse, name='str_d2__str_d1')
    str_d2__str_d2    = Projection(pre=str_d2, post=str_d2,    target='gaba', synapse=factor_synapse, name='str_d2__str_d2')
    ### str fsi output
    str_fsi__str_d1  = Projection(pre=str_fsi, post=str_d1,  target='gaba', synapse=factor_synapse, name='str_fsi__str_d1')
    str_fsi__str_d2  = Projection(pre=str_fsi, post=str_d2,  target='gaba', synapse=factor_synapse, name='str_fsi__str_d2')
    str_fsi__str_fsi = Projection(pre=str_fsi, post=str_fsi, target='gaba', synapse=factor_synapse, name='str_fsi__str_fsi')
    ### stn output
    stn__snr       = Projection(pre=stn, post=snr,       target='ampa', synapse=factor_synapse, name='stn__snr')
    stn__gpe_proto = Projection(pre=stn, post=gpe_proto, target='ampa', synapse=factor_synapse, name='stn__gpe_proto')
    stn__gpe_arky  = Projection(pre=stn, post=gpe_arky,  target='ampa', synapse=factor_synapse, name='stn__gpe_arky')
    stn__gpe_cp    = Projection(pre=stn, post=gpe_cp,    target='ampa', synapse=factor_synapse, name='stn__gpe_cp')
    ### gpe proto output
    gpe_proto__stn      = Projection(pre=gpe_proto, post=stn,      target='gaba', synapse=factor_synapse, name='gpe_proto__stn')
    gpe_proto__snr      = Projection(pre=gpe_proto, post=snr,      target='gaba', synapse=factor_synapse, name='gpe_proto__snr')
    gpe_proto__gpe_arky = Projection(pre=gpe_proto, post=gpe_arky, target='gaba', synapse=factor_synapse, name='gpe_proto__gpe_arky')
    gpe_proto__gpe_cp   = Projection(pre=gpe_proto, post=gpe_cp,   target='gaba', synapse=factor_synapse, name='gpe_proto__gpe_cp')
    gpe_proto__str_fsi  = Projection(pre=gpe_proto, post=str_fsi,  target='gaba', synapse=factor_synapse, name='gpe_proto__str_fsi')
    ### gpe arky output
    gpe_arky__str_d1    = Projection(pre=gpe_arky, post=str_d1,    target='gaba', synapse=factor_synapse, name='gpe_arky__str_d1')
    gpe_arky__str_d2    = Projection(pre=gpe_arky, post=str_d2,    target='gaba', synapse=factor_synapse, name='gpe_arky__str_d2')
    gpe_arky__str_fsi   = Projection(pre=gpe_arky, post=str_fsi,   target='gaba', synapse=factor_synapse, name='gpe_arky__str_fsi')
    gpe_arky__gpe_proto = Projection(pre=gpe_arky, post=gpe_proto, target='gaba', synapse=factor_synapse, name='gpe_arky__gpe_proto')
    gpe_arky__gpe_cp    = Projection(pre=gpe_arky, post=gpe_cp,    target='gaba', synapse=factor_synapse, name='gpe_arky__gpe_cp')
    ### gpe cp output
    gpe_cp__str_d1          = Projection(pre=gpe_cp, post=str_d1,           target='gaba', synapse=factor_synapse, name='gpe_cp__str_d1')
    gpe_cp__str_d2          = Projection(pre=gpe_cp, post=str_d2,           target='gaba', synapse=factor_synapse, name='gpe_cp__str_d2')
    gpe_cp__str_fsi         = Projection(pre=gpe_cp, post=str_fsi,          target='gaba', synapse=factor_synapse, name='gpe_cp__str_fsi')
    gpe_cp__gpe_proto       = Projection(pre=gpe_cp, post=gpe_proto,        target='gaba', synapse=factor_synapse, name='gpe_cp__gpe_proto')
    gpe_cp__gpe_arky        = Projection(pre=gpe_cp, post=gpe_arky,         target='gaba', synapse=factor_synapse, name='gpe_cp__gpe_arky')
    gpe_cp__integrator_stop = Projection(pre=gpe_cp, post=integrator_stop,  target='ampa', synapse=factor_synapse, name='gpe_cp__integrator_stop')
    ### snr output
    snr__thal = Projection(pre=snr, post=thal, target='gaba', synapse=factor_synapse, name='snr__thal')
    ### thal output
    thal__integrator_go = Projection(pre=thal, post=integrator_go, target='ampa', synapse=factor_synapse, name='thal__integrator_go')
    thal__str_d1        = Projection(pre=thal, post=str_d1,        target='ampa', synapse=factor_synapse, name='thal__str_d1')
    thal__str_d2        = Projection(pre=thal, post=str_d2,        target='ampa', synapse=factor_synapse, name='thal__str_d2')
    thal__str_fsi       = Projection(pre=thal, post=str_fsi,       target='ampa', synapse=factor_synapse, name='thal__str_fsi')






























































