"""
In this example, the DBS stimulator is tested with a simple spiking and rate-coded
model. The spiking model is based on the Izhikevich model (Izhikevich, 2007) with
conductance-based synapses. The rate-coded model is based on neurons including membrane
potential and a resulting firing rate. The DBS stimulator is tested with different
stimulation parameters. The resulting activity of the populations is compared to the
expected activity (not part of example, included for testing purposes only). The
resulting activity of the populations is plotted. The figures are saved in the
DBS_spiking_figure and DBS_rate_figure folders. The different DBS conditions are:
    - no stimulation
    - orthodromic stimulation of efferents
    - orthodromic stimulation of afferents
    - orthodromic stimulation of efferents and afferents
    - orthodromic stimulation of passing fibres
    - depolarization of the stimulated population
    - antidromic stimulation of efferents
    - antidromic stimulation of afferents
    - antidromic stimulation of efferents and afferents
    - antidromic stimulation of passing fibres
    - antidromic stimulation of passing fibres with lower strength
    - full dbs stimulation
    - full dbs stimulation without axon spikes (only effective for spiking model)
    - full dbs stimulation without axon_rate_amp (only effective for rate-coded model)

For rate-coded models, antidromic stimulation of projections is not available.
"""
from ANNarchy import (
    Neuron,
    Population,
    setup,
    simulate,
    Projection,
    get_population,
    get_projection,
    DefaultRateCodedSynapse,
    DefaultSpikingSynapse,
    dt,
    Constant,
)
from CompNeuroPy import (
    CompNeuroMonitors,
    PlotRecordings,
    CompNeuroModel,
    cnp_clear,
    DBSstimulator,
)
from CompNeuroPy.monitors import RecordingTimes
import numpy as np

### setup ANNarchy
setup(dt=0.1, seed=12345)


### create dbs test model
class dbs_test_model_class:
    """
    Class to create dbs test model.

    The used neuron models have the following constraints:
        The neuron model has to contain the following parameters:
        - base_mean: mean of the base current
        - base_noise: standard deviation of the base current noise
        Spiking neuron models have to contain conductance based synapses using the
        following conductance variables:
        - g_ampa: excitatory synapse
        - g_gaba: inhibitory synapse
        Rate neuron models have to contain the following input variables:
        - sum(ampa): excitatory input
        - sum(gaba): inhibitory input
        For DBS rate-coded models have to contain a membrane potential variable mp
        and spiking models have to be Izhihkevich models.

    Model structure:
    -------------------------
            POP1       POP2
            |          |
            o          v
    DBS--->POP3------oPOP4
                .----.
                |    |
            POP5   '-->POP6

    -o = inhibitory synapse
    -> = excitatory synapse
    .-> = passing fibre excitatory synapse

    Attributes:
        model (CompNeuroModel):
            dbs test model
    """

    def __init__(self, mode) -> None:
        """
        Initialize dbs test model

        Args:
            mode (str):
                Mode of the dbs test model, either "spiking" or "rate-coded"
        """
        ### constants should still be available after DBSstimulator recreates the model
        ### test this by creating this constant
        Constant("my_important_const", 0.0)

        ### check if model to create is spiking or rate-coded
        if mode == "spiking":
            self.model = CompNeuroModel(
                model_creation_function=self.create_model,
                model_kwargs={
                    "neuron_model": self.get_neuron_model_spiking(),
                    "base_current_list": [40, 100, 200, 50, 40, 40],
                    "base_current_noise": 40,
                },
                name="dbs_test_spiking",
                description="Simple spiking model to test dbs",
                do_compile=False,
            )
        elif mode == "rate-coded":
            self.model = CompNeuroModel(
                model_creation_function=self.create_model,
                model_kwargs={
                    "neuron_model": self.get_neuron_model_rate_coded(),
                    "base_current_list": [0.35, 0.7, 1.1, 0.85, 0.35, 0.35],
                    "base_current_noise": 0.01,
                    "weight_list": [0.3, 0.4, 0.3, 0.1],
                    "prob_list": [0.5, 0.7, 0.7, 0.5],
                },
                name="dbs_test_rate-coded",
                description="Simple rate-coded model to test dbs",
                do_compile=False,
            )
        else:
            raise ValueError("Neuron model not recognized")

    def create_model(
        self,
        neuron_model: Neuron,
        pop_size: int = 10,
        base_current_list: list = [0, 0, 0, 0, 0, 0],
        base_current_noise: float = 0.0,
        prob_list: list = [0.5, 0.5, 0.5, 0.5],
        weight_list: list = [1.0, 1.0, 1.0, 1.0],
    ):
        """
        Create dbs test model

        Args:
            neuron_model (Neuron):
                Neuron model to use for the dbs test model
            pop_size (int, optional):
                Number of neurons in each population. Default: 10
            base_current_list (list, optional):
                List of base currents for the four populations.
                Default: [0, 0, 0, 0, 0, 0]
            base_current_noise (float, optional):
                Standard deviation of the base current noise. Default: 0
            prob_list (list, optional):
                List of connection probabilities for the inhibitory and excitatory path.
                Default: [0.5, 0.5, 0.5, 0.5]
            weight_list (list, optional):
                List of connection weights for the inhibitory and excitatory path.
                Default: [0.1, 0.1, 0.1, 0.1]
        """
        ### create populations
        pop1 = Population(pop_size, neuron_model, name=f"pop1_{neuron_model.name}")
        pop2 = Population(pop_size, neuron_model, name=f"pop2_{neuron_model.name}")
        pop3 = Population(pop_size, neuron_model, name=f"pop3_{neuron_model.name}")
        pop4 = Population(pop_size, neuron_model, name=f"pop4_{neuron_model.name}")
        pop5 = Population(pop_size, neuron_model, name=f"pop5_{neuron_model.name}")
        pop6 = Population(pop_size, neuron_model, name=f"pop6_{neuron_model.name}")

        ### create projections of inhhibitory path
        proj_1_3 = Projection(
            pre=pop1,
            post=pop3,
            target="gaba",
            name=f"proj_1_3_{neuron_model.name}",
            synapse=self.get_synapse(neuron_model.name),
        )
        proj_1_3.connect_fixed_probability(
            probability=prob_list[0],
            weights=weight_list[0],
        )
        proj_3_4 = Projection(
            pre=pop3,
            post=pop4,
            target="gaba",
            name=f"proj_3_4_{neuron_model.name}",
            synapse=self.get_synapse(neuron_model.name),
        )
        proj_3_4.connect_fixed_probability(
            probability=prob_list[1],
            weights=weight_list[1],
        )
        ### create projections of excitatory path
        proj_2_4 = Projection(
            pre=pop2,
            post=pop4,
            target="ampa",
            name=f"proj_2_4_{neuron_model.name}",
            synapse=self.get_synapse(neuron_model.name),
        )
        proj_2_4.connect_fixed_probability(
            probability=prob_list[2],
            weights=weight_list[2],
        )
        ### create projection of passing fibres
        proj_5_6 = Projection(
            pre=pop5,
            post=pop6,
            target="ampa",
            name=f"proj_5_6_{neuron_model.name}",
            synapse=self.get_synapse(neuron_model.name),
        )
        proj_5_6.connect_fixed_probability(
            probability=prob_list[3],
            weights=weight_list[3],
        )

        ### set baseline activity parameters
        pop1.base_mean = base_current_list[0]
        pop2.base_mean = base_current_list[1]
        pop3.base_mean = base_current_list[2]
        pop4.base_mean = base_current_list[3]
        pop5.base_mean = base_current_list[4]
        pop6.base_mean = base_current_list[5]
        pop1.base_noise = base_current_noise
        pop2.base_noise = base_current_noise
        pop3.base_noise = base_current_noise
        pop4.base_noise = base_current_noise
        pop5.base_noise = base_current_noise
        pop6.base_noise = base_current_noise

    def get_neuron_model_spiking(self):
        """
        Get neuron model with spiking dynamics

        Returns
            neuron_model (Neuron):
                Neuron model with spiking dynamics
        """
        neuron_model = Neuron(
            parameters="""
                C      = 100     : population # pF
                k      = 0.7     : population # pS * mV**-1
                v_r    = -60     : population # mV
                v_t    = -40     : population # mV
                a      = 0.03     : population # ms**-1
                b      = -2     : population # nS
                c      = -50     : population # mV
                d      = 100     : population # pA
                v_peak = 35     : population # mV
                I_app  = 0     # pA
                tau_ampa = 10  : population # ms
                tau_gaba = 10  : population # ms
                E_ampa   = 0   : population # mV
                E_gaba   = -90 : population # mV
                base_mean       = 0 # pA
                base_noise      = 0 # pA
                rate_base_noise = 100 # Hz
            """,
            equations="""
                ### noisy base input
                offset_base = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rate_base_noise, offset_base, Normal(0., 1.) * base_noise)
                I_base      = base_mean + offset_base + my_important_const
                ### input conductances
                dg_ampa/dt = -g_ampa/tau_ampa
                dg_gaba/dt = -g_gaba/tau_gaba
                ### input currents
                I = I_app - g_ampa*neg(v - E_ampa) - g_gaba*pos(v - E_gaba) + I_base
                ### membrane potential and recovery variable
                C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I
                du/dt      = a*(b*(v - v_r) - u)
            """,
            spike="v >= v_peak",
            reset="""
                v = c
                u = u + d
            """,
            name="spiking",
            description="""
                Simple neuron model equations from Izhikevich (2007) using regular-spiking parameters
                with conductance-based AMPA and GABA synapses/currents.
            """,
        )
        return neuron_model

    def get_neuron_model_rate_coded(self):
        """
        Get neuron model with rate-coded dynamics

        Returns:
            neuron_model (Neuron):
                Neuron model with rate-coded dynamics
        """
        neuron_model = Neuron(
            parameters="""
                tau = 10.0 : population
                sigma = 0.6 : population
                I_0 = 0.2 : population
                I_app = 0.
                base_mean       = 0
                base_noise      = 0
                rate_base_noise = 100 # Hz
                # = (sigma*I_0 + I_0)/(sigma - sigma*I_0) : population
                c = (0.6*0.2 + 0.2)/(0.6 - 0.6*0.2) : population
            """,
            equations="""
                ### noisy base input
                offset_base = ite(Uniform(0.0, 1.0) * 1000.0 / dt > rate_base_noise, offset_base, Normal(0., 1.) * base_noise)
                I_base      = base_mean + offset_base + my_important_const
                ### input currents
                I = sum(ampa) - sum(gaba) + I_base + I_app
                ### membrane potential
                tau * dmp/dt = -mp + I
                mp_r = mp: min=-0.99*sigma
                ### activation function
                r = activation(mp_r,sigma,c) : max=1., min=0.
            """,
            name="rate-coded",
            functions="""
                activation(x,sigma,c) = ((sigma*x + x)/(sigma + x)) * (1 + c) - c
            """,
            description="Rate-coded neuron with excitatory (ampa) and inhibitory (gaba) inputs plus baseline and noise.",
        )
        return neuron_model

    def get_synapse(self, mode):
        """
        Create a synapse.

        Args:
            mode (str):
                Mode of the dbs test model, either "spiking" or "rate-coded"

        Returns:
            synapse (DefaultRateCodedSynapse or DefaultSpikingSynapse):
                Synapse object
        """
        if mode == "rate-coded":
            return DefaultRateCodedSynapse()
        elif mode == "spiking":
            return DefaultSpikingSynapse()
        else:
            raise ValueError("Neuron model not recognized")


def do_simulation(
    mon: CompNeuroMonitors,
    dbs: DBSstimulator,
    dbs_val_list: list[list],
    dbs_key_list: list[str],
):
    """
    Do the simulation

    Args:
        mon (CompNeuroMonitors):
            CompNeuroMonitors object
        dbs (DBSstimulator):
            DBS stimulator object
        dbs_val_list (list[list]):
            List of lists with DBS stimulation values used by the dbs.on() function
        dbs_key_list (list[str]):
            List of DBS stimulation keys used by the dbs.on() function

    Returns:
        recordings (list):
            List of recordings from the monitors
        recording_times (RecordingTimes):
            Recording times object
    """
    ### run initial ramp up simulation
    simulate(2000.0)

    ### start monitors
    mon.start()

    ### loop over trials
    for trial in range(len(dbs_val_list)):
        ### 1000 ms with DBS off
        simulate(1000.0)
        ### 500 ms with DBS on
        dbs.on(
            **{
                dbs_key_list[i]: dbs_val_list[trial][i]
                for i in range(len(dbs_key_list))
            }
        )
        simulate(500.0)
        ### 1000 ms with DBS off
        dbs.off()
        simulate(1000.0)
        mon.reset(model=False)

    ### get data from monitors
    recordings = mon.get_recordings()
    recording_times = mon.get_recording_times()

    return recordings, recording_times


def check_dbs_effects_spiking(
    dbs_val_list: list[list],
    recordings: list,
    model: CompNeuroModel,
    recording_times: RecordingTimes,
):
    """
    Check if the dbs effects are as expecteds.

    Args:
        dbs_val_list (list[list]):
            List of lists with DBS stimulation values used by the dbs.on() function
        recordings (list):
            List of recordings from the monitors
        model (CompNeuroModel):
            Model used for the simulation
        recording_times (RecordingTimes):
            Recording times object
    """
    ### effects_on_activity_list contains the expected effects of dbs on the activity of the populations for each trial
    ### 0 means no effect, 1 means increase, -1 means decrease
    effects_on_activity = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0],
        [0, 0, -1, 1, 0, 0],
        [0, 0, -1, -1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, -1, 1, 0, 0],
        [0, 0, -1, 1, 0, 0],
        [-1, 0, 0, 0, 0, 0],
        [-1, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0],
        [-1, 0, -1, -1, -1, 1],
        [0, 0, -1, 1, 0, 0],
        [-1, 0, -1, -1, -1, 1],
    ]
    ### check if the expected effects are present in the data
    effect_list = []
    high_effect_list = []
    low_effect_list = []
    for trial_idx, trial in enumerate(range(len(dbs_val_list))):
        effect_list.append([])
        for pop_name_idx, pop_name in enumerate(model.populations):
            v_arr = recordings[trial][f"{pop_name};v"]
            ### mean over neurons
            v_arr = np.mean(v_arr, axis=1)
            ### mean of first period
            v_mean_1 = np.mean(
                v_arr[
                    recording_times.idx_lims(chunk=trial)[0]
                    + int(round(500 / dt())) : recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1000 / dt()))
                ]
            )
            v_std_1 = np.std(
                v_arr[
                    recording_times.idx_lims(chunk=trial)[0]
                    + int(round(500 / dt())) : recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1000 / dt()))
                ]
            )
            ### mean of second period
            v_mean_2 = np.mean(
                v_arr[
                    recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1000 / dt())) : recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1500 / dt()))
                ]
            )
            ### mean of third period
            v_mean_3 = np.mean(
                v_arr[
                    recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1500 / dt())) : recording_times.idx_lims(chunk=trial)[0]
                    + int(round(2000 / dt()))
                ]
            )
            v_std_3 = np.std(
                v_arr[
                    recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1500 / dt())) : recording_times.idx_lims(chunk=trial)[0]
                    + int(round(2000 / dt()))
                ]
            )
            ### get meand depending on dbs
            mean_on = v_mean_2
            mean_off = (v_mean_1 + v_mean_3) / 2
            std_off = (v_std_1 + v_std_3) / 2
            ### calculate effect
            effect = (mean_on - mean_off) / std_off
            if effect > 1:
                high_effect_list.append(abs(effect))
                effect = 1
            elif effect < -1:
                high_effect_list.append(abs(effect))
                effect = -1
            else:
                low_effect_list.append(abs(effect))
                effect = 0

            effect_list[trial_idx].append(effect)

    assert (
        np.array(effects_on_activity).astype(int) == np.array(effect_list).astype(int)
    ).all(), "Effects on activity not as expected for spiking model"


def check_dbs_effects_rate_coded(
    dbs_val_list: list[list],
    recordings: list,
    model: CompNeuroModel,
    recording_times: RecordingTimes,
):
    """
    Check if the dbs effects are as expected.

    Args:
        dbs_val_list (list[list]):
            List of lists with DBS stimulation values used by the dbs.on() function
        recordings (list):
            List of recordings from the monitors
        model (CompNeuroModel):
            Model used for the simulation
        recording_times (RecordingTimes):
            Recording times object
    """
    ### effects_on_activity_list contains the expected effects of dbs on the activity of the populations for each trial
    ### 0 means no effect, 1 means increase, -1 means decrease
    effects_on_activity = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0],
        [0, 0, -1, 1, 0, 0],
        [0, 0, -1, -1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, 0, 1],
        [0, 0, -1, -1, 0, 1],
        [0, 0, -1, 1, 0, 0],
    ]
    ### check if the expected effects are present in the data
    effect_list = []
    high_effect_list = []
    low_effect_list = []
    for trial_idx, trial in enumerate(range(len(dbs_val_list))):
        effect_list.append([])
        for pop_name_idx, pop_name in enumerate(model.populations):
            v_arr = recordings[trial][f"{pop_name};r"]
            ### mean over neurons
            v_arr = np.mean(v_arr, axis=1)
            ### mean of first period
            v_mean_1 = np.mean(
                v_arr[
                    recording_times.idx_lims(chunk=trial)[0]
                    + int(round(500 / dt())) : recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1000 / dt()))
                ]
            )
            v_std_1 = np.std(
                v_arr[
                    recording_times.idx_lims(chunk=trial)[0]
                    + int(round(500 / dt())) : recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1000 / dt()))
                ]
            )
            ### mean of second period
            v_mean_2 = np.mean(
                v_arr[
                    recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1000 / dt())) : recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1500 / dt()))
                ]
            )
            ### mean of third period
            v_mean_3 = np.mean(
                v_arr[
                    recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1500 / dt())) : recording_times.idx_lims(chunk=trial)[0]
                    + int(round(2000 / dt()))
                ]
            )
            v_std_3 = np.std(
                v_arr[
                    recording_times.idx_lims(chunk=trial)[0]
                    + int(round(1500 / dt())) : recording_times.idx_lims(chunk=trial)[0]
                    + int(round(2000 / dt()))
                ]
            )
            ### get meand depending on dbs
            mean_on = v_mean_2
            mean_off = (v_mean_1 + v_mean_3) / 2
            std_off = (v_std_1 + v_std_3) / 2
            ### calculate effect
            effect = (mean_on - mean_off) / std_off
            if effect > 2.5:
                high_effect_list.append(abs(effect))
                effect = 1
            elif effect < -2.5:
                high_effect_list.append(abs(effect))
                effect = -1
            else:
                low_effect_list.append(abs(effect))
                effect = 0

            effect_list[trial_idx].append(effect)
    assert (
        np.array(effects_on_activity).astype(int) == np.array(effect_list).astype(int)
    ).all(), "Effects on activity not as expected for rate-coded model"


def plot_spiking(
    dbs_val_list: list[list],
    recordings: list,
    recording_times: RecordingTimes,
    model: CompNeuroModel,
    plotting: bool,
):
    """
    Plot spiking data.

    Args:
        dbs_val_list (list[list]):
            List of lists with DBS stimulation values used by the dbs.on() function
        recordings (list):
            List of recordings from the monitors
        recording_times (RecordingTimes):
            Recording times object
        model (CompNeuroModel):
            Model used for the simulation
        plotting (bool):
            If True, plots are created
    """
    if not plotting:
        return

    ### plot data
    for trial in range(len(dbs_val_list)):
        PlotRecordings(
            figname=f"DBS_spiking_figure/membrane_trial_{trial}.png",
            recordings=recordings,
            recording_times=recording_times,
            chunk=trial,
            shape=(3, 2),
            plan={
                "position": np.arange(len(model.populations), dtype=int) + 1,
                "compartment": model.populations,
                "variable": ["v"] * len(model.populations),
                "format": ["matrix"] * len(model.populations),
            },
            time_lim=(
                recording_times.time_lims(chunk=trial)[0] + 500,
                recording_times.time_lims(chunk=trial)[1] - 500,
            ),
        )
        PlotRecordings(
            figname=f"DBS_spiking_figure/axon_spikes_{trial}.png",
            recordings=recordings,
            recording_times=recording_times,
            chunk=trial,
            shape=(3, 2),
            plan={
                "position": np.arange(len(model.populations), dtype=int) + 1,
                "compartment": model.populations,
                "variable": ["axon_spike"] * len(model.populations),
                "format": ["raster"] * len(model.populations),
            },
            time_lim=(
                recording_times.time_lims(chunk=trial)[0] + 1000,
                recording_times.time_lims(chunk=trial)[0] + 1030,
            ),
        )


def plot_rate_coded(
    dbs_val_list: list[list],
    recordings: list,
    recording_times: RecordingTimes,
    model: CompNeuroModel,
    plotting: bool,
):
    """
    Plot rate-coded data.

    Args:
        dbs_val_list (list[list]):
            List of lists with DBS stimulation values used by the dbs.on() function
        recordings (list):
            List of recordings from the monitors
        recording_times (RecordingTimes):
            Recording times object
        model (CompNeuroModel):
            Model used for the simulation
        plotting (bool):
            If True, plots are created
    """
    if not plotting:
        return

    ### plot data
    for trial in range(len(dbs_val_list)):
        PlotRecordings(
            figname=f"DBS_rate_figure/activity_trial_{trial}.png",
            recordings=recordings,
            recording_times=recording_times,
            chunk=trial,
            shape=(3, 2),
            plan={
                "position": np.arange(len(model.populations), dtype=int) + 1,
                "compartment": model.populations,
                "variable": ["r"] * len(model.populations),
                "format": ["matrix"] * len(model.populations),
            },
            time_lim=(
                recording_times.time_lims(chunk=trial)[0] + 500,
                recording_times.time_lims(chunk=trial)[1] - 500,
            ),
        )


def main(plotting: bool = False):
    """
    Main function

    Args:
        plotting (bool, optional):
            If True, plots are created. Default: False
    """
    ### define simulations
    ### i.e. the parameters for the dbs stimulator on function
    ### do simulate calls repeatedly dbs.on() and dbs.off() with different parameters
    ### specified in dbs_val_list
    dbs_key_list = [
        "population_proportion",
        "dbs_depolarization",
        "orthodromic",
        "antidromic",
        "efferents",
        "afferents",
        "passing_fibres",
        "passing_fibres_strength",
        "axon_spikes_per_pulse",
        "axon_rate_amp",
    ]
    dbs_val_list = [
        # 0 - nothing
        [None, 0, False, False, False, False, False, 0.2, 1, 1],
        # 1 - orthodromic efferents
        [None, 0, True, False, True, False, False, 0.2, 1, 1],
        # 2 - orthodromic afferents
        [None, 0, True, False, False, True, False, 0.2, 1, 1],
        # 3 - orthodromic efferents and afferents
        [None, 0, True, False, True, True, False, 0.2, 1, 1],
        # 4 - orthodromic passing fibres
        [None, 0, True, False, False, False, True, 0.2, 1, 1],
        # 5 - depolarization
        [None, 100, False, False, False, False, False, 0.2, 1, 1],
        # 6 - antidromic efferents
        [None, 0, False, True, True, False, False, 0.2, 1, 1],
        # 7 - antidromic afferents
        [None, 0, False, True, False, True, False, 0.2, 1, 1],
        # 8 - antidromic efferents and afferents
        [None, 0, False, True, True, True, False, 0.2, 1, 1],
        # 9 - antidromic passing fibres
        [None, 0, False, True, False, False, True, 0.2, 1, 1],
        # 10 - antidromic passing fibres lower strength
        [None, 0, False, True, False, False, True, 0.01, 1, 1],
        # 11 - all
        [None, 100, True, True, True, True, True, 0.2, 1, 1],
        # 12 - all without axon spikes, should not affect rate-coded model
        [None, 100, True, True, True, True, True, 0.2, 0, 1],
        # 13 - all without axon_rate_amp, should not affect spiking model
        [None, 100, True, True, True, True, True, 0.2, 1, 0],
    ]

    spiking_model = True
    rate_coded_model = True

    if spiking_model:
        ### create the spiking network
        model = dbs_test_model_class("spiking").model
        dbs = DBSstimulator(
            stimulated_population=get_population("pop3_spiking"),
            passing_fibres_list=[get_projection("proj_5_6_spiking")],
            passing_fibres_strength=0.2,
            auto_implement=True,
            model=model,
        )
        model = dbs.model

        ### compile model
        model.compile(compile_folder_name="DBS_test_spiking")

        ### create monitors
        mon_dict = {}
        for pop_name in model.populations:
            mon_dict[pop_name] = ["v", "spike", "axon_spike"]
        mon = CompNeuroMonitors(mon_dict)

        ### run simulation and get data from monitors
        recordings, recording_times = do_simulation(
            mon, dbs, dbs_val_list, dbs_key_list
        )

        ### plot data
        plot_spiking(
            dbs_val_list=dbs_val_list,
            recordings=recordings,
            recording_times=recording_times,
            model=model,
            plotting=plotting,
        )

        ### check dbs effects
        check_dbs_effects_spiking(
            dbs_val_list,
            recordings,
            model,
            recording_times,
        )

    if rate_coded_model:
        ### create the rate-coded network
        cnp_clear()
        model = dbs_test_model_class("rate-coded").model
        dbs = DBSstimulator(
            stimulated_population=get_population("pop3_rate-coded"),
            passing_fibres_list=[get_projection("proj_5_6_rate-coded")],
            passing_fibres_strength=0.2,
            model=model,
            auto_implement=True,
        )
        model = dbs.model

        ### compile model
        model.compile(compile_folder_name="DBS_test_rate_coded")

        ### create monitors
        mon_dict = {}
        for pop_name in model.populations:
            mon_dict[pop_name] = ["r"]
        mon = CompNeuroMonitors(mon_dict)

        ### run simulation and get data from monitors
        recordings, recording_times = do_simulation(
            mon, dbs, dbs_val_list, dbs_key_list
        )

        ### plot data
        plot_rate_coded(
            dbs_val_list=dbs_val_list,
            recordings=recordings,
            recording_times=recording_times,
            model=model,
            plotting=plotting,
        )

        ### check dbs effects
        check_dbs_effects_rate_coded(
            dbs_val_list,
            recordings,
            model,
            recording_times,
        )
    return 1


if __name__ == "__main__":
    main(plotting=True)
