from CompNeuroPy import (
    cnp_clear,
    compile_in_folder,
    find_folder_with_prefix,
    data_obj,
    replace_names_with_dict,
)
from ANNarchy import (
    Population,
    get_population,
    Monitor,
    Network,
    get_projection,
    dt,
    parallel_run,
    simulate,
    reset,
    Neuron,
    simulate_until,
)
from ANNarchy.core.Global import _network
import numpy as np
from scipy.interpolate import interp1d, interpn
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import inspect
import textwrap
import os
import itertools
from tqdm import tqdm
import multiprocessing
import importlib.util
from time import time, strftime
import datetime
from sympy import symbols, Symbol, sympify, solve
from hyperopt import fmin, tpe, hp
import pandas as pd
from scipy.stats import poisson


class model_configurator:
    def __init__(
        self,
        model,
        target_firing_rate_dict,
        interpolation_grid_points=10,
        max_psp=10,
        do_not_config_list=[],
        print_guide=False,
        I_app_variable="I_app",
    ) -> None:
        """
        Args:
            model: CompNeuroPy generate_model object
                it's not important if the model is created or compiled but after running
                the model_configurator only the given model will exist so do not create
                something else in ANNarchy!

            target_firing_rate_dict: dict
                keys = population names of model which should be configured, values = target firing rates in Hz

            interpolation_grid_points: int, optional, default=10
                how many points should be used for the interpolation of the f-I-g curve on a single axis

            max_psp: int, optional, default=10
                maximum post synaptic potential in mV

            do_not_config_list: list, optional, default=[]
                list with strings containing population names of populations which should not be configured

            print_guide: bool, optional, default=False
                if you want to get information about what you could do with model_configurator

            I_app_variable: str, optional, default="I_app"
                the name of the varaible in the populations which represents the applied current
                TODO: not implemented yet, default value is always used

        Functions:
            get_max_syn:
                returns a dictionary with weight ranges for all afferent projections of the configured populations
        """
        self.model = model
        self.target_firing_rate_dict = target_firing_rate_dict
        self.pop_name_list = list(target_firing_rate_dict.keys())
        for do_not_pop_name in do_not_config_list:
            self.pop_name_list.remove(do_not_pop_name)
        self.I_app_max_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.g_max_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.tau_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.nr_afferent_proj_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.net_many_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.net_single_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.net_single_v_clamp_dict = {
            pop_name: None for pop_name in self.pop_name_list
        }
        self.max_weight_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.variable_init_sampler_dict = {
            pop_name: None for pop_name in self.pop_name_list
        }
        self.f_I_g_curve_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.I_f_g_curve_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.afferent_projection_dict = {
            pop_name: None for pop_name in self.pop_name_list
        }
        self.neuron_model_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.neuron_model_parameters_dict = {
            pop_name: None for pop_name in self.pop_name_list
        }
        self.neuron_model_attributes_dict = {
            pop_name: None for pop_name in self.pop_name_list
        }
        self.max_psp_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.possible_rates_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.extreme_firing_rates_df_dict = {
            pop_name: None for pop_name in self.pop_name_list
        }
        ### set max psp for a single spike
        self.max_psp_dict = {pop_name: max_psp for pop_name in self.pop_name_list}
        ### print things
        self.log_exist = False
        self.caller_name = ""
        self.log("model configurator log:")
        self.print_guide = print_guide
        ### simulation things
        self.simulation_dur = 5000
        self.simulation_dur_estimate_time = 50
        self.nr_neurons_per_net = 10000

        ### do things for which the model needs to be created (it will not be available later)
        self.analyze_model()

        ### print guide
        self._p_g(_p_g_1)

    def get_max_syn(self):
        """
        get the weight dictionary for all populations given in target_firing_rate_dict
        keys = population names, values = dict which contain values = afferent projection names, values = lists with w_min and w_max
        """

        ### create single neuron networks
        self.create_single_neuron_networks()

        ### get max synaptic things with single neuron networks
        for pop_name in self.pop_name_list:
            self.log(pop_name)
            ### get max synaptic currents (I and gs) using the single neuron networks
            txt = (
                f"get max I_app, g_ampa and g_gaba using network_single for {pop_name}"
            )
            print(txt)
            self.log(txt)
            I_app_max, g_ampa_max, g_gaba_max = self.get_max_syn_currents(
                pop_name=pop_name
            )
            self.I_app_max_dict[pop_name] = I_app_max
            self.g_max_dict[pop_name] = {
                "ampa": g_ampa_max,
                "gaba": g_gaba_max,
            }
            ### get the max_weight dict
            self.log(f"get the max_weight_dict for {pop_name}")
            self.max_weight_dict[pop_name] = self.get_max_weight_dict_for_pop(pop_name)

        ### print next steps
        ### create the synaptic load template dict
        self.syn_load_dict = {}
        for pop_name in self.pop_name_list:
            self.syn_load_dict[pop_name] = []
            if "ampa" in self.afferent_projection_dict[pop_name]["target"]:
                self.syn_load_dict[pop_name].append("ampa_load")
            if "gaba" in self.afferent_projection_dict[pop_name]["target"]:
                self.syn_load_dict[pop_name].append("gaba_load")
        ### create the synaptic contribution template dict
        self.syn_contr_dict = {}
        for pop_name in self.max_weight_dict.keys():
            self.syn_contr_dict[pop_name] = {}

            for target_type in ["ampa", "gaba"]:
                self.syn_contr_dict[pop_name][target_type] = {}

                nr_afferent_proj_with_target = len(
                    list(self.max_weight_dict[pop_name][target_type].keys())
                )
                for proj_name in self.max_weight_dict[pop_name][target_type].keys():
                    self.syn_contr_dict[pop_name][target_type][proj_name] = (
                        1 / nr_afferent_proj_with_target
                    )
        ### only return synaptic contributions smaller 1
        template_synaptic_contribution_dict = (
            self.get_template_synaptic_contribution_dict(given_dict=self.syn_contr_dict)
        )

        self._p_g(
            _p_g_after_get_weights(
                template_weight_dict=self.max_weight_dict,
                template_synaptic_load_dict=self.syn_load_dict,
                template_synaptic_contribution_dict=template_synaptic_contribution_dict,
            )
        )

        return self.max_weight_dict

    def create_single_neuron_networks(self):
        ### clear ANNarchy
        cnp_clear()
        ### create the single neuron networks
        for pop_name in self.pop_name_list:
            txt = f"create network_single for {pop_name}"
            print(txt)
            self.log(txt)
            ### the network with the standard neuron
            self.net_single_dict[pop_name] = self.create_net_single(pop_name=pop_name)
            ### the network with the voltage clamp version neuron
            self.net_single_v_clamp_dict[
                pop_name
            ] = self.create_net_single_voltage_clamp(pop_name=pop_name)

    def create_net_single(self, pop_name):
        """
        creates a network with the neuron type of the population given by pop_name
        the number of neurons is 1

        Args:
            pop_name: str
                population name
        """
        ### create the single neuron population
        single_neuron = Population(
            1,
            neuron=self.neuron_model_dict[pop_name],
            name=f"single_neuron_{pop_name}",
        )
        ### set the attributes of the neuron
        for attr_name, attr_val in self.neuron_model_parameters_dict[pop_name]:
            setattr(single_neuron, attr_name, attr_val)

        ### create Monitor for single neuron
        mon_single = Monitor(single_neuron, ["spike"])

        ### create network with single neuron
        net_single = Network()
        net_single.add([single_neuron, mon_single])
        compile_in_folder(
            folder_name=f"single_net_{pop_name}", silent=True, net=net_single
        )

        ### get the values of the variables after 2000 ms simulation
        variable_init_sampler = self.get_init_neuron_variables(
            net_single, net_single.get(single_neuron)
        )

        ### network dict
        net_single_dict = {
            "net": net_single,
            "population": net_single.get(single_neuron),
            "monitor": net_single.get(mon_single),
            "variable_init_sampler": variable_init_sampler,
        }

        return net_single_dict

    def get_init_neuron_variables(self, net, pop):
        """
        get the variables of the given population after simulating 2000 ms

        Args:
            net: ANNarchy network
                the network which contains the pop

            pop: ANNarchy population
                the population whose variables are obtained

        """
        ### reset neuron and deactivate input
        net.reset()
        pop.I_app = 0

        ### 1000 ms init duration
        net.simulate(1000)

        ### simulate 2000 ms and check every dt the variables of the neuron
        time_steps = int(2000 / dt())
        var_name_list = list(pop.variables)
        var_arr = np.zeros((time_steps, len(var_name_list)))
        for time_idx in range(time_steps):
            net.simulate(dt())
            get_arr = np.array([getattr(pop, var_name) for var_name in pop.variables])
            var_arr[time_idx, :] = get_arr[:, 0]
        net.reset()

        ### create a sampler with the data samples of from the 1000 ms simulation
        sampler = self.var_arr_sampler(var_arr)
        return sampler

    def create_net_single_voltage_clamp(self, pop_name):
        """
        creates a network with the neuron type of the population given by pop_name
        the number of neurons is 1

        The equation wich defines the chagne of v is set to zero and teh change of v
        is stored in the new variable v_clamp_rec

        Args:
            pop_name: str
                population name
        """

        ### get the initial arguments of the neuron
        neuron_model = self.neuron_model_dict[pop_name]
        ### names of arguments
        init_arguments_name_list = list(Neuron.__init__.__code__.co_varnames)
        init_arguments_name_list.remove("self")
        init_arguments_name_list.remove("name")
        init_arguments_name_list.remove("description")
        ### arguments dict
        init_arguments_dict = {
            init_arguments_name: getattr(neuron_model, init_arguments_name)
            for init_arguments_name in init_arguments_name_list
        }
        ### get new equations for voltage clamp
        equations_new = self.get_voltage_clamp_equations(init_arguments_dict, pop_name)
        init_arguments_dict["equations"] = equations_new
        ### add v_clamp_rec_thresh to the parameters
        parameters_line_split_list = str(init_arguments_dict["parameters"]).splitlines()
        parameters_line_split_list.append("v_clamp_rec_thresh = 0 : population")
        init_arguments_dict["parameters"] = "\n".join(parameters_line_split_list)

        ### create neuron model with new equations
        neuron_model_new = Neuron(**init_arguments_dict)

        ### create the single neuron population
        single_neuron_v_clamp = Population(
            1,
            neuron=neuron_model_new,
            name=f"single_neuron_v_clamp_{pop_name}",
            stop_condition="(abs(v_clamp_rec-v_clamp_rec_thresh)<1) and (abs(v_clamp_rec_pre-v_clamp_rec_thresh)>1) : any",
        )

        ### set the attributes of the neuron
        for attr_name, attr_val in self.neuron_model_parameters_dict[pop_name]:
            setattr(single_neuron_v_clamp, attr_name, attr_val)

        ### create Monitor for single neuron
        mon_single = Monitor(single_neuron_v_clamp, ["v_clamp_rec"])

        ### create network with single neuron
        net_single = Network()
        net_single.add([single_neuron_v_clamp, mon_single])
        compile_in_folder(
            folder_name=f"single_v_clamp_net_{pop_name}", silent=True, net=net_single
        )

        ### find v where dv/dt is minimal (best = 0)
        self.log("search v_rest with y(X) = delta_v_2000(v=X) using hyperopt")
        best = fmin(
            fn=lambda X_val: self.get_v_clamp_2000(
                v=X_val,
                net=net_single,
                population=net_single.get(single_neuron_v_clamp),
            ),
            space=hp.normal("v", -70, 30),
            algo=tpe.suggest,
            timeout=5,
            show_progressbar=False,
        )
        v_rest = best["v"]
        detla_v_rest = self.get_v_clamp_2000(
            v=v_rest,
            net=net_single,
            population=net_single.get(single_neuron_v_clamp),
        )
        self.log(f"found v_rest={v_rest} with delta_v_2000(v=v_rest)={detla_v_rest}")
        if detla_v_rest > 1:
            ### there seems to be no restign potential --> use -60 mV
            v_rest = -60
            self.log(f"since there is seems to be no v_rest --> set v_rest={v_rest}")

        ### get the variable_init_sampler for v=v_rest
        variable_init_sampler = self.get_init_neuron_variables_v_clamp(
            net_single, net_single.get(single_neuron_v_clamp), v_rest=v_rest
        )

        ### network dict
        net_single_dict = {
            "net": net_single,
            "population": net_single.get(single_neuron_v_clamp),
            "monitor": net_single.get(mon_single),
            "variable_init_sampler": variable_init_sampler,
        }

        return net_single_dict

    def log(self, txt):
        caller_frame = inspect.currentframe().f_back
        caller_name = caller_frame.f_code.co_name

        if caller_name == self.caller_name:
            txt = f"{textwrap.indent(str(txt), '    ')}"
        else:
            txt = f"[{caller_name}]:\n{textwrap.indent(str(txt), '    ')}"

        self.caller_name = caller_name

        if self.log_exist:
            with open("model_conf_log", "a") as f:
                print(txt, file=f)
        else:
            with open("model_conf_log", "w") as f:
                print(txt, file=f)
                self.log_exist = True

    def _p_g(self, txt):
        """
        prints guiding text
        """
        print_width = min([os.get_terminal_size().columns, 80])

        if self.print_guide:
            print("\n[model_configurator guide]:")
            for line in txt.splitlines():
                wrapped_text = textwrap.fill(
                    line, width=print_width - 5, replace_whitespace=False
                )
                wrapped_text = textwrap.indent(wrapped_text, "    |")
                print(wrapped_text)
        print("")

    def _p_w(self, txt):
        """
        prints warning
        """
        print_width = min([os.get_terminal_size().columns, 80])

        print("\n[model_configurator WARNING]:")
        for line in str(txt).splitlines():
            wrapped_text = textwrap.fill(
                line, width=print_width - 5, replace_whitespace=False
            )
            wrapped_text = textwrap.indent(wrapped_text, "    |")
            print(wrapped_text)
        print("")

    def get_base(self):
        """
        Obtain the baseline currents for the configured populations to obtian the target firing rates
        with the currently set weights, set by .set_weights or .set_syn_load

        return:
            I_base_dict, dict
                Dictionary with baseline curretns for all configured populations.
        """
        ### use interpolation to get baseline currents
        I_base_dict = {}
        target_firing_rate_changed = True
        nr_max_iter = 100
        nr_iter = 0
        while target_firing_rate_changed and nr_iter < nr_max_iter:
            ### predict baseline current values
            (
                target_firing_rate_changed,
                I_base_dict,
            ) = self.find_base_current()
            nr_iter += 1

        return I_base_dict

    def find_base_current(self):

        ### search through whole I_app space
        ### for each population simulate a network with 10000 neurons, each neuron has a different I_app value
        ### g_ampa and g_gaba values are internally created using
        ### the weigths stored in the afferent_projection dict
        ### and target firing rates stored in the target_firing_rate_dict
        I_app_arr_list = []
        weight_list_list = []
        proj_name_list_list = []
        rate_list_list = []
        ### TODO get lists which define the weights to the afferent populations
        ### TODO get lists which define the rates of the afferent populations
        ### maybe lists with the projection names and lists with the weight values
        ### the length of the lists has to be the number of networks i.e. the number of populations
        for pop_name in self.pop_name_list:
            ### get the weights, names and rates of the afferent populations
            weight_list = self.afferent_projection_dict[pop_name]["weights"]
            proj_name_list = self.afferent_projection_dict[pop_name]["projection_names"]
            rate_list = self.get_rate_list_for_pop(pop_name)
            ### append these lists to the list for all populations i.e. networks
            weight_list_list.append(weight_list)
            proj_name_list_list.append(proj_name_list)
            rate_list_list.append(rate_list)

        ### create model
        net_many_dict = self.create_many_neuron_network()

        ### create list with variable_init_samplers of populations
        variable_init_sampler_list = [
            self.net_single_dict[pop_name]["variable_init_sampler"]
            for pop_name in self.pop_name_list
        ]

        ### get firing rates obtained with all I_app values
        ### rates depend on the current weights and the current target firing rates
        nr_networks = len(self.pop_name_list)
        possible_firing_rates_list_list = parallel_run(
            method=get_rate_parallel,
            networks=net_many_dict["network_list"],
            **{
                "population": net_many_dict["population_list"],
                "variable_init_sampler": variable_init_sampler_list,
                "monitor": net_many_dict["monitor_list"],
                "I_app_arr": I_app_arr_list,
                "weight_list": weight_list_list,
                "proj_name_list": proj_name_list_list,
                "rate_list": rate_list_list,
                "simulation_dur": [self.simulation_dur] * nr_networks,
            },
        )

        ### catch if target firing rate in any population cannot be reached
        I_app_best_dict = {}
        target_firing_rate_changed = False
        for pop_idx, pop_name in enumerate(self.pop_name_list):
            target_firing_rate = self.target_firing_rate_dict[pop_name]
            possible_firing_rates_arr = np.array(
                possible_firing_rates_list_list[pop_idx]
            )
            I_app_arr = I_app_arr_list[pop_idx]
            possible_f_min = possible_firing_rates_arr.min()
            possible_f_max = possible_firing_rates_arr.max()
            if not (
                target_firing_rate >= possible_f_min
                and target_firing_rate <= possible_f_max
            ):
                new_target_firing_rate = np.array([possible_f_min, possible_f_max])[
                    np.argmin(
                        np.absolute(
                            np.array([possible_f_min, possible_f_max])
                            - target_firing_rate
                        )
                    )
                ]
                ### if the possible firing rates are too small --> what (high) firing rate could be maximally reached with a hypothetical g_ampa_max and I_app_max
                ### if the possible firing rates are too large --> waht (low) firing rate could be reached with g_gaba_max and -I_app_max
                warning_txt = f"WARNING get_possible_rates: target firing rate of population {pop_name}({target_firing_rate}) cannot be reached.\nPossible range with current synaptic load: [{round(possible_f_min,1)},{round(possible_f_max,1)}].\nSet firing rate to {round(new_target_firing_rate,1)}."
                self._p_w(warning_txt)
                self.log(warning_txt)
                self.target_firing_rate_dict[pop_name] = new_target_firing_rate
                target_firing_rate = self.target_firing_rate_dict[pop_name]
                target_firing_rate_changed = True
            ### find best I_app for reaching target firing rate
            best_idx = np.argmin(
                np.absolute(possible_firing_rates_arr - target_firing_rate)
            )
            I_app_best_dict[pop_name] = I_app_arr[best_idx]

        return [target_firing_rate_changed, I_app_best_dict]

    def get_rate_list_for_pop(self, pop_name):
        """
        get the rate list for the afferent populations of the given population
        """
        rate_list = []
        for proj_name in self.afferent_projection_dict[pop_name]["projection_names"]:
            proj_dict = self.get_proj_dict(proj_name)
            pre_pop_name = proj_dict["pre_pop_name"]
            pre_rate = self.target_firing_rate_dict[pre_pop_name]
            rate_list.append(pre_rate)
        return rate_list

    def set_base(self, I_base_dict=None, I_base_variable="base_mean"):
        """
        Set baseline currents in model, compile model and set weights in model.

        Args:
            I_base_dict: dict, optional, default=None
                Dictionary with baseline currents for all populations, if None the baselines are obtained by .get_base

            I_base_variable: str, optional, default="mean_base"
                Name of the variable which represents the baseline current in the configured populations. They all have to have the same variable.
        """
        ### check I_base_dict
        if isinstance(I_base_dict, type(None)):
            I_base_dict = self.get_base()

        ### clear annarchy, create model and set baselines and weights
        cnp_clear()
        self.model.create(do_compile=False)
        ### set initial variables of populations
        for pop_name in self.pop_name_list:
            population = get_population(pop_name)
            variable_init_sampler = self.net_single_dict[pop_name][
                "variable_init_sampler"
            ]
            variable_init_arr = variable_init_sampler.sample(len(population), seed=0)
            for var_idx, var_name in enumerate(population.variables):
                set_val = variable_init_arr[:, var_idx]
                setattr(population, var_name, set_val)
        ### set baselines
        for pop_name in I_base_dict.keys():
            get_val = getattr(get_population(pop_name), I_base_variable)
            try:
                set_val = np.ones(len(get_val)) * I_base_dict[pop_name]
            except:
                set_val = I_base_dict[pop_name]
            setattr(get_population(pop_name), I_base_variable, set_val)
        ### compile
        self.model.compile()
        ### set weights
        for pop_name in self.pop_name_list:
            for proj_idx, proj_name in enumerate(
                self.afferent_projection_dict[pop_name]["projection_names"]
            ):
                weight_val = self.afferent_projection_dict[pop_name]["weights"][
                    proj_idx
                ]
                get_projection(proj_name).w = weight_val

        return I_base_dict

    def get_time_in_x_sec(self, x):
        """
        Args:
            x: int
                how many seconds add to the current time

        return:
            formatted_future_time: str
                string of the future time in HH:MM:SS
        """
        # Get the current time
        current_time = datetime.datetime.now()

        # Add 10 seconds to the current time
        future_time = current_time + datetime.timedelta(seconds=x)

        # Format future_time as HH:MM:SS
        formatted_future_time = future_time.strftime("%H:%M:%S")

        return formatted_future_time

    def get_interpolation(self):
        """
        get the interpolations to
        predict f with I_app, g_ampa and g_gaba

        sets the class variable self.f_I_g_curve_dict --> for each population a f_I_g_curve function
        """

        ### create model
        net_many_dict = self.create_many_neuron_network()

        ### get interpolation data
        txt = "get interpolation data..."
        print(txt)
        self.log(txt)
        ### for each population get the input arrays for I_app, g_ampa and g_gaba
        ### while getting inputs define which values should be used later
        input_dict = self.get_input_for_many_neurons_net()

        ### create list with variable_init_samplers of populations
        variable_init_sampler_list = [
            self.net_single_dict[pop_name]["variable_init_sampler"]
            for pop_name in self.pop_name_list
        ]

        ### run the run_parallel with a reduced simulation duration and obtain a time estimate for the full duration
        ### TODO use directly measureing simulation time to get time estimate
        start = time()
        parallel_run(
            method=get_rate_parallel,
            number=self.nr_networks,
            **{
                "pop_name_list": [self.pop_name_list] * self.nr_networks,
                "population_list": [list(net_many_dict["population_dict"].values())]
                * self.nr_networks,
                "variable_init_sampler_list": [variable_init_sampler_list]
                * self.nr_networks,
                "monitor_list": [list(net_many_dict["monitor_dict"].values())]
                * self.nr_networks,
                "I_app_list": input_dict["I_app_list"],
                "g_ampa_list": input_dict["g_ampa_list"],
                "g_gaba_list": input_dict["g_gaba_list"],
                "simulation_dur": [dt()] * self.nr_networks,
            },
        )
        reset()
        end = time()
        offset_time = end - start
        start = time()
        parallel_run(
            method=get_rate_parallel,
            number=self.nr_networks,
            **{
                "pop_name_list": [self.pop_name_list] * self.nr_networks,
                "population_list": [list(net_many_dict["population_dict"].values())]
                * self.nr_networks,
                "variable_init_sampler_list": [variable_init_sampler_list]
                * self.nr_networks,
                "monitor_list": [list(net_many_dict["monitor_dict"].values())]
                * self.nr_networks,
                "I_app_list": input_dict["I_app_list"],
                "g_ampa_list": input_dict["g_ampa_list"],
                "g_gaba_list": input_dict["g_gaba_list"],
                "simulation_dur": [self.simulation_dur_estimate_time]
                * self.nr_networks,
            },
        )
        reset()
        end = time()
        time_estimate = np.clip(
            round(
                (end - start - offset_time)
                * (self.simulation_dur / self.simulation_dur_estimate_time),
                0,
            ),
            0,
            None,
        )

        txt = f"start parallel_run of many neurons network on {self.nr_networks} threads, will take approx. {time_estimate} s (end: {self.get_time_in_x_sec(x=time_estimate)})..."
        print(txt)
        self.log(txt)
        ### simulate the many neurons network with the input arrays splitted into the network populations sizes
        ### and get the data of all populations
        ### run_parallel
        start = time()
        f_rec_arr_list_list = parallel_run(
            method=get_rate_parallel,
            number=self.nr_networks,
            **{
                "pop_name_list": [self.pop_name_list] * self.nr_networks,
                "population_list": [list(net_many_dict["population_dict"].values())]
                * self.nr_networks,
                "variable_init_sampler_list": [variable_init_sampler_list]
                * self.nr_networks,
                "monitor_list": [list(net_many_dict["monitor_dict"].values())]
                * self.nr_networks,
                "I_app_list": input_dict["I_app_list"],
                "g_ampa_list": input_dict["g_ampa_list"],
                "g_gaba_list": input_dict["g_gaba_list"],
                "simulation_dur": [self.simulation_dur] * self.nr_networks,
            },
        )
        end = time()
        txt = f"took {end-start} s"
        print(txt)
        self.log(txt)

        ### combine the list of outputs from parallel_run to one output per population
        output_of_populations_dict = self.get_output_of_populations(
            f_rec_arr_list_list, input_dict
        )

        ### create interpolation for each population
        ### it can be a 1D to 3D interpolation, default (if everything works fine) is
        ### 3D interpolation with "x": "I_app", "y": "g_ampa", "z": "g_gaba"
        for pop_name in self.pop_name_list:
            ### get whole input arrays
            I_app_value_array = None
            g_ampa_value_array = None
            g_gaba_value_array = None
            if self.I_app_max_dict[pop_name] > 0:
                I_app_value_array = input_dict["I_app_arr_dict"][pop_name]
            if self.g_max_dict[pop_name]["ampa"] > 0:
                g_ampa_value_array = input_dict["g_ampa_arr_dict"][pop_name]
            if self.g_max_dict[pop_name]["gaba"] > 0:
                g_gaba_value_array = input_dict["g_gaba_arr_dict"][pop_name]

            ### get the interpolation
            self.f_I_g_curve_dict[pop_name] = self.get_interp_3p(
                values=output_of_populations_dict[pop_name],
                model_conf_obj=self,
                var_name_dict={"x": "I_app", "y": "g_ampa", "z": "g_gaba"},
                x=I_app_value_array,
                y=g_ampa_value_array,
                z=g_gaba_value_array,
            )

        self.did_get_interpolation = True

        ### with interpolation get the firing rates for all extreme values of I_app, g_ampa, g_gaba
        for pop_name in self.pop_name_list:
            self.extreme_firing_rates_df_dict[
                pop_name
            ] = self.get_extreme_firing_rates_df(pop_name)

    def get_extreme_firing_rates_df(self, pop_name):
        """
        get the firing rates for all extreme values of I_app, g_ampa, g_gaba

        Args:
            pop_name: str
                popualtion name

        return:
            table_df: pandas dataframe
                containing the firing rates for all extreme values of I_app, g_ampa, g_gaba
        """
        I_app_list = [-self.I_app_max_dict[pop_name], self.I_app_max_dict[pop_name]]
        g_ampa_list = [0, self.g_max_dict[pop_name]["ampa"]]
        g_gaba_list = [0, self.g_max_dict[pop_name]["gaba"]]
        ### create all combiniations of I_app_list, g_ampa_list, g_gaba_list in a single list
        comb_list = self.get_all_combinations_of_lists(
            [I_app_list, g_ampa_list, g_gaba_list]
        )

        ### get the firing rates for all combinations
        f_list = []
        for I_app, g_ampa, g_gaba in comb_list:
            f_list.append(
                self.f_I_g_curve_dict[pop_name](x=I_app, y=g_ampa, z=g_gaba)[0]
            )

        ### now get the same for names
        I_app_name_list = ["min", "max"]
        g_ampa_name_list = ["min", "max"]
        g_gaba_name_list = ["min", "max"]
        ### create all combiniations of I_app_name_list, g_ampa_name_list, g_gaba_name_list in a single list
        comb_name_list = self.get_all_combinations_of_lists(
            [I_app_name_list, g_ampa_name_list, g_gaba_name_list]
        )

        ### create a dict as table with header I_app, g_ampa, g_gaba
        table_dict = {
            "I_app": np.array(comb_name_list)[:, 0].tolist(),
            "g_ampa": np.array(comb_name_list)[:, 1].tolist(),
            "g_gaba": np.array(comb_name_list)[:, 2].tolist(),
            "f": f_list,
        }

        ### create a pandas dataframe from the table_dict
        table_df = pd.DataFrame(table_dict)

        return table_df

    def get_all_combinations_of_lists(self, list_of_lists):
        """
        get all combinations of lists in a single list
        example: [[1,2],[3,4],[5,6]] --> [[1,3,5],[1,3,6],[1,4,5],[1,4,6],[2,3,5],[2,3,6],[2,4,5],[2,4,6]]
        """
        return list(itertools.product(*list_of_lists))

    def get_output_of_populations(self, f_rec_arr_list_list, input_dict):
        """
        restructure the output of run_parallel so that for each population a single array with firing rates is obtained

        Args:
            f_rec_arr_list_list: list of lists of arrays
                first lists contain different network runs, second level lists contain arrays for the different populations
        return:
            output_pop_dict: dict of arrays
                for each population a single array with firing rates
        """
        output_pop_dict = {}
        for pop_name in self.pop_name_list:
            output_pop_dict[pop_name] = []
        ### first loop selecting the network
        for f_rec_arr_list in f_rec_arr_list_list:
            ### second loop selecting the population
            for pop_idx, pop_name in enumerate(self.pop_name_list):
                ### append the recorded values to the array of the corresponding population
                output_pop_dict[pop_name].append(f_rec_arr_list[pop_idx])

        ### concatenate the arrays of the individual populations
        for pop_name in self.pop_name_list:
            output_pop_dict[pop_name] = np.concatenate(output_pop_dict[pop_name])

        ### use the input dict to only use values which should be used
        ### lis of lists, first list level = networks, second list level = populations then you get array with input values
        ### so same format as f_rec_arr_list_list
        use_I_app_arr_list_list = input_dict["use_I_app_list"]
        use_g_ampa_arr_list_list = input_dict["use_g_ampa_list"]
        use_g_gaba_arr_list_list = input_dict["use_g_gaba_list"]

        ### now get for each population an array which contains the info if the values should be used
        use_output_pop_dict = {}
        for pop_name in self.pop_name_list:
            use_output_pop_dict[pop_name] = []
        ### first loop selecting the network
        for net_idx in range(len(use_I_app_arr_list_list)):
            use_I_app_arr_list = use_I_app_arr_list_list[net_idx]
            use_g_ampa_arr_list = use_g_ampa_arr_list_list[net_idx]
            use_g_gaba_arr_list = use_g_gaba_arr_list_list[net_idx]
            ### second loop selecting the population
            for pop_idx, pop_name in enumerate(self.pop_name_list):
                ### only use values if for all input values use is True
                use_I_app_arr = use_I_app_arr_list[pop_idx]
                use_g_ampa_arr = use_g_ampa_arr_list[pop_idx]
                use_g_gaba_arr = use_g_gaba_arr_list[pop_idx]
                use_value_arr = np.logical_and(use_I_app_arr, use_g_ampa_arr)
                use_value_arr = np.logical_and(use_value_arr, use_g_gaba_arr)
                ### append the recorded values to the array of the corresponding population
                use_output_pop_dict[pop_name].append(use_value_arr)

        ### concatenate the arrays of the individual populations
        for pop_name in self.pop_name_list:
            use_output_pop_dict[pop_name] = np.concatenate(
                use_output_pop_dict[pop_name]
            )

        ### finaly only use values defined by ues_output...
        for pop_name in self.pop_name_list:
            output_pop_dict[pop_name] = output_pop_dict[pop_name][
                use_output_pop_dict[pop_name]
            ]

        return output_pop_dict

    def get_input_for_many_neurons_net(self):
        """
        get the inputs for the parallel many neurons network simulation

        need a list of dicts, keys=pop_name, lsit=number of networks
        """

        ### create dicts with lists for the populations
        I_app_arr_list_dict = {}
        g_ampa_arr_list_dict = {}
        g_gaba_arr_list_dict = {}
        use_I_app_arr_list_dict = {}
        use_g_ampa_arr_list_dict = {}
        use_g_gaba_arr_list_dict = {}
        I_app_arr_dict = {}
        g_ampa_arr_dict = {}
        g_gaba_arr_dict = {}
        for pop_name in self.pop_name_list:
            ### prepare grid for I, g_ampa and g_gaba
            ### bounds
            g_ampa_max = self.g_max_dict[pop_name]["ampa"]
            g_gaba_max = self.g_max_dict[pop_name]["gaba"]
            I_max = self.I_app_max_dict[pop_name]

            ### create value_arrays
            I_app_value_array = np.linspace(
                -I_max, I_max, self.nr_vals_interpolation_grid
            )
            g_ampa_value_array = np.linspace(
                0, g_ampa_max, self.nr_vals_interpolation_grid
            )
            g_gaba_value_array = np.linspace(
                0, g_gaba_max, self.nr_vals_interpolation_grid
            )

            ### store these value arrays for each pop
            I_app_arr_dict[pop_name] = I_app_value_array
            g_ampa_arr_dict[pop_name] = g_ampa_value_array
            g_gaba_arr_dict[pop_name] = g_gaba_value_array

            ### create use values arrays
            use_I_app_array = np.array([I_max > 0] * self.nr_vals_interpolation_grid)
            use_g_ampa_array = np.array(
                [g_ampa_max > 0] * self.nr_vals_interpolation_grid
            )
            use_g_gaba_array = np.array(
                [g_gaba_max > 0] * self.nr_vals_interpolation_grid
            )
            ### use at least a single value
            use_I_app_array[0] = True
            use_g_ampa_array[0] = True
            use_g_gaba_array[0] = True

            ### get all combinations (grid) of value_arrays
            I_g_arr = np.array(
                list(
                    itertools.product(
                        *[I_app_value_array, g_ampa_value_array, g_gaba_value_array]
                    )
                )
            )

            ### get all combinations (grid) of the use values arrays
            use_I_g_arr = np.array(
                list(
                    itertools.product(
                        *[use_I_app_array, use_g_ampa_array, use_g_gaba_array]
                    )
                )
            )

            ### individual value arrays from combinations
            I_app_arr = I_g_arr[:, 0]
            g_ampa_arr = I_g_arr[:, 1]
            g_gaba_arr = I_g_arr[:, 2]

            ### individual use values arrays from combinations
            use_I_app_arr = use_I_g_arr[:, 0]
            use_g_ampa_arr = use_I_g_arr[:, 1]
            use_g_gaba_arr = use_I_g_arr[:, 2]

            ### split the arrays for the networks
            networks_size_list = np.array(
                [self.nr_neurons_of_pop_per_net] * self.nr_networks
            )
            split_idx_arr = np.cumsum(networks_size_list)[:-1]
            ### after this split the last array may be smaller than the others --> append zeros
            ### value arrays
            I_app_arr_list = np.split(I_app_arr, split_idx_arr)
            g_ampa_arr_list = np.split(g_ampa_arr, split_idx_arr)
            g_gaba_arr_list = np.split(g_gaba_arr, split_idx_arr)
            ### use value arrays
            use_I_app_arr_list = np.split(use_I_app_arr, split_idx_arr)
            use_g_ampa_arr_list = np.split(use_g_ampa_arr, split_idx_arr)
            use_g_gaba_arr_list = np.split(use_g_gaba_arr, split_idx_arr)

            ### check if last network is smaler
            if self.nr_last_network < self.nr_neurons_of_pop_per_net:
                ### if yes --> append zeros to value arrays
                ### and append False to use values arrays
                nr_of_zeros_append = round(
                    self.nr_neurons_of_pop_per_net - self.nr_last_network, 0
                )
                ### value arrays
                I_app_arr_list[-1] = np.concatenate(
                    [I_app_arr_list[-1], np.zeros(nr_of_zeros_append)]
                )
                g_ampa_arr_list[-1] = np.concatenate(
                    [g_ampa_arr_list[-1], np.zeros(nr_of_zeros_append)]
                )
                g_gaba_arr_list[-1] = np.concatenate(
                    [g_gaba_arr_list[-1], np.zeros(nr_of_zeros_append)]
                )
                ### use values arrays
                use_I_app_arr_list[-1] = np.concatenate(
                    [use_I_app_arr_list[-1], np.array([False] * nr_of_zeros_append)]
                )
                use_g_ampa_arr_list[-1] = np.concatenate(
                    [use_g_ampa_arr_list[-1], np.array([False] * nr_of_zeros_append)]
                )
                use_g_gaba_arr_list[-1] = np.concatenate(
                    [use_g_gaba_arr_list[-1], np.array([False] * nr_of_zeros_append)]
                )

            ### store the array lists into the population dicts
            ### value arrays
            I_app_arr_list_dict[pop_name] = I_app_arr_list
            g_ampa_arr_list_dict[pop_name] = g_ampa_arr_list
            g_gaba_arr_list_dict[pop_name] = g_gaba_arr_list
            ### use value arrays
            use_I_app_arr_list_dict[pop_name] = use_I_app_arr_list
            use_g_ampa_arr_list_dict[pop_name] = use_g_ampa_arr_list
            use_g_gaba_arr_list_dict[pop_name] = use_g_gaba_arr_list

        ### restructure the dict of lists into a list for networks of list for populations
        I_app_list = []
        g_ampa_list = []
        g_gaba_list = []
        use_I_app_list = []
        use_g_ampa_list = []
        use_g_gaba_list = []
        for net_idx in range(self.nr_networks):
            ### value arrays
            I_app_list.append(
                [
                    I_app_arr_list_dict[pop_name][net_idx]
                    for pop_name in self.pop_name_list
                ]
            )
            g_ampa_list.append(
                [
                    g_ampa_arr_list_dict[pop_name][net_idx]
                    for pop_name in self.pop_name_list
                ]
            )
            g_gaba_list.append(
                [
                    g_gaba_arr_list_dict[pop_name][net_idx]
                    for pop_name in self.pop_name_list
                ]
            )
            ### use values arrays
            use_I_app_list.append(
                [
                    use_I_app_arr_list_dict[pop_name][net_idx]
                    for pop_name in self.pop_name_list
                ]
            )
            use_g_ampa_list.append(
                [
                    use_g_ampa_arr_list_dict[pop_name][net_idx]
                    for pop_name in self.pop_name_list
                ]
            )
            use_g_gaba_list.append(
                [
                    use_g_gaba_arr_list_dict[pop_name][net_idx]
                    for pop_name in self.pop_name_list
                ]
            )

        return {
            "I_app_list": I_app_list,
            "g_ampa_list": g_ampa_list,
            "g_gaba_list": g_gaba_list,
            "use_I_app_list": use_I_app_list,
            "use_g_ampa_list": use_g_ampa_list,
            "use_g_gaba_list": use_g_gaba_list,
            "I_app_arr_dict": I_app_arr_dict,
            "g_ampa_arr_dict": g_ampa_arr_dict,
            "g_gaba_arr_dict": g_gaba_arr_dict,
        }

        for pop_name in self.pop_name_list:
            ### prepare grid for I, g_ampa and g_gaba
            ### bounds
            g_ampa_max = self.g_max_dict[pop_name]["ampa"]
            g_gaba_max = self.g_max_dict[pop_name]["gaba"]
            I_max = self.I_app_max_dict[pop_name]
            ### number of points for individual value arrays: I, g_ampa and g_gaba
            number_of_points = np.round(
                self.nr_neurons_net_many_total ** (1 / 3), 0
            ).astype(int)
            ### create value_arrays
            I_app_value_array = np.linspace(-I_max, I_max, number_of_points)
            g_ampa_value_array = np.linspace(0, g_ampa_max, number_of_points)
            g_gaba_value_array = np.linspace(0, g_gaba_max, number_of_points)
            ### get all combinations (grid) of value_arrays
            I_g_arr = np.array(
                list(
                    itertools.product(
                        *[I_app_value_array, g_ampa_value_array, g_gaba_value_array]
                    )
                )
            )
            ### individual value arrays from combinations
            I_app_arr = I_g_arr[:, 0]
            g_ampa_arr = I_g_arr[:, 1]
            g_gaba_arr = I_g_arr[:, 2]

            ### split the arrays into the sizes of the many-neuron networks
            split_idx_arr = np.cumsum(self.nr_many_neurons_list[pop_name])[:-1]

            I_app_arr_list = np.split(I_app_arr, split_idx_arr)
            g_ampa_arr_list = np.split(g_ampa_arr, split_idx_arr)
            g_gaba_arr_list = np.split(g_gaba_arr, split_idx_arr)

    class get_interp_3p:
        def __init__(
            self, values, model_conf_obj, var_name_dict, x=None, y=None, z=None
        ) -> None:
            """
            x, y, and z are the increasing gid steps on the interpolation grid
            set z=None to get 2D interpiolation
            set y and z = None to get 1D interpolation
            """
            self.x = x
            self.y = y
            self.z = z
            self.values = values
            self.model_conf_obj = model_conf_obj
            self.var_name_dict = var_name_dict

            if (
                isinstance(self.x, type(None))
                and isinstance(self.y, type(None))
                and isinstance(self.z, type(None))
            ):
                error_msg = (
                    "ERROR get_interp_3p: at least one of x,y,z has to be an array"
                )
                model_conf_obj.log(error_msg)
                raise AssertionError(error_msg)

        def __call__(self, x=None, y=None, z=None):
            ### check x
            if isinstance(x, type(None)):
                if not isinstance(self.x, type(None)):
                    error_msg = f"ERROR get_interp_3p: interpolation values for {self.var_name_dict['x']} were given but sample points are missing!"
                    self.model_conf_obj.log(error_msg)
                    raise AssertionError(error_msg)
                tmp_x = 0
            else:
                if isinstance(self.x, type(None)):
                    warning_txt = f"WARNING get_interp_3p: sample points for {self.var_name_dict['x']} are given but no interpolation values for {self.var_name_dict['x']} were given!"
                    self.model_conf_obj.log(warning_txt)
                    x = None
                    tmp_x = 0
                else:
                    tmp_x = x

            ### check y
            if isinstance(y, type(None)):
                if not isinstance(self.y, type(None)):
                    error_msg = f"ERROR get_interp_3p: interpolation values for {self.var_name_dict['y']} were given but sample points are missing!"
                    self.model_conf_obj.log(error_msg)
                    raise AssertionError(error_msg)
                tmp_y = 0
            else:
                if isinstance(self.y, type(None)):
                    warning_txt = f"WARNING get_interp_3p: sample points for {self.var_name_dict['y']} are given but no interpolation values for {self.var_name_dict['y']} were given!"
                    self.model_conf_obj.log(warning_txt)
                    y = None
                    tmp_y = 0
                else:
                    tmp_y = y

            ### check z
            if isinstance(z, type(None)):
                if not isinstance(self.y, type(None)):
                    error_msg = f"ERROR get_interp_3p: interpolation values for {self.var_name_dict['z']} were given but sample points are missing!"
                    self.model_conf_obj.log(error_msg)
                    raise AssertionError(error_msg)
                tmp_z = 0
            else:
                if isinstance(self.z, type(None)):
                    warning_txt = f"WARNING get_interp_3p: sample points for {self.var_name_dict['z']} are given but no interpolation values for {self.var_name_dict['z']} were given!"
                    self.model_conf_obj._p_w(warning_txt)
                    self.model_conf_obj.log(warning_txt)
                    z = None
                    tmp_z = 0
                else:
                    tmp_z = z

            ### get input arrays
            input_arr_dict = {
                "x": np.array(tmp_x).reshape(-1),
                "y": np.array(tmp_y).reshape(-1),
                "z": np.array(tmp_z).reshape(-1),
            }

            ### check if the arrays with size larger 1 have same size
            size_arr = np.array([val.size for val in input_arr_dict.values()])
            mask = size_arr > 1
            if True in mask:
                input_size = size_arr[mask][0]
                if not (input_size == size_arr[mask]).all():
                    raise ValueError(
                        "ERROR model_configurator get_interp_3p: x,y,z sample points have to be either single values or arrays. All arrays have to have same size"
                    )

            ### if there are inputs only consisting of a single value --> duplicate to increase size if there are also array inputs
            for idx, larger_1 in enumerate(mask):
                if not larger_1 and True in mask:
                    val = input_arr_dict[list(input_arr_dict.keys())[idx]][0]
                    input_arr_dict[list(input_arr_dict.keys())[idx]] = (
                        np.ones(input_size) * val
                    )

            ### get the sample points
            use_variable_names_list = ["x", "y", "z"]
            if isinstance(x, type(None)):
                use_variable_names_list.remove("x")
            if isinstance(y, type(None)):
                use_variable_names_list.remove("y")
            if isinstance(z, type(None)):
                use_variable_names_list.remove("z")
            point_arr = np.array(
                [input_arr_dict[var_name] for var_name in use_variable_names_list]
            ).T

            ### get the grid points, only use these which are not None
            use_variable_names_list = ["x", "y", "z"]
            if isinstance(self.x, type(None)):
                use_variable_names_list.remove("x")
            if isinstance(self.y, type(None)):
                use_variable_names_list.remove("y")
            if isinstance(self.z, type(None)):
                use_variable_names_list.remove("z")

            interpolation_grid_arr_dict = {
                "x": self.x,
                "y": self.y,
                "z": self.z,
            }
            points = tuple(
                [
                    interpolation_grid_arr_dict[var_name]
                    for var_name in use_variable_names_list
                ]
            )

            ### get shape of values
            values_shape = tuple(
                [
                    interpolation_grid_arr_dict[var_name].size
                    for var_name in use_variable_names_list
                ]
            )

            return interpn(
                points=points,
                values=self.values.reshape(values_shape),
                xi=point_arr,
            )

    def set_syn_load(self, synaptic_load_dict, synaptic_contribution_dict=None):
        """
        Args:
            synaptic_load_dict: dict or number
                either a dictionary with keys = all population names the model_configurator should configure
                or a single number between 0 and 1
                The dictionary values should be lists which contain either 2 values for ampa and gaba load,
                only 1 value if the population has only ampa or gaba input.
                For the strucutre of the dictionary check the print_guide

            synaptic_contribution_dict: dict, optional, default=None
                by default the synaptic contributions of all afferent projections is equal
                one can define other contributions in this dict
                give for each affernt projection the contribution to the synaptic load of the target population
                For the strucutre of the dictionary check the print_guide
        """

        ### synaptic load
        ### is dict --> replace internal dict values
        if isinstance(synaptic_load_dict, dict):
            ### check if correct number of population
            if len(list(synaptic_load_dict.keys())) != len(
                list(self.syn_load_dict.keys())
            ):
                error_msg = f"ERROR set_syn_load: wrong number of populations given with 'synaptic_load_dict' given={len(list(synaptic_load_dict.keys()))}, expected={len(list(self.syn_load_dict.keys()))}"
                self.log(error_msg)
                raise ValueError(error_msg)
            ### loop over all populations
            for pop_name in synaptic_load_dict.keys():
                ### cehck pop name
                if pop_name not in list(self.syn_load_dict.keys()):
                    error_msg = f"ERROR set_syn_load: the given population {pop_name} is not within the list of populations which should be configured {self.pop_name_list}"
                    self.log(error_msg)
                    raise ValueError(error_msg)
                value_list = synaptic_load_dict[pop_name]
                ### check value list
                if len(value_list) != len(self.syn_load_dict[pop_name]):
                    error_msg = f"ERROR set_syn_load: for population {pop_name}, {len(self.syn_load_dict[pop_name])} syn load values should be given but {len(value_list)} were given"
                    self.log(error_msg)
                    raise ValueError(error_msg)
                if not (
                    (np.array(value_list) <= 1).all()
                    and (np.array(value_list) >= 0).all()
                ):
                    error_msg = f"ERROR set_syn_load: the values for synaptic loads should be equal or smaller than 1, given for population {pop_name}: {value_list}"
                    self.log(error_msg)
                    raise ValueError(error_msg)
                ### replace internal values with given values
                self.syn_load_dict[pop_name] = value_list
        else:
            ### is not a dict --> check number
            try:
                synaptic_load = float(synaptic_load_dict)
            except:
                error_msg = "ERROR set_syn_load: if synaptic_load_dict is not a dictionary it should be a single number!"
                self.log(error_msg)
                raise ValueError(error_msg)
            if not (synaptic_load <= 1 and synaptic_load >= 0):
                error_msg = "ERROR set_syn_load: value for synaptic_loadshould be equal or smaller than 1"
                self.log(error_msg)
                raise ValueError(error_msg)
            ### replace internal values with given value
            for pop_name in self.syn_load_dict.keys():
                for idx in range(len(self.syn_load_dict[pop_name])):
                    self.syn_load_dict[pop_name][idx] = synaptic_load
        ### transform syn load dict in correct form with projection target type keys
        syn_load_dict = {}
        for pop_name in self.pop_name_list:
            syn_load_dict[pop_name] = {}
            if (
                "ampa" in self.afferent_projection_dict[pop_name]["target"]
                and "gaba" in self.afferent_projection_dict[pop_name]["target"]
            ):
                syn_load_dict[pop_name]["ampa"] = self.syn_load_dict[pop_name][0]
                syn_load_dict[pop_name]["gaba"] = self.syn_load_dict[pop_name][1]
            elif "ampa" in self.afferent_projection_dict[pop_name]["target"]:
                syn_load_dict[pop_name]["ampa"] = self.syn_load_dict[pop_name][0]
                syn_load_dict[pop_name]["gaba"] = None
            elif "gaba" in self.afferent_projection_dict[pop_name]["target"]:
                syn_load_dict[pop_name]["ampa"] = None
                syn_load_dict[pop_name]["gaba"] = self.syn_load_dict[pop_name][0]
        self.syn_load_dict = syn_load_dict

        ### synaptic contribution
        if not isinstance(synaptic_contribution_dict, type(None)):
            ### loop over all given populations
            for pop_name in synaptic_contribution_dict.keys():
                ### check pop_name
                if pop_name not in list(self.syn_contr_dict.keys()):
                    error_msg = f"ERROR set_syn_load: the given population {pop_name} is not within the list of populations which should be configured {self.pop_name_list}"
                    self.log(error_msg)
                    raise ValueError(error_msg)
                ### loop over given projeciton target type (ampa,gaba)
                for given_proj_target_type in synaptic_contribution_dict[
                    pop_name
                ].keys():
                    ### check given target type
                    if not (
                        given_proj_target_type == "ampa"
                        or given_proj_target_type == "gaba"
                    ):
                        error_msg = f"ERROR set_syn_load: with the synaptic_contribution_dict for each given population a 'ampa' and/or 'gaba' dictionary contianing the corresponding afferent projections should be given, given key={given_proj_target_type}"
                        self.log(error_msg)
                        raise ValueError(error_msg)
                    ### check if for the projection target type the correct number of projections is given
                    given_proj_name_list = list(
                        synaptic_contribution_dict[pop_name][
                            given_proj_target_type
                        ].keys()
                    )
                    internal_proj_name_list = list(
                        self.syn_contr_dict[pop_name][given_proj_target_type].keys()
                    )
                    if len(given_proj_name_list) != len(internal_proj_name_list):
                        error_msg = f"ERROR set_syn_load: in synaptic_contribution_dict for population {pop_name} and target_type {given_proj_target_type} wrong number of projections is given\ngiven={given_proj_name_list}, expected={internal_proj_name_list}"
                        self.log(error_msg)
                        raise ValueError(error_msg)
                    ### check if given contributions for the target type sum up to 1
                    given_contribution_arr = np.array(
                        list(
                            synaptic_contribution_dict[pop_name][
                                given_proj_target_type
                            ].values()
                        )
                    )
                    if round(given_contribution_arr.sum(), 6) != 1:
                        error_msg = f"ERROR set_syn_load: given synaptic contributions for population {pop_name} and target_type {given_proj_target_type} do not sum up to 1: given={given_contribution_arr}-->{round(given_contribution_arr.sum(),6)}"
                        self.log(error_msg)
                        raise ValueError(error_msg)
                    ### loop over given afferent projections
                    for proj_name in given_proj_name_list:
                        ### check if projection name exists
                        if proj_name not in internal_proj_name_list:
                            error_msg = f"ERROR set_syn_load: given projection {proj_name} given with synaptic_contribution_dict no possible projection, possible={internal_proj_name_list}"
                            self.log(error_msg)
                            raise ValueError(error_msg)
                        ### replace internal value of the projection with given value
                        self.syn_contr_dict[pop_name][given_proj_target_type][
                            proj_name
                        ] = synaptic_contribution_dict[pop_name][
                            given_proj_target_type
                        ][
                            proj_name
                        ]

        ### create weight dict from synaptic load/contributions
        self.weight_dict = {}
        for pop_name in self.max_weight_dict.keys():
            self.weight_dict[pop_name] = {}
            for proj_target_type in self.max_weight_dict[pop_name].keys():
                self.weight_dict[pop_name][proj_target_type] = {}
                for proj_name in self.max_weight_dict[pop_name][
                    proj_target_type
                ].keys():
                    weight_val = self.max_weight_dict[pop_name][proj_target_type][
                        proj_name
                    ]
                    syn_load = self.syn_load_dict[pop_name][proj_target_type]
                    syn_contr = self.syn_contr_dict[pop_name][proj_target_type][
                        proj_name
                    ]
                    weight_val_scaled = weight_val * syn_load * syn_contr
                    self.weight_dict[pop_name][proj_target_type][
                        proj_name
                    ] = weight_val_scaled

        ### set the weights in the afferent_projection_dict
        for pop_name in self.max_weight_dict.keys():
            weight_list = []
            for proj_name in self.afferent_projection_dict[pop_name][
                "projection_names"
            ]:
                ### get proj info
                proj_dict = self.get_proj_dict(proj_name)
                proj_target_type = proj_dict["proj_target_type"]
                ### store the weight from the weight_dict within weight_list
                weight_list.append(
                    self.weight_dict[pop_name][proj_target_type][proj_name]
                )
            self.afferent_projection_dict[pop_name]["weights"] = weight_list

        ### print guide
        self._p_g(_p_g_after_set_syn_load)

    def get_template_synaptic_contribution_dict(self, given_dict):
        """
        converts the full template dict with all keys for populations, target-types and projections into a reduced dict
        which only contains the keys which lead to values smaller 1
        """

        ret_dict = {}
        for key in given_dict.keys():
            if isinstance(given_dict[key], dict):
                rec_dict = self.get_template_synaptic_contribution_dict(given_dict[key])
                if len(rec_dict) > 0:
                    ret_dict[key] = self.get_template_synaptic_contribution_dict(
                        given_dict[key]
                    )
            else:
                if given_dict[key] < 1:
                    ret_dict[key] = "contr"

        return ret_dict

    def divide_almost_equal(self, number, num_parts):
        # Calculate the quotient and remainder
        quotient, remainder = divmod(number, num_parts)

        # Initialize a list to store the almost equal integers
        result = [quotient] * num_parts

        # Distribute the remainder evenly among the integers
        for i in range(remainder):
            result[i] += 1

        return result

    def analyze_model(self):
        """
        prepares the creation of the single neuron and many neuron networks
        """

        ### clear ANNarchy and create the model
        cnp_clear()
        self.model.create(do_compile=False)

        ### get the neuron models from the model
        for pop_name in self.pop_name_list:
            self.neuron_model_dict[pop_name] = get_population(pop_name).neuron_type
            self.neuron_model_parameters_dict[pop_name] = get_population(
                pop_name
            ).init.items()
            self.neuron_model_attributes_dict[pop_name] = get_population(
                pop_name
            ).attributes

        ### do further things for which the model needs to be created
        ### get the afferent projection dict for the populations (model needed!)
        for pop_name in self.pop_name_list:
            ### get afferent projection dict
            self.log(f"get the afferent_projection_dict for {pop_name}")
            self.afferent_projection_dict[pop_name] = self.get_afferent_projection_dict(
                pop_name=pop_name
            )

        ### create dictionary with timeconstants of g_ampa and g_gaba of the populations
        for pop_name in self.pop_name_list:
            self.tau_dict[pop_name] = {
                "ampa": get_population(pop_name).tau_ampa,
                "gaba": get_population(pop_name).tau_gaba,
            }

        ### get the post_pop_name_dict
        self.post_pop_name_dict = {}
        for proj_name in self.model.projections:
            self.post_pop_name_dict[proj_name] = get_projection(proj_name).post.name

        ### get the pre_pop_name_dict
        self.pre_pop_name_dict = {}
        for proj_name in self.model.projections:
            self.pre_pop_name_dict[proj_name] = get_projection(proj_name).pre.name

        ### clear ANNarchy --> the model is not available anymore
        cnp_clear()

    def compile_net_many_sequential(self):
        network_list = [
            net_many_dict["net"]
            for net_many_dict_list in self.net_many_dict.values()
            for net_many_dict in net_many_dict_list
        ]
        for net in network_list:
            self.compile_net_many(net=net)

    def compile_net_many_parallel(self):
        nr_available_workers = int(multiprocessing.cpu_count() / 2)
        network_list = [
            net_many_dict["net"]
            for net_many_dict_list in self.net_many_dict.values()
            for net_many_dict in net_many_dict_list
        ]
        with multiprocessing.Pool(nr_available_workers) as p:
            p.map(self.compile_net_many, network_list)

        ### for each network have network idx
        ### network 0 is base network
        ### netork 1,2,3...N are the single neuron networks for the N populations
        ### start idx = N+1 (inclusive), end_idx = number many networks + N (inclusive)
        for net_idx in range(
            len(self.pop_name_list) + 1, len(network_list) + len(self.pop_name_list) + 1
        ):
            ### get the name of the run folder of the network
            ### search for a folder which starts with run_
            ### there should only be 1 --> get run_folder_name as str
            run_folder_name = find_folder_with_prefix(
                base_path=f"annarchy_folders/many_net_{net_idx}", prefix="run_"
            )
            run_folder_name = f"/scratch/olmai/Projects/PhD/CompNeuroPy/CompNeuroPy/examples/model_configurator/annarchy_folders/many_net_{net_idx}//{run_folder_name}"

            print(run_folder_name)
            ### import the ANNarchyCore.so module from this folder
            spec = importlib.util.spec_from_file_location(
                f"ANNarchyCore{net_idx}", f"{run_folder_name}/ANNarchyCore{net_idx}.so"
            )
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)

            ### overwrite the entries in the network manager
            _network[net_idx]["instance"] = foo
            _network[net_idx]["compiled"] = True
            _network[net_idx]["directory"] = run_folder_name

    def get_afferent_projection_dict(self, pop_name):
        """
        creates a dictionary containing
            projection_names
            target firing rate
            probability
            size
            target
        for each afferent projection (=first level of keys) of the specified population

        Args:
            pop_name: str
                populaiton name

        return: dict of dicts
        """
        ### check if model is available
        if not self.model.created:
            error_msg = "ERROR model_configurator get_afferent_projection_dict: the model has to be created!"
            self.log(error_msg)
            raise AssertionError(error_msg)
        ### get projection names
        afferent_projection_dict = {}
        afferent_projection_dict["projection_names"] = []
        for projection in self.model.projections:
            if get_projection(projection).post.name == pop_name:
                afferent_projection_dict["projection_names"].append(projection)

        self.nr_afferent_proj_dict[pop_name] = len(
            afferent_projection_dict["projection_names"]
        )

        ### get target firing rates resting-state for afferent projections
        afferent_projection_dict["target firing rate"] = []
        afferent_projection_dict["probability"] = []
        afferent_projection_dict["size"] = []
        afferent_projection_dict["target"] = []
        for projection in afferent_projection_dict["projection_names"]:
            pre_pop_name = get_projection(projection).pre.name
            ### target firing rate
            afferent_projection_dict["target firing rate"].append(
                self.target_firing_rate_dict[pre_pop_name]
            )
            ### probability, _connection_args only if connect_fixed_prob (i.e. connector_name==Random)
            afferent_projection_dict["probability"].append(
                get_projection(projection)._connection_args[0]
            )
            ### size
            afferent_projection_dict["size"].append(len(get_projection(projection).pre))
            ### target type
            afferent_projection_dict["target"].append(get_projection(projection).target)

        return afferent_projection_dict

    def get_max_syn_currents(self, pop_name):
        """
        obtain I_app_max, g_ampa_max and g_gaba max.
        f_max = f_0 + f_t + 100
        I_app_max causes f_max (increases f from f_0 to f_max)
        g_ampa_max causes f_max (increases f from f_0 to f_max)
        I_gaba_max causes f_0 while I_app=I_app_max (decreases f from f_max to f_0)

        TODO:
            reaching f_max does not work well sometimes extreme values of I are needed and then the max values for the conductances gets extreme as well
            --> first get max g_ampa and g_gaba with a max PSP:
                a single spike causes conductance to increase shortly and the resulting snypatic current changes v
                start at resting potential than produce a single spike and record v and get the peak of v
                for g_ampa a peak = a minimum of v
                for g_gaba a peak = a minimum of v
                with max of g_ampa get max firing rate and with this max firing rate get max I_app

        Args:
            pop_name: str
                population name from original model

        return:
            list containing [I_may, g_ampa_max, g_gaba_max]

        Abbreviations:
            f_max: max firing rate

            f_0: firing rate without syn currents

            f_t: target firing rate
        """
        ### find g_ampa max
        self.log("search g_ampa_max with y(X) = PSP(g_ampa=X, g_gaba=0)")
        g_ampa_max = self.incremental_continuous_bound_search(
            y_X=lambda X_val: self.get_psp(
                net=self.net_single_v_clamp_dict[pop_name]["net"],
                population=self.net_single_v_clamp_dict[pop_name]["population"],
                variable_init_sampler=self.net_single_v_clamp_dict[pop_name][
                    "variable_init_sampler"
                ],
                monitor=self.net_single_v_clamp_dict[pop_name]["monitor"],
                g_ampa=X_val,
            ),
            y_bound=self.max_psp_dict[pop_name],
            X_0=0,
            y_0=0,
            alpha_abs=0.1,
            X_increase=0.1,
        )

        ### find g_gaba max
        self.log("search g_gaba_max with y(X) = PSP(g_ampa=0, g_gaba=X)")
        g_gaba_max = self.incremental_continuous_bound_search(
            y_X=lambda X_val: self.get_psp(
                net=self.net_single_v_clamp_dict[pop_name]["net"],
                population=self.net_single_v_clamp_dict[pop_name]["population"],
                variable_init_sampler=self.net_single_v_clamp_dict[pop_name][
                    "variable_init_sampler"
                ],
                monitor=self.net_single_v_clamp_dict[pop_name]["monitor"],
                g_gaba=X_val,
            ),
            y_bound=self.max_psp_dict[pop_name],
            X_0=0,
            y_0=0,
            alpha_abs=0.1,
            X_increase=0.1,
        )

        ### get f_0 and f_max
        f_0 = self.get_rate(
            net=self.net_single_dict[pop_name]["net"],
            population=self.net_single_dict[pop_name]["population"],
            variable_init_sampler=self.net_single_dict[pop_name][
                "variable_init_sampler"
            ],
            monitor=self.net_single_dict[pop_name]["monitor"],
        )[0]
        f_max = f_0 + self.target_firing_rate_dict[pop_name] + 100

        ### find I_max with f_0, and f_max using incremental_continuous_bound_search
        self.log("search I_app_max with y(X) = f(I_app=X, g_ampa=0, g_gaba=0)")
        I_max = self.incremental_continuous_bound_search(
            y_X=lambda X_val: self.get_rate(
                net=self.net_single_dict[pop_name]["net"],
                population=self.net_single_dict[pop_name]["population"],
                variable_init_sampler=self.net_single_dict[pop_name][
                    "variable_init_sampler"
                ],
                monitor=self.net_single_dict[pop_name]["monitor"],
                I_app=X_val,
            )[0],
            y_bound=f_max,
            X_0=0,
            y_0=f_0,
            alpha_abs=1,
        )

        return [I_max, g_ampa_max, g_gaba_max]

    def incremental_continuous_bound_search(
        self,
        y_X,
        y_bound,
        X_0,
        y_0,
        alpha_rel=0.01,
        alpha_abs=None,
        n_it_max=100,
        X_increase=1,
        saturation_thresh=10,
        saturation_warning=True,
    ):
        """
        you have system X --> y
        you want X for y=y_bound (either upper or lower bound)
        if you increase X (from starting point) y gets closer to y_bound!

        expectes a continuous funciton without from P_0(X_0,y_0) to P_bound(X_bound, y_bound)
        if it finds a saturation or non-continuous "step" on the way to P_bound it will return
        the X_bound for the end of the continuous part from P_0 to P_bound --> y_bound will not
        be reached

        Args:
            y_X: function
                returns a single number given a single number, call like y = y_X(X)
                increasing X should bring y closer to y_bound

            y_bound: number
                the bound for y for which an X_bound should be found

            X_0: number
                start value of X, from where the search should start

            y_0: number
                start value of y which results from X_0

            alpha_rel: number, optional, default=0.001
                allowed relative tolerance for deviations of y from y_bound
                if alpha_abs is given it overrides alpha_rel

            alpha_abs: number, optional, default=None
                allowed absolute tolerance for deviations of y from y_bound
                if alpha_abs is given it overrides alpha_rel

            n_it_max: number, optional, default=100
                maximum of iterations to find X_bound

            X_increase: number, optional, default=1
                the first increase of X (starting from X_0) to obtain the first new y_val
                i.e. first calculation is: y_val = y_X(X_0+X_increase)

            saturation_thresh: number, optional, default=5
                if y does not change while increasing X by X_increase the search will stop
                after this number of trials

            saturation_warning: bool, optional, default=True
                if you want to get a warning when the saturation is reached during search

        return:
            X_bound:
                X value which causes y=y_bound
        """
        ### TODO catch difference to target goes up in both directions
        ### then nothing new is predicted --> fails

        self.log(
            f"find X_bound for: y_0(X_0={X_0})={y_0} --> y_bound(X_bound=??)={y_bound}"
        )

        ### get tolerance
        tolerance = abs(y_bound - y_0) * alpha_rel
        if not isinstance(alpha_abs, type(None)):
            tolerance = alpha_abs

        ### define stop condition
        stop_condition = (
            lambda y_val, n_it: (
                ((y_bound - tolerance) <= y_val) and (y_val <= (y_bound + tolerance))
            )
            or n_it >= n_it_max
        )

        ### search for X_val
        X_list_predict = [X_0]
        y_list_predict = [y_0]
        X_list_all = [X_0]
        y_list_all = [y_0]
        n_it = 0
        X_val = X_0 + X_increase
        y_val = y_0
        y_not_changed_counter = 0
        X_change_predicted = X_increase
        while not stop_condition(y_val, n_it):
            ### get y_val for X
            y_val_pre = y_val
            y_val = y_X(X_val)
            y_change = y_val_pre - y_val

            ### store search history
            X_list_all.append(X_val)
            y_list_all.append(y_val)

            ### get next X_val depending on if y_val changed or not
            if abs(y_change) > 0:
                ### append X_val and y_val to y_list/X_list
                y_list_predict.append(y_val)
                X_list_predict.append(X_val)
                ### predict new X_val using y_bound as predictor
                X_val_pre = X_val
                X_val = self.predict_1d(
                    X=y_list_predict, y=X_list_predict, X_pred=y_bound
                )[0]
                X_change_predicted = X_val - X_val_pre
            else:
                ### just increase X_val
                X_val = X_val + X_change_predicted

            ### check saturation of y_val
            if abs(y_change) < tolerance:
                ### increase saturation counter
                y_not_changed_counter += 1
            else:
                ### reset saturation counter
                y_not_changed_counter = 0

            ### break if y_val saturated
            if y_not_changed_counter >= saturation_thresh:
                break

            ### increase iterator
            n_it += 1

        ### catch the initial point already satisified stop condition
        if len(X_list_all) == 1:
            warning_txt = f"WARNING incremental_continuous_bound_search: search did not start because initial point already satisfied stop condition!"
            self._p_w(warning_txt)
            self.log(warning_txt)
            return X_0

        ### warning if search saturated
        if (y_not_changed_counter >= saturation_thresh) and saturation_warning:
            warning_txt = f"WARNING incremental_continuous_bound_search: search saturated at y={y_list_predict[-1]} while searching for X_val for y_bound={y_bound}"
            self._p_w(warning_txt)
            self.log(warning_txt)
        self.log("initial search lists:")
        self.log("all:")
        self.log(np.array([X_list_all, y_list_all]).T)
        self.log("predict:")
        self.log(np.array([X_list_predict, y_list_predict]).T)

        ### if search saturated right at the begining --> search failed (i.e. y did not change while increasing X)
        if (y_not_changed_counter >= saturation_thresh) and len(X_list_predict) == 1:
            error_msg = "ERROR incremental_continuous_bound_search: search failed because changing X_val did not change y_val"
            self.log(error_msg)
            raise AssertionError(error_msg)

        ### get best X value for which y is closest to y_bound
        idx_best = np.argmin(np.absolute(np.array(y_list_predict) - y_bound))
        X_bound = X_list_predict[idx_best]

        ### if y cannot get larger or smaller than y_bound one has to check if you not "overshoot" with X_bound
        ### --> fine tune result by investigating the space between X_0 and X_bound and preict a new X_bound
        self.log(f"X_0: {X_0}, X_bound:{X_bound} for final predict list")
        X_space_arr = np.linspace(X_0, X_bound, 100)
        y_val = y_0
        X_list_predict = []
        y_list_predict = []
        y_list_all = []
        for X_val in X_space_arr:
            y_val_pre = y_val
            y_val = y_X(X_val)
            y_list_all.append(y_val)
            if y_val != y_val_pre:
                ### if y_val changed
                ### append X_val and y_val to y_list/X_list
                y_list_predict.append(y_val)
                X_list_predict.append(X_val)
        self.log("final predict lists:")
        self.log("all:")
        self.log(np.array([X_space_arr, y_list_all]).T)
        self.log("predict:")
        self.log(np.array([X_list_predict, y_list_predict]).T)

        ### check if there is a discontinuity in y_all, starting with the first used value in y_predict
        ### update all values with first predict value
        first_y_used_in_predict = y_list_predict[0]
        idx_first_y_in_all = y_list_all.index(first_y_used_in_predict)
        y_list_all = y_list_all[idx_first_y_in_all:]
        X_space_arr = y_list_all[idx_first_y_in_all:]
        ### get discontinuity
        discontinuity_idx_list = self.get_discontinuity_idx_list(y_list_all)
        self.log("discontinuity_idx_list")
        self.log(f"{discontinuity_idx_list}")
        if len(discontinuity_idx_list) > 0:
            ### there is a discontinuity
            discontinuity_idx = discontinuity_idx_list[0]
            ### only use values until discontinuity
            y_bound_new = y_list_all[discontinuity_idx]
            idx_best = y_list_predict.index(y_bound_new)
            X_val_best = X_list_predict[idx_best]
            y_val_best = y_list_predict[idx_best]
            ### print warning
            warning_txt = f"WARNING incremental_continuous_bound_search: found discontinuity, only reached y={y_bound_new} while searching for y_bound={y_bound}"
            self._p_w(warning_txt)
            ### log
            self.log(warning_txt)
            self.log(
                f"discontinuities detected --> only use last values until first discontinuity: X={X_val_best}, y={y_val_best}"
            )
        else:
            ### there is no discontinuity
            ### now predict final X_val
            X_val = self.predict_1d(X=y_list_predict, y=X_list_predict, X_pred=y_bound)[
                0
            ]
            y_val = y_X(X_val)

            ### append it to lists
            X_list_predict.append(X_val)
            y_list_predict.append(y_val)

            ### ifnd best
            idx_best = np.argmin(np.absolute(np.array(y_list_predict) - y_bound))
            X_val_best = X_list_predict[idx_best]
            y_val_best = y_list_predict[idx_best]

            ### log
            self.log(f"final values: X={X_val_best}, y={y_val_best}")

        ### warning for max iteration search
        if not (n_it < n_it_max):
            warning_txt = f"WARNING incremental_continuous_bound_search: reached max iterations to find X_bound to get y_bound={y_bound}, found X_bound causes y={y_val_best}"
            self._p_w(warning_txt)
            self.log(warning_txt)

        return X_val_best

    def get_discontinuity_idx_list(self, arr):
        """
        Args:
            arr: array-like
                array for which its checked if there are discontinuities
        """
        arr = np.array(arr)
        range_data = arr.max() - arr.min()
        diff_arr = np.diff(arr)
        diff_rel_range_arr = diff_arr / range_data
        diff_rel_range_abs_arr = np.absolute(diff_rel_range_arr)
        peaks = find_peaks(
            diff_rel_range_abs_arr, prominence=10 * np.mean(diff_rel_range_abs_arr)
        )
        peaks_idx_list = peaks[0]

        return peaks_idx_list

    def predict_1d(self, X, y, X_pred):
        """
        Args:
            X: array-like
                X values
            y: array-like
                y values, same size as X_values
            X_pred: array-like or number
                X value(s) for which new y value(s) are predicted

        return:
            Y_pred_arr: array
                predicted y values for X_pred
        """
        y_X = interp1d(x=X, y=y, fill_value="extrapolate")
        y_pred_arr = y_X(X_pred)
        return y_pred_arr.reshape(1)

    def get_rate_dict(
        self,
        net,
        population_dict,
        variable_init_sampler_dict,
        monitor_dict,
        I_app_dict,
        g_ampa_dict,
        g_gaba_dict,
    ):
        """
        function to obtain the firing rates of the populations of
        the network given with 'idx' for given I_app, g_ampa and g_gaba values

        Args:
            idx: int
                network index given by the parallel_run function

            net: object
                network object given by the parallel_run function

            net_many_dict: dict
                dictionary containing a population_dict and a monitor_dict
                which contain for each population name the
                - ANNarchy Population object of the magic network
                - ANNarchy Monitor object of the magic network

            I_app_arr_dict: dict of arrays
                dictionary containing for each population the array with input values for I_app

            g_ampa_arr_dict: dict of arrays
                dictionary containing for each population the array with input values for g_ampa

            g_gaba_arr_dict: dict of arrays
                dictionary containing for each population the array with input values for g_gaba

            variable_init_sampler_dict: dict
                dictionary containing for each population the initial variables sampler object
                with the function.sample() to get initial values of the neurons

            self: object
                the model_configurator object

        return:
            f_rec_arr_dict: dict of arrays
                dictionary containing for each population the array with the firing rates for the given inputs
        """
        ### reset and set init values
        net.reset()
        for pop_name, varaible_init_sampler in variable_init_sampler_dict.items():
            population = net.get(population_dict[pop_name])
            variable_init_arr = varaible_init_sampler.sample(len(population), seed=0)
            for var_idx, var_name in enumerate(population.variables):
                set_val = variable_init_arr[:, var_idx]
                setattr(population, var_name, set_val)

        ### slow down conductances (i.e. make them constant)
        for pop_name in population_dict.keys():
            population = net.get(population_dict[pop_name])
            population.tau_ampa = 1e20
            population.tau_gaba = 1e20
        ### apply given variables
        for pop_name in population_dict.keys():
            population = net.get(population_dict[pop_name])
            population.I_app = I_app_dict[pop_name]
            population.g_ampa = g_ampa_dict[pop_name]
            population.g_gaba = g_gaba_dict[pop_name]
        ### simulate 500 ms initial duration + X ms
        net.simulate(500 + self.simulation_dur)
        ### get rate for the last X ms
        f_arr_dict = {}
        for pop_name in population_dict.keys():
            population = net.get(population_dict[pop_name])
            monitor = net.get(monitor_dict[pop_name])
            spike_dict = monitor.get("spike")
            f_arr = np.zeros(len(population))
            for idx_n, n in enumerate(spike_dict.keys()):
                time_list = np.array(spike_dict[n])
                nbr_spks = np.sum((time_list > (500 / dt())).astype(int))
                rate = nbr_spks / (self.simulation_dur / 1000)
                f_arr[idx_n] = rate
            f_arr_dict[pop_name] = f_arr

        return f_arr_dict

    def get_rate(
        self,
        net,
        population,
        variable_init_sampler,
        monitor,
        I_app=0,
        g_ampa=0,
        g_gaba=0,
    ):
        """
        simulates a population for X+500 ms and returns the firing rate of each neuron for the last X ms
        X is defined with self.simulation_dur

        Args:
            net: ANNarchy network
                network which contains the population and monitor

            population: ANNarchy population
                population which is recorded and stimulated

            variable_init_sampler: object
                containing the initial values of the population neuron, use .sample() to get values

            monitor: ANNarchy monitor
                to record spikes from population

            I_app: number or arr, optional, default = 0
                applied current to the population neurons, has to have the same size as the population

            g_ampa: number or arr, optional, default = 0
                applied ampa conductance to the population neurons, has to have the same size as the population

            g_gaba: number or arr, optional, default = 0
                applied gaba conductance to the population neurons, has to have the same size as the population
        """
        ### reset and set init values
        net.reset()
        variable_init_arr = variable_init_sampler.sample(len(population), seed=0)
        for var_idx, var_name in enumerate(population.variables):
            set_val = variable_init_arr[:, var_idx]
            setattr(population, var_name, set_val)
        ### slow down conductances (i.e. make them constant)
        population.tau_ampa = 1e20
        population.tau_gaba = 1e20
        ### apply given variables
        population.I_app = I_app
        population.g_ampa = g_ampa
        population.g_gaba = g_gaba
        ### simulate 500 ms initial duration + X ms
        net.simulate(500 + self.simulation_dur)
        ### get rate for the last X ms
        spike_dict = monitor.get("spike")
        f_arr = np.zeros(len(population))
        for idx_n, n in enumerate(spike_dict.keys()):
            time_list = np.array(spike_dict[n])
            nbr_spks = np.sum((time_list > (500 / dt())).astype(int))
            rate = nbr_spks / (self.simulation_dur / 1000)
            f_arr[idx_n] = rate

        return f_arr

    def get_psp(
        self,
        net,
        population,
        variable_init_sampler,
        monitor,
        g_ampa=0,
        g_gaba=0,
        do_plot=False,
    ):
        """
        simulates a single spike at t=50 ms and records the change of v within a voltage_clamp neuron

        Args:
            net: ANNarchy network
                network which contains the population and monitor

            population: ANNarchy population
                population which is recorded and stimulated

            variable_init_sampler: object
                containing the initial values of the population neuron, use .sample() to get values

            monitor: ANNarchy monitor
                to record v_clamp_rec from population

            g_ampa: number, optional, default = 0
                applied ampa conductance to the population neuron at t=50 ms

            g_gaba: number, optional, default = 0
                applied gaba conductance to the population neurons at t=50 ms
        """
        ### reset network and set initial values
        net.reset()
        variable_init_arr = variable_init_sampler.sample(len(population), seed=0)
        for var_idx, var_name in enumerate(population.variables):
            set_val = variable_init_arr[:, var_idx]
            setattr(population, var_name, set_val)
        ### apply no input
        population.I_app = 0
        population.g_ampa = 0
        population.g_gaba = 0
        ### simulate 50 ms initial duration
        net.simulate(50)
        ### apply given conductances --> changes v_clamp_rec
        v_clamp_rec_rest = population.v_clamp_rec[0]
        population.v_clamp_rec_thresh = v_clamp_rec_rest
        population.g_ampa = g_ampa
        population.g_gaba = g_gaba
        ### simulate until v_clamp_rec is near v_clamp_rec_rest again
        net.simulate_until(max_duration=self.simulation_dur, population=population)
        ### get the psp = maximum of difference of v_clamp_rec and v_clamp_rec_rest
        v_clamp_rec = monitor.get("v_clamp_rec")[:, 0]
        psp = float(np.absolute(v_clamp_rec - v_clamp_rec_rest).max())

        if do_plot:
            plt.figure()
            plt.title(
                f"g_ampa={g_ampa}, g_gaba={g_gaba}, v_clamp_rec_rest={v_clamp_rec_rest}, psp={psp}"
            )
            plt.plot(v_clamp_rec)
            plt.savefig(
                f"tmp_psp_{population.name}_{int(g_ampa*1000)}_{int(g_gaba*1000)}.png"
            )
            plt.close("all")

        return psp

    def compile_net_many(self, net):
        compile_in_folder(
            folder_name=f"many_net_{net.id}", net=net, clean=True, silent=True
        )

    def create_many_neuron_network(self):
        """
        creates a ANNarchy magic network with all popualtions which should be configured the size
        of the populations is equal and is obtianed by dividing the number of the
        interpolation values by the number of networks which will be used during run_parallel

        return:
            net_many_dict: dict
                contains
                - population_dict: for all population names the created population in the magic network
                - monitor_dict: for all population names the created monitors in the magic network
        """
        self.log("create many neurons network")

        ### clear ANNarchy
        cnp_clear()

        ### for each population of the given model which should be configured
        ### create a population with a given size
        ### create a monitor recording spikes
        ### create a network containing the population and the monitor
        many_neuron_population_list = []
        many_neuron_monitor_list = []
        many_neuron_network_list = []
        for pop_name in self.pop_name_list:

            ### create the neuron model with poisson spike trains
            ### get the initial arguments of the neuron
            neuron_model = self.neuron_model_dict[pop_name]
            ### names of arguments
            init_arguments_name_list = list(Neuron.__init__.__code__.co_varnames)
            init_arguments_name_list.remove("self")
            init_arguments_name_list.remove("name")
            init_arguments_name_list.remove("description")
            ### arguments dict
            init_arguments_dict = {
                init_arguments_name: getattr(neuron_model, init_arguments_name)
                for init_arguments_name in init_arguments_name_list
            }
            ### get the afferent populations
            afferent_population_list = []
            proj_target_type_list = []
            for proj_name in self.afferent_projection_dict[pop_name]["projeciton_name"]:
                proj_dict = self.get_proj_dict(proj_name)
                pre_pop_name = proj_dict["pre_pop_name"]
                afferent_population_list.append(pre_pop_name)
                proj_target_type_list.append(proj_dict["proj_target_type"])
            
            ### for each afferent population create a poisson spike train equation string
            ### add it to the equations
            ### and add the related parameters to the parameters

            equations_line_split_list = str(
                init_arguments_dict["equations"]
            ).splitlines()

            parameters_line_split_list = str(
                init_arguments_dict["parameters"]
            ).splitlines()

            for pre_pop_name in afferent_population_list:
                ### TODO currently I get rate as target firing rate but this needs to be a spike train which consideres the nummber of synapses i.e. probability and number of pre neurons
                poisson_equation_str = f"{pre_pop_name}_spike_train = ite(Uniform(0.0, 1.0) * 1000.0 / dt > {pre_pop_name}_rate, 0, {pre_pop_name}_weight"
                
                equations_line_split_list.insert(1,poisson_equation_str)
                parameters_line_split_list.append(f"{pre_pop_name}_rate = 0")
                parameters_line_split_list.append(f"{pre_pop_name}_weight = 0")

            ### change the g_ampa and g_gaba line, they additionally are the sum of the spike trains
            for equation_line in equations_line_split_list:
                ### remove whitespaces
                line = equation_line.replace(" ", "")
                ### check if line contains g_ampa
                if "dg_ampa/dt" in line:
                    ### get the right side of the equation
                    line_right = line.split("=")[1]
                    ### remove and store tags_str
                    tags_str=""
                    if len(line_right.split(":"))>1:
                        line_right, tags_str = line_right.split(":")
                    ### get the populations whose spike train should be appended
                    afferent_population_to_append_list = []
                    for pre_pop_name in afferent_population_list:
                        if proj_target_type_list[] == "ampa":
                            afferent_population_to_append_list.append(pre_pop_name)
                    if len(afferent_population_to_append_list)>0:
                        ### change right side, add the sum of the spike trains
                        line_right = f"{line_right} + {'+'.join([f'{pre_pop_name}_spike_train' for pre_pop_name in afferent_population_to_append_list])}"
                    ### add tags_str again
                    line_right = f"{line_right}:{tags_str}"

            ### combine string lines to multiline strings again
            init_arguments_dict["parameters"] = "\n".join(parameters_line_split_list)
            init_arguments_dict["equations"] = "\n".join(equations_line_split_list)

            ### create neuron model with new equations
            neuron_model_new = Neuron(**init_arguments_dict)

            ### create the many neuron population
            many_neuron_population_list.append(
                Population(
                    geometry=self.nr_neurons_per_net,
                    neuron=neuron_model_new,
                    name=f"many_neuron_{pop_name}",
                )
            )

            ### set the attributes of the neurons
            for attr_name, attr_val in self.neuron_model_parameters_dict[pop_name]:
                setattr(many_neuron_population_list[pop_name], attr_name, attr_val)

            ### create Monitor for many neuron
            many_neuron_monitor_list.append(
                Monitor(many_neuron_population_list[pop_name], ["spike"])
            )

            ### create the network with population and monitor
            many_neuron_network_list[pop_name] = Network()
            many_neuron_network_list[pop_name].add(many_neuron_population_list[-1])
            many_neuron_network_list[pop_name].add(many_neuron_monitor_list[-1])

        net_many_dict = {
            "network_list": many_neuron_network_list,
            "population_list": many_neuron_population_list,
            "monitor_list": many_neuron_monitor_list,
        }
        return net_many_dict

    def get_v_clamp_2000(self, v, net, population):
        net.reset()
        population.v = v
        net.simulate(2000)
        return population.v_clamp_rec[0]

    def get_voltage_clamp_equations(self, init_arguments_dict, pop_name):
        """
        works with
        dv/dt = ...
        v += ...
        """
        ### get the dv/dt equation from equations
        ### find the line with dv/dt= or v+= or v=
        eq = str(init_arguments_dict["equations"])
        eq = eq.splitlines()
        line_is_v_list = [False] * len(eq)
        ### check in which lines v is defined
        for line_idx, line in enumerate(eq):
            line_is_v_list[line_idx] = self.get_line_is_v(line)
        ### raise error if no v or multiple times v
        if True not in line_is_v_list or sum(line_is_v_list) > 1:
            raise ValueError(
                f"ERROR model_configurator create_net_single_voltage_clamp: In the equations of the neurons has to be exactly a single line which defines dv/dt or v, not given for population {pop_name}"
            )
        ### set the v equation
        eq_v = eq[line_is_v_list.index(True)]

        ### if equation type is v += ... --> just take right side
        if "+=" in eq_v:
            ### create the new equations for the ANNarchy neuron
            ### create two lines, the voltage clamp line v+=0 and the
            ### right sight of v+=... separately
            eq_new_0 = "v_clamp_rec_pre = v_clamp_rec"
            eq_new_1 = f"v_clamp_rec = abs({eq_v.split('+=')[1]})"
            eq_new_2 = "v+=0"
            ### remove old v line and insert new lines
            del eq[line_is_v_list.index(True)]
            eq.insert(line_is_v_list.index(True), eq_new_0)
            eq.insert(line_is_v_list.index(True), eq_new_1)
            eq.insert(line_is_v_list.index(True), eq_new_2)
            eq = "\n".join(eq)
            ### return new neuron equations
            return eq

        ### if equation type is dv/dt = ... --> get the right side of dv/dt=...
        ### transform eq_v
        ### remove whitespaces
        ### remove tags and store them for later
        ### TODO replace random distributions and mathematical expressions which may be on the left side
        eq_v = eq_v.replace(" ", "")
        eq_v = eq_v.replace("dv/dt", "delta_v")
        eq_tags_list = eq_v.split(":")
        eq_v = eq_tags_list[0]
        if len(eq_tags_list) > 1:
            tags = eq_tags_list[1]
        else:
            tags = None

        ### split the equation at "=" and move everything on one side (other side = 0)
        eq_v_splitted = eq_v.split("=")
        left_side = eq_v_splitted[0]
        right_side = "right_side"
        eq_v_one_side = f"{right_side}-({left_side})"

        ### prepare the sympy equation generation
        attributes_name_list = self.neuron_model_attributes_dict[pop_name]
        attributes_tuple = symbols(",".join(attributes_name_list))
        ### for each attribute of the neuron a sympy symbol
        attributes_sympy_dict = {
            key: attributes_tuple[attributes_name_list.index(key)]
            for key in attributes_name_list
        }
        ### furhter create symbols for dv/dt and right_side
        attributes_sympy_dict["delta_v"] = Symbol("delta_v")
        attributes_sympy_dict["right_side"] = Symbol("right_side")

        ### now replace the symbolds in the eq_v string with the dictionary items
        eq_v_replaced = replace_names_with_dict(
            expression=eq_v_one_side,
            name_of_dict="attributes_sympy_dict",
            dictionary=attributes_sympy_dict,
        )

        ### from this string get the sympy equation expression
        eq_sympy = eval(eq_v_replaced)

        ### solve the equation to delta_v
        result = solve(eq_sympy, attributes_sympy_dict["delta_v"], dict=True)
        if len(result) != 1:
            raise ValueError(
                f"ERROR model_configurator create_net_single_voltage_clamp: Could not find solution for dv/dt for neuronmodel of population {pop_name}!"
            )
        result = str(result[0][attributes_sympy_dict["delta_v"]])

        ### replace right_side by the original right side
        result = result.replace("right_side", f"({eq_v_splitted[1]})")

        ### TODO replace mathematical expressions and random distributions back to previous

        ### now create the new equations for the ANNarchy neuron
        ### create two lines, the voltage clamp line dv/dt=0 and the
        ### obtained line which would be the right side of dv/dt
        ### v_clamp_rec should be an absolute value
        eq_new_0 = f"v_clamp_rec = fabs({result})"
        eq_new_1 = "v_clamp_rec_pre = v_clamp_rec"
        ### add stored tags to dv/dt equation
        if not isinstance(tags, type(None)):
            eq_new_2 = f"dv/dt=0 : {tags}"
        else:
            eq_new_2 = "dv/dt=0"
        ### remove old v line and insert new lines
        del eq[line_is_v_list.index(True)]
        eq.insert(line_is_v_list.index(True), eq_new_0)
        eq.insert(line_is_v_list.index(True), eq_new_1)
        eq.insert(line_is_v_list.index(True), eq_new_2)
        eq = "\n".join(eq)
        ### return new neuron equations
        return eq

    def get_line_is_v(self, line: str):
        """
        check if a equation string contains dv/dt or v= or v+=
        """
        if "v" not in line:
            return False

        ### remove whitespaces
        line = line.replace(" ", "")

        ### check for dv/dt
        if "dv/dt" in line:
            return True

        ### check for v update
        if ("v=" in line or "v+=" in line) and line.startswith("v"):
            return True

        return False

    def get_line_is_g_ampa(self, line: str):
        """
        check if a equation string contains dg_ampa/dt
        """

        ### remove whitespaces
        line = line.replace(" ", "")

        ### check for dv/dt
        if "dv/dt" in line:
            return True

        ### check for v update
        if ("v=" in line or "v+=" in line) and line.startswith("v"):
            return True

        return False

    def get_init_neuron_variables_v_clamp(self, net, pop, v_rest):
        """
        get the variables of the given population after simulating 2000 ms

        Args:
            net: ANNarchy network
                the network which contains the pop

            pop: ANNarchy population
                the population whose variables are obtained

        """
        ### reset neuron and deactivate input and set v_rest
        net.reset()
        pop.I_app = 0
        pop.v = v_rest

        ### get the variables of the neuron after 5000 ms
        net.simulate(5000)
        var_name_list = list(pop.variables)
        var_arr = np.zeros((1, len(var_name_list)))
        get_arr = np.array([getattr(pop, var_name) for var_name in pop.variables])
        var_arr[0, :] = get_arr[:, 0]

        ### create a sampler with the one data sample
        sampler = self.var_arr_sampler(var_arr)
        return sampler

    class var_arr_sampler:
        def __init__(self, var_arr) -> None:
            self.var_arr_shape = var_arr.shape
            self.is_const = (
                np.std(var_arr, axis=0) <= np.mean(np.absolute(var_arr), axis=0) / 1000
            )
            self.constant_arr = var_arr[0, self.is_const]
            self.not_constant_val_arr = var_arr[:, np.logical_not(self.is_const)]

        def sample(self, n, seed=0):
            """
            Args:
                n: int
                    number of samples

                seed: int, optional, default=0
                    seed for rng
            """
            ### get random idx
            rng = np.random.default_rng(seed=seed)
            random_idx_arr = rng.integers(low=0, high=self.var_arr_shape[0], size=n)
            ### sample with random idx
            sample_arr = self.not_constant_val_arr[random_idx_arr]
            ### create return array
            ret_arr = np.zeros((n,) + self.var_arr_shape[1:])
            ### add samples to return array
            ret_arr[:, np.logical_not(self.is_const)] = sample_arr
            ### add constant values to return array
            ret_arr[:, self.is_const] = self.constant_arr

            return ret_arr

    def get_nr_many_neurons(self, nr_neurons, nr_networks):
        """
        Splits the number of neurons in almost equally sized parts.

        Args:
            nr_neurons: int
                number of neurons which should be splitted

            nr_networks: int
                number of networks over which the neurons should be equally distributed
        """
        return self.divide_almost_equal(number=nr_neurons, num_parts=nr_networks)

    def get_max_weight_dict_for_pop(self, pop_name):
        """
        get the weight dict for a single population

        Args:
            pop_name: str
                population name

        return: dict
            keys = afferent projection names, values = max weights
        """

        ### loop over afferent projections
        max_w_list = []
        for proj_name in self.afferent_projection_dict[pop_name]["projection_names"]:
            ### find max weight for projection
            max_weight_of_proj = self.get_max_weight_of_proj(proj_name=proj_name)
            max_w_list.append(max_weight_of_proj)
        self.afferent_projection_dict[pop_name]["max_weight"] = max_w_list

        ### remove weight key from self.afferent_projection_dict[pop_name] which was added during the process
        self.afferent_projection_dict[pop_name].pop("weights")

        ### now create the dictionary structure for return
        # {
        #     "ampa": {"projection_name": "max_weight value"...},
        #     "gaba": {"projection_name": "max_weight value"...},
        # }
        max_weight_dict_for_pop = {"ampa": {}, "gaba": {}}
        ### loop over all afferent projections
        for proj_name in self.afferent_projection_dict[pop_name]["projection_names"]:
            proj_dict = self.get_proj_dict(proj_name)
            proj_target_type = proj_dict["proj_target_type"]
            proj_max_weight = proj_dict["proj_max_weight"]
            ### add max weight of projection to the corresponding target type in the return dict
            max_weight_dict_for_pop[proj_target_type][proj_name] = proj_max_weight

        return max_weight_dict_for_pop

    def get_proj_dict(self, proj_name):
        """
        get a dictionary for a specified projection which contains following information:
            post_pop_name
            proj_target_type
            idx_proj
            spike_frequency
            proj_weight
            g_max

        Args:
            proj_name: str
                projection name

        return:
            proj_dict: dict
                keys see above
        """
        ### get pre_pop_name
        pre_pop_name = self.pre_pop_name_dict[proj_name]
        ### get post_pop_name
        post_pop_name = self.post_pop_name_dict[proj_name]
        ### get idx_proj and proj_target_type
        idx_proj = self.afferent_projection_dict[post_pop_name][
            "projection_names"
        ].index(proj_name)
        proj_target_type = self.afferent_projection_dict[post_pop_name]["target"][
            idx_proj
        ]
        ### get spike frequency
        f_t = self.afferent_projection_dict[post_pop_name]["target firing rate"][
            idx_proj
        ]
        p = self.afferent_projection_dict[post_pop_name]["probability"][idx_proj]
        s = self.afferent_projection_dict[post_pop_name]["size"][idx_proj]
        spike_frequency = f_t * p * s
        ### get weight
        try:
            proj_weight = self.afferent_projection_dict[post_pop_name]["weights"][
                idx_proj
            ]
        except:
            proj_weight = None
        ### g_max
        g_max = self.g_max_dict[post_pop_name][proj_target_type]
        ### get max weight
        try:
            proj_max_weight = self.afferent_projection_dict[post_pop_name][
                "max_weight"
            ][idx_proj]
        except:
            proj_max_weight = None

        return {
            "pre_pop_name": pre_pop_name,
            "post_pop_name": post_pop_name,
            "proj_target_type": proj_target_type,
            "idx_proj": idx_proj,
            "spike_frequency": spike_frequency,
            "proj_weight": proj_weight,
            "g_max": g_max,
            "proj_max_weight": proj_max_weight,
        }

    def get_max_weight_of_proj(self, proj_name):
        """
        find the max weight of a specified projection using incremental_continuous_bound_search
        increasing weights of projection increases conductance g of projection --> increase
        until g_max is found

        Args:
            proj_name: str
                projection name

        return:
            w_max: number
        """
        ### log task
        self.log(f"get w_max for {proj_name}")

        ### g_max for projection
        proj_dict = self.get_proj_dict(proj_name)
        g_max = proj_dict["g_max"]

        ### find max weight with incremental_continuous_bound_search
        ### increase weights until g_max is reached
        self.log("search w_max with y(X) = g(w=X)")
        w_max = self.incremental_continuous_bound_search(
            y_X=lambda X_val: self.get_g_of_single_proj(
                weight=X_val,
                proj_name=proj_name,
            ),
            y_bound=g_max,
            X_0=0,
            y_0=0,
        )

        return w_max

    def get_g_of_single_proj(self, weight, proj_name):
        """
        given a weight for a specified projection get the resulting conductance value g
        in the target population

        Args:
            weight: number
                the weight of the projection

            proj_name: str
                projection name

        return:
            g_val: number
        """
        ### get some projection infos
        proj_dict = self.get_proj_dict(proj_name)
        pop_name = proj_dict["post_pop_name"]
        idx_proj = proj_dict["idx_proj"]
        proj_target_type = proj_dict["proj_target_type"]

        ### set weights in the afferent_projection_dict
        ### set all weights to zero except the weight of the current proj which is set to the given weight
        weight_list = [0] * self.nr_afferent_proj_dict[pop_name]
        weight_list[idx_proj] = weight
        self.afferent_projection_dict[pop_name]["weights"] = weight_list

        ### get the g_ampa and g_gaba values based on the current afferent_projection_dict weights
        mean_g = self.get_g_values_of_pop(pop_name)

        ### then return the conductance related to the specified projection
        return mean_g[proj_target_type]

    def get_g_values_of_pop(self, pop_name):
        """
        calculate the average g_ampa and g_gaba values of the specified population based on the weights
        defined in the afferent_projection_dict

        Args:
            pop_name: str
                population name
        """
        spike_times_dict = {"ampa": [np.array([])], "gaba": [np.array([])]}
        spike_weights_dict = {"ampa": [np.array([])], "gaba": [np.array([])]}
        ### loop over afferent projections
        for proj_name in self.afferent_projection_dict[pop_name]["projection_names"]:
            ### get projection infos
            proj_dict = self.get_proj_dict(proj_name)
            proj_weight = proj_dict["proj_weight"]
            proj_target_type = proj_dict["proj_target_type"]
            spike_frequency = proj_dict["spike_frequency"]
            ### get spike times over the simulation duration for the spike frequency
            if spike_frequency > 0:
                spike_times_arr = self.get_spike_times_arr(
                    spike_frequency=spike_frequency
                )
            else:
                spike_times_arr = np.array([])
            ### get weights array
            spike_weights_arr = np.ones(len(spike_times_arr)) * proj_weight
            ### store spike times and weights for the target type of the projection
            spike_times_dict[proj_target_type].append(spike_times_arr)
            spike_weights_dict[proj_target_type].append(spike_weights_arr)

        mean_g = {}
        for target_type in ["ampa", "gaba"]:
            ### concatenate spike times and corresponding weights of different afferent projections
            spike_times_arr = np.concatenate(spike_times_dict[target_type])
            spike_weights_arr = np.concatenate(spike_weights_dict[target_type])

            ### sort the spike times and corresponding weights
            sort_idx = np.argsort(spike_times_arr)
            spike_times_arr = spike_times_arr[sort_idx]
            spike_weights_arr = spike_weights_arr[sort_idx]

            ### calculate mean g values from the spike times and corresponding weights
            mean_g[target_type] = self.get_mean_g(
                spike_times_arr=spike_times_arr,
                spike_weights_arr=spike_weights_arr,
                tau=self.tau_dict[pop_name][target_type],
            )

        return mean_g

    def get_spike_times_arr(self, spike_frequency):
        """
        get spike times for a given spike frequency

        Args:
            spike_frequency: number
                spike frequency in Hz
        """
        expected_nr_spikes = int(
            round((500 + self.simulation_dur) * (spike_frequency / 1000), 0)
        )
        ### isi_arr in timesteps
        isi_arr = poisson.rvs(
            (1 / (spike_frequency * (dt() / 1000))), size=expected_nr_spikes
        )
        ### convert to ms
        isi_arr = isi_arr * dt()

        ### get spike times from isi_arr
        spike_times_arr = np.cumsum(isi_arr)

        ### only use spikes which are in the simulation time
        spike_times_arr = spike_times_arr[spike_times_arr < (self.simulation_dur + 500)]

        return spike_times_arr

    def get_mean_g(self, spike_times_arr, spike_weights_arr, tau):
        """
        calculates the mean conductance g for given spike times, corresponding weights (increases of g) and time constant

        Args:
            spike_times_arr: arr
                1d array containing spike times in ms

            spike_weights_arr: arr
                1d array containing the weights corresponding to the spike times

            tau: number
                time constant of the exponential decay of the conductance g in ms
        """
        ### TODO instead of calculating the mean, create a conductance trace for the simulation time
        if np.sum(spike_weights_arr) > 0:
            ### get inter spike interval array
            isis_g_arr = np.diff(spike_times_arr)
            ### calc mean g
            mean_w = np.mean(spike_weights_arr)
            mean_isi = np.mean(isis_g_arr)
            mean_g = mean_w / ((1 / np.exp(-mean_isi / tau)) - 1)
        else:
            mean_g = 0

        return mean_g


def get_rate_parallel(
    idx,
    net,
    population,
    variable_init_sampler,
    monitor,
    I_app_arr,
    weight_list,
    proj_name_list,
    rate_list,
    simulation_dur,
):
    """
    function to obtain the firing rates of the populations of
    the network given with 'idx' for given I_app, g_ampa and g_gaba values

    Args:
        idx: int
            network index given by the parallel_run function

        net: object
            network object given by the parallel_run function

        pop_name_list: list of str
            list with population names of network

        population_list: list of ANNarchy Population object
            list of population objets of magic network

        variable_init_sampler_list: list of sampler objects
            for each population a sampler object with function .sample to get initial variable values

        monitor_list: list of ANNarchy Monitor objects
            list of monitor objets of magic network recording spikes from the populations

        I_app_list: list of arrays
            list containing for each population the array with input values for I_app

        g_ampa_list: list of arrays
            list containing for each population the array with input values for g_ampa

        g_gaba_list: list of arrays
            list containing for each population the array with input values for g_gaba

        simulation_dur: int
            simulation duration

    return:
        f_rec_arr_list: list of arrays
            list containing for each population the array with the firing rates for the given inputs
    """
    ### reset and set init values
    net.reset()
    ### sample init values, one could sample different values for multiple neurons
    ### but here we sample a single sample and use it for all neurons
    variable_init_arr = variable_init_sampler.sample(1, seed=0)
    variable_init_arr = np.array([variable_init_arr[0]] * len(population))
    for var_idx, var_name in enumerate(population.variables):
        set_val = variable_init_arr[:, var_idx]
        setattr(population, var_name, set_val)

    ### set the weights and rates of the poisson spike traces of the afferent populations
    for proj_idx, proj_name in enumerate(proj_name_list):
        setattr(population, f"{proj_name}_rate", rate_list[proj_idx])
        setattr(population, f"{proj_name}_weight", weight_list[proj_idx])

    ### set the I_app
    population.I_app = I_app_arr

    ### simulate 500 ms initial duration + X ms
    net.simulate(500 + simulation_dur)

    ### get rate for the last X ms
    spike_dict = monitor.get("spike")
    f_arr = np.zeros(len(population))
    for idx_n, n in enumerate(spike_dict.keys()):
        time_list = np.array(spike_dict[n])
        nbr_spks = np.sum((time_list > (500 / dt())).astype(int))
        rate = nbr_spks / (simulation_dur / 1000)
        f_arr[idx_n] = rate
    return f_arr


_p_g_1 = """First call get_max_syn.
This determines max synaptic conductances and weights of all afferent projections of the model populations and returns a dictionary with max weights."""

_p_g_after_get_weights = (
    lambda template_weight_dict, template_synaptic_load_dict, template_synaptic_contribution_dict: f"""Now either set the weights of all projections directly or first set the synaptic load of the populations and the synaptic contributions of the afferent projections.
You can set the weights using the function .set_weights() which requires a weight_dict as argument.
Use this template for the weight_dict:

{template_weight_dict}

The values within the template are the maximum weight values.


You can set the synaptic load and contribution using the function .set_syn_load() which requires a synaptic_load_dict or a single number between 0 and 1 for the synaptic load of the populations and a synaptic_contribution_dict for the synaptic contributions to the synaptic load of the afferent projections.
Use this template for the synaptic_load_dict:

{template_synaptic_load_dict}

'ampa_load' and 'gaba_load' are placeholders, replace them with values between 0 and 1.

Use this template for the synaptic_contribution_dict:

{template_synaptic_contribution_dict}

'contr' are placeholders, replace them with the contributions of the afferent projections. The contributions of all afferent projections of a single population have to sum up to 1!
"""
)

_p_g_after_set_syn_load = """Synaptic loads and contributions, i.e. weights set. Now call .get_base to obtain the baseline currents for the model populations. With .set_base you can directly set these baselines and the current weights in the model and compile the model.
"""
