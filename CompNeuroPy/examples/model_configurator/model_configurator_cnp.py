from CompNeuroPy import cnp_clear
from ANNarchy import Population, get_population, Monitor, Network, get_projection
from time import time
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import inspect
import textwrap
import os


class model_configurator:
    def __init__(
        self, model, target_firing_rate_dict, do_not_config_list=[], print_guide=False
    ) -> None:
        """
        Args:
            model: CompNeuroPy generate_model object
                it's not important if the model is created or compiled but after running
                the model_configurator only the given model will exist so do not create
                something else in ANNarchy!

            target_firing_rate_dict: dict
                keys = population names of model which should be configured, values = target firing rates in Hz

            do_not_config_list: list, optional, default=[]
                list with strings containing population names of populations which should not be configured

            print_guide: bool, optional, default=False
                if you want to get information about what you could do with model_configurator

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
        self.max_weight_dict = {pop_name: None for pop_name in self.pop_name_list}
        self.afferent_projection_dict = {
            pop_name: None for pop_name in self.pop_name_list
        }
        self.log_exist = False
        self.caller_name = ""
        self.log("model configurator log:")
        self.print_guide = print_guide

        ### print guide
        self._p_g(_p_g_1)

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

        if self.print_guide:
            print("\n[model_configurator WARNING]:")
            for line in str(txt).splitlines():
                wrapped_text = textwrap.fill(
                    line, width=print_width - 5, replace_whitespace=False
                )
                wrapped_text = textwrap.indent(wrapped_text, "    |")
                print(wrapped_text)
        print("")

    def set_synaptic_load(self, synaptic_load):
        """
        Args:
            synaptic_load: dict or number
                either a dictionary with keys = all populaiton names the model_configurator should configure
                or a single number between 0 and 1
                The dictionary values should be lists which contain either
        """
        pass

    def set_synaptic_contribution(self, synaptic_contribution):
        pass

    def get_max_syn(self):
        """
        get the weight dictionary for all populations given in target_firing_rate_dict
        keys = population names, values = dict which contain values = afferent projection names, values = lists with w_min and w_max
        """

        ### clear ANNarchy and create the model
        cnp_clear()
        self.model.create(do_compile=False)

        for pop_name in self.pop_name_list:
            self.log(pop_name)
            ### get the sizes of the many neuron population
            self.log(
                f"create network_1000 and get nr of neurons for network_many for {pop_name}"
            )
            nr_many_neurons = self.get_nr_many_neurons(pop_name)

            ### create the many and single neuron network
            self.log(f"create network_many and network_single for {pop_name}")
            self.net_many_dict[pop_name] = self.create_net_many(
                pop_name=pop_name, nr_neurons=nr_many_neurons
            )
            net_single_dict = self.create_net_single(pop_name=pop_name)

            ### get max synaptic currents (I and gs) using the single neuron network
            self.log(
                f"get max I_app, g_ampa and g_gaba using network_single for {pop_name}"
            )
            I_app_max, g_ampa_max, g_gaba_max = self.get_max_syn_currents(
                net_single_dict=net_single_dict, pop_name=pop_name
            )
            self.I_app_max_dict[pop_name] = I_app_max
            self.g_max_dict[pop_name] = {
                "ampa": g_ampa_max,
                "gaba": g_gaba_max,
            }

            ### get afferent projection dict
            self.log(f"get the afferent_projection_dict for {pop_name}")
            self.afferent_projection_dict[pop_name] = self.get_afferent_projection_dict(
                pop_name=pop_name
            )

            ### get the weight dict
            self.log(f"get the max_weight_dict for {pop_name}")
            self.max_weight_dict[pop_name] = self.get_max_weight_dict_for_pop(pop_name)

        ### print next steps
        template_weight_dict = self.max_weight_dict
        template_synaptic_load_dict = {}
        for pop_name in self.pop_name_list:
            template_synaptic_load_dict[pop_name] = []
            if "ampa" in self.afferent_projection_dict[pop_name]["target"]:
                template_synaptic_load_dict[pop_name].append("ampa_load")
            if "gaba" in self.afferent_projection_dict[pop_name]["target"]:
                template_synaptic_load_dict[pop_name].append("gaba_load")

        self._p_g(
            _p_g_after_get_weights(
                template_weight_dict=template_weight_dict,
                template_synaptic_load_dict=template_synaptic_load_dict,
            )
        )

        return self.max_weight_dict

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

    def get_max_syn_currents(self, net_single_dict, pop_name):
        """
        obtain I_app_max, g_ampa_max and g_gaba max.
        f_max = f_0 + f_t + 100
        I_app_max causes f_max (increases f from f_0 to f_max)
        g_ampa_max causes f_max (increases f from f_0 to f_max)
        I_gaba_max causes f_0 while I_app=I_app_max (decreases f from f_max to f_0)

        Args:
            net_single_dict: dict
                dictionary which contains the network/population to simulate and record from
                items: {
                    "net": ANNarchy network,
                    "population": ANNarchy population,
                    "monitor": ANNarchy monitor,
                    "variable_init_dict": initial values for population
                }

            pop_name: str
                population name from original model

        return:
            list containing [I_may, g_ampa_max, g_gaba_max]

        Abbreviations:
            f_max: max firing rate

            f_0: firing rate without syn currents

            f_t: target firing rate
        """

        ### extrace values from net_single_dict
        net = net_single_dict["net"]
        population = net_single_dict["population"]
        monitor = net_single_dict["monitor"]
        variable_init_dict = net_single_dict["variable_init_dict"]

        ### get f_0 and f_max
        f_0 = self.get_rate_1000(net, population, variable_init_dict, monitor)[0]
        f_max = f_0 + self.target_firing_rate_dict[pop_name] + 100

        ### find I_max with incremental_continuous_bound_search
        self.log("search I_app_max with y(X) = f(I_app=X, g_ampa=0, g_gaba=0)")
        I_max = self.incremental_continuous_bound_search(
            y_X=lambda X_val: self.get_rate_1000(
                net,
                population,
                variable_init_dict,
                monitor,
                I_app=X_val,
            )[0],
            y_bound=f_max,
            X_0=0,
            y_0=f_0,
            alpha_abs=1,
        )

        ### find g_ampa with incremental_continuous_bound_search
        ### dificulties with g_ampa:
        ### increasing until f_max is reached or increasing until f_0 is reached while I_app=-I_max
        ### DOES NOT WORK! because I_gaba is not well suited to compensate negative currents
        ### because I_gaba drives v only to 0 mV, the nearer v gets to 0 the smaller I_gaba gets
        ### if there is a negative current which is large enough close to v=0 this negative current
        ### will always prevent spiking, a large g_ampa cannot cause spiking in this case
        ### neuron models like Izhikevich have recovery variables i.e. negative currents after spiking
        ### --> faster spiking --> larger negative currents --> g_ampa cannot increase spiking
        ### therefore increase g_ampa either until f_max is reached or the firing rate saturates
        ### further difficulty: increasing g_ampa also leads to discontinuities in firing rate
        ### (sudden jumps in firing rate to new plateau... I think it's numerical inprecision)
        ### only get values until first discontinuity
        self.log("search g_ampa_max with y(X) = f(I_app=0, g_ampa=X, g_gaba=0)")
        g_ampa_max = self.incremental_continuous_bound_search(
            y_X=lambda X_val: self.get_rate_1000(
                net,
                population,
                variable_init_dict,
                monitor,
                g_ampa=X_val,
            )[0],
            y_bound=f_max,
            X_0=0,
            y_0=f_0,
            alpha_abs=1,
        )

        ### find g_gaba with incremental_continuous_bound_search
        self.log("search g_gaba_max with y(X) = f(I_app=I_max, g_ampa=0, g_gaba=X)")
        g_gaba_max = self.incremental_continuous_bound_search(
            y_X=lambda X_val: self.get_rate_1000(
                net,
                population,
                variable_init_dict,
                monitor,
                g_gaba=X_val,
                I_app=I_max,
            )[0],
            y_bound=f_0,
            X_0=0,
            y_0=f_max,
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
        ### log the task
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

        ### check if there is a discontinuity in y
        discontinuity_idx_list = self.get_discontinuity_idx_list(y_list_all)
        self.log("discontinuity_idx_list")
        self.log(f"{discontinuity_idx_list}")
        if len(discontinuity_idx_list) > 0:
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

    def get_rate_1000(
        self, net, population, variable_init_dict, monitor, I_app=0, g_ampa=0, g_gaba=0
    ):
        """
        simulates 1000 ms a population and returns the firing rate of each neuron

        Args:
            net: ANNarchy network
                network which contains the population and monitor

            population: ANNarchy population
                population which is recorded and stimulated

            variable_init_dict: dict
                containing the initial values of the population neuron

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
        for var_name, var_val in variable_init_dict.items():
            tmp = getattr(population, var_name)
            set_val = np.ones(len(tmp)) * var_val
            setattr(population, var_name, set_val)
        ### slow down conductances (i.e. make them constant for 1000 ms)
        population.tau_ampa = 1e20
        population.tau_gaba = 1e20
        ### apply given variables
        population.I_app = I_app
        population.g_ampa = g_ampa
        population.g_gaba = g_gaba
        ### simulate 1000 ms
        net.simulate(1000)
        ### rate is nbr spikes since simulated 1000 ms
        spike_dict = monitor.get("spike")
        f_arr = np.zeros(len(population))
        for idx_n, n in enumerate(spike_dict.keys()):
            rate = len(spike_dict[n])
            f_arr[idx_n] = rate

        return f_arr

    def create_net_many(self, pop_name, nr_neurons):
        """
        creates a network with the neuron type of the population given by pop_name
        the number of neurons is selected so that the simulation of 1000 ms takes
        arbout 10 s

        Args:
            pop_name: str
                population name

            nr_neurons: int
                size of the created population
        """
        ### create the many neuron population
        many_neuron = Population(
            nr_neurons,
            neuron=get_population(pop_name).neuron_type,
            name=f"many_neuron_{pop_name}",
        )

        ### set the attributes of the neurons
        for attr_name, attr_val in get_population(pop_name).init.items():
            setattr(many_neuron, attr_name, attr_val)

        ### create Monitor for many neuron
        mon_many = Monitor(many_neuron, ["spike"])

        ### create network with many neuron
        net_many = Network()
        net_many.add([many_neuron, mon_many])
        net_many.compile()

        ### network dict
        net_many_dict = {
            "net": net_many,
            "population": net_many.get(many_neuron),
            "monitor": net_many.get(mon_many),
        }

        return net_many_dict

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
            neuron=get_population(pop_name).neuron_type,
            name=f"single_neuron_{pop_name}",
        )
        ### set the attributes of the neuron
        for attr_name, attr_val in get_population(pop_name).init.items():
            setattr(single_neuron, attr_name, attr_val)

        ### create Monitor for single neuron
        mon_single = Monitor(single_neuron, ["spike"])

        ### create network with single neuron
        net_single = Network()
        net_single.add([single_neuron, mon_single])
        net_single.compile()

        ### get the values of the variables after 2000 ms simulation
        variable_init_dict = self.get_init_neuron_variables(
            net_single, net_single.get(single_neuron)
        )

        ### network dict
        net_single_dict = {
            "net": net_single,
            "population": net_single.get(single_neuron),
            "monitor": net_single.get(mon_single),
            "variable_init_dict": variable_init_dict,
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
        net.reset()
        pop.I_app = 0
        net.simulate(2000)
        variable_init_dict = {
            var_name: getattr(pop, var_name) for var_name in pop.variables
        }
        net.reset()
        return variable_init_dict

    def get_nr_many_neurons(self, pop_name):
        """
        gets population size of a population given by the neuron type of pop_name
        so that simulating 1000 ms takes about 10 s
        and that the 3rd root of population size is an int

        Args:
            pop_name: str
                population name
        """
        duration_1000 = self.get_duration_1000(pop_name=pop_name, pop_size=1000)
        nr_factor = 10 / duration_1000
        nr_neuron_10s = int(int(int(1000 * nr_factor) ** (1 / 3)) ** 3)
        return nr_neuron_10s

    def get_duration_1000(self, pop_name, pop_size):
        """
        gets the duration for simulating 1000 ms with a population given
        by the neuron type of pop_name and the size pop_size

        Args:
            pop_name: str
                population name

            pop_size: int
                population size i.e. number of neurons
        """
        ### create many neuron population with determined neuron_type and given size
        many_neuron = Population(
            pop_size,
            neuron=get_population(pop_name).neuron_type,
            name=f"duration_1000_{pop_name}",
        )
        ### set the attributes of the neurons
        for attr_name, attr_val in get_population(pop_name).init.items():
            setattr(many_neuron, attr_name, attr_val)

        ### create Monitor for many neuron population
        mon_many = Monitor(many_neuron, ["spike"])

        ### create network and record duration for 1000 ms simulation
        net = Network()
        net.add([many_neuron, mon_many])
        net.compile()
        start = time()
        net.simulate(1000)
        end = time()
        duration = end - start

        return duration

    def get_max_weight_dict_for_pop(self, pop_name):
        """
        get the weight dict for a single population

        Args:
            pop_name: str
                population name

        return: dict
            keys = afferent projection names, values = max weights
        """
        ### create dictionary with timeconstants of g_ampa and g_gaba within the defined population
        self.tau_dict[pop_name] = {
            "ampa": get_population(pop_name).tau_ampa,
            "gaba": get_population(pop_name).tau_gaba,
        }

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
        ### get post_pop_name
        post_pop_name = get_projection(proj_name).post.name
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
            ### get spike times over 1s for the spike frequency
            ### and transform them into ms
            if spike_frequency > 0:
                spike_times_arr = np.arange(0, 1, 1 / spike_frequency)
                spike_times_arr = spike_times_arr * 1000
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


_p_g_1 = """First call get_max_syn.
This determines max synaptic conductances and weights of all afferent projections of the model populations and returns a dictionary with max weights."""

_p_g_after_get_weights = (
    lambda template_weight_dict, template_synaptic_load_dict: f"""Now either set the weights of all projections directly or first set the synaptic load of the populations and then the synaptic contributions of the afferent projections.
You can set the weights using the function .set_weights() which requires a weight_dict as argument.
Use this template for the weight_dict:

{template_weight_dict}

The values within the template are the maximum weight values.


You can set the synaptic load using the function .set_syn_load() which requires a synaptic_load_dict or a single number between 0 and 1.
Use this template for the synaptic_load_dict:

{template_synaptic_load_dict}

'ampa_load' and 'gaba_load' are placeholders, replace them with values between 0 and 1
"""
)
