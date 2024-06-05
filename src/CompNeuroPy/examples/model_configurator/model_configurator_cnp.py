from CompNeuroPy.generate_model import CompNeuroModel
from CompNeuroPy import model_functions as mf
from CompNeuroPy import extra_functions as ef

from ANNarchy import (
    Population,
    Projection,
    Synapse,
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
    Uniform,
    get_current_step,
    projections,
    populations,
    Binomial,
    CurrentInjection,
)

from ANNarchy.core import ConnectorMethods

# from ANNarchy.core.Global import _network
import numpy as np
from scipy.interpolate import interp1d, interpn
from scipy.signal import find_peaks, argrelmin
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
from ANNarchy.extensions.bold import BoldMonitor
from sklearn.linear_model import LinearRegression
import sympy as sp


class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.caller_name = ""
        with open(log_file, "w") as f:
            print("Logger file:", file=f)

    def log(self, txt):
        caller_frame = inspect.currentframe().f_back
        caller_name = caller_frame.f_code.co_name

        if caller_name == self.caller_name:
            txt = f"{textwrap.indent(str(txt), '    ')}"
        else:
            txt = f"[{caller_name}]:\n{textwrap.indent(str(txt), '    ')}"

        self.caller_name = caller_name

        with open(self.log_file, "a") as f:
            print(txt, file=f)


class ModelConfigurator:
    def __init__(
        self,
        model: CompNeuroModel,
        target_firing_rate_dict: dict,
        max_psp: float = 10.0,
        do_not_config_list: list[str] = [],
        print_guide: bool = False,
        I_app_variable: str = "I_app",
        cache: bool = False,
        clear_cache: bool = False,
        log_file: str | None = None,
    ):
        ### create logger
        self._logger = Logger(log_file=log_file)
        ### analyze the given model, create model before analyzing, then clear ANNarchy
        self._analyze_model = AnalyzeModel(model=model)
        ### create the CompNeuroModel object for the reduced model (the model itself is
        ### not created yet)
        self._model_reduced = CreateReducedModel(
            model=model,
            analyze_model=self._analyze_model,
            reduced_size=100,
            do_create=False,
            do_compile=False,
            verbose=True,
        )
        ### create the single neuron networks (networks are compiled and ready to be
        ### simulated), normal model for searching for max conductances, max input
        ### current, resting firing rate; voltage clamp model for preparing the PSP
        ### simulationssearching, i.e., for resting potential and corresponding input
        ### current I_hold (for self-active neurons)
        self._single_nets = CreateSingleNeuronNetworks(
            model=model,
            analyze_model=self._analyze_model,
            do_not_config_list=do_not_config_list,
        )
        ### define the simulator with all simulations with the single neuron networks
        self._simulator = Simulator(single_nets=self._single_nets)
        ### get the resting potential and corresponding I_hold for each population using
        ### the voltage clamp networks
        self._v_rest = PreparePSP(
            simulator=self._simulator,
            model=model,
            single_nets=self._single_nets,
            do_not_config_list=do_not_config_list,
            logger=self._logger,
        )
        self._max_syn = GetMaxSyn()
        self._weight_templates = GetWeightTemplates()


class AnalyzeModel:
    """
    Class to analyze the given model to be able to reproduce it.
    """

    _connector_methods_dict = {
        "One-to-One": ConnectorMethods.connect_one_to_one,
        "All-to-All": ConnectorMethods.connect_all_to_all,
        "Gaussian": ConnectorMethods.connect_gaussian,
        "Difference-of-Gaussian": ConnectorMethods.connect_dog,
        "Random": ConnectorMethods.connect_fixed_probability,
        "Random Convergent": ConnectorMethods.connect_fixed_number_pre,
        "Random Divergent": ConnectorMethods.connect_fixed_number_post,
        "User-defined": ConnectorMethods.connect_with_func,
        "MatrixMarket": ConnectorMethods.connect_from_matrix_market,
        "Connectivity matrix": ConnectorMethods.connect_from_matrix,
        "Sparse connectivity matrix": ConnectorMethods.connect_from_sparse,
        "From File": ConnectorMethods.connect_from_file,
    }

    def __init__(self, model: CompNeuroModel):
        ### clear ANNarchy and create the model
        self._clear_model(model=model, do_create=True)

        ### get population info (eq, params etc.)
        self._analyze_populations(model=model)

        ### get projection info
        self._analyze_projections(model=model)

        ### clear ANNarchy
        self._clear_model(model=model, do_create=False)

    def _clear_model(self, model: CompNeuroModel, do_create: bool = True):
        mf.cnp_clear(functions=False, neurons=True, synapses=True, constants=False)
        model.create(do_create=do_create, do_compile=False)

    def _analyze_populations(self, model: CompNeuroModel):
        """
        Get info of each population

        Args:
            model (CompNeuroModel):
                Model to be analyzed
        """
        ### values of the paramters and variables of the population's neurons, keys are
        ### the names of paramters and variables
        self.neuron_model_attr_dict: dict[str, dict] = {}
        ### arguments of the __init__ function of the Neuron class
        self.neuron_model_init_parameter_dict: dict[str, dict] = {}
        ### arguments of the __init__ function of the Population class
        self.pop_init_parameter_dict: dict[str, dict] = {}

        ### for loop over all populations
        for pop_name in model.populations:
            pop: Population = get_population(pop_name)
            ### get the neuron model attributes (parameters/variables)
            ### old: self.neuron_model_parameters_dict
            ### old: self.neuron_model_attributes_dict = keys()
            self.neuron_model_attr_dict[pop.name] = pop.init
            ### get a dict of all arguments of the __init__ function of the Neuron
            ### ignore self
            ### old: self.neuron_model_dict[pop_name]
            init_params = inspect.signature(Neuron.__init__).parameters
            self.neuron_model_init_parameter_dict[pop.name] = {
                param: getattr(pop.neuron_type, param)
                for param in init_params
                if param != "self"
            }
            ### get a dict of all arguments of the __init__ function of the Population
            ### ignore self, storage_order and copied
            init_params = inspect.signature(Population.__init__).parameters
            self.pop_init_parameter_dict[pop.name] = {
                param: getattr(pop, param)
                for param in init_params
                if param != "self" and param != "storage_order" and param != "copied"
            }
            ### get the afferent projections dict of the population TODO do we still need this?
            # self.afferent_projection_dict[pop_name] = (
            #     self._get_afferent_projection_dict(pop_name=pop_name)
            # )

    def _analyze_projections(self, model: CompNeuroModel):
        """
        Get info of each projection

        Args:
            model (CompNeuroModel):
                Model to be analyzed
        """
        ### parameters of the __init__ function of the Projection class
        self.proj_init_parameter_dict: dict[str, dict] = {}
        ### parameters of the __init__ function of the Synapse class
        self.synapse_init_parameter_dict: dict[str, dict] = {}
        ### values of the paramters and variables of the synapse, keys are the names of
        ### paramters and variables
        self.synapse_model_attr_dict: dict[str, dict] = {}
        ### connector functions of the projections
        self.connector_function_dict: dict = {}
        ### parameters of the connector functions of the projections
        self.connector_function_parameter_dict: dict = {}
        ### names of pre- and post-synaptic populations of the projections
        ### old: self.post_pop_name_dict and self.pre_pop_name_dict
        self.pre_post_pop_name_dict: dict[str, tuple] = {}
        ### sizes of pre- and post-synaptic populations of the projections
        ### old: self.pre_pop_size_dict
        self.pre_post_pop_size_dict: dict[str, tuple] = {}

        ### loop over all projections
        for proj_name in model.projections:
            proj: Projection = get_projection(proj_name)
            ### get the synapse model attributes (parameters/variables)
            self.synapse_model_attr_dict[proj.name] = proj.init
            ### get a dict of all paramters of the __init__ function of the Synapse
            init_params = inspect.signature(Synapse.__init__).parameters
            self.synapse_init_parameter_dict[proj.name] = {
                param: getattr(proj.synapse_type, param)
                for param in init_params
                if param != "self"
            }
            ### get a dict of all paramters of the __init__ function of the Projection
            init_params = inspect.signature(Projection.__init__).parameters
            self.proj_init_parameter_dict[proj_name] = {
                param: getattr(proj, param)
                for param in init_params
                if param != "self" and param != "synapse" and param != "copied"
            }

            ### get the connector function of the projection and its parameters
            ### raise errors for not supported connector functions
            if (
                proj.connector_name == "User-defined"
                or proj.connector_name == "MatrixMarket"
                or proj.connector_name == "From File"
            ):
                raise ValueError(
                    f"Connector function '{self._connector_methods_dict[proj.connector_name].__name__}' not supported yet"
                )

            ### get the connector function
            self.connector_function_dict[proj.name] = self._connector_methods_dict[
                proj.connector_name
            ]

            ### get the parameters of the connector function
            self.connector_function_parameter_dict[proj.name] = (
                self._get_connector_parameters(proj)
            )

            ### get the names of the pre- and post-synaptic populations
            self.pre_post_pop_name_dict[proj.name] = (proj.pre.name, proj.post.name)

            ### get the sizes of the pre- and post-synaptic populations
            self.pre_post_pop_size_dict[proj.name] = (
                proj.pre.size,
                proj.post.size,
            )

    def _get_connector_parameters(self, proj: Projection):
        """
        Get the parameters of the given connector function.

        Args:
            proj (Projection):
                Projection for which the connector parameters are needed

        Returns:
            connector_parameters_dict (dict):
                Parameters of the given connector function
        """

        if proj.connector_name == "One-to-One":
            return {
                "weights": proj._connection_args[0],
                "delays": proj._connection_args[1],
                "force_multiple_weights": not (proj._single_constant_weight),
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "All-to-All":
            return {
                "weights": proj._connection_args[0],
                "delays": proj._connection_args[1],
                "allow_self_connections": proj._connection_args[2],
                "force_multiple_weights": not (proj._single_constant_weight),
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "Gaussian":
            return {
                "amp": proj._connection_args[0],
                "sigma": proj._connection_args[1],
                "delays": proj._connection_args[2],
                "limit": proj._connection_args[3],
                "allow_self_connections": proj._connection_args[4],
                "storage_format": proj._storage_format,
            }
        elif proj.connector_name == "Difference-of-Gaussian":
            return {
                "amp_pos": proj._connection_args[0],
                "sigma_pos": proj._connection_args[1],
                "amp_neg": proj._connection_args[2],
                "sigma_neg": proj._connection_args[3],
                "delays": proj._connection_args[4],
                "limit": proj._connection_args[5],
                "allow_self_connections": proj._connection_args[6],
                "storage_format": proj._storage_format,
            }
        elif proj.connector_name == "Random":
            return {
                "probability": proj._connection_args[0],
                "weights": proj._connection_args[1],
                "delays": proj._connection_args[2],
                "allow_self_connections": proj._connection_args[3],
                "force_multiple_weights": not (proj._single_constant_weight),
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "Random Convergent":
            return {
                "number": proj._connection_args[0],
                "weights": proj._connection_args[1],
                "delays": proj._connection_args[2],
                "allow_self_connections": proj._connection_args[3],
                "force_multiple_weights": not (proj._single_constant_weight),
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "Random Divergent":
            return {
                "number": proj._connection_args[0],
                "weights": proj._connection_args[1],
                "delays": proj._connection_args[2],
                "allow_self_connections": proj._connection_args[3],
                "force_multiple_weights": not (proj._single_constant_weight),
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "Connectivity matrix":
            return {
                "weights": proj._connection_args[0],
                "delays": proj._connection_args[1],
                "pre_post": proj._connection_args[2],
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "Sparse connectivity matrix":
            return {
                "weights": proj._connection_args[0],
                "delays": proj._connection_args[1],
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }


class CreateSingleNeuronNetworks:
    """
    Class to create single neuron networks for normal and voltage clamp mode.

    Attributes:
        single_net_dict (dict):
            Nested dict containing the single neuron networks for normal and voltage
            clamp mode
            keys: mode (str)
                normal or v_clamp
            values: dict
                keys: pop_name (str)
                    population name
                values: dict
                    keys: net, population, monitor, init_sampler
                    values: Network, Population, Monitor, ArrSampler
    """

    def __init__(
        self,
        model: CompNeuroModel,
        analyze_model: AnalyzeModel,
        do_not_config_list: list[str],
    ):
        """
        Args:
            model (CompNeuroModel):
                Model to be analyzed
            analyze_model (AnalyzeModel):
                Analyzed model
            do_not_config_list (list[str]):
                List of population names which should not be configured
        """
        self._single_net_dict = {}
        ### create the single neuron networks for normal and voltage clamp mode
        for mode in ["normal", "v_clamp"]:
            self._single_net_dict[mode] = {}
            self.create_single_neuron_networks(
                model=model,
                analyze_model=analyze_model,
                do_not_config_list=do_not_config_list,
                mode=mode,
            )

    def single_net_dict(self, pop_name: str, mode: str):
        """
        Get the single neuron network dict for the given population and mode.

        Args:
            pop_name (str):
                Name of the population
            mode (str):
                Mode for which the network should be returned

        Returns:
            net_single_dict (dict):
                Dict containing the Network, Population, Monitor and ArrSampler objects
        """
        return self.ReturnSingleNeuronNetworks(self._single_net_dict[mode][pop_name])

    class ReturnSingleNeuronNetworks:
        def __init__(self, single_net_dict):
            self.net: Network = single_net_dict["net"]
            self.population: Population = single_net_dict["population"]
            self.monitor: Monitor = single_net_dict["monitor"]
            self.init_sampler: ArrSampler = single_net_dict["init_sampler"]

    def create_single_neuron_networks(
        self,
        model: CompNeuroModel,
        analyze_model: AnalyzeModel,
        do_not_config_list: list[str],
        mode: str,
    ):
        """
        Create the single neuron networks for the given mode. Sets the single_net_dict.

        Args:
            model (CompNeuroModel):
                Model to be analyzed
            analyze_model (AnalyzeModel):
                Analyzed model
            do_not_config_list (list[str]):
                List of population names which should not be configured
            mode (str):
                Mode for which the single neuron networks should be created
        """

        ### loop over populations which should be configured
        for pop_name in model.populations:
            ### skip populations which should not be configured
            if pop_name in do_not_config_list:
                continue
            ### store the dict containing the network etc
            self._single_net_dict[mode][pop_name] = self.create_net_single(
                pop_name=pop_name, analyze_model=analyze_model, mode=mode
            )

    def create_net_single(self, pop_name: str, analyze_model: AnalyzeModel, mode: str):
        """
        Creates a network with the neuron type of the population given by pop_name for
        the given mode. The population size is set to 1.

        Args:
            pop_name (str):
                Name of the population
            analyze_model (AnalyzeModel):
                Analyzed model
            mode (str):
                Mode for which the network should be created

        Returns:
            net_single_dict (dict):
                Dict containing the Network, Population, Monitor and ArrSampler objects
        """
        ### create the adjusted neuron model for the stop condition
        neuron_model_new = self.get_single_neuron_neuron_model(
            pop_name=pop_name, analyze_model=analyze_model, mode=mode
        )

        ### create the single neuron population
        pop_single_neuron = self.get_single_neuron_population(
            pop_name=pop_name,
            neuron_model_new=neuron_model_new,
            analyze_model=analyze_model,
            mode=mode,
        )

        ### create Monitor for single neuron
        if mode == "normal":
            mon_single = Monitor(pop_single_neuron, ["spike", "v"])
        elif mode == "v_clamp":
            mon_single = Monitor(pop_single_neuron, ["v_clamp_rec_sign"])

        ### create network with single neuron and compile it
        net_single = Network()
        net_single.add([pop_single_neuron, mon_single])
        mf.compile_in_folder(
            folder_name=f"single_net_{mode}_{pop_name}", silent=True, net=net_single
        )

        ### network dict
        net_single_dict = {
            "net": net_single,
            "population": net_single.get(pop_single_neuron),
            "monitor": net_single.get(mon_single),
            "init_sampler": None,
        }

        ### for v_clamp we are done here
        if mode == "v_clamp":
            return net_single_dict

        ### for normal neuron get the init sampler for the variables of the neuron model
        ### (to initialize a population of the neuron model)
        init_sampler = self.get_neuron_model_init_sampler(
            net=net_single, pop=net_single.get(pop_single_neuron)
        )
        net_single_dict["init_sampler"] = init_sampler

        return net_single_dict

    def get_single_neuron_neuron_model(
        self, pop_name: str, analyze_model: AnalyzeModel, mode=str
    ):
        """
        Create the adjusted neuron model for the given mode.

        Args:
            pop_name (str):
                Name of the population
            analyze_model (AnalyzeModel):
                Analyzed model
            mode (str):
                Mode for which the neuron model should be created

        Returns:
            neuron_model_new (Neuron):
                Adjusted neuron model
        """
        ### get the stored parameters of the __init__ function of the Neuron
        neuron_model_init_parameter_dict = (
            analyze_model.neuron_model_init_parameter_dict[pop_name].copy()
        )
        ### Define the attributes of the neuron model as sympy symbols
        neuron_model_attributes_name_list = list(
            analyze_model.neuron_model_attr_dict[pop_name].keys()
        )
        ### add v_before_psp and v_psp_thresh to equations/parameters, for the stop
        ### condition below
        self.adjust_neuron_model(
            neuron_model_init_parameter_dict,
            neuron_model_attributes_name_list,
            mode=mode,
        )
        ### create the adjusted neuron model
        neuron_model_new = Neuron(**neuron_model_init_parameter_dict)
        return neuron_model_new

    def get_single_neuron_population(
        self,
        pop_name: str,
        neuron_model_new: Neuron,
        analyze_model: AnalyzeModel,
        mode: str,
    ):
        """
        Create the single neuron population for the given mode.

        Args:
            pop_name (str):
                Name of the population
            neuron_model_new (Neuron):
                Adjusted neuron model
            analyze_model (AnalyzeModel):
                Analyzed model
            mode (str):
                Mode for which the population should be created

        Returns:
            pop_single_neuron (Population):
                Single neuron population
        """
        if mode == "normal":
            pop_single_neuron = Population(
                1,
                neuron=neuron_model_new,
                name=f"single_neuron_{pop_name}",
                stop_condition="((abs(v-v_psp_thresh)<0.01) and (abs(v_before_psp-v_psp_thresh)>0.01)): any",
            )
        elif mode == "v_clamp":
            ### create the single neuron population
            pop_single_neuron = Population(
                1,
                neuron=neuron_model_new,
                name=f"single_neuron_v_clamp_{pop_name}",
            )

        ### get the stored parameters and variables
        neuron_model_attr_dict = analyze_model.neuron_model_attr_dict[pop_name]
        ### set the parameters and variables
        for attr_name, attr_val in neuron_model_attr_dict.items():
            setattr(pop_single_neuron, attr_name, attr_val)
        return pop_single_neuron

    def get_neuron_model_init_sampler(self, net: Network, pop: Population):
        """
        Create a sampler for the initial values of the variables of the neuron model by
        simulating the neuron for 10000 ms and afterwards simulating 2000 ms and
        sampling the variables every dt.

        Args:
            net (Network):
                Network with the single neuron population
            pop (Population):
                Single neuron population

        Returns:
            sampler (ArrSampler):
                Sampler for the initial values of the variables of the neuron model
        """

        ### reset network and deactivate input
        net.reset()
        pop.I_app = 0

        ### 10000 ms init duration
        net.simulate(10000)

        ### simulate 2000 ms and check every dt the variables of the neuron
        time_steps = int(2000 / dt())
        var_name_list = list(pop.variables)
        var_arr = np.zeros((time_steps, len(var_name_list)))
        for time_idx in range(time_steps):
            net.simulate(dt())
            get_arr = np.array([getattr(pop, var_name) for var_name in pop.variables])
            var_arr[time_idx, :] = get_arr[:, 0]

        ### reset network after simulation
        net.reset()

        ### create a sampler with the data samples of from the 1000 ms simulation
        sampler = ArrSampler(arr=var_arr, var_name_list=var_name_list)
        return sampler

    def adjust_neuron_model(
        self,
        neuron_model_init_parameter_dict: dict,
        neuron_model_attributes_name_list: list[str],
        mode: str,
    ):
        """
        Adjust the parameters and equations of the neuron model for the given mode.

        Args:
            neuron_model_init_parameter_dict (dict):
                Dict with the parameters and equations of the neuron model
            neuron_model_attributes_name_list (list[str]):
                List of the names of the attributes of the neuron model
            mode (str):
                Mode for which the neuron model should be adjusted
        """
        ### get the equations of the neuron model as a list of strings
        equations_line_split_list = str(
            neuron_model_init_parameter_dict["equations"]
        ).splitlines()
        ### get the parameters of the neuron model as a list of strings
        parameters_line_split_list = str(
            neuron_model_init_parameter_dict["parameters"]
        ).splitlines()

        if mode == "normal":
            ### add v_before_psp=v at the beginning of the equations
            equations_line_split_list.insert(0, "v_before_psp = v")
            ### add v_psp_thresh to the parameters
            parameters_line_split_list.append("v_psp_thresh = 0 : population")
        elif mode == "v_clamp":
            ### get new equations for voltage clamp
            equations_new_list = CreateVoltageClampEquations(
                equations_line_split_list, neuron_model_attributes_name_list
            ).eq_new
            neuron_model_init_parameter_dict["equations"] = equations_new_list
            ### add v_clamp_rec_thresh to the parameters
            parameters_line_split_list.append("v_clamp_rec_thresh = 0 : population")

        ### join equations and parameters to a string and store them in the dict
        neuron_model_init_parameter_dict["equations"] = "\n".join(
            equations_line_split_list
        )
        neuron_model_init_parameter_dict["parameters"] = "\n".join(
            parameters_line_split_list
        )


class Simulator:
    def __init__(self, single_nets: CreateSingleNeuronNetworks):
        self.single_nets = single_nets

    def get_v_clamp_2000(
        self,
        pop_name: str,
        v: float | None = None,
        I_app: float | None = None,
    ) -> float:
        """
        Simulates the v_clamp single neuron network of the given pop_name for 2000 ms
        and returns the v_clamp_rec value of the single neuron after 2000 ms. The
        returned values is "dv/dt". Therefore, to get the hypothetical change of v for a
        single time step multiply it with dt!

        Args:
            pop_name (str):
                Name of the population
            v (float):
                Membrane potential (does not change over time due to voltage clamp)
            I_app (float):
                Applied current

        Returns:
            v_clamp_rec (float):
                v_clamp_rec value of the single neuron after 2000 ms
        """
        ### get the network, population, init_sampler
        net = self.single_nets.single_net_dict(pop_name=pop_name, mode="v_clamp").net
        population = self.single_nets.single_net_dict(
            pop_name=pop_name, mode="v_clamp"
        ).population
        init_sampler = self.single_nets.single_net_dict(
            pop_name=pop_name, mode="v_clamp"
        ).init_sampler
        ### reset network
        net.reset()
        net.set_seed(0)
        ### set the initial variables of the neuron model
        if init_sampler is not None:
            init_sampler.set_init_variables(population)
        ### set v and I_app
        if v is not None:
            population.v = v
        if I_app is not None:
            population.I_app = I_app
        ### simulate 2000 ms
        net.simulate(2000)
        ### return the v_clamp_rec value of the single neuron after 2000 ms
        return population.v_clamp_rec[0]

    def get_v_2000(self, pop_name, initial_variables, I_app=None, do_plot=True):
        """
        Simulate normal single neuron 2000 ms and return v for this duration.

        Args:
            pop_name (str):
                Name of the population
            initial_variables (dict):
                Initial variables of the neuron model
            I_app (float):
                Applied current
            do_plot (bool):
                If True, plot the membrane potential

        Returns:
            v_arr (np.array):
                Membrane potential for the 2000 ms simulation with shape: (time_steps,)
        """
        ### get the network, population, monitor
        net = self.single_nets.single_net_dict(pop_name=pop_name, mode="normal").net
        population = self.single_nets.single_net_dict(
            pop_name=pop_name, mode="normal"
        ).population
        monitor = self.single_nets.single_net_dict(
            pop_name=pop_name, mode="normal"
        ).monitor
        ### reset network
        net.reset()
        net.set_seed(0)
        ### set the initial variables of the neuron model
        for var_name, var_val in initial_variables.items():
            if var_name in population.variables:
                setattr(population, var_name, var_val)
        ### set I_app
        if I_app is not None:
            population.I_app = I_app
        ### simulate
        net.simulate(2000)
        v_arr = monitor.get("v")[:, 0]

        if do_plot:
            plt.figure()
            plt.title(f"{population.I_app}")
            plt.plot(v_arr)
            plt.savefig(f"tmp_v_rest_{pop_name}.png")
            plt.close("all")

        return v_arr


class CreateReducedModel:
    """
    Class to create a reduced model from the original model. It is accessable via the
    attribute model_reduced.

    Attributes:
        model_reduced (CompNeuroModel):
            Reduced model, created but not compiled
    """

    def __init__(
        self,
        model: CompNeuroModel,
        analyze_model: AnalyzeModel,
        reduced_size: int,
        do_create: bool = False,
        do_compile: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Prepare model for reduction.

        Args:
            model (CompNeuroModel):
                Model to be reduced
            reduced_size (int):
                Size of the reduced populations
        """
        ### set the attributes
        self._model = model
        self._analyze_model = analyze_model
        self._reduced_size = reduced_size
        self._verbose = verbose
        ### recreate model with reduced populations and projections
        self.model_reduced = CompNeuroModel(
            model_creation_function=self.recreate_model,
            name=f"{model.name}_reduced",
            description=f"{model.description}\nWith reduced populations and projections.",
            do_create=do_create,
            do_compile=do_compile,
            compile_folder_name=f"{model.compile_folder_name}_reduced",
        )

    def recreate_model(self):
        """
        Recreates the model with reduced populations and projections.
        """
        ### 1st for each population create a reduced population
        for pop_name in self._model.populations:
            self.create_reduced_pop(pop_name)
        ### 2nd for each population which is a presynaptic population, create a spikes collecting aux population
        for pop_name in self._model.populations:
            self.create_spike_collecting_aux_pop(pop_name)
        ## 3rd for each population which has afferents create a population for incoming spikes for each target type
        for pop_name in self._model.populations:
            self.create_conductance_aux_pop(pop_name, target="ampa")
            self.create_conductance_aux_pop(pop_name, target="gaba")

    def create_reduced_pop(self, pop_name: str):
        """
        Create a reduced population from the given population.

        Args:
            pop_name (str):
                Name of the population to be reduced
        """
        if self._verbose:
            print(f"create_reduced_pop for {pop_name}")
        ### 1st check how the population is connected
        _, is_postsynaptic, ampa, gaba = self.how_pop_is_connected(pop_name)

        ### 2nd recreate neuron model
        ### get the stored parameters of the __init__ function of the Neuron
        neuron_model_init_parameter_dict = (
            self._analyze_model.neuron_model_init_parameter_dict[pop_name].copy()
        )
        ### if the population is a postsynaptic population adjust the synaptic
        ### conductance equations
        if is_postsynaptic:
            neuron_model_init_parameter_dict = self.adjust_neuron_model(
                neuron_model_init_parameter_dict, ampa=ampa, gaba=gaba
            )
        ### create the new neuron model
        neuron_model_new = Neuron(**neuron_model_init_parameter_dict)

        ### 3rd recreate the population
        ### get the stored parameters of the __init__ function of the Population
        pop_init_parameter_dict = self._analyze_model.pop_init_parameter_dict[
            pop_name
        ].copy()
        ### replace the neuron model with the new neuron model
        pop_init_parameter_dict["neuron"] = neuron_model_new
        ### replace the size with the reduced size (if reduced size is smaller than the
        ### original size)
        ### TODO add model requirements somewhere, here requirements = geometry = int
        pop_init_parameter_dict["geometry"] = min(
            [pop_init_parameter_dict["geometry"][0], self._reduced_size]
        )
        ### append _reduce to the name
        pop_init_parameter_dict["name"] = f"{pop_name}_reduced"
        ### create the new population
        pop_new = Population(**pop_init_parameter_dict)

        ### 4th set the parameters and variables of the population's neurons
        ### get the stored parameters and variables
        neuron_model_attr_dict = self._analyze_model.neuron_model_attr_dict[pop_name]
        ### set the parameters and variables
        for attr_name, attr_val in neuron_model_attr_dict.items():
            setattr(pop_new, attr_name, attr_val)

    def create_spike_collecting_aux_pop(self, pop_name: str):
        """
        Create a spike collecting population for the given population.

        Args:
            pop_name (str):
                Name of the population for which the spike collecting population should be created
        """
        ### get all efferent projections
        efferent_projection_list = [
            proj_name
            for proj_name, pre_post_pop_name_dict in self._analyze_model.pre_post_pop_name_dict.items()
            if pre_post_pop_name_dict[0] == pop_name
        ]
        ### check if pop has efferent projections
        if len(efferent_projection_list) == 0:
            return
        if self._verbose:
            print(f"create_spike_collecting_aux_pop for {pop_name}")
        ### create the spike collecting population
        pop_aux = Population(
            1,
            neuron=self.SpikeProbCalcNeuron(
                reduced_size=min(
                    [
                        self._analyze_model.pop_init_parameter_dict[pop_name][
                            "geometry"
                        ][0],
                        self._reduced_size,
                    ]
                )
            ),
            name=f"{pop_name}_spike_collecting_aux",
        )
        ### create the projection from reduced pop to spike collecting aux pop
        proj = Projection(
            pre=get_population(pop_name + "_reduced"),
            post=pop_aux,
            target="ampa",
            name=f"proj_{pop_name}_spike_collecting_aux",
        )
        proj.connect_all_to_all(weights=1)

    def create_conductance_aux_pop(self, pop_name: str, target: str):
        """
        Create a conductance calculating population for the given population and target.

        Args:
            pop_name (str):
                Name of the population for which the conductance calculating population should be created
            target (str):
                Target type of the conductance calculating population
        """
        ### get all afferent projections
        afferent_projection_list = [
            proj_name
            for proj_name, pre_post_pop_name_dict in self._analyze_model.pre_post_pop_name_dict.items()
            if pre_post_pop_name_dict[1] == pop_name
        ]
        ### check if pop has afferent projections
        if len(afferent_projection_list) == 0:
            return
        ### get all afferent projections with target type
        afferent_target_projection_list = [
            proj_name
            for proj_name in afferent_projection_list
            if self._analyze_model.proj_init_parameter_dict[proj_name]["target"]
            == target
        ]
        ### check if there are afferent projections with target type
        if len(afferent_target_projection_list) == 0:
            return
        if self._verbose:
            print(f"create_conductance_aux_pop for {pop_name} target {target}")
        ### get projection informations
        ### TODO somewhere add model requirements, here requirements = geometry = int and connection = fixed_probability i.e. random (with weights and probability)
        projection_dict = {
            proj_name: {
                "pre_size": self._analyze_model.pop_init_parameter_dict[
                    self._analyze_model.pre_post_pop_name_dict[proj_name][0]
                ]["geometry"][0],
                "connection_prob": self._analyze_model.connector_function_parameter_dict[
                    proj_name
                ][
                    "probability"
                ],
                "weights": self._analyze_model.connector_function_parameter_dict[
                    proj_name
                ]["weights"],
                "pre_name": self._analyze_model.pre_post_pop_name_dict[proj_name][0],
            }
            for proj_name in afferent_target_projection_list
        }
        ### create the conductance calculating population
        pop_aux = Population(
            self._analyze_model.pop_init_parameter_dict[pop_name]["geometry"][0],
            neuron=self.InputCalcNeuron(projection_dict=projection_dict),
            name=f"{pop_name}_{target}_aux",
        )
        ### set number of synapses parameter for each projection
        for proj_name, vals in projection_dict.items():
            number_synapses = Binomial(
                n=vals["pre_size"], p=vals["connection_prob"]
            ).get_values(
                self._analyze_model.pop_init_parameter_dict[pop_name]["geometry"][0]
            )
            setattr(pop_aux, f"number_synapses_{proj_name}", number_synapses)
        ### create the "current injection" projection from conductance calculating
        ### population to the reduced post population
        proj = CurrentInjection(
            pre=pop_aux,
            post=get_population(f"{pop_name}_reduced"),
            target=f"incomingaux{target}",
            name=f"proj_{pop_name}_{target}_aux",
        )
        proj.connect_current()
        ### create projection from spike_prob calculating aux neurons of presynaptic
        ### populations to conductance calculating aux population
        for proj_name, vals in projection_dict.items():
            pre_pop_name = vals["pre_name"]
            pre_pop_spike_collecting_aux = get_population(
                f"{pre_pop_name}_spike_collecting_aux"
            )
            proj = Projection(
                pre=pre_pop_spike_collecting_aux,
                post=pop_aux,
                target=f"spikeprob_{pre_pop_name}",
                name=f"{proj_name}_spike_collecting_to_conductance",
            )
            proj.connect_all_to_all(weights=1)

    def how_pop_is_connected(self, pop_name):
        """
        Check how a population is connected. If the population is a postsynaptic and/or
        presynaptic population, check if it gets ampa and/or gaba input.

        Args:
            pop_name (str):
                Name of the population to check

        Returns:
            is_presynaptic (bool):
                True if the population is a presynaptic population, False otherwise
            is_postsynaptic (bool):
                True if the population is a postsynaptic population, False otherwise
            ampa (bool):
                True if the population gets ampa input, False otherwise
            gaba (bool):
                True if the population gets gaba input, False otherwise
        """
        is_presynaptic = False
        is_postsynaptic = False
        ampa = False
        gaba = False
        ### loop over all projections
        for proj_name in self._model.projections:
            ### check if the population is a presynaptic population in any projection
            if self._analyze_model.pre_post_pop_name_dict[proj_name][0] == pop_name:
                is_presynaptic = True
            ### check if the population is a postsynaptic population in any projection
            if self._analyze_model.pre_post_pop_name_dict[proj_name][1] == pop_name:
                is_postsynaptic = True
                ### check if the projection target is ampa or gaba
                if (
                    self._analyze_model.proj_init_parameter_dict[proj_name]["target"]
                    == "ampa"
                ):
                    ampa = True
                elif (
                    self._analyze_model.proj_init_parameter_dict[proj_name]["target"]
                    == "gaba"
                ):
                    gaba = True

        return is_presynaptic, is_postsynaptic, ampa, gaba

    def adjust_neuron_model(
        self, neuron_model_init_parameter_dict, ampa=True, gaba=True
    ):
        """
        Add the new synaptic input coming from the auxillary population to the neuron
        model.

        Args:
            neuron_model_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Neuron
            ampa (bool):
                True if the population gets ampa input and therefore the ampa conductance
                needs to be adjusted, False otherwise
            gaba (bool):
                True if the population gets gaba input and therefore the gaba conductance
                needs to be adjusted, False otherwise

        Returns:
            neuron_model_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Neuron
                with DBS mechanisms added
        """
        ### 1st adjust the conductance equations
        ### get the equations of the neuron model as a list of strings
        equations_line_split_list = str(
            neuron_model_init_parameter_dict["equations"]
        ).splitlines()
        ### search for equation with dg_ampa/dt and dg_gaba/dt
        for line_idx, line in enumerate(equations_line_split_list):
            if (
                self.get_line_is_dvardt(line, var_name="g_ampa", tau_name="tau_ampa")
                and ampa
            ):
                ### add " + tau_ampa*g_incomingauxampa/dt"
                ### TODO add model requirements somewhere, here requirements = tau_ampa * dg_ampa/dt = -g_ampa
                equations_line_split_list[line_idx] = (
                    "tau_ampa*dg_ampa/dt = -g_ampa + tau_ampa*g_incomingauxampa/dt"
                )
            if (
                self.get_line_is_dvardt(line, var_name="g_gaba", tau_name="tau_gaba")
                and gaba
            ):
                ### add " + tau_gaba*g_incomingauxgaba/dt"
                ### TODO add model requirements somewhere, here requirements = tau_gaba * dg_gaba/dt = -g_gaba
                equations_line_split_list[line_idx] = (
                    "tau_gaba*dg_gaba/dt = -g_gaba + tau_gaba*g_incomingauxgaba/dt"
                )
        ### join list to a string
        neuron_model_init_parameter_dict["equations"] = "\n".join(
            equations_line_split_list
        )

        ### 2nd extend description
        neuron_model_init_parameter_dict["description"] = (
            f"{neuron_model_init_parameter_dict['description']}\nWith incoming auxillary population input implemented."
        )

        return neuron_model_init_parameter_dict

    def get_line_is_dvardt(self, line: str, var_name: str, tau_name: str):
        """
        Check if a equation string has the form "tau*dvar/dt = -var".

        Args:
            line (str):
                Equation string
            var_name (str):
                Name of the variable
            tau_name (str):
                Name of the time constant

        Returns:
            is_solution_correct (bool):
                True if the equation is as expected, False otherwise
        """
        if var_name not in line:
            return False

        # Define the variables
        var, _, _, _ = sp.symbols(f"{var_name} d{var_name} dt {tau_name}")

        # Given equation as a string
        equation_str = line

        # Parse the equation string
        lhs, rhs = equation_str.split("=")
        lhs = sp.sympify(lhs)
        rhs = sp.sympify(rhs)

        # Form the equation
        equation = sp.Eq(lhs, rhs)

        # Solve the equation for var
        try:
            solution = sp.solve(equation, var)
        except:
            ### equation is not solvable with variables means it is not as expected
            return False

        # Given solution to compare
        expected_solution_str = f"-{tau_name}*d{var_name}/dt"
        expected_solution = sp.sympify(expected_solution_str)

        # Check if the solution is as expected
        is_solution_correct = solution[0] == expected_solution

        return is_solution_correct

    class SpikeProbCalcNeuron(Neuron):
        """
        Neuron model to calculate the spike probabilities of the presynaptic neurons.
        """

        def __init__(self, reduced_size=1):
            """
            Args:
                reduced_size (int):
                    Reduced size of the associated presynaptic population
            """
            parameters = f"""
                reduced_size = {reduced_size} : population
                tau= 1.0 : population
            """
            equations = """
                tau*dr/dt = g_ampa/reduced_size - r
                g_ampa = 0
            """
            super().__init__(parameters=parameters, equations=equations)

    class InputCalcNeuron(Neuron):
        """
        This neurons gets the spike probabilities of the pre neurons and calculates the
        incoming spikes for each projection. It accumulates the incoming spikes of all
        projections (of the same target type) and calculates the conductance increase
        for the post neuron.
        """

        def __init__(self, projection_dict):
            """
            Args:
                projection_dict (dict):
                    keys: names of afferent projections (of the same target type)
                    values: dict with keys "weights", "pre_name"
            """

            ### create parameters
            parameters = [
                f"""
                number_synapses_{proj_name} = 0
                weights_{proj_name} = {vals['weights']}
            """
                for proj_name, vals in projection_dict.items()
            ]
            parameters = "\n".join(parameters)

            ### create equations
            equations = [
                f"""
                incoming_spikes_{proj_name} = number_synapses_{proj_name} * sum(spikeprob_{vals['pre_name']}) + Normal(0, 1)*sqrt(number_synapses_{proj_name} * sum(spikeprob_{vals['pre_name']}) * (1 - sum(spikeprob_{vals['pre_name']}))) : min=0, max=number_synapses_{proj_name}
            """
                for proj_name, vals in projection_dict.items()
            ]
            equations = "\n".join(equations)
            sum_of_conductance_increase = (
                "r = "
                + "".join(
                    [
                        f"incoming_spikes_{proj_name} * weights_{proj_name} + "
                        for proj_name in projection_dict.keys()
                    ]
                )[:-3]
            )
            equations = equations + "\n" + sum_of_conductance_increase

            super().__init__(parameters=parameters, equations=equations)


class PreparePSP:
    """
    Find v_rest, corresponding I_hold (in case of self-active neurons) and an
    init_sampler to initialize the neuron model for the PSP calculation for each
    population.
    """

    def __init__(
        self,
        simulator: Simulator,
        model: CompNeuroModel,
        single_nets: CreateSingleNeuronNetworks,
        do_not_config_list: list[str],
        logger: Logger,
        do_plot: bool,
    ):

        for pop_name in model.populations:
            ### skip populations which should not be configured
            if pop_name in do_not_config_list:
                continue
            ### find initial v_rest
            logger.log(
                f"search v_rest with y(X) = delta_v_2000(v=X) using grid search for pop {pop_name}"
            )
            v_rest, delta_v_v_rest, variables_v_rest = self.find_v_rest_initial(
                pop_name=pop_name,
                simulator=simulator,
                single_nets=single_nets,
                do_plot=do_plot,
            )
            logger.log(
                f"for {pop_name} found v_rest={v_rest} with delta_v_2000(v=v_rest)={delta_v_v_rest}"
            )
            ### check if v is constant after setting v to v_rest
            v_rest_is_constant, v_rest_arr = self.get_v_rest_is_const()

            if v_rest_is_constant:
                ### v_rest found, no I_app_hold needed
                v_rest = v_rest_arr[-1]
                I_app_hold = 0
            else:
                ### there is no resting_state i.e. neuron is self-active --> find
                ### smallest negative I_app to silence neuron
                logger.log(
                    "neuron seems to be self-active --> find smallest I_app to silence the neuron"
                )
                v_rest, I_app_hold = self.find_I_app_hold()
            logger.log(f"I_app_hold = {I_app_hold}, resulting v_rest = {v_rest}")

            ### get the sampler for the initial variables
            variable_init_sampler = self.get_init_neuron_variables_for_psp(
                net=self.net_single_dict[pop_name]["net"],
                pop=self.net_single_dict[pop_name]["population"],
                v_rest=v_rest,
                I_app_hold=I_app_hold,
            )

            return {
                "v_rest": v_rest,
                "I_app_hold": I_app_hold,
                "variable_init_sampler": variable_init_sampler,
            }

    def find_I_app_hold(self):
        # TODO continue
        ### negative current initially reduces v
        ### then v climbs back up
        ### check if the second half of v is constant if yes fine if not increase negative I_app
        ### find I_app_hold with incremental_continuous_bound_search
        self.log("search I_app_hold with y(X) = CHANGE_OF_V(I_app=X)")
        I_app_hold = -self.incremental_continuous_bound_search(
            y_X=lambda X_val: self.get_v_rest_arr_const(
                pop_name=pop_name,
                obtained_variables=obtained_variables,
                I_app=-X_val,
            ),
            y_bound=0,
            X_0=0,
            y_0=self.get_v_rest_arr_const(
                pop_name=pop_name,
                obtained_variables=obtained_variables,
                I_app=0,
            ),
            X_increase=detla_v_rest,
            accept_non_dicontinuity=True,
            bound_type="greater",
        )
        ### again simulate the neuron with the obtained I_app_hold to get the new v_rest
        v_rest_arr = self.get_new_v_rest_2000(
            pop_name, obtained_variables, I_app=I_app_hold
        )
        v_rest = v_rest_arr[-1]

    def find_v_rest_initial(
        self,
        pop_name: str,
        simulator: Simulator,
        single_nets: CreateSingleNeuronNetworks,
        do_plot: bool,
    ):
        """
        Find the initial v_rest with the voltage clamp single neuron network for the
        given population. Furthermore, get the change of v durign setting v_rest and the
        stady state variables of the neuron.

        Args:
            pop_name (str):
                Name of the population
            simulator (Simulator):
                Simulator object
            single_nets (CreateSingleNeuronNetworks):
                Single neuron networks
            do_plot (bool):
                True if plots should be created, False otherwise

        Returns:
            v_rest (float):
                Resting membrane potential
            detla_v_v_rest (float):
                Change of the membrane potential during setting v_rest as membrane
                potential
            variables_v_rest (dict):
                Stady state variables of the neuron during setting v_rest as membrane
                potential
        """
        ### find v where dv/dt is minimal with voltage clamp network (best = 0, it can
        ### only be >= 0)
        v_arr = np.linspace(-90, -20, 200)
        v_clamp_arr = np.array(
            [simulator.get_v_clamp_2000(pop_name=pop_name, v=v_val) for v_val in v_arr]
        )
        v_clamp_min_idx = argrelmin(v_clamp_arr)[0]
        v_rest = np.min(v_arr[v_clamp_min_idx])
        if do_plot:
            plt.figure()
            plt.plot(v_arr, v_clamp_arr)
            plt.axvline(v_rest, color="k")
            plt.axhline(0, color="k", ls="dashed")
            plt.savefig(f"v_clamp_{pop_name}.png")
            plt.close("all")

        ### do again the simulation only with the obtained v_rest to get the detla_v for
        ### v_rest
        detla_v_v_rest = simulator.get_v_clamp_2000(pop_name=pop_name, v=v_rest) * dt()
        population = single_nets.single_net_dict(
            pop_name=pop_name, mode="v_clamp"
        ).population
        ### and the stady state variables of the neuron
        variables_v_rest = {
            var_name: getattr(population, var_name) for var_name in population.variables
        }
        return v_rest, detla_v_v_rest, variables_v_rest

    def get_v_rest_is_const(
        self, simulator: Simulator, pop_name: str, variables_v_rest: dict, do_plot=bool
    ):
        """
        Check if the membrane potential is constant after setting it to v_rest.

        Returns:
            v_rest_is_constant (bool):
                True if the membrane potential is constant, False otherwise
            v_rest_arr (np.array):
                Membrane potential for the 2000 ms simulation with shape: (time_steps,)
        """
        ### check if the neuron stays at v_rest with normal neuron
        v_rest_arr = simulator.get_v_2000(
            pop_name=pop_name,
            initial_variables=variables_v_rest,
            I_app=0,
            do_plot=do_plot,
        )
        v_rest_arr_is_const = (
            np.std(v_rest_arr) <= np.mean(np.absolute(v_rest_arr)) / 1000
        )
        return v_rest_arr_is_const, v_rest_arr


class GetMaxSyn:
    def __init__(self):
        pass


class GetWeightTemplates:
    def __init__(self):
        pass


class ArrSampler:
    """
    Class to store an array and sample from it.
    """

    def __init__(self, arr: np.ndarray, var_name_list: list[str]) -> None:
        """
        Args:
            arr (np.ndarray)
                array with shape (n_samples, n_variables)
            var_name_list (list[str])
                list of variable names
        """
        self.arr_shape = arr.shape
        self.var_name_list = var_name_list
        ### check values of any variable are constant
        self.is_const = np.std(arr, axis=0) <= np.mean(np.absolute(arr), axis=0) / 1000
        ### for the constant variables only the first value is used
        self.constant_arr = arr[0, self.is_const]
        ### array without the constant variables
        self.not_constant_val_arr = arr[:, np.logical_not(self.is_const)]

    def sample(self, n=1, seed=0):
        """
        Sample n samples from the array.

        Args:
            n (int)
                number of samples to be drawn
            seed (int)
                seed for the random number generator

        Returns:
            ret_arr (np.ndarray)
                array with shape (n, n_variables)
        """
        ### get n random indices along the n_samples axis
        rng = np.random.default_rng(seed=seed)
        random_idx_arr = rng.integers(low=0, high=self.arr_shape[0], size=n)
        ### sample with random idx
        sample_arr = self.not_constant_val_arr[random_idx_arr]
        ### create return array
        ret_arr = np.zeros((n,) + self.arr_shape[1:])
        ### add samples to return array
        ret_arr[:, np.logical_not(self.is_const)] = sample_arr
        ### add constant values to return array
        ret_arr[:, self.is_const] = self.constant_arr

        return ret_arr

    def set_init_variables(self, population: Population):
        """
        Set the initial variables of the given population to the given values.
        """
        variable_init_arr = self.sample(len(population), seed=0)
        var_name_list = self.var_name_list
        for var_name in population.variables:
            if var_name in var_name_list:
                set_val = variable_init_arr[:, var_name_list.index(var_name)]
                setattr(population, var_name, set_val)


class CreateVoltageClampEquations:
    """
    Class to create voltage clamp equations from the given equations of a neuron model.
    The equations of the neuron model have to contain the voltage change equation in the
    form of ... dv/dt ... = ...

    Attributes:
        eq_new (list[str])
            new equations of the neuron model with the voltage clamp
    """

    def __ini__(self, eq: list[str], neuron_model_attributes_name_list: list[str]):
        """
        Args:
            eq (list[str])
                equations of the neuron model
            neuron_model_attributes_name_list (list[str])
                list of the names of the attributes of the neuron model
        """
        ### get the dv/dt equation from equations
        eq_v, eq_v_idx = self.get_eq_v(eq=eq)

        ### prepare the equation string for solving
        ### TODO replace random distributions and mathematical expressions which may be on the left side
        eq_v, tags = self.prepare_eq_v(eq_v=eq_v)

        ### solve equation to delta_v (which is dv/dt)
        result = self.solve_delta_v(eq_v, neuron_model_attributes_name_list)

        ### insert the new equation lines for v_clamp and remove the old dv/dt line
        self.eq_new = self.replace_delta_v(
            result=result, eq=eq, eq_v_idx=eq_v_idx, tags=tags
        )

    def replace_delta_v(
        self, result: str, eq: list[str], eq_v_idx: int, tags: str = None
    ):
        """
        Replace the dv/dt line with the voltage clamp lines.

        Args:
            result (str)
                right side of the dv/dt equation
            eq (list[str])
                equations of the neuron model
            eq_v_idx (int)
                index of the dv/dt line
            tags (str)
                tags of the dv/dt line

        Returns:
            eq (list[str])
                new equations of the neuron model with the voltage clamp
        """
        ### create the line for recording voltage clamp (right side of dv/dt)
        eq_new_0 = f"v_clamp_rec_sign = {result}"
        ### create the line for the absolute voltage clamp
        eq_new_1 = f"v_clamp_rec = fabs({result})"
        ### create the line for the absolute voltage clamp from the previous time step
        eq_new_2 = "v_clamp_rec_pre = v_clamp_rec"
        ### create the voltage clamp line "dv/dt=0" with tags if they exist
        if not isinstance(tags, type(None)):
            eq_new_3 = f"dv/dt=0 : {tags}"
        else:
            eq_new_3 = "dv/dt=0"
        ### remove old v line and insert new three lines, order is important
        del eq[eq_v_idx]
        eq.insert(eq_v_idx, eq_new_0)
        eq.insert(eq_v_idx, eq_new_1)
        eq.insert(eq_v_idx, eq_new_2)
        eq.insert(eq_v_idx, eq_new_3)
        ### return new neuron equations
        return eq

    def get_line_is_v(self, line: str):
        """
        Check if the line contains the definition of dv/dt.

        Args:
            line (str)
                line of the equations of the neuron model

        Returns:
            line_is_v (bool)
                True if the line contains the definition of dv/dt, False otherwise
        """
        if "v" not in line:
            return False

        ### remove whitespaces
        line = line.replace(" ", "")

        ### check for dv/dt
        if "dv/dt" in line:
            return True

        return False

    def get_eq_v(self, eq: list[str]):
        """
        Get the dv/dt equation from the equations of the neuron model.

        Args:
            eq (list[str])
                equations of the neuron model

        Returns:
            eq_v (str)
                dv/dt equation
            eq_v_idx (int)
                index of the dv/dt line
        """
        ### get the dv/dt equation from equations
        ### find the line with dv/dt= or v+= or v=
        line_is_v_list = [False] * len(eq)
        ### check in which lines v is defined
        for line_idx, line in enumerate(eq):
            line_is_v_list[line_idx] = self.get_line_is_v(line)
        ### raise error if no v or multiple times v
        if True not in line_is_v_list or sum(line_is_v_list) > 1:
            raise ValueError(
                "In the equations of the neurons has to be exactly a single line which defines dv/dt!"
            )
        ### get the index of the line with dv/dt
        eq_v_idx = line_is_v_list.index(True)
        ### set the v equation
        eq_v = eq.copy()[eq_v_idx]
        return eq_v, eq_v_idx

    def prepare_eq_v(self, eq_v: str):
        """
        Prepare the equation string for solving with sympy.

        Args:
            eq_v (str)
                dv/dt equation

        Returns:
            eq_v (str)
                dv/dt equation
            tags (str)
                tags of the dv/dt equation
        """
        ### remove whitespaces
        eq_v = eq_v.replace(" ", "")
        ### replace dv/dt by delta_v
        eq_v = eq_v.replace("dv/dt", "delta_v")
        ### separate equation and tags
        eq_tags_list = eq_v.split(":")
        eq_v = eq_tags_list[0]
        if len(eq_tags_list) > 1:
            tags = eq_tags_list[1]
        else:
            tags = None
        return eq_v, tags

    def solve_delta_v(self, eq_v: str, neuron_model_attributes_name_list: list[str]):
        """
        Solve the dv/dt equation for delta_v (which is dv/dt).

        Args:
            eq_v (str)
                dv/dt equation
            neuron_model_attributes_name_list (list[str])
                list of the names of the attributes of the neuron model

        Returns:
            solution_str (str)
                right side of the dv/dt equation
        """
        ### Define the attributes of the neuron model as sympy symbols
        sp.symbols(",".join(neuron_model_attributes_name_list))
        ### Define delta_v and right_side as sympy symbols
        delta_v, _ = sp.symbols("delta_v right_side")

        ### Parse the equation string
        lhs, rhs_string = eq_v.split("=")
        lhs = sp.sympify(lhs)
        rhs = sp.sympify(rhs_string)

        ### Form the equation
        equation = sp.Eq(lhs, rhs)

        ### Solve the equation for delta_v
        try:
            solution = sp.solve(equation, delta_v)[0]
        except:
            raise ValueError("Could not find solution for dv/dt!")

        ### Get the solution as a string
        solution_str = str(solution)

        ### replace right_side by the actual right side string
        solution_str = solution_str.replace("right_side", f"({rhs_string})")

        return solution_str
