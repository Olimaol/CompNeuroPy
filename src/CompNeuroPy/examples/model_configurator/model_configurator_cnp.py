from CompNeuroPy.generate_model import CompNeuroModel
from CompNeuroPy import model_functions as mf

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


class ModelConfigurator:
    def __init__(
        self,
        model: CompNeuroModel,
        target_firing_rate_dict: dict,
        max_psp: float = 10.0,
        do_not_config_list: list[str] = [],
        print_guide: bool = False,
        I_app_variable: str = "I_app",
    ):
        self._analyze_model = AnalyzeModel(model=model)
        self._single_neuron_networks = CreateSingleNeuronNetworks()
        self._reduced_model = CreateReducedModel()
        self._v_rest = GetVRest()
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
        self._clear_model(model=model)

        ### get population info (eq, params etc.)
        self._analyze_populations(model=model)

        ### get projection info
        self._analyze_projections(model=model)

    def _clear_model(self, model: CompNeuroModel):
        mf.cnp_clear(functions=False, neurons=True, synapses=True, constants=False)
        model.create(do_compile=False)

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
            ### get the afferent projections dict of the population TODO
            self.afferent_projection_dict[pop_name] = (
                self._get_afferent_projection_dict(pop_name=pop_name)
            )

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
    def __init__(self):
        pass


class CreateReducedModel:
    def __init__(self):
        pass


class GetVRest:
    def __init__(self):
        pass


class GetMaxSyn:
    def __init__(self):
        pass


class GetWeightTemplates:
    def __init__(self):
        pass
